"""Export logic for fine-tuned coin VLMs: merge LoRA adapters into a standalone
16-bit checkpoint, optionally quantize to AWQ W4A16, and convert to GGUF.

Merge notes (QLoRA on a bnb 4-bit base):
 - The QLoRA base is a bnb 4-bit (NF4) checkpoint. You cannot merge a LoRA into
   4-bit weights, so the base is loaded 4-bit and dequantized to bf16 first; the
   adapter (trained against those exact NF4 weights) is then merged in.
 - Merging uses plain transformers + peft.merge_and_unload(). Unsloth's
   save_pretrained_merged did NOT merge on the installed version (it re-saved
   only the adapter), so it is not used.
 - The base path is resolved from the model config (`unsloth_name`), not the
   adapter_config, whose baked-in path is stale after the base is moved.
 - The vision projector (`visual_merger.mlp.*`) is trained too; everything is
   baked into the weights so vLLM can serve a plain model without runtime LoRA.

AWQ notes:
 - Quantize only the language-model Linear layers; keep the vision tower/merger
   at full precision. Load the merged model in bf16 (fp16 overflows during AWQ
   smoothing -> NaN). Calibration reuses the project's CoinDataset.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


def _resolve_base_model(config, adapter_path: Path, override: str | None) -> str:
    """Pick the base model to merge into.

    Priority: explicit override > model config (unsloth_name, then name) >
    adapter_config's base_model_name_or_path. The model config is preferred over
    adapter_config because the latter bakes in the absolute path from TRAINING
    time, which is often stale after the checkpoint/base is moved to another
    machine; the model config is what the user keeps current.
    """
    if override:
        return override

    config_base = config.model.get("unsloth_name", None) or config.model.get("name", None)
    if config_base and Path(config_base).exists():
        return config_base

    cfg_file = adapter_path / "adapter_config.json"
    if cfg_file.exists():
        with cfg_file.open() as f:
            adapter_base = json.load(f).get("base_model_name_or_path")
        if adapter_base:
            if config_base and not Path(config_base).exists():
                logger.warning(
                    "Model-config base '%s' not found on disk; "
                    "falling back to adapter_config base '%s'.",
                    config_base, adapter_base,
                )
            return adapter_base

    if config_base:
        return config_base
    raise ValueError(
        "Could not resolve a base model: no --base_model, no model.unsloth_name/"
        "name, and no base_model_name_or_path in adapter_config.json."
    )


def _save_processor(adapter_path: Path, output_dir: Path, base_model: str | None = None) -> None:
    """Ensure the merged dir carries the processor (image processor + tokenizer
    + chat template) so vLLM can serve multimodal requests. Prefer the adapter
    checkpoint (carries the trained tokenizer/template), fall back to the base."""
    from transformers import AutoProcessor

    proc = None
    for src in (str(adapter_path), base_model):
        if not src:
            continue
        try:
            proc = AutoProcessor.from_pretrained(src, trust_remote_code=True, local_files_only=True)
            break
        except Exception:
            continue
    if proc is not None:
        proc.save_pretrained(str(output_dir))
    else:
        logger.warning("Could not load a processor from adapter or base; "
                       "merged dir may be missing tokenizer/processor files.")


def _sanitize_config_for_json(cfg) -> None:
    """Recursively make a (possibly composite) HF config JSON-serializable after
    dequantization: drop bnb quantization metadata and convert any leftover
    torch.dtype attributes (e.g. `dtype`, `_pre_quantization_dtype`) to strings.
    Nested sub-configs (text_config, vision_config, ...) are handled too."""
    import torch

    if cfg is None:
        return
    for attr in ("quantization_config", "_pre_quantization_dtype"):
        if hasattr(cfg, attr):
            try:
                delattr(cfg, attr)
            except Exception:
                setattr(cfg, attr, None)

    for key, val in list(vars(cfg).items()):
        if key in ("dtype", "torch_dtype"):
            # We dequantized to bf16; force bf16 so the config matches the actual
            # weights (the bnb base often advertised float16, which would make
            # vLLM serve this bf16-native model in fp16).
            setattr(cfg, key, "bfloat16")
        elif isinstance(val, torch.dtype):
            setattr(cfg, key, str(val).split(".")[-1])  # torch.bfloat16 -> "bfloat16"
        elif val.__class__.__name__.endswith("Config") and hasattr(val, "to_dict"):
            _sanitize_config_for_json(val)


def _merge_hf(base_model: str, adapter_path: Path, output_dir: Path) -> None:
    """Merge with plain transformers + peft. Loads the base at bf16 from its
    LOCAL path, attaches the adapter onto that already-loaded model (so the
    stale base path inside adapter_config is irrelevant), merges, and saves a
    standalone 16-bit checkpoint. This is the reliable path: Unsloth's
    save_pretrained_merged did not actually merge on the installed version."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForImageTextToText
    from peft import PeftModel

    # The base may be a bnb 4-bit (NF4) checkpoint — that is what QLoRA trained
    # on. You cannot merge a LoRA into 4-bit weights, so if the base is
    # quantized we load it 4-bit and dequantize back to bf16 first. Since the
    # adapter was trained against exactly these NF4 weights, dequantizing and
    # merging is the faithful reconstruction of (base + adapter).
    base_cfg = Path(base_model) / "config.json"
    is_quantized = False
    if base_cfg.exists():
        is_quantized = bool(json.loads(base_cfg.read_text()).get("quantization_config"))

    if is_quantized:
        logger.info("Base '%s' is bnb-quantized; loading 4-bit then dequantizing to bf16...", base_model)
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
        model = model.dequantize()      # Linear4bit -> bf16 Linear, drops quantization_config
    else:
        logger.info("Loading base '%s' at bf16 via transformers (local, offline)...", base_model)
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )

    logger.info("Applying LoRA adapter from '%s'...", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)

    logger.info("Merging adapter into base model (16-bit)...")
    model = model.merge_and_unload()

    # After dequantize() the model is plain bf16, but the config still carries
    # bnb quantization metadata and raw torch.dtype objects that are not
    # JSON-serializable (breaking save_pretrained). Sanitize the whole config
    # tree so the saved config reflects a clean 16-bit model.
    _sanitize_config_for_json(model.config)

    model.save_pretrained(str(output_dir), safe_serialization=True)
    _save_processor(adapter_path, output_dir, base_model=base_model)


def merge_lora(config, adapter_path: str, output_dir: str, base_model: str | None = None) -> None:
    """Merge a LoRA/QLoRA adapter into its base model and save a standalone
    16-bit checkpoint ready for vLLM serving."""
    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (adapter_path / "adapter_config.json").exists():
        raise ValueError(
            f"Could not find adapter_config.json in: {adapter_path}. "
            "Is this a LoRA checkpoint?"
        )

    base_model = _resolve_base_model(config, adapter_path, base_model)

    # Always use the transformers + peft merge. It produces a clean 16-bit
    # checkpoint regardless of training backend; Unsloth's save_pretrained_merged
    # did not actually merge (it re-saved only the adapter) on the installed
    # version, so it is no longer used here.
    _merge_hf(base_model, adapter_path, output_dir)

    logger.info("Merge complete. Merged model saved to: %s", output_dir)
    logger.info(
        "Serve with vLLM, e.g.:\n"
        "  vllm serve %s --trust-remote-code --limit-mm-per-prompt image=1",
        output_dir,
    )


def _build_awq_calibration_dataset(config, processor, num_samples: int, split: str):
    """Build a HuggingFace Dataset of already-processed multimodal inputs for
    AWQ calibration. llmcompressor expects a HF Dataset (with column_names),
    not a torch Dataset, and for vision models the rows must already be
    tokenized (input_ids, attention_mask, pixel_values, image_grid_thw, ...).

    We reuse CoinDataCollator only to build the SAME prompt/messages as training,
    then run the processor per sample and store the tensors as nested lists. A
    batch-size-1 data_collator reconstructs the tensors at calibration time.
    """
    from datasets import Dataset as HFDataset

    from src.data import CoinDataset, CoinDataCollator

    coin = CoinDataset(config, split=split)
    msg_builder = CoinDataCollator(processor, config)  # for _build_messages only
    n = min(num_samples, len(coin))

    rows = []
    for i in range(n):
        item = coin[i]
        messages = msg_builder._build_messages(item["label"])
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        inputs = processor(text=[text], images=[item["image"]], return_tensors="pt")
        rows.append({k: v.tolist() for k, v in inputs.items()})

    logger.info("Built %d calibration samples for AWQ.", len(rows))
    return HFDataset.from_list(rows)


def export_awq(
    config,
    model_path: str,
    output_dir: str,
    num_calibration_samples: int = 256,
    max_seq_length: int | None = None,
    calibration_split: str = "train",
) -> None:
    """Quantize a merged 16-bit checkpoint to AWQ W4A16 for vLLM serving.

    Only the language-model Linear layers are quantized; the vision tower and
    multimodal merger are kept at full precision (quantizing them tends to hurt
    fine-grained reading of digits / mint marks). Calibration reuses the
    project's own CoinDataset so activation statistics match real coin inputs.
    """
    # Dataset + base are cached locally; stay offline so HF lookups fail fast
    # instead of burning ~30s on connect-timeout retries.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading merged model for AWQ from '%s'...", model_path)
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
    )
    # Force bf16: Qwen3-VL is bf16-native and AWQ smoothing overflows in fp16
    # ("No finite loss / NaN in forward pass"). Do NOT rely on config dtype.
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    logger.info("Building calibration set from split '%s' (%d samples)...",
                calibration_split, num_calibration_samples)
    calib_dataset = _build_awq_calibration_dataset(
        config, processor, num_calibration_samples, calibration_split,
    )

    def data_collator(batch):
        # Calibration runs one sample at a time; rebuild tensors from the stored
        # nested lists, preserving each key's original (multimodal) shape.
        assert len(batch) == 1, "AWQ calibration expects batch_size=1"
        return {k: torch.tensor(v) for k, v in batch[0].items()}

    # Keep vision encoder + merger and the output head in full precision.
    # NOTE: if AWQ errors on an unmatched module, inspect model.named_modules()
    # and adjust these ignore patterns (Qwen3-VL vision lives under `*.visual.*`).
    recipe = [
        AWQModifier(
            targets=["Linear"],
            scheme="W4A16",
            ignore=["lm_head", "re:.*visual.*", "re:.*merger.*"],
        )
    ]

    logger.info("Running AWQ oneshot quantization (W4A16)...")
    oneshot(
        model=model,
        dataset=calib_dataset,
        data_collator=data_collator,
        recipe=recipe,
        max_seq_length=max_seq_length or config.model.get("max_seq_length", 2048),
        num_calibration_samples=min(num_calibration_samples, len(calib_dataset)),
        output_dir=str(output_dir),
    )

    # Make sure the processor lands next to the quantized weights for vLLM.
    processor.save_pretrained(str(output_dir))

    logger.info("AWQ export complete. Quantized model saved to: %s", output_dir)
    logger.info(
        "Serve with vLLM, e.g.:\n"
        "  vllm serve %s --trust-remote-code --quantization awq --limit-mm-per-prompt image=1",
        output_dir,
    )


def export_gguf(model_path: str, output_dir: str, quantization: str) -> None:
    """Convert a merged HF checkpoint to GGUF via llama.cpp."""
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gguf_name = f"{model_path.name}-{quantization}.gguf"
    output_path = output_dir / gguf_name

    llamacpp_convert = Path("llama.cpp/convert_hf_to_gguf.py")
    if not llamacpp_convert.exists():
        raise FileNotFoundError(
            "Could not find llama.cpp/convert_hf_to_gguf.py. "
            "Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
        )

    logger.info("Converting '%s' to GGUF (%s)...", model_path, quantization)
    cmd = [
        sys.executable, str(llamacpp_convert),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", quantization.lower(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("GGUF conversion failed:\n%s", result.stderr)
        raise RuntimeError("GGUF conversion failed.")

    logger.info("GGUF export complete. Saved to: %s", output_path)