"""Export logic for fine-tuned coin VLMs: merge LoRA adapters into a standalone
checkpoint, and convert merged checkpoints to GGUF.

Merge notes (QLoRA + Unsloth):
 - Adapters trained with QLoRA (load_in_4bit=True) must be merged into the base
   loaded at 16-bit, NOT 4-bit — merging into a 4-bit base degrades accuracy.
 - The base is resolved from the adapter's own adapter_config.json
   (`base_model_name_or_path`) so we merge into the SAME weights used to train.
 - For Unsloth-trained adapters (which also touch the vision projector
   `visual_merger.mlp.*`), everything is baked into the weights here so vLLM can
   serve a plain merged model without applying LoRA at runtime.
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


def _save_processor(adapter_path: Path, output_dir: Path, fallback=None) -> None:
    """Ensure the merged dir carries the processor (image processor + tokenizer
    + chat template) so vLLM can serve multimodal requests."""
    from transformers import AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained(str(adapter_path))
    except Exception:
        proc = fallback
    if proc is not None:
        proc.save_pretrained(str(output_dir))


def _merge_unsloth(base_model: str, adapter_path: Path, output_dir: Path) -> None:
    # The base lives locally; force offline so Unsloth's telemetry/statistics
    # call to HuggingFace fails fast instead of hanging 120s when HF is
    # unreachable (which otherwise aborts the whole merge).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

    from unsloth import FastVisionModel

    # Load the BASE via Unsloth (so the model keeps Unsloth's
    # `save_pretrained_merged`), then attach the trained adapter IN PLACE with
    # load_adapter. We do NOT load straight from the adapter folder because its
    # adapter_config bakes in the stale training-time base path; and we do NOT
    # wrap with a vanilla PeftModel because that drops save_pretrained_merged
    # (forcing a fallback that saves a 4-bit checkpoint).
    logger.info("Loading base '%s' via Unsloth...", base_model)
    model, processor = FastVisionModel.from_pretrained(
        base_model,
        load_in_4bit=False,          # request 16-bit; merged_16bit also dequantizes
        use_gradient_checkpointing=False,
        local_files_only=True,
    )

    if not hasattr(model, "save_pretrained_merged"):
        raise RuntimeError(
            "Unsloth model has no save_pretrained_merged(); cannot guarantee a "
            "16-bit merge. Check the Unsloth version."
        )

    logger.info("Attaching LoRA adapter from '%s' (in place)...", adapter_path)
    model.load_adapter(str(adapter_path))

    logger.info("Merging + dequantizing to 16-bit (save_method='merged_16bit')...")
    model.save_pretrained_merged(
        str(output_dir),
        processor,
        save_method="merged_16bit",
    )


def _merge_hf(base_model: str, adapter_path: Path, output_dir: Path) -> None:
    import torch
    from transformers import AutoModelForImageTextToText
    from peft import PeftModel

    logger.info("Loading base '%s' at bf16 via transformers...", base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Applying LoRA adapter from '%s'...", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path))

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir), safe_serialization=True)
    _save_processor(adapter_path, output_dir)


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
    backend = config.model.get("backend", None)

    if backend == "unsloth":
        _merge_unsloth(base_model, adapter_path, output_dir)
    else:
        _merge_hf(base_model, adapter_path, output_dir)

    logger.info("Merge complete. Merged model saved to: %s", output_dir)
    logger.info(
        "Serve with vLLM, e.g.:\n"
        "  vllm serve %s --trust-remote-code --limit-mm-per-prompt image=1",
        output_dir,
    )


class _CalibrationCollator:
    """Wrap CoinDataCollator for AWQ calibration: same prompt/image pipeline as
    training, but drop the `labels` key (calibration only needs forward-pass
    activations, not a loss)."""

    def __init__(self, processor, config):
        from src.data import CoinDataCollator

        self._inner = CoinDataCollator(processor, config)

    def __call__(self, batch):
        out = self._inner(batch)
        out.pop("labels", None)
        return out


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
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    from src.data import CoinDataset

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading merged model for AWQ from '%s'...", model_path)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Building calibration set from split '%s' (%d samples)...",
                calibration_split, num_calibration_samples)
    dataset = CoinDataset(config, split=calibration_split)
    collator = _CalibrationCollator(processor, config)

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
        dataset=dataset,
        data_collator=collator,
        recipe=recipe,
        max_seq_length=max_seq_length or config.model.get("max_seq_length", 2048),
        num_calibration_samples=num_calibration_samples,
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
