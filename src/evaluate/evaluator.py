import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import compute_metrics, parse_response

logger = logging.getLogger(__name__)

class CoinEvaluator:
    def __init__(self, config, model=None, processor=None, checkpoint_path=None, load_base=False):
        self.config = config

        if model is not None and processor is not None:
            # Caller (e.g. training callback) owns this processor object; do NOT
            # mutate its image_processor here or we may silently change the
            # resolution used by the running training collator.
            self.model = model
            self.processor = processor
        elif checkpoint_path is not None:
            self.model, self.processor = self._load_from_checkpoint(checkpoint_path)
            self._apply_processor_overrides(self.processor)
        elif load_base:
            self.model, self.processor = self._load_base_model()
            self._apply_processor_overrides(self.processor)
        else:
            raise ValueError(
                "Must provide either (model + processor), checkpoint_path, or load_base=True"
            )

        self.device = next(self.model.parameters()).device

    def _apply_processor_overrides(self, processor) -> None:
        """Ensure eval-time image resolution matches the configured processor
        (config.processor or config.model.processor). Without this, a
        processor loaded from an adapter/checkpoint directory may carry stale
        min_pixels/max_pixels saved at training time, silently downsampling
        and hurting fine-grained mint-mark recognition.
        """
        if processor is None:
            return
        processor_config = self.config.get("processor", None)
        if processor_config is None:
            processor_config = self.config.model.get("processor", None)
        if processor_config is None:
            return
        try:
            processor.image_processor.min_pixels = processor_config.min_pixels * 28 * 28
            processor.image_processor.max_pixels = processor_config.max_pixels * 28 * 28
            logger.info(
                "Eval processor overrides applied: min_pixels=%d, max_pixels=%d",
                processor_config.min_pixels,
                processor_config.max_pixels,
            )
        except AttributeError:
            logger.warning("Processor has no image_processor; skipping resolution override.")

    def evaluate(self, dataset) -> dict:
        logger.info("Running evaluation on %d samples...", len(dataset))

        dataloader = DataLoader(
            dataset,
            batch_size = self.config.training.per_device_eval_batch_size,
            shuffle = False,
            num_workers = self.config.training.dataloader_num_workers,
            collate_fn = self._eval_collate_fn,
        )

        all_predictions = []
        all_references = []
        all_details = []

        self.model.eval()
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Evaluating"):
                responses = self._generate(batch["images"], batch["texts"])

                for response, gold in zip(responses, batch["ground_truths"]):
                    all_predictions.append(response)
                    all_references.append(gold)
                    all_details.append({
                        "ground_truth": gold,
                        "raw_response": response,
                        "parsed": parse_response(response),
                    })

        metrics = compute_metrics(all_predictions, all_references)
        self._log_metrics(metrics)

        return {
            "metrics": metrics,
            "predictions": all_details,
        }

    def save_results(self, results: dict, output_dir: str) -> Path:
        """
        Lưu metrics + predictions vào outputs/results/.
        File name: eval_{timestamp}.json
        """
        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"eval_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to: %s", output_path)
        return output_path


    def _load_base_model(self) -> tuple:
        """Load the un-finetuned base model via the configured backend loader.

        Mirrors training's loading path (unsloth/hf_peft/full_finetune) so that
        baseline eval runs at the same VRAM footprint as training.
        """
        from src.model.factory import get_model_loader

        loader = get_model_loader(self.config)
        model, processor = loader.load_for_inference()
        model.eval()
        return model, processor


    def _load_from_checkpoint(self, checkpoint_path: str) -> tuple:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Loading model from checkpoint: %s", checkpoint_path)

        is_lora = (checkpoint_path / "adapter_config.json").exists()

        if is_lora:
            logger.info(
                "Detected LoRA adapter. Loading base via [%s] loader + attaching adapter...",
                self.config.model.backend.upper(),
            )
            from peft import PeftModel
            from src.model.factory import get_model_loader

            loader = get_model_loader(self.config)
            base_model, processor = loader.load_for_inference()

            model = PeftModel.from_pretrained(base_model, str(checkpoint_path))

            # Unsloth fast-inference path expects the merged/wrapped model too.
            if self.config.model.backend == "unsloth":
                from unsloth import FastVisionModel
                FastVisionModel.for_inference(model)

            # Prefer processor from the adapter dir if it has one (it may carry
            # tokenizer changes saved during training); fall back to loader's.
            try:
                processor = AutoProcessor.from_pretrained(checkpoint_path)
            except Exception:
                logger.info("No processor in adapter dir, keeping loader's processor.")
        else:
            # Full model checkpoint (no adapter)
            logger.info("Loading full model from checkpoint...")
            processor = AutoProcessor.from_pretrained(checkpoint_path)
            model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        model.eval()
        return model, processor


    def _generate(self, images: list, texts: list[str]) -> list[str]:
        generation_config = self.config.get("generation", {})

        # Decoder-only models need left padding for correct generation.
        # Use try/finally so an exception in generate() doesn't leak
        # padding_side="left" into subsequent training batches.
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        try:
            inputs = self.processor(
                images = images,
                text = texts,
                return_tensors = "pt",
                padding = True,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens = generation_config.get("max_new_tokens", 256),
                do_sample = generation_config.get("do_sample", False),
                temperature = generation_config.get("temperature", 1.0),
                top_p = generation_config.get("top_p", 1.0),
                pad_token_id = self.processor.tokenizer.pad_token_id,
                eos_token_id = self.processor.tokenizer.eos_token_id,
            )

            input_len = inputs["input_ids"].shape[1]
            return self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
            )
        finally:
            # Restore original padding side for training, even on exception.
            self.processor.tokenizer.padding_side = original_padding_side

    def _eval_collate_fn(self, batch: list[dict]) -> dict:

        images = [item["image"] for item in batch]
        ground_truths = [
            {
                "year": item["label"]["year"],
                "mint_mark": item["label"]["mint_mark"],
            }
            for item in batch
        ]

        texts = [
            self.processor.apply_chat_template(
                self._build_eval_messages(),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for _ in batch
        ]

        return {
            "images": images,
            "texts": texts,
            "ground_truths": ground_truths,
        }

    def _build_eval_messages(self) -> list[dict]:
        prompt = self._resolve_prompt()
        return [
            {
                "role": "system",
                "content": prompt.system,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt.user},
                ],
            },
        ]

    def _resolve_prompt(self):
        """Eval mirrors inference: prefer the inference prompt
        (config.prompt from inference.yaml), fall back to model.prompt
        if not set. The training prompt is for updating LoRA params; at
        eval/inference we test the model under deployment conditions.
        """
        inference_prompt = self.config.get("prompt", None)
        if inference_prompt is not None:
            return inference_prompt

        model_prompt = self.config.model.get("prompt", None)
        if model_prompt is not None:
            logger.info("No inference prompt set, falling back to model.prompt.")
            return model_prompt

        raise ValueError(
            "Prompt not found. Set `prompt:` in inference.yaml or `model.prompt` in model config."
        )

    def _log_metrics(self, metrics: dict) -> None:
        logger.info("=" * 50)
        logger.info("Evaluation Results")
        logger.info("=" * 50)
        logger.info("Samples: %d", metrics["n_samples"])
        logger.info("Parse errors: %d (%.1f%%)",
                    metrics["n_parse_errors"],
                    metrics["parse_error_rate"] * 100)
        logger.info("Extract match:  %.2f%%", metrics["extract_match"] * 100)
        logger.info("Year accuracy: %.2f%%", metrics["year_accuracy"] * 100)
        logger.info("Mint mark accuracy: %.2f%%", metrics["mint_mark_accuracy"] * 100)
        logger.info("Confusion matrix (mint_mark):")
        for gold, preds in metrics["confusion_matrix"].items():
            logger.info("  gold=%-5s %s", gold, preds)
        logger.info("=" * 50)

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "eval/extract_match": metrics["extract_match"],
                    "eval/year_accuracy": metrics["year_accuracy"],
                    "eval/mint_mark_accuracy": metrics["mint_mark_accuracy"],
                    "eval/parse_error_rate": metrics["parse_error_rate"],
                })
                rows = [
                    [gold, pred, count]
                    for gold, preds in metrics["confusion_matrix"].items()
                    for pred, count in preds.items()
                ]
                wandb.log({
                    "eval/confusion_matrix": wandb.Table(
                        columns=["gold", "pred", "count"],
                        data=rows,
                    )
                })
        except ImportError:
            pass