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
    def __init__(self, config, model, processor, checkpoint_path=None):
        self.config = config

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        elif checkpoint_path is not None:
            self.model, self.processor = self._load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError(
                "Must provide either (model + processor) or checkpoint_path"
            )

        self.device = next(self.model.parameters()).device

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


    def _load_from_checkpoint(self, checkpoint_path: str) -> tuple:

        from transformers import AutoProcessor, AutoModelForImageTextToText

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Loading model from checkpoint: %s", checkpoint_path)

        processor = AutoProcessor.from_pretrained(checkpoint_path)

        if (checkpoint_path / "adapter_config.json").exists():
            # LoRA adapter
            logger.info("Detected LoRA adapter, loading base model + adapter...")
            import torch
            from peft import PeftModel

            base_model_name = self.config.model.name
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_name,
                torch_dtype = torch.bfloat16,
                device_map  = "auto",
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            # Full model
            logger.info("Loading full model from checkpoint...")
            import torch
            model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_path,
                torch_dtype = torch.bfloat16,
                device_map  = "auto",
            )

        model.eval()
        return model, processor


    def _generate(self, images: list, texts: list[str]) -> list[str]:
        generation_config = self.config.get("generation", {})

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
            )
            for _ in batch
        ]

        return {
            "images": images,
            "texts": texts,
            "ground_truths": ground_truths,
        }

    def _build_eval_messages(self) -> list[dict]:
        prompt = self.config.get("prompt", None)
        if prompt is None:
            prompt = self.config.data.prompt
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

    def _log_metrics(self, metrics: dict) -> None:
        logger.info("=" * 50)
        logger.info("Evaluation Results")
        logger.info("=" * 50)
        logger.info("Samples: %d", metrics["n_samples"])
        logger.info("Parse errors: %d (%.1f%%)",
                    metrics["n_parse_errors"],
                    metrics["parse_error_rate"] * 100)
        logger.info("Exact match:  %.2f%%", metrics["exact_match"] * 100)
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
                    "eval/exact_match": metrics["exact_match"],
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