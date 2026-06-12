from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from src.utils import get_logger
from omegaconf import DictConfig, OmegaConf, open_dict, read_write
from .callbacks import GradNormCallback, MemoryCallback

logger = get_logger(__name__)

class BaseTrainer(ABC):
    def __init__(self, config, model, processor, train_loader, val_loader):
        self.config = config
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Give every run its own folder under the configured checkpoints root so
        # successive trainings don't overwrite each other.
        self._resolve_run_output_dir()

    def _resolve_run_output_dir(self):
        training_cfg = self.config.training
        base_dir = Path(training_cfg.output_dir)
        run_name = training_cfg.get("run_name", "run") or "run"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = base_dir / f"{run_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if OmegaConf.is_config(training_cfg):
            with read_write(training_cfg), open_dict(training_cfg):
                training_cfg.output_dir = str(run_dir)
        else:
            training_cfg.output_dir = str(run_dir)
        logger.info(f"Run output dir: {run_dir}")

    def train(self):
        logger.info("Starting training...")
        args = self._build_args()
        self.trainer = self._build_trainer(args)
        self._maybe_wrap_evaluate_with_generation()

        logger.info("Starting training with args: %s", args)
        result = self._run_training()
        return result

    def _maybe_wrap_evaluate_with_generation(self):
        """Monkey-patch trainer.evaluate to also run generation-based eval and
        inject extract_match/year_accuracy/mint_mark_accuracy into the metrics
        dict that HF Trainer reads for `metric_for_best_model`.

        Why not a TrainerCallback? In Unsloth's compiled SFTTrainer the
        callback dispatch around evaluate() runs AFTER `_determine_best_metric`,
        so adding the metric from `on_evaluate` is too late and HF raises
        KeyError on `eval_extract_match`. Overriding evaluate() at the trainer
        instance guarantees the keys are present before HF reads them.
        """
        gen_eval_cfg = self.config.training.get("generation_eval", None)
        if gen_eval_cfg is None or not gen_eval_cfg.get("enabled", False):
            return

        max_samples = gen_eval_cfg.get("max_samples", None)
        trainer = self.trainer
        config = self.config
        val_dataset = self.val_loader
        processor = self.processor
        original_evaluate = trainer.evaluate

        def evaluate(eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            metrics = original_evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            # Only run generation eval for the standard "eval" pass; skip
            # ad-hoc evaluate() calls with custom prefixes.
            if metric_key_prefix != "eval":
                return metrics

            from torch.utils.data import Subset
            from src.evaluate.evaluator import CoinEvaluator

            ds = val_dataset
            if max_samples is not None and len(ds) > max_samples:
                stride = max(1, len(ds) // max_samples)
                indices = list(range(0, len(ds), stride))[:max_samples]
                ds = Subset(ds, indices)

            model = trainer.model
            was_training = model.training
            try:
                try:
                    from unsloth import FastVisionModel
                    FastVisionModel.for_inference(model)
                except Exception:
                    pass

                evaluator = CoinEvaluator(config, model=model, processor=processor)
                gen_results = evaluator.evaluate(ds)
                gen = gen_results["metrics"]
            except Exception as e:
                # Never let generation eval crash the training loop. Log loudly
                # and return the eval_loss-only metrics; HF will fall back to
                # warning about the missing best-model metric but training
                # continues.
                logger.exception("Generation eval failed at step %d: %s",
                                 trainer.state.global_step, e)
                return metrics
            finally:
                try:
                    from unsloth import FastVisionModel
                    FastVisionModel.for_training(model)
                except Exception:
                    pass
                if was_training:
                    model.train()

            metrics["eval_extract_match"] = gen["extract_match"]
            metrics["eval_mint_mark_accuracy"] = gen["mint_mark_accuracy"]
            metrics["eval_year_accuracy"] = gen["year_accuracy"]
            metrics["eval_parse_error_rate"] = gen["parse_error_rate"]

            logger.info(
                "[GenEval @ step %d on %d samples] exact=%.4f | year=%.4f | mint=%.4f | parse_err=%.4f",
                trainer.state.global_step,
                len(ds),
                gen["extract_match"],
                gen["year_accuracy"],
                gen["mint_mark_accuracy"],
                gen["parse_error_rate"],
            )

            try:
                trainer.log({
                    "eval_extract_match": gen["extract_match"],
                    "eval_mint_mark_accuracy": gen["mint_mark_accuracy"],
                    "eval_year_accuracy": gen["year_accuracy"],
                    "eval_parse_error_rate": gen["parse_error_rate"],
                })
            except Exception:
                pass

            return metrics

        trainer.evaluate = evaluate
        logger.info(
            "Patched trainer.evaluate to add generation metrics (max_samples=%s).",
            max_samples,
        )
    
    def get_model(self):
        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call train() first.")
        return self.trainer.model
    
    def _build_args(self):
        from transformers import TrainingArguments
        training_args = self.config.training
        return TrainingArguments(
            output_dir = training_args.output_dir,
            num_train_epochs = training_args.num_train_epochs,
            per_device_train_batch_size = training_args.per_device_train_batch_size,
            per_device_eval_batch_size = training_args.per_device_eval_batch_size,
            gradient_accumulation_steps= training_args.gradient_accumulation_steps,
            learning_rate= training_args.learning_rate,
            lr_scheduler_type= training_args.lr_scheduler_type,
            warmup_steps= training_args.warmup_steps,
            weight_decay= training_args.weight_decay,
            max_grad_norm= training_args.max_grad_norm,
            bf16= training_args.bf16,
            fp16= training_args.fp16,
            eval_strategy= training_args.eval_strategy,
            eval_steps= training_args.eval_steps,
            save_strategy= training_args.save_strategy,
            save_steps= training_args.save_steps,
            save_total_limit= training_args.save_total_limit,
            load_best_model_at_end= training_args.load_best_model_at_end,
            metric_for_best_model= training_args.metric_for_best_model,
            greater_is_better= training_args.get("greater_is_better", None),
            logging_steps= training_args.logging_steps,
            dataloader_num_workers= training_args.dataloader_num_workers,
            dataloader_pin_memory= training_args.dataloader_pin_memory,
            seed= training_args.seed,
            remove_unused_columns= False, # dataset return image + label dict, no need to remove columns
            report_to= training_args.report_to,
            run_name= training_args.run_name,
        )

    def _run_training(self):
        result = self.trainer.train()
        logger.info(
            "Training complete. "
            "Steps: %d | Train loss: %.4f",
            result.global_step,
            result.training_loss,
        )
        # Save the processor into every remaining checkpoint dir so evaluator
        # can load any of them standalone (weights are already there from Trainer).
        run_dir = Path(self.trainer.args.output_dir)
        for ckpt_dir in sorted(run_dir.glob("checkpoint-*")):
            self.processor.save_pretrained(ckpt_dir)
            logger.info(f"Processor saved into: {ckpt_dir}")

        best_ckpt = getattr(self.trainer.state, "best_model_checkpoint", None)
        logger.info(
            "Best checkpoint: %s | Run dir: %s",
            best_ckpt or "(unknown)",
            run_dir,
        )
        return result

    @abstractmethod
    def _build_trainer(self, args):
        pass
    
    def _get_callbacks(self):
        # Note: generation-based eval is now injected by overriding
        # trainer.evaluate (see _maybe_wrap_evaluate_with_generation) instead
        # of via a TrainerCallback — callbacks fire too late under Unsloth's
        # SFTTrainer wrapper for `metric_for_best_model` to pick them up.
        callbacks = [GradNormCallback(), MemoryCallback()]

        if self.config.training.early_stopping_patience.enabled:
            from transformers import EarlyStoppingCallback
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping_patience.patience,
                    early_stopping_threshold=self.config.training.early_stopping_patience.threshold,
                )
            )
        return callbacks

