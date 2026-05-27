import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.utils import get_logger

logger = get_logger(__name__)

class GradNormCallback(TrainerCallback):
    """
    Log gradient norm for trainable parameters only.
    """

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if model is None:
            return

        if state.global_step % args.logging_steps != 0:
            return

        grad_norms = [
            torch.norm(p.grad.detach(), 2)
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]

        if not grad_norms:
            logger.warning(
                "Step %d | No gradients found for trainable parameters.",
                state.global_step,
            )
            return

        total_norm = torch.norm(torch.stack(grad_norms), 2).item()

        logger.info(
            "Step %d | trainable_grad_norm: %.6f",
            state.global_step,
            total_norm,
        )

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {"train/trainable_grad_norm": total_norm},
                    step=state.global_step,
                )
                
        except ImportError:
            pass

class GenerationMetricsCallback(TrainerCallback):
    """Run generation-based eval on (a subsample of) val during training and
    inject task metrics into the eval dict so HF Trainer can use them for
    `metric_for_best_model`.

    Mutates the `metrics` dict passed by HF Trainer in-place — HF will read
    `eval_mint_mark_accuracy` (etc.) from it for best-checkpoint selection.
    """

    def __init__(self, config, val_dataset, processor, max_samples=None):
        self.config = config
        self.val_dataset = val_dataset
        self.processor = processor
        self.max_samples = max_samples

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        metrics=None,
        **kwargs,
    ):
        if model is None or metrics is None:
            return

        from torch.utils.data import Subset
        from src.evaluate.evaluator import CoinEvaluator

        ds = self.val_dataset
        if self.max_samples is not None and len(ds) > self.max_samples:
            stride = max(1, len(ds) // self.max_samples)
            indices = list(range(0, len(ds), stride))[: self.max_samples]
            ds = Subset(ds, indices)

        was_training = model.training
        try:
            # Unsloth FastVisionModel needs to flip to inference mode for fast
            # generation kernels. Best-effort; ignored if not unsloth.
            try:
                from unsloth import FastVisionModel
                FastVisionModel.for_inference(model)
            except Exception:
                pass

            evaluator = CoinEvaluator(
                self.config, model=model, processor=self.processor
            )
            results = evaluator.evaluate(ds)
            gen_metrics = results["metrics"]
        finally:
            try:
                from unsloth import FastVisionModel
                FastVisionModel.for_training(model)
            except Exception:
                pass
            if was_training:
                model.train()

        # HF Trainer's `metric_for_best_model` lookup auto-prefixes "eval_" if
        # missing, so emitting both the prefixed and bare keys is fine; we
        # emit prefixed to be explicit.
        metrics["eval_mint_mark_accuracy"] = gen_metrics["mint_mark_accuracy"]
        metrics["eval_year_accuracy"] = gen_metrics["year_accuracy"]
        metrics["eval_extract_match"] = gen_metrics["extract_match"]
        metrics["eval_parse_error_rate"] = gen_metrics["parse_error_rate"]

        logger.info(
            "[GenEval @ step %d on %d samples] mint=%.4f | year=%.4f | exact=%.4f | parse_err=%.4f",
            state.global_step,
            len(ds),
            gen_metrics["mint_mark_accuracy"],
            gen_metrics["year_accuracy"],
            gen_metrics["extract_match"],
            gen_metrics["parse_error_rate"],
        )

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/mint_mark_accuracy": gen_metrics["mint_mark_accuracy"],
                        "eval/year_accuracy": gen_metrics["year_accuracy"],
                        "eval/extract_match": gen_metrics["extract_match"],
                        "eval/parse_error_rate": gen_metrics["parse_error_rate"],
                    },
                    step=state.global_step,
                )
        except ImportError:
            pass


class MemoryCallback(TrainerCallback):
    """
    Log GPU memory usage at the end of each epoch.
    """
    def on_evaluate(
      self, 
      args: TrainingArguments,
      state: TrainerState,
      control: TrainerControl,
      **kwargs,      
    ):
        if not torch.cuda.is_available():
            return
        
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            logger.info(
                f"GPU {i} | Memory Allocated: {mem_alloc:.2f} GB | Memory Reserved: {mem_reserved:.2f} GB"
            )

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(
                        {
                            f"gpu_{i}/memory_allocated_gb": mem_alloc,
                            f"gpu_{i}/memory_reserved_gb": mem_reserved,
                        },
                        step=state.global_step,
                    )
            except ImportError:
                pass