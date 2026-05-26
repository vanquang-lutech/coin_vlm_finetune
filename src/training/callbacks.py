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