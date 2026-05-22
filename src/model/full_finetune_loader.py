import torch 
from src.utils import get_logger
from .base import BaseModelLoader

logger = get_logger(__name__)

_MIN_VRAM_GB = 40

class FullFinetuneLoader(BaseModelLoader):
    def _load_model(self):
        from transformers import AutoModelForImageTextToText
        model_config = self.config.model
        torch_dtype = getattr(torch, model_config.torch_dtype, torch.float16)
        self._warn_vram()
        kwargs = dict(
            torch_dtype = torch_dtype,
            device_map = model_config.device_map,
        )
        att_implementation = model_config.get("attn_implementation", None)
        if att_implementation:
            kwargs["attn_implementation"] = att_implementation
        
        model = AutoModelForImageTextToText.from_pretrained(
            model_config.name, 
            **kwargs
        )

        if model_config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,  # Disable reentrant to reduce VRAM usage
                }
            )
            logger.info("Gradient checkpointing enabled with non-reentrant mode to save VRAM.")
        return model

    def _warn_vram(self):
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Full fine-tuning may be very slow on CPU.")
            return
        total_vram_gb = sum(torch.data.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / (1024 ** 3)
        if total_vram_gb < _MIN_VRAM_GB:
            logger.warning(
                f"Total VRAM across all GPUs is {total_vram_gb:.1f} GB, which may be insufficient for full fine-tuning. Consider using a smaller model or enabling gradient checkpointing."
            )
        else:
            logger.info(f"Total VRAM across all GPUs: {total_vram_gb:.1f} GB")