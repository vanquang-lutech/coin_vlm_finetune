import torch
from src.utils import get_logger
from .base import BaseModelLoader

logger = get_logger(__name__)

class HFPeftModelLoader(BaseModelLoader):
    def _load_model(self):
        from transformers import AutoModelForImageTextToText
        from transformers import BitsAndBytesConfig

        model_config = self.config.model 
        torch_dtype = getattr(torch, model_config.torch_dtype, torch.bfloat16)
        kwargs = dict(
            torch_dtype = torch_dtype,
            device_map = model_config.device_map,
        )

        trust_remote_code = model_config.get("trust_remote_code", None)
        if trust_remote_code is not None:
            kwargs["trust_remote_code"] = trust_remote_code

        attn_implementation = model_config.get("attn_implementation", None)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        quant_config = model_config.get("quantization", None)
        if quant_config:
            kwargs["quantization_config"] = BitsAndBytesConfig(**quant_config)
        elif model_config.get("load_in_4bit", False):
            kwargs["load_in_4bit"] = True
        
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_config.name, 
                **kwargs
            )
        except Exception:
            from transformers import AutoModel
            logger.warning(
                "AutoModelForImageTextToText failed to load. Falling back to AutoModel. This may cause issues if the model architecture is not compatible."
            )
            model = AutoModel.from_pretrained(
                model_config.name, 
                **kwargs
            )

        if model_config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        
        # Enable input gradients for LoRA fine-tuning
        model.enable_input_require_grads()
        return model
    
    def _apply_adapter(self):
        from peft import LoraConfig, get_peft_model

        lora_config = self.config.get("lora", None)
        if lora_config is None:
            lora_config = self.config.model.get("lora", None)
        logger.info(
            "Applying HF PEFT LoRA adapter (r=%d)...", lora_config.r
        )

        lora_config = LoraConfig(
            r = lora_config.r,
            lora_alpha = lora_config.lora_alpha,
            lora_dropout= lora_config.lora_dropout,
            target_modules = list(lora_config.target_modules),
            bias = lora_config.bias,
            task_type = lora_config.task_type,
        )

        self.model = get_peft_model(self.model, lora_config)
