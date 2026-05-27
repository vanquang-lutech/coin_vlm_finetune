from src.utils import get_logger
from .base import BaseModelLoader

logger = get_logger(__name__)

class UnslothModelLoader(BaseModelLoader):
    def _load_processor(self):
        return None

    def _load_model(self):
        try:
            from unsloth import FastVisionModel
        except ImportError:
            raise ImportError("Please install unsloth library to use UnslothModelLoader.")
        
        model_config = self.config.model
        unsloth_name = model_config.get("unsloth_name", None)
        if not unsloth_name:
            raise ValueError(
                "Unsloth backend requires model.unsloth_name to be set to an Unsloth repo."
            )
        processor_config = self.config.get("processor", None)
        if processor_config is None:
            processor_config = self.config.model.get("processor", None)
        logger.info("Loading Unsloth model repo: %s", unsloth_name)
        model, processor = FastVisionModel.from_pretrained(
            model_name = unsloth_name,
            load_in_4bit = model_config.load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )

        self.processor = processor
        if processor_config is not None:
            self.processor.image_processor.min_pixels = processor_config.min_pixels * 28 * 28
            self.processor.image_processor.max_pixels = processor_config.max_pixels * 28 * 28
            logger.info(
                "Processor loaded. min_pixels: %d, max_pixels: %d",
                self.processor.image_processor.min_pixels,
                self.processor.image_processor.max_pixels,
            )
        else:
            logger.info("Processor loaded with default image processor settings.")
        return model
    
    def load_for_inference(self) -> tuple:
        from unsloth import FastVisionModel

        model_config = self.config.model
        logger.info(
            "[UNSLOTH] Loading '%s' for inference...", model_config.name,
        )
        self.processor = self._load_processor()
        self.model = self._load_model()
        FastVisionModel.for_inference(self.model)
        return self.model, self.processor

    def _apply_adapter(self):
        from unsloth import FastVisionModel
        lora_config = self.config.get("lora", None)
        if lora_config is None:
            lora_config = self.config.model.get("lora", None)
        logger.info(
            "Applying Unsloth QloRA Adapter (r=%d)...", lora_config.r
        )
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers     = True, 
            finetune_language_layers   = True, 
            finetune_attention_modules = True, 
            finetune_mlp_modules       = True, 
            r = lora_config.r,
            lora_alpha = lora_config.lora_alpha,
            lora_dropout = lora_config.lora_dropout,
            target_modules = list(lora_config.target_modules),
            bias = lora_config.bias,
            use_gradient_checkpointing = lora_config.use_gradient_checkpointing,
            random_state = self.config.training.get("seed", 42),
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )