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
        from_pretrained_kwargs = dict(
            load_in_4bit = model_config.get("load_in_4bit", False),
            use_gradient_checkpointing = "unsloth",
        )

        # Models whose config class is not registered with the default
        # AutoModelForImageTextToText mapping (e.g. InternVL's InternVLChatConfig)
        # must be loaded through an explicit auto class. See unsloth issue #3401:
        # passing auto_model + trust_remote_code lets FastVisionModel resolve the
        # architecture via the model's remote code.
        auto_model_name = model_config.get("auto_model", None)
        if auto_model_name:
            import transformers
            auto_model_cls = getattr(transformers, auto_model_name, None)
            if auto_model_cls is None:
                raise ValueError(
                    f"model.auto_model='{auto_model_name}' is not a class exported "
                    "by transformers (expected e.g. 'AutoModel' or "
                    "'AutoModelForImageTextToText')."
                )
            from_pretrained_kwargs["auto_model"] = auto_model_cls
            from_pretrained_kwargs["trust_remote_code"] = model_config.get(
                "trust_remote_code", True
            )
            from_pretrained_kwargs["unsloth_force_compile"] = model_config.get(
                "unsloth_force_compile", False
            )
            logger.info(
                "Loading via auto_model=%s (trust_remote_code=%s, force_compile=%s)",
                auto_model_name,
                from_pretrained_kwargs["trust_remote_code"],
                from_pretrained_kwargs["unsloth_force_compile"],
            )

        logger.info("Loading Unsloth model repo: %s", unsloth_name)
        model, processor = FastVisionModel.from_pretrained(
            unsloth_name,
            **from_pretrained_kwargs,
        )

        self.processor = processor

        if processor_config is not None:
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is not None:
                import transformers
                from packaging.version import parse as parse_version

                min_pixels = processor_config.min_pixels * 28 * 28
                max_pixels = processor_config.max_pixels * 28 * 28
                is_v5 = parse_version(transformers.__version__).major >= 5

                if is_v5:
                    # transformers >= 5: min_pixels/max_pixels are read-only
                    # properties backed by `size`, which is a SizeDict (a dataclass,
                    # NOT a dict subclass) holding shortest_edge/longest_edge. It
                    # supports `in` and item assignment, so avoid isinstance(dict).
                    size = getattr(image_processor, "size", None)
                    if size is not None and "shortest_edge" in size and "longest_edge" in size:
                        size["shortest_edge"] = min_pixels
                        size["longest_edge"] = max_pixels
                        logger.info(
                            "Processor loaded (transformers v5 size dict). "
                            "min_pixels: %d, max_pixels: %d",
                            min_pixels,
                            max_pixels,
                        )
                    else:
                        logger.warning(
                            "transformers v5 image_processor has no size dict with "
                            "shortest_edge/longest_edge; skipping resolution override."
                        )
                else:
                    # transformers < 5: writable min_pixels/max_pixels attributes.
                    if hasattr(image_processor, "min_pixels"):
                        image_processor.min_pixels = min_pixels
                        image_processor.max_pixels = max_pixels
                        logger.info(
                            "Processor loaded (transformers v4 attributes). "
                            "min_pixels: %d, max_pixels: %d",
                            min_pixels,
                            max_pixels,
                        )
                    else:
                        logger.warning(
                            "transformers v4 image_processor has no min_pixels "
                            "attribute; skipping resolution override."
                        )
            else:
                logger.warning(
                    "Processor has no image_processor; skipping resolution override."
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
            "Applying Unsloth LoRA Adapter (r=%d)...", lora_config.r
        )
        peft_kwargs = dict(
            finetune_vision_layers     = lora_config.get("finetune_vision_layers", True),
            finetune_language_layers   = lora_config.get("finetune_language_layers", True),
            finetune_attention_modules = lora_config.get("finetune_attention_modules", True),
            finetune_mlp_modules       = lora_config.get("finetune_mlp_modules", True),
            r = lora_config.r,
            lora_alpha = lora_config.lora_alpha,
            lora_dropout = lora_config.lora_dropout,
            bias = lora_config.bias,
            use_gradient_checkpointing = lora_config.use_gradient_checkpointing,
            random_state = self.config.training.get("seed", 42),
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        # target_modules is optional. When omitted, Unsloth auto-selects the
        # linear layers to adapt based on the finetune_* flags above. This is
        # required for InternVL, whose vision/projector module names differ from
        # Qwen's — passing the Qwen target lists would error with "target modules
        # not found". Qwen configs still pass an explicit list.
        target_modules = lora_config.get("target_modules", None)
        if target_modules:
            peft_kwargs["target_modules"] = list(target_modules)

        self.model = FastVisionModel.get_peft_model(self.model, **peft_kwargs)