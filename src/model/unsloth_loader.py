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
        load_kwargs = dict(
            load_in_4bit = model_config.get("load_in_4bit", False),
            use_gradient_checkpointing = "unsloth",
        )
        # Unsloth's canonical bf16 call for hybrid/MoE models (e.g. Qwen3.5,
        # which should NOT be QLoRA'd in 4-bit) passes load_in_16bit=True. Only
        # forward it when the config sets it, so existing 4-bit configs and
        # older Unsloth builds (which may not accept the kwarg) are unaffected.
        if "load_in_16bit" in model_config:
            load_kwargs["load_in_16bit"] = model_config.get("load_in_16bit")
        model, processor = FastVisionModel.from_pretrained(
            unsloth_name,
            **load_kwargs,
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

        # target_modules may be the string "all-linear" (let Unsloth's
        # get_peft_regex discover EVERY Linear in both towers — driven by the
        # finetune_* flags below) OR an explicit list of name suffixes. A list
        # is fine to list(); a string must NOT be list()'d or "all-linear"
        # explodes into individual characters.
        tm = lora_config.target_modules
        target_modules = tm if isinstance(tm, str) else list(tm)

        # Which sub-networks LoRA adapts. Default True (finetune both towers),
        # now config-controllable. IMPORTANT: these flags only take effect when
        # target_modules == "all-linear"; with an explicit list Unsloth uses the
        # list verbatim and ignores them (so an explicit list that names no
        # vision modules silently leaves the vision tower frozen).
        finetune_vision_layers = lora_config.get("finetune_vision_layers", True)
        finetune_language_layers = lora_config.get("finetune_language_layers", True)
        finetune_attention_modules = lora_config.get("finetune_attention_modules", True)
        finetune_mlp_modules = lora_config.get("finetune_mlp_modules", True)

        logger.info(
            "Applying Unsloth LoRA (r=%d, target_modules=%s, vision=%s "
            "language=%s attn=%s mlp=%s)...",
            lora_config.r,
            target_modules if isinstance(target_modules, str) else f"{len(target_modules)} names",
            finetune_vision_layers, finetune_language_layers,
            finetune_attention_modules, finetune_mlp_modules,
        )
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers     = finetune_vision_layers,
            finetune_language_layers   = finetune_language_layers,
            finetune_attention_modules = finetune_attention_modules,
            finetune_mlp_modules       = finetune_mlp_modules,
            r = lora_config.r,
            lora_alpha = lora_config.lora_alpha,
            lora_dropout = lora_config.lora_dropout,
            target_modules = target_modules,
            bias = lora_config.bias,
            use_gradient_checkpointing = lora_config.use_gradient_checkpointing,
            random_state = self.config.training.get("seed", 42),
            use_rslora = False,
            loftq_config = None,
        )
        self._log_lora_coverage()

    def _log_lora_coverage(self) -> None:
        """Log which modules actually received LoRA, split vision vs language,
        so "is vision being finetuned?" is verifiable instead of a guess.

        Catches silent non-matches (a mistyped target suffix that matches
        nothing) and confirms hybrid token-mixing layers (Qwen3.5 GatedDeltaNet
        in_proj_qkvz / in_proj_ba) got adapters. PEFT names each adapted Linear
        ``....<proj>.lora_A`` — counting those is an exact tally.
        """
        from collections import Counter

        vision_kw = ("visual", "vision", "patch_embed", "merger", "image")
        vision_names: Counter = Counter()
        text_names: Counter = Counter()
        for name, _ in self.model.named_modules():
            if not name.endswith(".lora_A"):
                continue
            leaf = name.split(".")[-2]
            if any(k in name.lower() for k in vision_kw):
                vision_names[leaf] += 1
            else:
                text_names[leaf] += 1

        logger.info(
            "[LoRA coverage] vision adapters=%d %s | language adapters=%d %s",
            sum(vision_names.values()), dict(vision_names),
            sum(text_names.values()), dict(text_names),
        )
        if sum(vision_names.values()) == 0:
            logger.warning(
                "[LoRA coverage] NO vision modules received LoRA — the vision "
                "tower is effectively FROZEN. If you intended to finetune vision, "
                "set target_modules='all-linear' with finetune_vision_layers=true."
            )