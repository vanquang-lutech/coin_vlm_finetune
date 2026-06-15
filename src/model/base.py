from abc import ABC, abstractmethod

from omegaconf import DictConfig

from src.utils import get_logger

logger = get_logger(__name__)

class BaseModelLoader(ABC):

    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.processor = None


    def load(self) -> tuple:

        model_config = self.config.model
        logger.info(
            "[%s] Loading '%s'...",
            self.config.model.backend.upper(),
            model_config.name,
        )

        self.processor = self._load_processor()
        self.model = self._load_model()
        self._apply_adapter()

        self.print_trainable_params()
        return self.model, self.processor

    def load_for_inference(self) -> tuple:
        """Load model + processor for evaluation/inference.

        Mirrors load() but skips _apply_adapter() (no training LoRA wrap).
        Subclasses (e.g. UnslothModelLoader) may override to enable
        backend-specific fast inference kernels.
        """
        model_config = self.config.model
        logger.info(
            "[%s] Loading '%s' for inference...",
            self.config.model.backend.upper(),
            model_config.name,
        )
        self.processor = self._load_processor()
        self.model = self._load_model()
        return self.model, self.processor


    def _load_processor(self):
   
        from transformers import AutoProcessor

        processor_config = self.config.get("processor", None)
        if processor_config is None:
            processor_config = self.config.model.get("processor", None)
        processor = AutoProcessor.from_pretrained(self.config.model.name)

        img_proc = getattr(processor, "image_processor", None)
        has_min_pixels = img_proc is not None and hasattr(img_proc, "min_pixels")

        if processor_config is not None and has_min_pixels:
            img_proc.min_pixels = processor_config.min_pixels * 28 * 28
            img_proc.max_pixels = processor_config.max_pixels * 28 * 28

            logger.info(
                "Processor loaded. min_pixels=%d, max_pixels=%d",
                processor_config.min_pixels,
                processor_config.max_pixels,
            )
        elif processor_config is not None:
            # Fixed-resolution models (e.g. PaliGemma's SigLIP processor) have no
            # min_pixels/max_pixels; resolution is baked into the checkpoint. The
            # global inference.yaml `processor:` block is then a harmless no-op.
            logger.info(
                "Processor has no min_pixels/max_pixels (fixed-resolution model); "
                "ignoring processor.min_pixels/max_pixels config."
            )
        else:
            logger.info("Processor loaded with default image processor settings.")
        return processor


    @abstractmethod
    def _load_model(self):

        pass
    def _apply_adapter(self) -> None:

        pass


    def get_trainable_params(self) -> tuple[int, int]:
        self._check_loaded()
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def print_trainable_params(self) -> None:
        trainable, total = self.get_trainable_params()
        pct = 100 * trainable / total if total > 0 else 0
        logger.info(
            "Trainable params: %s / %s (%.4f%%)",
            f"{trainable:,}",
            f"{total:,}",
            pct,
        )

    def _check_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "Model not loaded. Call loader.load() first."
            )