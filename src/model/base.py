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


    def _load_processor(self):
   
        from transformers import AutoProcessor

        processor_config = self.config.get("processor", None)
        if processor_config is None:
            processor_config = self.config.model.get("processor", None)
        processor = AutoProcessor.from_pretrained(self.config.model.name)

        if processor_config is not None:
            processor.image_processor.min_pixels = processor_config.min_pixels * 28 * 28
            processor.image_processor.max_pixels = processor_config.max_pixels * 28 * 28

            logger.info(
                "Processor loaded. min_pixels=%d, max_pixels=%d",
                processor_config.min_pixels,
                processor_config.max_pixels,
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