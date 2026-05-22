from omegaconf import OmegaConf
from src.utils import get_logger
from .base import BaseModelLoader
from .unsloth_loader import UnslothModelLoader
from .hf_peft_loader import HFPeftModelLoader
from .full_finetune_loader import FullFinetuneLoader

logger = get_logger(__name__)

_LOADERS: dict[str, type[BaseModelLoader]] = {
    "unsloth": UnslothModelLoader,
    "hf_peft": HFPeftModelLoader,
    "full_finetune": FullFinetuneLoader,
}

def get_model_loader(config: OmegaConf) -> BaseModelLoader:
    backend = config.model.backend

    if backend not in _LOADERS:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends: {list(_LOADERS.keys())}")
    
    logger.info(f"Using model loader: {_LOADERS[backend].__name__} for backend: {backend}")
    return _LOADERS[backend](config)
