from omegaconf import DictConfig
from src.utils import get_logger
from torch.utils.data import Dataset
from .base import BaseTrainer
from .unsloth_trainer import UnslothTrainer
from .hf_peft_trainer import HFPeftTrainer
from .full_finetune_trainer import FullFinetuneTrainer

logger = get_logger(__name__)
_TRAINERS: dict[str, type[BaseTrainer]] = {
    "unsloth": UnslothTrainer,
    "hf_peft": HFPeftTrainer,
    "full_finetune": FullFinetuneTrainer,
}

def get_trainer(
    config,
    model,
    processor,
    train_loader: Dataset,
    val_loader: Dataset,    
    ):
    
    backend = config.training.backend
    if backend not in _TRAINERS:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends: {list(_TRAINERS.keys())}")
    

    logger.info("Using trainer: '%s'", backend)
    return _TRAINERS[backend](config, model, processor, train_loader, val_loader)