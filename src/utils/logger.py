import logging
import os
from typing import Any
import sys
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = {"token", "password", "secret", "key", "api_key", "hf_token"}
 
 
def setup_logging(level: str = "INFO") -> None:

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,    
    )
 
    for noisy_lib in ("transformers", "datasets", "peft", "urllib3", "PIL"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)
 
 
def get_logger(name: str) -> logging.Logger:

    return logging.getLogger(name)
 
 
def init_wandb(cfg: DictConfig) -> None:
    if cfg.training.report_to != "wandb":
        return
 
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb not installed. Run: pip install wandb"
            " or set training.report_to=none in config."
        )
 
    config_dict = _mask_sensitive(OmegaConf.to_container(cfg, resolve=True))
 
    wandb.init(
        project=cfg.training.get("wandb_project", "coin-vlm-finetune"),
        name=cfg.training.get("run_name", None),
        config=config_dict,
        resume="allow",
    )
 
    logger = get_logger(__name__)
    logger.info("W&B run initialized: %s", wandb.run.url)
 
 
def finish_wandb() -> None:
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
 
def _mask_sensitive(obj: Any, depth: int = 0) -> Any:

    if depth > 10:     
        return obj
 
    if isinstance(obj, dict):
        return {
            k: "***" if any(s in k.lower() for s in _SENSITIVE_KEYS)
            else _mask_sensitive(v, depth + 1)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_mask_sensitive(i, depth + 1) for i in obj]
 
    return obj