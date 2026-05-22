import logging
import os
import sys
from typing import Any
from omegaconf import OmegaConf

_SENSITIVE_KEYS = {"token", "password", "secret", "key", "api_key", "hf_token"}

def setup_logging(level):
    logging.basicConfig(
        level = getattr(logging, level.upper(), logging.INFO),
        format = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers = [logging.StreamHandler(sys.stdout)],
        force = True,
    )

    for noisy_lib in ["transformers", "datasets", "peft", "urllib3", "PIL"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)
    
def get_logger(name):
    return logging.getLogger(name)

def init_wandb(config):
    if config.training.report_to != "wandb":
        return 
    try:
        import wandb
    except ImportError:
        logging.warning("wandb not installed, skipping initialization.")
        return
    
    if not os.environ.get("WANDB_API_KEY"):
        logging.warning("WANDB_API_KEY not set, skipping wandb initialization.")
        return
    
    config_dict = _mask_sensitive(OmegaConf.to_container(config, resolve=True))

    wandb.init(
        project = config.training.get("wandb_project", "coin-vlm-finetune"),
        name = config.training.get("run_name", None),
        resume  = "allow",
    )

    logger = get_logger(__name__)
    logger.info("W&B run initialized: %s", wandb.run.url)

def finish_wandb():
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

def _mask_sensitive(obj, depth) -> Any:
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