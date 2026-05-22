import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


logger = logging.getLogger(__name__)

class ConfigLoader:
    @staticmethod
    def load(
        data_config: str,
        model_config: str,
        training_config: str,
        method_config: str | None = None,
        inference_config: str | None = None,
        overrides: list[str] | None = None,
    ) -> DictConfig:
        config = {
            "data": ConfigLoader._load_yaml(data_config),
            "model": ConfigLoader._load_yaml(model_config),
            "training": ConfigLoader._load_yaml(training_config),
        }

        merged = OmegaConf.create(config)

        if method_config:
            method_cfg = ConfigLoader._load_yaml(method_config)
            merged = OmegaConf.merge(merged, method_cfg)

        if inference_config:
            inference_cfg = ConfigLoader._load_yaml(inference_config)
            merged = OmegaConf.merge(merged, inference_cfg)

        if overrides:
            override_conf = OmegaConf.from_dotlist(overrides)
            merged = OmegaConf.merge(merged, override_conf)

        OmegaConf.set_readonly(merged, True)
        ConfigLoader._log_config(merged)

        return merged

    @staticmethod
    def _load_yaml(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        config = OmegaConf.load(path)
        logger.debug("Loaded config from %s:\n%s", path, OmegaConf.to_yaml(config))
        return config
    
    @staticmethod
    def _log_config(config):
        logger.info(
            "Final config:\n%s",
            OmegaConf.to_yaml(config, resolve=True),
        )