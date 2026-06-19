import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
import sys
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = {"token", "password", "secret", "key", "api_key", "hf_token"}


def _date_key(rotation: str, now: datetime | None = None) -> str:
    """The granularity key that identifies the current log bucket: 'YYYY-MM' for
    monthly, 'YYYY-MM-DD' for daily (default)."""
    now = now or datetime.now()
    return now.strftime("%Y-%m") if rotation == "monthly" else now.strftime("%Y-%m-%d")


def dated_log_path(log_dir, stem: str, ext: str, rotation: str = "daily") -> Path:
    """Build a date-partitioned log path.

    daily   -> {log_dir}/{YYYY-MM}/{stem}-{YYYY-MM-DD}.{ext}
    monthly -> {log_dir}/{YYYY}/{stem}-{YYYY-MM}.{ext}
    """
    now = datetime.now()
    log_dir = Path(log_dir)
    if rotation == "monthly":
        return log_dir / now.strftime("%Y") / f"{stem}-{now.strftime('%Y-%m')}.{ext}"
    return log_dir / now.strftime("%Y-%m") / f"{stem}-{now.strftime('%Y-%m-%d')}.{ext}"


class DatedFileHandler(logging.Handler):
    """File handler that writes to a date-partitioned path and rolls over to a
    new file when the day/month changes — so a long-running server splits its
    logs by date without an external rotator."""

    def __init__(self, log_dir, stem: str, ext: str = "log", rotation: str = "daily"):
        super().__init__()
        self.log_dir = log_dir
        self.stem = stem
        self.ext = ext
        self.rotation = rotation
        self._key: str | None = None
        self._stream = None
        self._open()

    def _open(self) -> None:
        path = dated_log_path(self.log_dir, self.stem, self.ext, self.rotation)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._key = _date_key(self.rotation)
        self._stream = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Reopen when the calendar bucket changed, OR when the stream was
            # closed out from under us (uvicorn's dictConfig closes existing
            # handlers on startup, but this handler stays attached to root).
            stream_dead = self._stream is None or self._stream.closed
            if stream_dead or _date_key(self.rotation) != self._key:
                if self._stream is not None and not self._stream.closed:
                    self._stream.close()
                self._open()
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)

    def close(self) -> None:
        try:
            if self._stream:
                self._stream.close()
        finally:
            super().close()


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_dir: str | None = None,
    log_stem: str = "app",
    log_rotation: str = "daily",
) -> None:

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    sink_desc = None
    if log_dir:
        # Date-partitioned, self-rotating file split by day/month.
        handlers.append(DatedFileHandler(log_dir, log_stem, "log", log_rotation))
        sink_desc = str(dated_log_path(log_dir, log_stem, "log", log_rotation))
    elif log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        sink_desc = log_file

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    if sink_desc:
        logging.getLogger(__name__).info("Logging to file: %s", sink_desc)

    for noisy_lib in ("transformers", "datasets", "peft", "urllib3", "PIL"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)
 
 
def get_logger(name: str) -> logging.Logger:

    return logging.getLogger(name)
 
 
def report_to_list(report_to) -> list[str]:
    """Normalize a ``report_to`` config (str | list | None) to a plain list of
    backend names. ``none``/``null``/``""`` -> ``[]``. Handles OmegaConf
    ListConfig so it can be passed straight to TrainingArguments."""
    if report_to is None:
        return []
    if isinstance(report_to, str):
        return [] if report_to.lower() in ("none", "null", "") else [report_to]
    try:
        return [str(x) for x in report_to]
    except TypeError:
        return []


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log scalar metrics to whichever trackers are active (W&B + MLflow).

    Single place for hybrid logging — call this instead of ``wandb.log`` so
    both backends stay in sync. Non-scalar values are skipped for MLflow.
    Never raises.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:  # noqa: BLE001 - logging must never crash training
        pass
    try:
        import mlflow
        if mlflow.active_run() is not None:
            scalars = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if scalars:
                mlflow.log_metrics(scalars, step=step)
    except Exception:  # noqa: BLE001
        pass


def init_mlflow(cfg: DictConfig) -> None:
    """Start an MLflow run + attach lineage, when "mlflow" is in report_to.

    The HF MLflowCallback (enabled via report_to) logs per-step metrics and
    TrainingArguments params automatically and REUSES this active run. Here we
    only add what the callback doesn't: a named run, lineage tags (git/data),
    and the full merged config as an artifact (logged as an artifact, not
    params, to avoid name clashes with the callback). Never raises.
    """
    if "mlflow" not in report_to_list(cfg.training.get("report_to", None)):
        return

    try:
        import mlflow
    except ImportError:
        get_logger(__name__).warning(
            "mlflow not installed; skipping MLflow init. Run: pip install mlflow-skinny"
        )
        return

    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment = (
            os.getenv("MLFLOW_EXPERIMENT_NAME")
            or cfg.training.get("mlflow_experiment", "coin-vlm-finetune")
        )
        mlflow.set_experiment(experiment)

        if mlflow.active_run() is None:
            mlflow.start_run(run_name=cfg.training.get("run_name", None))

        from .lineage import collect_lineage  # local import: avoid import cycle
        lin = collect_lineage(cfg)
        tags = {
            "git_commit": lin.get("git_commit"),
            "git_branch": lin.get("git_branch"),
            "git_dirty": lin.get("git_dirty"),
            "hf_dataset_name": lin.get("hf_dataset_name"),
            "hf_revision": lin.get("hf_revision"),
            "host": lin.get("host"),
        }
        mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})
        mlflow.log_dict(lin.get("config", {}), "config.json")

        get_logger(__name__).info(
            "MLflow run started (experiment=%s, uri=%s)",
            experiment, tracking_uri or "default",
        )
    except Exception:  # noqa: BLE001
        get_logger(__name__).exception("MLflow init failed; continuing without it.")


def log_model_artifact(cfg: DictConfig, ckpt_path, artifact_path: str = "model") -> bool:
    """Upload the best checkpoint to the ACTIVE MLflow run as an artifact, so the
    run shows a model in the UI and its "Register Model" button works.

    This is the manual, gate-free path (register straight from the MLflow UI),
    complementing scripts/register_model.py which applies the quality gate and
    registers from the CLI / Airflow. Logging the SAME checkpoint dir keeps the
    two paths' artifacts identical.

    Gated by training.mlflow_log_model (default True); only runs when "mlflow"
    is in report_to and a run is active. Never raises.
    """
    if "mlflow" not in report_to_list(cfg.training.get("report_to", None)):
        return False
    if not cfg.training.get("mlflow_log_model", True):
        return False
    try:
        import mlflow
    except ImportError:
        return False
    try:
        if mlflow.active_run() is None:
            get_logger(__name__).warning("log_model_artifact: no active MLflow run; skipping.")
            return False
        p = Path(ckpt_path)
        if not p.exists():
            get_logger(__name__).warning("log_model_artifact: checkpoint not found: %s", p)
            return False
        mlflow.log_artifacts(str(p), artifact_path=artifact_path)
        get_logger(__name__).info(
            "Logged model to MLflow run under '%s/'. Register from the UI: open "
            "the run -> Artifacts -> select '%s' -> 'Register Model'.",
            artifact_path, artifact_path,
        )
        return True
    except Exception:  # noqa: BLE001
        get_logger(__name__).exception("log_model_artifact failed; continuing.")
        return False


def finish_mlflow() -> None:
    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:  # noqa: BLE001
        pass


def init_wandb(cfg: DictConfig) -> None:
    if "wandb" not in report_to_list(cfg.training.get("report_to", None)):
        return

    try:
        import wandb
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY environment variable is not set")
        else:
            wandb.login(key=wandb_api_key, relogin=True)
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