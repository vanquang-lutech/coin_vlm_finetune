"""Run lineage: capture git commit, data revision, env and the merged config
into a ``run_metadata.json`` next to each run's checkpoints.

This is the cheap reproducibility backbone (Phase 0.2): given a checkpoint you
can answer "which code, which data, which config produced this" without relying
on an external tracker. MLflow (Phase 3) logs the same fields as tags/params,
but this file lives with the weights so it survives even if MLflow doesn't.
"""

import json
import logging
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .logger import _mask_sensitive

logger = logging.getLogger(__name__)

# Repo root: src/utils/lineage.py -> src/utils -> src -> <root>
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _git(*args: str) -> str | None:
    """Run a git command in the repo root; return stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - lineage must never crash training
        return None


def _safe_get(config: DictConfig, *keys: str) -> Any:
    node: Any = config
    for key in keys:
        try:
            node = node.get(key, None)
        except Exception:  # noqa: BLE001
            return None
        if node is None:
            return None
    return node


def collect_lineage(config: DictConfig) -> dict[str, Any]:
    """Gather code + data + env + config snapshot for one run."""
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "hf_dataset_name": _safe_get(config, "data", "hf_dataset_name"),
        "hf_revision": _safe_get(config, "data", "hf_revision"),
        "run_name": _safe_get(config, "training", "run_name"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "config": _mask_sensitive(OmegaConf.to_container(config, resolve=True)),
    }


def write_run_metadata(config: DictConfig, output_dir: str | Path) -> Path:
    """Write ``run_metadata.json`` into the run's output dir. Returns the path.

    Never raises — lineage capture must not break a training run.
    """
    path = Path(output_dir) / "run_metadata.json"
    try:
        meta = collect_lineage(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Wrote run metadata (lineage) to %s", path)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to write run metadata to %s", path)
    return path
