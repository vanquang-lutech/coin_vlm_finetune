"""Quality gate + model registration — the final step of CT (Phase 3.3 / 5.3).

Reads the eval metrics, and IF the model clears the quality threshold, packages
the checkpoint (weights + processor + inference config + run_metadata) into a
new MLflow Registry version with lineage tags. Below threshold -> no register,
exit code 1 so the pipeline (make / Airflow) knows it failed the gate.

This is CT, not CD: registering = cataloguing a trained model for storage +
lineage. It does NOT promote to a Production stage or deploy anything (that's
CD, deferred until there is a serving target).

Standalone + lightweight (no GPU, no model load): it uploads the checkpoint dir
as artifacts and registers it. Needs mlflow-skinny + MLFLOW_* env.

Usage:
  python scripts/register_model.py \
      --checkpoint_path outputs/checkpoints/coin-vlm_<ts>/checkpoint-best \
      --metrics outputs/results/metrics.json \
      --metric extract_match --threshold 0.90 --model-name coin-vlm
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Quality gate + MLflow model registration.")
    p.add_argument("--checkpoint_path", required=True, help="Checkpoint dir to register.")
    p.add_argument("--metrics", default="outputs/results/metrics.json",
                   help="Flat metrics JSON written by scripts/evaluate.py.")
    p.add_argument("--metric", default="extract_match", help="Gate metric key.")
    p.add_argument("--threshold", type=float, default=0.90, help="Min metric to register.")
    p.add_argument("--model-name", default="coin-vlm", help="MLflow registered model name.")
    p.add_argument("--run-metadata", default=None,
                   help="run_metadata.json for lineage tags (default: <checkpoint>/run_metadata.json).")
    return p.parse_args()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint_path)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}. Run evaluate first.")

    metrics = _load_json(metrics_path)
    # Accept both the flat metrics dict and a {"metrics": {...}} wrapper.
    if args.metric not in metrics and "metrics" in metrics:
        metrics = metrics["metrics"]

    if args.metric not in metrics:
        raise SystemExit(f"Metric '{args.metric}' not in {metrics_path}. Keys: {list(metrics)}")

    score = float(metrics[args.metric])
    logger.info("Quality gate: %s = %.4f (threshold %.4f)", args.metric, score, args.threshold)

    # --- Quality gate ---------------------------------------------------- #
    if score < args.threshold:
        logger.warning(
            "GATE FAILED: %s=%.4f < %.4f → NOT registering %s.",
            args.metric, score, args.threshold, args.model_name,
        )
        sys.exit(1)

    # --- Register to MLflow --------------------------------------------- #
    try:
        import mlflow
    except ImportError:
        raise SystemExit("mlflow not installed. Run: pip install mlflow-skinny")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "coin-vlm-finetune")
    mlflow.set_experiment(experiment)

    # Lineage tags from run_metadata.json (written at train time, Phase 0.2).
    meta_path = Path(args.run_metadata) if args.run_metadata else ckpt / "run_metadata.json"
    tags = {"gate_metric": args.metric, "gate_score": f"{score:.4f}"}
    if meta_path.exists():
        meta = _load_json(meta_path)
        for k in ("git_commit", "git_branch", "hf_dataset_name", "hf_revision", "run_name"):
            if meta.get(k) is not None:
                tags[k] = str(meta[k])
    else:
        logger.warning("No run_metadata.json at %s — registering without git/data lineage.", meta_path)

    with mlflow.start_run(run_name=f"register-{args.model_name}") as run:
        mlflow.set_tags(tags)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
        # Package the checkpoint (weights + processor + inference config +
        # run_metadata) as the model artifact, then register that version.
        mlflow.log_artifacts(str(ckpt), artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, args.model_name, tags=tags)
        logger.info(
            "Registered %s version %s (run %s). %s=%.4f passed gate.",
            args.model_name, mv.version, run.info.run_id, args.metric, score,
        )


if __name__ == "__main__":
    main()
