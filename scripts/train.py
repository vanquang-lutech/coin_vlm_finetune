import os

# Disable TorchDynamo for the whole job. Hybrid linear-attn models (Qwen3.5
# Gated DeltaNet) recompile causal_conv1d per prompt-length during the periodic
# in-training generation eval; on the unsloth backend that path ignores the
# evaluator's set_stance("force_eager") guard, and the accumulated recompiles
# leak host RAM until the process is OOM-killed (this is a host-RAM, not GPU,
# OOM). Disabling dynamo trades the training compile speedup for a flat host-RAM
# profile. Must be set BEFORE torch/unsloth import or torch._dynamo reads the
# old value. Re-enable compile by exporting TORCHDYNAMO_DISABLE=0.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Unsloth MUST be imported before transformers/peft (which the src.* imports
# below pull in transitively) so all of its patches/optimizations — including
# memory savings — are applied. Keep this as the first heavy import (the env
# set above only touches os, so unsloth's patches still apply correctly).
import unsloth  # noqa: E402,F401

import argparse
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.utils import (
    ConfigLoader,
    finish_mlflow,
    finish_wandb,
    get_logger,
    init_mlflow,
    init_wandb,
    log_model_artifact,
    set_seed,
    write_run_metadata,
)
from src.data.dataset import CoinDataset
from src.model.factory import get_model_loader
from src.training.factory import get_trainer
from src.evaluate.evaluator import CoinEvaluator
 
logger = get_logger(__name__)
 
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune VLM for coin recognition.")
 
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument(
        "--method_config",
        default=None,
        help="Optional backend config to set model/training backend and backend-specific settings.",
    )
    parser.add_argument(
        "--inference_config",
        default="config/inference/inference.yaml",
        help="Optional inference config to merge for generation defaults.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config. E.g. --override training.learning_rate=1e-4",
    )

    parser.add_argument(
        "--skip_test_eval",
        action="store_true",
        default=False,
        help="Whether to skip evaluation on test set after training.",
    )
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    config = ConfigLoader.load(
        data_config = args.data_config,
        model_config = args.model_config,
        training_config = args.training_config,
        method_config = args.method_config,
        inference_config = args.inference_config,
        overrides = args.override,
    )

    if config.model.backend == "unsloth" or config.training.backend == "unsloth":
        import unsloth
 
    set_seed(
        config.training.seed,
        deterministic=config.training.get("deterministic", False),
    )
    init_wandb(config)
    init_mlflow(config)

    loader = get_model_loader(config)
    model, processor = loader.load()
 
    logger.info("Loading datasets...")
    train_ds = CoinDataset(config, split="train")
    val_ds = CoinDataset(config, split="validation")
 
    trainer = get_trainer(config, model, processor, train_ds, val_ds)

    # Lineage: trainer.__init__ resolves training.output_dir to the timestamped
    # run dir; write run_metadata.json there before training so it survives even
    # if the run later crashes.
    write_run_metadata(config, config.training.output_dir)

    trainer.train()

    # Stable pointer to the best checkpoint so `make register` / Airflow can
    # find it without parsing timestamped run dirs.
    best_ckpt = None
    try:
        result_dir = Path(config.training.get("result_dir", "outputs/results"))
        result_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = trainer.get_best_checkpoint()
        (result_dir / "best_checkpoint.txt").write_text(best_ckpt, encoding="utf-8")
        logger.info("Best checkpoint pointer written: %s -> %s",
                    result_dir / "best_checkpoint.txt", best_ckpt)
    except Exception:
        logger.exception("Could not write best_checkpoint.txt pointer.")

    if not args.skip_test_eval:
        logger.info("Running evaluation on test set...")
        test_ds = CoinDataset(config, split="test")
        evaluator = CoinEvaluator(config, model=trainer.get_model(), processor=processor)
        results = evaluator.evaluate(test_ds)
        evaluator.save_results(results, config.training.get("result_dir", "outputs/results/"))

    # Attach the best checkpoint to the (still-active) MLflow run so it appears
    # in the UI and can be registered via the "Register Model" button. Gated by
    # training.mlflow_log_model. Must run BEFORE finish_mlflow ends the run.
    if best_ckpt:
        log_model_artifact(config, best_ckpt)

    finish_wandb()
    finish_mlflow()
    logger.info("Done.")
 
if __name__ == "__main__":
    main()