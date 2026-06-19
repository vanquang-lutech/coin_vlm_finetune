import argparse
import os
import sys
from pathlib import Path

# Disable TorchDynamo before any torch import. Standalone eval never benefits
# from compile (generation is forced eager anyway), and on the unsloth backend
# the compiled generate path recompiles per prompt-length and leaks host RAM.
# Must precede the src.* imports below, which pull in torch transitively.
# Re-enable compile by exporting TORCHDYNAMO_DISABLE=0.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.utils import ConfigLoader, get_logger, set_seed
from src.data.dataset import CoinDataset
from src.evaluate.evaluator import CoinEvaluator
 
logger = get_logger(__name__)
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned coin VLM.")
 
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
        "--checkpoint_path",
        default=None,
        help="Path to checkpoint. Omit when using --base_model.",
    )
    parser.add_argument(
        "--base_model",
        action="store_true",
        help="Evaluate the un-finetuned base model from config.model.name (ignores --checkpoint_path).",
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output_dir", default="outputs/results/")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
 
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

    # Mirror train.py: unsloth must be imported BEFORE transformers for its
    # patches to apply. Without this, FastVisionModel falls back to the slow
    # path and 4-bit kernels won't engage during eval.
    if config.model.backend == "unsloth" or config.training.backend == "unsloth":
        import unsloth  # noqa: F401

    set_seed(
        config.training.seed,
        deterministic=config.training.get("deterministic", False),
    )
 
    if not args.base_model and not args.checkpoint_path:
        raise SystemExit("Must provide --checkpoint_path or --base_model.")

    logger.info("Loading '%s' split...", args.split)
    dataset = CoinDataset(config, split=args.split)

    if args.base_model:
        logger.info("Evaluating BASE (un-finetuned) model: %s", config.model.name)
        evaluator = CoinEvaluator(config, load_base=True)
        output_dir = str(Path(args.output_dir) / "baseline")
    else:
        evaluator = CoinEvaluator(config, checkpoint_path=args.checkpoint_path)
        output_dir = args.output_dir

    results = evaluator.evaluate(dataset)
    evaluator.save_results(results, output_dir)
 
 
if __name__ == "__main__":
    main()