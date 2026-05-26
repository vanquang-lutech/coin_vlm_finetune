import argparse
import sys
from pathlib import Path
 
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
    parser.add_argument("--checkpoint_path", required=True, help="Path to checkpoint.")
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
 
    set_seed(
        config.training.seed,
        deterministic=config.training.get("deterministic", False),
    )
 
    logger.info("Loading '%s' split...", args.split)
    dataset = CoinDataset(config, split=args.split)
 
    evaluator = CoinEvaluator(config, checkpoint_path=args.checkpoint_path)
    results = evaluator.evaluate(dataset)
 
    evaluator.save_results(results, args.output_dir)
 
 
if __name__ == "__main__":
    main()