import argparse
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.utils import ConfigLoader, finish_wandb, get_logger, init_wandb, set_seed
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
 
    set_seed(config.training.seed)
    init_wandb(config)
 
    loader = get_model_loader(config)
    model, processor = loader.load()
 
    logger.info("Loading datasets...")
    train_ds = CoinDataset(config, split="train")
    val_ds = CoinDataset(config, split="validation")
 
    trainer = get_trainer(config, model, processor, train_ds, val_ds)
    trainer.train()

    
    if not args.skip_test_eval:
        logger.info("Running evaluation on test set...")
        test_ds = CoinDataset(config, split="test")
        evaluator = CoinEvaluator(config, model=trainer.get_model(), processor=processor)
        results = evaluator.evaluate(test_ds)
        evaluator.save_results(results, config.training.get("result_dir", "outputs/results/"))
 
    finish_wandb()
    logger.info("Done.")
 
if __name__ == "__main__":
    main()