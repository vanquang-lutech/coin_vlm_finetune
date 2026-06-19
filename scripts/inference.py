import argparse
import json
import os
import sys
from pathlib import Path

# Disable TorchDynamo before any torch import. Inference does not benefit from
# compile here, and on the unsloth backend the compiled generate path recompiles
# per prompt-length and leaks host RAM. Must precede the src.* imports below,
# which pull in torch transitively. Re-enable by exporting TORCHDYNAMO_DISABLE=0.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.utils import ConfigLoader, get_logger
from src.inference.predictor import CoinPredictor
 
logger = get_logger(__name__)
 
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
 
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned coin VLM.")
 
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
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
 
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image")
    group.add_argument("--image_dir", help="Path to a directory containing images to run batch inference on.")
 
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", default="outputs/results/predictions.json",
                        help="Path to save inference results.")
 
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
 
    predictor = CoinPredictor(config, checkpoint_path=args.checkpoint_path)
 
    if args.image:
        result = predictor.predict(args.image)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
 
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
 
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
 
    if not image_paths:
        logger.warning("No images found in: %s", image_dir)
        return
 
    logger.info("Found %d images in '%s'", len(image_paths), image_dir)
 
    results = predictor.predict_batch(image_paths, batch_size=args.batch_size)
 
    output_data = [
        {"file": p.name, **r}
        for p, r in zip(image_paths, results)
    ]
 
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
 
    logger.info("Saved %d predictions to: %s", len(output_data), output_path)
 
if __name__ == "__main__":
    main()