import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import merge_lora, export_awq, export_gguf
from src.utils import ConfigLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Export fine-tuned coin VLM.")

    parser.add_argument("--mode", required=True, choices=["merge_lora", "awq", "gguf"])
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
    parser.add_argument("--output_dir", required=True, help="Directory to save output.")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")

    # merge_lora args
    parser.add_argument("--adapter_path", help="[merge_lora] Path to LoRA adapter checkpoint.")
    parser.add_argument(
        "--base_model",
        default=None,
        help="[merge_lora] Override base model to merge into. "
             "Defaults to base_model_name_or_path in adapter_config.json.",
    )

    # awq args
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=256,
        help="[awq] Number of calibration samples drawn from the dataset.",
    )
    parser.add_argument(
        "--calibration_split",
        default="train",
        help="[awq] Dataset split to draw calibration samples from.",
    )

    # gguf / awq shared args
    parser.add_argument("--model_path", help="[awq|gguf] Path to merged model.")
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
        help="[gguf] Quantization type.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = ConfigLoader.load(
        data_config=args.data_config,
        model_config=args.model_config,
        training_config=args.training_config,
        method_config=args.method_config,
        inference_config=args.inference_config,
        overrides=args.override,
    )

    if args.mode == "merge_lora":
        if not args.adapter_path:
            raise ValueError("--adapter_path is required with mode=merge_lora.")
        merge_lora(config, args.adapter_path, args.output_dir, base_model=args.base_model)

    elif args.mode == "awq":
        if not args.model_path:
            raise ValueError("--model_path (merged model) is required with mode=awq.")
        export_awq(
            config,
            args.model_path,
            args.output_dir,
            num_calibration_samples=args.num_calibration_samples,
            calibration_split=args.calibration_split,
        )

    elif args.mode == "gguf":
        if not args.model_path:
            raise ValueError("--model_path is required with mode=gguf.")
        export_gguf(args.model_path, args.output_dir, args.quantization)


if __name__ == "__main__":
    main()
