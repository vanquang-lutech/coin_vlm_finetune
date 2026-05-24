import argparse
import subprocess
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.utils import ConfigLoader, get_logger
 
logger = get_logger(__name__)
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Export fine-tuned coin VLM.")
 
    parser.add_argument("--mode", required=True, choices=["merge_lora", "gguf"])
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
 
    # gguf args
    parser.add_argument("--model_path",   help="[gguf] Path to merged model.")
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
        help="[gguf] Quantization type.",
    )
 
    return parser.parse_args()
 
 
def merge_lora(config, adapter_path: str, output_dir: str):

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
 
    adapter_path = Path(adapter_path)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    if not (adapter_path / "adapter_config.json").exists():
        raise ValueError(
            f"Could not find adapter_config.json in: {adapter_path}. "
            "Is this a LoRA checkpoint?"
        )
 
    logger.info("Loading base model '%s'...", config.model.name)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model.name,
        torch_dtype = torch.bfloat16,
        device_map  = "auto",
    )
 
    logger.info("Loading LoRA adapter from '%s'...", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
 
    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()
 
    logger.info("Saving merged model to '%s'...", output_dir)
    model.save_pretrained(output_dir)
 
    processor = AutoProcessor.from_pretrained(adapter_path)
    processor.save_pretrained(output_dir)
 
    logger.info("Merge complete. Model saved to: %s", output_dir)
 
def export_gguf(model_path: str, output_dir: str, quantization: str):

    model_path  = Path(model_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    model_name  = model_path.name
    gguf_name   = f"{model_name}-{quantization}.gguf"
    output_path = output_dir / gguf_name
 
    llamacpp_convert = Path("llama.cpp/convert_hf_to_gguf.py")
    if not llamacpp_convert.exists():
        raise FileNotFoundError(
            "Could not find llama.cpp/convert_hf_to_gguf.py. "
            "Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
        )
 
    logger.info("Converting '%s' to GGUF (%s)...", model_path, quantization)
 
    cmd = [
        sys.executable, str(llamacpp_convert),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", quantization.lower(),
    ]
 
    result = subprocess.run(cmd, capture_output=True, text=True)
 
    if result.returncode != 0:
        logger.error("GGUF conversion failed:\n%s", result.stderr)
        raise RuntimeError("GGUF conversion failed.")
 
    logger.info("GGUF export complete. Saved to: %s", output_path)
 
 
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
 
    if args.mode == "merge_lora":
        if not args.adapter_path:
            raise ValueError("--adapter_path is required with mode=merge_lora.")
        merge_lora(config, args.adapter_path, args.output_dir)
 
    elif args.mode == "gguf":
        if not args.model_path:
            raise ValueError("--model_path is required with mode=gguf.")
        export_gguf(args.model_path, args.output_dir, args.quantization)
 
if __name__ == "__main__":
    main()