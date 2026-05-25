## Coin Fine-tune VLM

Fine-tune a vision-language model to extract coin metadata. The current pipeline targets a JSON response with:

```json
{"year": "YYYY", "mint_mark": "X"}
```

### Requirements

- Python 3.10+
- CUDA GPU recommended
- bitsandbytes for QLoRA (4-bit)
- Access token for gated models (if applicable)

### Install

Install requirements first, then pin `transformers`/`tokenizers` and install Unsloth + Unsloth Zoo.

```bash
pip install -r requirements.txt

python -m pip install -U --no-cache-dir --no-deps "transformers==4.56.2"
python -m pip install -U --no-cache-dir --no-deps "tokenizers>=0.22.0,<0.23.0"
python -m pip install --no-cache-dir --no-deps "unsloth==2026.5.7"
python -m pip install --no-cache-dir --no-deps "unsloth_zoo==2026.5.4"
```

### Config layout

- `config/data/coin_dataset.yaml`: HF dataset name and split settings
- `config/model/*.yaml`: base model identity, processor defaults, and prompt template
- `config/training/training.yaml`: training hyperparameters
- `config/backend/*.yaml`: backend overrides (hf_peft, unsloth, full_finetune)
- `config/inference/inference.yaml`: generation settings (optional prompt override)

Backend configs are merged on top of model/training configs when you pass
`--method_config`.

### Train

```bash
python scripts/train.py \
	--data_config config/data/coin_dataset.yaml \
	--model_config config/model/qwen3.5_9b.yaml \
	--training_config config/training/training.yaml \
	--method_config config/backend/hf_peft_lora.yaml
```

Override any config value:

```bash
python scripts/train.py ... \
	--override training.learning_rate=1e-4
```

### Evaluate

```bash
python scripts/evaluate.py \
	--data_config config/data/coin_dataset.yaml \
	--model_config config/model/qwen3.5_9b.yaml \
	--training_config config/training/training.yaml \
	--method_config config/backend/hf_peft_lora.yaml \
	--checkpoint_path outputs/checkpoints/checkpoint-1000
```

### Inference

```bash
python scripts/inference.py \
	--data_config config/data/coin_dataset.yaml \
	--model_config config/model/qwen3.5_9b.yaml \
	--training_config config/training/training.yaml \
	--method_config config/backend/hf_peft_lora.yaml \
	--checkpoint_path outputs/checkpoints/checkpoint-1000 \
	--image path/to/coin.jpg
```

### Export

Merge LoRA into base model:

```bash
python scripts/export.py \
	--mode merge_lora \
	--data_config config/data/coin_dataset.yaml \
	--model_config config/model/qwen3.5_9b.yaml \
	--training_config config/training/training.yaml \
	--method_config config/backend/hf_peft_lora.yaml \
	--adapter_path outputs/checkpoints/checkpoint-1000 \
	--output_dir outputs/merged_models/coin-vlm
```

Export to GGUF (requires llama.cpp):

```bash
python scripts/export.py \
	--mode gguf \
	--data_config config/data/coin_dataset.yaml \
	--model_config config/model/qwen3.5_9b.yaml \
	--training_config config/training/training.yaml \
	--method_config config/backend/hf_peft_lora.yaml \
	--model_path outputs/merged_models/coin-vlm \
	--output_dir outputs/merged_models \
	--quantization Q4_K_M
```

### Notes

- For Unsloth, set `model.unsloth_name` in the model config to an Unsloth repo.
- QLoRA requires bitsandbytes and a CUDA GPU.
- Generation settings come from `config/inference/inference.yaml`; set `prompt` there to override inference/eval prompts (training still uses model prompt).
- Set `prompt.assistant_start_token` in the model config if the assistant token is not inferred correctly.
