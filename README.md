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

```bash
pip install -r requirements.txt
```

### Config layout

- `config/data/coin_dataset.yaml`: HF dataset name and prompt template
- `config/model/*.yaml`: base model identity and processor defaults
- `config/training/training.yaml`: training hyperparameters
- `config/method/*.yaml`: backend overrides (hf_peft, unsloth, full_finetune)
- `config/inference/inference.yaml`: generation settings and inference prompt

Method configs are merged on top of model/training configs when you pass
`--method_config`.

### Train

```bash
python scripts/train.py \
	--data_config config/data/coin_dataset.yaml \
	--model_config config/model/qwen3.5_9b.yaml \
	--training_config config/training/training.yaml \
	--method_config config/method/hf_peft_lora.yaml
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
	--method_config config/method/hf_peft_lora.yaml \
	--checkpoint_path outputs/checkpoints/checkpoint-1000
```

### Inference

```bash
python scripts/inference.py \
	--data_config config/data/coin_dataset.yaml \
	--model_config config/model/qwen3.5_9b.yaml \
	--training_config config/training/training.yaml \
	--method_config config/method/hf_peft_lora.yaml \
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
	--method_config config/method/hf_peft_lora.yaml \
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
	--method_config config/method/hf_peft_lora.yaml \
	--model_path outputs/merged_models/coin-vlm \
	--output_dir outputs/merged_models \
	--quantization Q4_K_M
```

### Notes

- For Unsloth, set `model.name` in the method config to an Unsloth repo.
- QLoRA requires bitsandbytes and a CUDA GPU.
- Inference prompt and generation settings come from `config/inference/inference.yaml`.
