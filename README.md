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

### Serve (FastAPI + vLLM)

Serve a merged (+ optionally AWQ-quantized) checkpoint behind a FastAPI API.
vLLM loads the model once at startup and the engine is shared across requests.

First produce a servable checkpoint (merge, then optionally quantize):

```bash
# 1) merge LoRA adapter -> standalone bf16 checkpoint
python scripts/export.py --mode merge_lora ... \
	--adapter_path outputs/checkpoints/checkpoint-1000 \
	--output_dir outputs/merged_models/coin-vlm

# 2) (optional) AWQ W4A16 quantize for a smaller/faster serve
python scripts/export.py --mode awq ... \
	--model_path outputs/merged_models/coin-vlm \
	--output_dir outputs/merged_models/coin-vlm-awq
```

Point `serving.model_path` at the result and launch:

```bash
python scripts/serve.py \
	--data_config config/data/coin_dataset.yaml \
	--model_config config/model/qwen3_vl_8b.yaml \
	--training_config config/training/training.yaml \
	--serving_config config/serving/serving.yaml \
	--override serving.model_path=outputs/merged_models/coin-vlm-awq serving.quantization=awq
```

For a non-quantized merged model, set `serving.quantization=null`.

Query it:

```bash
curl http://localhost:49710/health
curl -X POST http://localhost:49710/predict -F "file=@path/to/coin.jpg"
# -> {"year": "1921", "mint_mark": "S", "raw": "...", "parse_ok": true}
```

**Public access via ngrok.** To test the API from outside the A100 box, the
server opens an ngrok tunnel on startup (config `serving.ngrok`). It uses a
reserved/static domain so the URL is stable. A reserved domain needs an
authtoken — set it once in the env:

```bash
export NGROK_AUTHTOKEN=<your-ngrok-token>
python scripts/serve.py ... --serving_config config/serving/serving.yaml
# -> log: "ngrok tunnel up: https://embezzle-eastcoast-armoire.ngrok-free.dev -> http://localhost:49710"

curl https://embezzle-eastcoast-armoire.ngrok-free.dev/health
```

Disable with `serving.ngrok.enabled=false` (or `--override serving.ngrok.enabled=false`).
The tunnel is best-effort: if pyngrok/authtoken is missing it logs an error and
the server still runs locally.

Generation settings and the prompt come from `config/inference/inference.yaml`,
so the served output matches `scripts/inference.py`. The image **resolution** is
also kept in sync: `processor.min_pixels/max_pixels` (in vision-token units) are
converted to raw pixels and passed to vLLM via `mm_processor_kwargs`, mirroring
`CoinPredictor._apply_processor_overrides` — so the served input resolution
equals the HF inference path rather than whatever was baked into the checkpoint.
vLLM engine settings (VRAM, max length, concurrency, quantization) live in
`config/serving/serving.yaml`. Install vLLM in its serving env: `pip install "vllm>=0.6.6"`.

**Input enhancement (important).** The training dataset was enhanced with CLAHE +
unsharp mask (`src/data/preprocessing.py:CoinEnhancer`). The API applies the
**same** enhancement to every uploaded image before inference — otherwise the
model sees a different image distribution than it trained on (train/serve skew).
Controlled by `serving.preprocess` (defaults match the dataset build: CLAHE
clip=2.0/tile=8×8, unsharp sigma=2.0/amount=1.5, `mode: full`). Set
`serving.preprocess.enhance=false` only if you serve a model trained on raw images.

**Logs.** Date-partitioned under `serving.log_dir` (default `outputs/logs/serving/`),
rolling over by `serving.log_rotation` (`daily` | `monthly`). Daily layout:

```
outputs/logs/serving/2026-05/serve-2026-05-29.log          # app + server logs (also stdout)
outputs/logs/serving/2026-05/predictions-2026-05-29.jsonl  # one line per request
```

Monthly layout: `outputs/logs/serving/2026/serve-2026-05.log`. A long-running
server rolls to a new file automatically when the day/month changes. Each
`predictions-*.jsonl` line is `{ts, file, year, mint_mark, parse_ok, raw, latency_ms}`
for auditing / offline re-scoring. Disable the JSONL sink with `serving.predictions_log=false`.

### Notes

- For Unsloth, set `model.unsloth_name` in the model config to an Unsloth repo.
- QLoRA requires bitsandbytes and a CUDA GPU.
- Generation settings come from `config/inference/inference.yaml`; set `prompt` there to override inference/eval prompts (training still uses model prompt).
- Set `prompt.assistant_start_token` in the model config if the assistant token is not inferred correctly.
