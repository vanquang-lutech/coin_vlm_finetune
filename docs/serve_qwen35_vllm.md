# Serve Qwen3.5 with vLLM

This project serves vLLM from a standalone checkpoint directory. Do not point
`serving.model_path` at a LoRA adapter checkpoint directly; export a merged
model first, then optionally quantize that merged model.

vLLM's stable supported-models table lists Qwen3.5
(`Qwen3_5ForConditionalGeneration`) as a multimodal text-generation model. The
Qwen3.5 implementation is present in vLLM `0.17.0` and newer; for this server's
CUDA 12.x driver, pin `vllm==0.19.1`. Newer `0.22.x` pulls a torch/CUDA-13
stack that imports but cannot initialize CUDA on the current driver.

## 1. Build the serving environment

Use a fresh env for serving. Do not install the training requirements into this
env, and do not manually install a CUDA-13 torch wheel on the A100 box with a
CUDA 12.x driver.

```bash
conda create -n coin-vlm-serve python=3.11 -y
conda activate coin-vlm-serve
pip install uv

uv pip install -r requirements-serve.txt
uv pip install "vllm==0.19.1" --torch-backend=cu128

python -c "import torch; print(torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())"
```

Expected: `torch.version.cuda` is `12.x` and `torch.cuda.is_available()` is
`True`. If it prints `+cu130` / CUDA `13.0`, uninstall the mismatched stack:

```bash
python -m pip uninstall -y torch torchvision torchaudio vllm xformers triton
uv pip install -r requirements-serve.txt
uv pip install "vllm==0.19.1" --torch-backend=cu128
```

## 2. Export a Qwen3.5 checkpoint for serving

For the 9B model:

```bash
python scripts/export.py \
  --mode merge_lora \
  --data_config config/data/coin_dataset.yaml \
  --model_config config/model/qwen3.5_9b.yaml \
  --training_config config/training/training.yaml \
  --method_config config/backend/unsloth_qlora_qwen35.yaml \
  --adapter_path outputs/checkpoints/checkpoint-1000 \
  --max_shard_size 5GB \
  --output_dir outputs/merged_models/qwen35-9b-coin
```

The number of `.safetensors` files is only checkpoint packaging. vLLM can load
one shard or many shards. For a bf16 Qwen3.5-9B merged checkpoint, the total
directory size should be roughly the full model size; if the output is tiny, it
is not a real merged checkpoint. `--max_shard_size 5GB` normally produces several
shards, similar to the base model.

Optional W4A16 compression for vLLM:

```bash
python scripts/export.py \
  --mode awq \
  --data_config config/data/coin_dataset.yaml \
  --model_config config/model/qwen3.5_9b.yaml \
  --training_config config/training/training.yaml \
  --method_config config/backend/unsloth_qlora_qwen35.yaml \
  --model_path outputs/merged_models/qwen35-9b-coin \
  --output_dir outputs/merged_models/qwen35-9b-coin-awq
```

For 4B, use `config/model/qwen3.5_4b.yaml` and change the output directory
names accordingly.

## 3. Start the FastAPI server

If vLLM fails with many warnings like
`Parameter language_model.language_model... not found in params_dict`, normalize
the exported checkpoint once before serving:

```bash
python scripts/normalize_qwen35_vllm_keys.py outputs/merged_models/qwen35-9b-coin-awq
```

This rewrites only safetensors key metadata so vLLM can map the Qwen3.5 text
tower. It does not change tensor payload bytes.

For the 9B AWQ output:

```bash
python scripts/serve.py \
  --data_config config/data/coin_dataset.yaml \
  --model_config config/model/qwen3.5_9b.yaml \
  --training_config config/training/training.yaml \
  --serving_config config/serving/serving.yaml \
  --override \
    serving.model_path=outputs/merged_models/qwen35-9b-coin-awq \
    serving.quantization=null \
    serving.max_model_len=2560 \
    serving.gpu_memory_utilization=0.90 \
    serving.max_num_seqs=4
```

For a merged bf16 checkpoint instead of AWQ, keep `serving.quantization=null`
and set `serving.model_path=outputs/merged_models/qwen35-9b-coin`.

## 4. Test

```bash
curl http://localhost:49710/health

curl -X POST http://localhost:49710/predict \
  -F "obverse=@front.jpg" \
  -F "reverse=@back.jpg"
```

The API expects two images per request. It applies the same per-image
enhancement used during training, concatenates obverse and reverse side by side,
then sends the single combined image to vLLM.

## Notes

- Keep `serving.quantization=null` for this repo's llmcompressor
  compressed-tensors outputs. Do not force `serving.quantization=awq` unless the
  checkpoint was exported by AutoAWQ.
- `config/model/qwen3.5_9b.yaml` already excludes the tiny Qwen3.5
  `in_proj_ba` GatedDeltaNet projection from AWQ; keep that ignore list, because
  quantizing it can trigger vLLM Marlin shape failures and saves almost no VRAM.
- If startup OOMs, lower `serving.max_num_seqs` first, then lower
  `serving.gpu_memory_utilization` only if you need to leave memory for another
  process.
