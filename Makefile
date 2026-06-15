# CT pipeline shortcuts — run on the GPU server (train container or conda env).
# See docs/mlops.md Phase 2. `make all` = train (+ test-eval) then gate+register.
.PHONY: train eval register all

# Configs — override on the command line, e.g.
#   make train MODEL=config/model/qwen3.5_4b.yaml BACKEND=config/backend/hf_peft_lora.yaml
DATA       ?= config/data/coin_dataset.yaml
MODEL      ?= config/model/qwen3_vl_8b.yaml
TRAIN      ?= config/training/training.yaml
BACKEND    ?= config/backend/unsloth_qlora.yaml
RESULTS    ?= outputs/results
METRICS    ?= $(RESULTS)/metrics.json

# Best checkpoint: train.py writes the path to $(RESULTS)/best_checkpoint.txt.
# Override: make register CKPT=outputs/checkpoints/coin-vlm_<ts>/checkpoint-500
CKPT       ?= $(shell cat $(RESULTS)/best_checkpoint.txt 2>/dev/null)

# Quality gate (Phase 3.3)
METRIC     ?= extract_match
THRESHOLD  ?= 0.90
MODEL_NAME ?= coin-vlm

train:
	python scripts/train.py \
	  --data_config $(DATA) --model_config $(MODEL) \
	  --training_config $(TRAIN) --method_config $(BACKEND)

eval:
	@test -n "$(CKPT)" || (echo "CKPT empty — run 'make train' first or pass CKPT=..."; exit 1)
	python scripts/evaluate.py \
	  --checkpoint_path $(CKPT) \
	  --data_config $(DATA) --model_config $(MODEL) --training_config $(TRAIN)

register:
	@test -n "$(CKPT)" || (echo "CKPT empty — run 'make train' first or pass CKPT=..."; exit 1)
	python scripts/register_model.py \
	  --checkpoint_path $(CKPT) --metrics $(METRICS) \
	  --metric $(METRIC) --threshold $(THRESHOLD) --model-name $(MODEL_NAME)

# Full CT: train (train.py also runs test-eval -> metrics.json + best_checkpoint.txt),
# then gate + register the best checkpoint.
all: train register
