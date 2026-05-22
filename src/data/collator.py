import json
import logging
from typing import Any
import torch

logger = logging.getLogger(__name__)

class CoinDataCollator:

    def __init__(self, processor, config):
        self.processor = processor
        self.config = config
        self.prompt = config.data.prompt

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        messages_batch = [self._build_messages(label) for label in labels]

        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,    
            ) for messages in messages_batch
        ]

        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        inputs["labels"] = self._mask_labels(
            input_ids=inputs["input_ids"],
            texts=texts,
        )

        return inputs

    def _build_messages(self, label: dict) -> list[dict]:

        return [
            {
                "role": "system",
                "content": self.prompt.system,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},                         
                    {"type": "text", "text": self.prompt.user},
                ],
            },
            {
                "role": "assistant",
                "content": self._build_response(label),
            },
        ]

    def _build_response(self, label: dict) -> str:

        template: dict = self.prompt.response_template

        resolved = {
            key: label[field.strip("{}")]
            for key, field in template.items()
            if field.strip("{}") in label
        }

        if self.prompt.response_format == "json":
            return json.dumps(resolved, ensure_ascii=False)

        return ", ".join(f"{k}: {v}" for k, v in resolved.items())

    def _mask_labels(
        self,
        input_ids: torch.Tensor,
        texts: list[str],
    ) -> torch.Tensor:

        labels = input_ids.clone()

        for i, text in enumerate(texts):
            instruction_text = self._get_instruction_only(text)

            instruction_ids = self.processor.tokenizer(
                instruction_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids

            instruction_len = instruction_ids.shape[-1]

            labels[i, :instruction_len] = -100

        return labels

    def _get_instruction_only(self, full_text: str) -> str:

        assistant_token = self._get_assistant_start_token()
        if assistant_token and assistant_token in full_text:
            idx = full_text.index(assistant_token) + len(assistant_token)
            return full_text[:idx]

        logger.warning(
            "Could not find assistant start token '%s'. "
            "Labels will not be masked correctly.",
            assistant_token,
        )
        return full_text

    def _get_assistant_start_token(self) -> str | None:

        model_name: str = self.config.model.model_name.lower()

        if "qwen" in model_name:
            return "<|im_start|>assistant"
        if "llama" in model_name:
            return "<|start_header_id|>assistant<|end_header_id|>"
        if "gemma" in model_name or "paligemma" in model_name:
            return "<start_of_turn>model"
        if "mistral" in model_name:
            return "[/INST]"

        logger.warning(
            "Unknown model '%s', cannot determine assistant start token.",
            self.config.model.model_name,
        )
        
        return None