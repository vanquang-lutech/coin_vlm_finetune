import json
import logging
from typing import Any
import torch

logger = logging.getLogger(__name__)

class CoinDataCollator:

    def __init__(self, processor, config):
        self.processor = processor
        self.config = config
        self.prompt = self._resolve_prompt()

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        messages_batch = [self._build_messages(label) for label in labels]

        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
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

    def _resolve_prompt(self):
        model_prompt = self.config.model.get("prompt", None)
        if model_prompt is not None:
            return model_prompt

        prompt = self.config.get("prompt", None)
        if prompt is not None:
            return prompt

        data_prompt = self.config.data.get("prompt", None)
        if data_prompt is not None:
            return data_prompt

        raise ValueError(
            "Prompt config not found. Set model.prompt in model config or prompt in inference/data config."
        )

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
        """Mask instruction tokens (system + user + assistant header) with -100,
        keeping only the assistant response content for training.

        Uses token-level search in input_ids to correctly handle image tokens
        that get expanded during processor encoding.
        """

        labels = input_ids.clone()

        assistant_token = self._get_assistant_start_token()
        if not assistant_token:
            logger.warning("No assistant start token found. Labels will not be masked.")
            return labels

        # Tokenize the assistant header (including trailing newline) to search in input_ids
        assistant_header = assistant_token + "\n"
        header_ids = self.processor.tokenizer.encode(
            assistant_header,
            add_special_tokens=False,
        )

        for i in range(input_ids.shape[0]):
            seq = input_ids[i].tolist()

            # Search for assistant header token IDs in input_ids (from end, in case of duplicates)
            match_pos = self._find_last_subsequence(seq, header_ids)

            if match_pos is None:
                # Fallback: try without trailing newline
                header_ids_no_nl = self.processor.tokenizer.encode(
                    assistant_token,
                    add_special_tokens=False,
                )
                match_pos = self._find_last_subsequence(seq, header_ids_no_nl)
                if match_pos is not None:
                    mask_end = match_pos + len(header_ids_no_nl)
                else:
                    # Last resort: fall back to text-based approach (may be inaccurate)
                    logger.warning(
                        "Could not find assistant header in input_ids (sample %d). "
                        "Falling back to text-based masking.",
                        i,
                    )
                    instruction_text = self._get_instruction_text(texts[i], assistant_token)
                    fallback_ids = self.processor.tokenizer(
                        instruction_text,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids
                    mask_end = fallback_ids.shape[-1]
            else:
                mask_end = match_pos + len(header_ids)

            labels[i, :mask_end] = -100

            # Debug: log masking info for first sample of first batch
            if i == 0 and not hasattr(self, '_mask_debug_done'):
                self._mask_debug_done = True
                n_total = input_ids[i].shape[0]
                n_masked = mask_end
                n_trained = n_total - n_masked
                response_ids = input_ids[i, mask_end:]
                response_text = self.processor.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                logger.info(
                    "[Label Masking] Total tokens: %d | Masked (instruction): %d | "
                    "Trained (response): %d | Response preview: '%s'",
                    n_total, n_masked, n_trained, response_text[:200],
                )

        return labels

    @staticmethod
    def _find_last_subsequence(seq: list[int], subseq: list[int]) -> int | None:
        """Find the start index of the last occurrence of subseq in seq."""
        n, m = len(seq), len(subseq)
        if m == 0 or m > n:
            return None
        for i in range(n - m, -1, -1):
            if seq[i:i + m] == subseq:
                return i
        return None

    def _get_instruction_text(self, full_text: str, assistant_token: str) -> str:
        """Fallback: extract instruction portion from raw text string."""
        if assistant_token in full_text:
            idx = full_text.index(assistant_token) + len(assistant_token)
            return full_text[:idx]

        logger.warning(
            "Could not find assistant start token '%s'. "
            "Labels will not be masked correctly.",
            assistant_token,
        )
        return full_text

    def _get_assistant_start_token(self) -> str | None:

        prompt_token = getattr(self.prompt, "assistant_start_token", None)
        if prompt_token:
            return prompt_token

        model_token = self.config.model.get("assistant_start_token", None)
        if model_token:
            return model_token

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