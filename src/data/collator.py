import json
import logging
from typing import Any
import torch

from src.utils import safe_template_kwargs, is_prefix_suffix, resolve_prefix

logger = logging.getLogger(__name__)

class CoinDataCollator:

    def __init__(self, processor, config):
        self.processor = processor
        self.config = config
        self.prompt = self._resolve_prompt()
        # PaliGemma-style models bypass chat templating entirely (see
        # _call_prefix_suffix). Resolve the fixed prefix once here.
        self.prefix_suffix = is_prefix_suffix(config)
        if self.prefix_suffix:
            self.prefix_text = resolve_prefix(config)
            logger.info(
                "[Collator] prefix_suffix mode (PaliGemma). prefix=%r",
                self.prefix_text,
            )
        # `enable_thinking` is a Qwen3 text / *-Thinking* chat-template variable.
        # Templates that don't actually read it (Qwen3-VL-Instruct, Qwen2.5,
        # InternVL3) make transformers>=5.4 warn on every apply_chat_template
        # call ("Kwargs passed to processor.__call__ have to be in
        # processor_kwargs dict, not in **kwargs"). safe_template_kwargs forwards
        # it only when the active template declares it as a free variable.
        self._template_kwargs = safe_template_kwargs(
            self.processor, {"enable_thinking": False}
        )
        self._maybe_limit_image_patches()

    def _maybe_limit_image_patches(self) -> None:
        """Cap InternVL dynamic tiling so one image can't blow past max_seq_length.

        InternVL splits an image into up to `max_patches` tiles (default 12) of
        `image_seq_length` (256) tokens each, plus a thumbnail when tiled. A
        high-tile image (e.g. 2560 tokens) then gets cut by the collator's
        truncation, and InternVL's processor raises "Mismatch in image token
        count ... Likely due to truncation". Bounding max_patches keeps every
        image comfortably under max_seq_length. No-op for processors without
        this attribute (e.g. Qwen), so it's safe in the shared collator.
        """
        max_patches = self.config.model.get("image_max_patches", None)
        if max_patches is None:
            return
        img_proc = getattr(self.processor, "image_processor", None)
        if img_proc is not None and hasattr(img_proc, "max_patches"):
            img_proc.max_patches = int(max_patches)
            logger.info(
                "[Collator] Capped InternVL image tiling: max_patches=%d", int(max_patches)
            )
        else:
            logger.warning(
                "[Collator] model.image_max_patches=%s set, but processor has no "
                "image_processor.max_patches; ignoring.", max_patches
            )

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        if self.prefix_suffix:
            return self._call_prefix_suffix(batch)

        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        messages_batch = [self._build_messages(label) for label in labels]

        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                **self._template_kwargs,
            ) for messages in messages_batch
        ]

        # Prompt-only version (system + user, with assistant header appended
        # via add_generation_prompt=True). Used to compute exact mask boundary
        # per sample without relying on BPE-fragile subsequence search.
        prompt_only_messages = [
            [m for m in msgs if m["role"] != "assistant"] for msgs in messages_batch
        ]
        prompt_only_texts = [
            self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                **self._template_kwargs,
            ) for msgs in prompt_only_messages
        ]

        # Sanity check (one-time): the prompt-only render with
        # add_generation_prompt=True must be a textual prefix of the full
        # render with add_generation_prompt=False. If the chat template
        # diverges between these two modes (e.g. inserts different
        # whitespace, role tags, or system additions only in one mode),
        # the prompt-only length we compute below will NOT correspond to
        # the assistant-response boundary in the full input_ids, and
        # masking will be silently wrong. Fail loudly on first batch.
        if not getattr(self, "_template_prefix_checked", False):
            self._template_prefix_checked = True
            for i, (full, prefix) in enumerate(zip(texts, prompt_only_texts)):
                if not full.startswith(prefix):
                    raise RuntimeError(
                        "Chat template divergence detected (sample %d): "
                        "prompt-only render (add_generation_prompt=True) is NOT "
                        "a prefix of the full render (add_generation_prompt=False). "
                        "Label masking would be incorrect.\n"
                        "--- prompt-only (last 200 chars) ---\n%s\n"
                        "--- full (last 200 chars) ---\n%s"
                        % (i, prefix[-200:], full[-200:])
                    )
            logger.info(
                "[Collator] Chat template prefix check passed "
                "(prompt-only render is a prefix of full render)."
            )

        max_seq_length = self.config.training.get("max_seq_length", None)
        processor_kwargs = dict(
            images=images,
            return_tensors="pt",
            padding=True,
        )
        if max_seq_length is not None:
            processor_kwargs["truncation"] = True
            processor_kwargs["max_length"] = max_seq_length

        inputs = self.processor(text=texts, **processor_kwargs)

        # Tokenize prompt-only with SAME images so image-token expansion is
        # identical to the full pass. The non-pad length of each prompt-only
        # row is the exact index in input_ids where the assistant response
        # content begins.
        prompt_inputs = self.processor(text=prompt_only_texts, **processor_kwargs)
        prompt_attn = prompt_inputs.get("attention_mask")
        if prompt_attn is None:
            # Defensive: fall back to full length, masking everything; the
            # all-masked safeguard inside _mask_labels will recover.
            prompt_lengths = [prompt_inputs["input_ids"].shape[1]] * len(texts)
        else:
            prompt_lengths = prompt_attn.sum(dim=1).tolist()

        inputs["labels"] = self._mask_labels(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            prompt_lengths=prompt_lengths,
        )

        return inputs

    def _call_prefix_suffix(self, batch: list[dict]) -> dict[str, Any]:
        """PaliGemma-style collation (prefix/suffix, no chat template).

        PaliGemma 2 is not a chat model: it ships no chat template and expects
        the prefix prompt as ``text`` and the target answer as ``suffix``. The
        processor expands ``<image>`` tokens, prepends ``<bos>``, appends
        ``<eos>`` to the suffix, and returns a ``labels`` tensor where the image
        tokens + prefix are ALREADY masked to -100 (via ``token_type_ids``). So
        we do no manual masking and no apply_chat_template here.
        """
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        prefixes = [self.prefix_text] * len(images)
        suffixes = [self._build_response(label) for label in labels]

        # No truncation on purpose: each image expands to a FIXED number of
        # image tokens (896px -> 4096). Truncating would cut image tokens and
        # trip PaliGemma's image-token-count assertion. The suffix (short JSON)
        # never needs truncation. Set training.max_seq_length high enough that
        # the model's position budget covers image_seq_length + prefix + suffix.
        inputs = self.processor(
            images=images,
            text=prefixes,
            suffix=suffixes,
            return_tensors="pt",
            padding=True,
        )

        if "labels" not in inputs:
            raise RuntimeError(
                "PaliGemma processor returned no `labels`. Expected the "
                "processor to build labels when `suffix=` is given (via "
                "token_type_ids). Check transformers/processor version."
            )

        self._log_prefix_suffix_debug_once(inputs)
        return inputs

    def _log_prefix_suffix_debug_once(self, inputs: dict[str, Any]) -> None:
        """One-time sanity log: confirm only the suffix carries loss."""
        if getattr(self, "_mask_debug_done", False):
            return
        self._mask_debug_done = True
        labels0 = inputs["labels"][0]
        n_total = int(labels0.shape[0])
        n_trained = int(labels0.ne(-100).sum().item())
        trained_ids = inputs["input_ids"][0][labels0.ne(-100)]
        decoded = self.processor.tokenizer.decode(
            trained_ids, skip_special_tokens=True
        )
        logger.info(
            "[PaliGemma Collation] Total tokens: %d | Ignored (image+prefix/pad): "
            "%d | Trained (suffix): %d | prefix=%r | suffix preview: '%s'",
            n_total, n_total - n_trained, n_trained, self.prefix_text, decoded[:200],
        )

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
        attention_mask: torch.Tensor | None,
        prompt_lengths: list[int],
    ) -> torch.Tensor:
        """Mask instruction tokens (system + user + assistant header) with -100,
        keeping only the assistant response content for training.

        ``prompt_lengths[i]`` is the number of non-pad tokens produced when
        the same image + prompt-only template (with ``add_generation_prompt=
        True``) is run through the processor. Because both the prompt-only
        pass and the full pass go through the SAME processor pipeline (same
        chat template, same image-token expansion, same right-padding), this
        length is the exact index in ``input_ids[i]`` where the assistant
        response content begins — no BPE-fragile subsequence search needed.
        """

        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask.eq(0), -100)

        seq_len = labels.shape[1]

        for i in range(input_ids.shape[0]):
            mask_end = min(int(prompt_lengths[i]), seq_len)
            labels[i, :mask_end] = -100

            # Safety: ensure at least one trainable token exists for this sample.
            # If everything ended up masked (truncation cut off the assistant
            # response, or header search overshot), unmask the final token so
            # the loss does not collapse to NaN (mean over empty selection).
            if not labels[i].ne(-100).any():
                logger.error(
                    "All tokens masked for sample %d (seq_len=%d, mask_end=%d). "
                    "This sample contributes no training signal; check max_seq_length "
                    "and assistant-header detection.",
                    i, seq_len, mask_end,
                )
                # Restore the final non-pad position so the batch loss stays finite.
                if attention_mask is not None:
                    last_valid = int(attention_mask[i].sum().item()) - 1
                else:
                    last_valid = seq_len - 1
                if last_valid >= 0:
                    labels[i, last_valid] = input_ids[i, last_valid]

            # Debug: log masking info for first sample of first batch
            if i == 0 and not hasattr(self, '_mask_debug_done'):
                self._mask_debug_done = True
                n_total = input_ids[i].shape[0]
                n_ignored = labels[i].eq(-100).sum().item()
                n_trained = labels[i].ne(-100).sum().item()
                response_ids = input_ids[i][labels[i].ne(-100)]
                response_text = self.processor.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                logger.info(
                    "[Label Masking] Total tokens: %d | Ignored (prompt/pad): %d | "
                    "Trained (response): %d | Response preview: '%s'",
                    n_total, n_ignored, n_trained, response_text[:200],
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
