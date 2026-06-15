"""Prompt-style helpers: ``chat`` vs ``prefix_suffix``.

Most VLMs in this repo (Qwen3-VL, InternVL3.5, Qwen3.5) are conversational and
consume inputs via ``processor.apply_chat_template()`` with system/user/assistant
roles. PaliGemma 2 is fundamentally different: it is NOT a chat model, ships no
chat template, and uses the original PaliGemma *prefix/suffix* format. There the
processor receives ``text=<prefix>`` and ``suffix=<answer>`` and builds the loss
``labels`` itself (masking the image tokens + prefix with -100 via
``token_type_ids``) — so the collator must NOT call ``apply_chat_template`` and
must NOT mask labels by hand.

A single ``model.prompt_style`` flag lets the shared collator / predictor /
evaluator pick the right path. Default is ``chat`` so existing configs are
unaffected.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_STYLE_CHAT = "chat"
PROMPT_STYLE_PREFIX_SUFFIX = "prefix_suffix"
_VALID_STYLES = {PROMPT_STYLE_CHAT, PROMPT_STYLE_PREFIX_SUFFIX}


def get_prompt_style(config) -> str:
    """Return ``model.prompt_style`` (default ``"chat"``).

    Unknown values raise — better to fail loudly than silently train PaliGemma
    through a chat template (which mis-masks labels) or vice-versa.
    """
    style = config.model.get("prompt_style", PROMPT_STYLE_CHAT)
    if style not in _VALID_STYLES:
        raise ValueError(
            f"Unknown model.prompt_style={style!r}. "
            f"Expected one of {sorted(_VALID_STYLES)}."
        )
    return style


def is_prefix_suffix(config) -> bool:
    """True when the model uses the PaliGemma-style prefix/suffix pipeline."""
    return get_prompt_style(config) == PROMPT_STYLE_PREFIX_SUFFIX


def resolve_prefix(config) -> str:
    """The prefix prompt text for a ``prefix_suffix`` model.

    Resolution order:
      1. ``prompt.prefix`` (an explicit inference-time prefix override), then
      2. ``model.prompt.prefix`` (the training prefix), then
      3. ``model.prompt.user`` (last-resort fallback on the training prompt).

    We deliberately do NOT fall back to a chat-style ``prompt.user`` from
    ``inference.yaml``: a fine-tuned PaliGemma must see the SAME prefix it was
    trained on, or accuracy degrades. Keeping the default at
    ``model.prompt.prefix`` guarantees train / eval / inference consistency.
    """
    infer = config.get("prompt", None)
    if infer is not None and infer.get("prefix", None) is not None:
        return infer.prefix

    model_prompt = config.model.get("prompt", None)
    if model_prompt is not None:
        if model_prompt.get("prefix", None) is not None:
            return model_prompt.prefix
        if model_prompt.get("user", None) is not None:
            logger.warning(
                "prefix_suffix: no `prefix` set; falling back to model.prompt.user."
            )
            return model_prompt.user

    raise ValueError(
        "prefix_suffix style requires a prefix. Set `model.prompt.prefix` "
        "(recommended) or `prompt.prefix`."
    )
