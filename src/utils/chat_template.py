"""Helpers for safely passing chat-template variables to a processor.

transformers>=5.4's ``apply_chat_template`` emits

    "Kwargs passed to `processor.__call__` have to be in `processor_kwargs`
     dict, not in `**kwargs`"

for any ``**kwargs`` entry that is NOT a free variable of the active chat
template — it assumes such a kwarg was meant for the processor's ``__call__``.
So forwarding a template variable like ``enable_thinking`` to a template that
does not actually read it (e.g. Qwen3-VL-Instruct, Qwen2.5, InternVL3) triggers
the warning on every call.

``safe_template_kwargs`` mirrors transformers' OWN detection — jinja2
undeclared-variable analysis (``_get_template_variables`` →
``jinja2.meta.find_undeclared_variables``) — so a variable is forwarded only
when the template truly reads it, and the warning never fires. A bare substring
check is not enough: a name that appears only in a ``{% set %}`` assignment or a
comment is matched by the substring but is NOT a free variable, which is exactly
the false positive that still produced the warning.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _chat_template(processor):
    """Return the processor's chat template (str or dict of named templates),
    or None. Falls back to the tokenizer's template if the processor has none.
    """
    tpl = getattr(processor, "chat_template", None)
    if tpl is None:
        tok = getattr(processor, "tokenizer", None)
        tpl = getattr(tok, "chat_template", None)
    if isinstance(tpl, (str, dict)):
        return tpl
    return None


def _declared_vars(tpl):
    """Set of free (undeclared) jinja variables in the template(s), or None if
    introspection is unavailable (caller then falls back to a substring match).

    Uses transformers' own ``_get_template_variables`` so detection matches the
    code that emits the warning exactly — including its sandboxed environment
    and extensions (loopcontrols, AssistantTracker), which a plain
    ``jinja2.Environment().parse`` would fail to parse on some templates.
    """
    try:
        from transformers.utils.chat_template_utils import _get_template_variables
    except Exception:
        return None
    try:
        templates = tpl.values() if isinstance(tpl, dict) else [tpl]
        out = set()
        for t in templates:
            out |= set(_get_template_variables(t))
        return out
    except Exception:
        return None


def safe_template_kwargs(processor, desired: dict) -> dict:
    """Filter ``desired`` apply_chat_template kwargs to those the active chat
    template actually declares as free variables.

    Returns the subset that is safe to forward as ``**kwargs`` to
    ``processor.apply_chat_template`` without triggering the transformers>=5.4
    "processor_kwargs" warning.
    """
    if not desired:
        return {}

    tpl = _chat_template(processor)
    if tpl is None:
        return {}

    declared = _declared_vars(tpl)
    if declared is None:
        # transformers<5.4 (predates the warning) or introspection failed:
        # fall back to a permissive substring match so we don't silently drop a
        # variable a template genuinely needs.
        tpl_str = " ".join(str(v) for v in tpl.values()) if isinstance(tpl, dict) else tpl
        return {k: v for k, v in desired.items() if k in tpl_str}

    return {k: v for k, v in desired.items() if k in declared}
