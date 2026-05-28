import json
import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# The 7 mint-mark labels actually present in this dataset (plus null/empty
# for "no mint mark"). Closed-set: anything outside this is a model
# hallucination (e.g. letters copied from "LIBERTY", "AMERICA", coin legends).
# Note the case: "Mo" (Mexico City) is mixed-case in the data — comparison
# is done after .upper() so any case is accepted, but this constant is the
# canonical surface form for display/prompting.
VALID_MINT_MARKS = {"", "D", "P", "S", "W", "O", "CC", "Mo"}


def parse_response(response: str) -> dict | None:

    if not response or not response.strip():
        return None

    # Strip thinking and tool_response tags if present (Qwen3 artifacts)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    response = re.sub(r'</?tool_response>', '', response).strip()

    if not response:
        return None

    try:
        parsed = json.loads(response.strip())
        return _extract_fields(parsed)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return _extract_fields(parsed)
        except json.JSONDecodeError:
            pass

    result = {}

    year_match = re.search(r'"?year"?\s*[:=]\s*"?(\d{4})"?', response, re.IGNORECASE)
    if year_match:
        result["year"] = year_match.group(1).strip()

    mint_match = re.search(r'"?mint_mark"?\s*[:=]\s*"?([^",\n}]*)"?', response, re.IGNORECASE)
    if mint_match:
        result["mint_mark"] = mint_match.group(1).strip()

    if result:
        logger.warning("JSON parse failed, used regex fallback: %s", response[:100])
        # Route through the same normalization as the JSON path so callers
        # always see canonical None/str values.
        return _extract_fields(result)

    logger.warning("Could not parse response: %s", response[:100])
    return None


def _extract_fields(parsed: dict) -> dict:
    """Normalize a parsed JSON object to the canonical schema.

    Preserves JSON null as Python None (so the saved `parsed` field mirrors
    the ground-truth shape, where absent mint marks are null — not the string
    "NONE" or ""). Strings like "none"/"null"/"" are also folded to None so
    downstream comparison is consistent regardless of how the model phrased
    "no mint mark".
    """
    year_raw = parsed.get("year", None)
    year = None if year_raw is None else str(year_raw).strip() or None

    mint_raw = parsed.get("mint_mark", None)
    if mint_raw is None:
        mint = None
    else:
        mint = str(mint_raw).strip().upper()
        if mint in {"", "NONE", "NULL"}:
            mint = None

    return {"year": year, "mint_mark": mint}

def _norm_year(v) -> str:
    """None/'' → ''. Otherwise stripped string form."""
    if v is None:
        return ""
    return str(v).strip()


def _norm_mint(v) -> str:
    """None / '' / 'none' / 'null' / 'NONE' all collapse to '' so that gold
    null and predicted null/none/empty are treated as equivalent.
    """
    if v is None:
        return ""
    s = str(v).strip().upper()
    if s in {"NONE", "NULL"}:
        return ""
    return s


def year_accuracy(pred: dict, gold: dict) -> bool:
    return _norm_year(pred.get("year")) == _norm_year(gold.get("year"))


def mint_mark_accuracy(pred: dict, gold: dict) -> bool:
    return _norm_mint(pred.get("mint_mark")) == _norm_mint(gold.get("mint_mark"))


def extract_match(pred: dict, gold: dict) -> bool:
    return year_accuracy(pred, gold) and mint_mark_accuracy(pred, gold)

def build_confusion_matrix(
    predictions: list[dict | None],
    references: list[dict],
) -> dict[str, dict[str, int]]:

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pred, gold in zip(predictions, references):
        # Show empty/null as the literal label "NONE" in the matrix so the
        # gold-null row stays readable, but use the normalized form for the
        # bucket key so "none"/null/"" all collapse to a single cell.
        gold_norm = _norm_mint(gold.get("mint_mark"))
        gold_key = gold_norm if gold_norm else "NONE"
        if pred is None:
            pred_key = "PARSE_ERROR"
        else:
            pred_norm = _norm_mint(pred.get("mint_mark"))
            pred_key = pred_norm if pred_norm else "NONE"
        matrix[gold_key][pred_key] += 1

    return {k: dict(v) for k, v in matrix.items()}


def compute_metrics(predictions, references):

    assert len(predictions) == len(references), (
        f"predictions ({len(predictions)}) and references ({len(references)}) "
        "must have the same length."
    )

    n = len(predictions)
    n_exact = 0
    n_year_correct = 0
    n_mint_correct = 0
    n_parse_errors = 0
    parsed_preds = []

    for pred_str, gold in zip(predictions, references):
        pred = parse_response(pred_str)
        parsed_preds.append(pred)

        if pred is None:
            n_parse_errors += 1
            continue

        if extract_match(pred, gold):
            n_exact += 1
        if year_accuracy(pred, gold):
            n_year_correct += 1
        if mint_mark_accuracy(pred, gold):
            n_mint_correct += 1

    valid_n = n - n_parse_errors

    return {
        "extract_match": round(n_exact / n, 4) if n > 0 else 0.0,
        "year_accuracy": round(n_year_correct / valid_n, 4) if valid_n > 0 else 0.0,
        "mint_mark_accuracy": round(n_mint_correct / valid_n, 4) if valid_n > 0 else 0.0,
        "parse_error_rate": round(n_parse_errors / n, 4) if n > 0 else 0.0,
        "confusion_matrix": build_confusion_matrix(parsed_preds, references),
        "n_samples": n,
        "n_parse_errors": n_parse_errors,
    }
