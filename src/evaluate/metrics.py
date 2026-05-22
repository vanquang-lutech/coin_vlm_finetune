import json
import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

VALID_MINT_MARKS = {"", "P", "D", "S", "W", "CC", "O", "C", "?"}


def parse_response(response: str) -> dict | None:

    if not response or not response.strip():
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
        return result

    logger.warning("Could not parse response: %s", response[:100])
    return None


def _extract_fields(parsed: dict) -> dict:
    return {
        "year": str(parsed.get("year", "")).strip(),
        "mint_mark": str(parsed.get("mint_mark", "")).strip().upper(),
    }

def year_accuracy(pred: dict, gold: dict) -> bool:
    return pred.get("year", "").strip() == str(gold.get("year", "")).strip()


def mint_mark_accuracy(pred: dict, gold: dict) -> bool:
    return (
        pred.get("mint_mark", "").strip().upper()
        == str(gold.get("mint_mark", "")).strip().upper()
    )


def exact_match(pred: dict, gold: dict) -> bool:
    return year_accuracy(pred, gold) and mint_mark_accuracy(pred, gold)

def build_confusion_matrix(
    predictions: list[dict | None],
    references: list[dict],
) -> dict[str, dict[str, int]]:

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pred, gold in zip(predictions, references):
        gold_mint = str(gold.get("mint_mark", "")).strip().upper()
        pred_mint = pred.get("mint_mark", "PARSE_ERROR").strip().upper() if pred else "PARSE_ERROR"
        matrix[gold_mint][pred_mint] += 1

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

        if exact_match(pred, gold):
            n_exact += 1
        if year_accuracy(pred, gold):
            n_year_correct += 1
        if mint_mark_accuracy(pred, gold):
            n_mint_correct += 1

    valid_n = n - n_parse_errors

    return {
        "exact_match": round(n_exact / n, 4) if n > 0 else 0.0,
        "year_accuracy": round(n_year_correct / valid_n, 4) if valid_n > 0 else 0.0,
        "mint_mark_accuracy": round(n_mint_correct / valid_n, 4) if valid_n > 0 else 0.0,
        "parse_error_rate": round(n_parse_errors / n, 4) if n > 0 else 0.0,
        "confusion_matrix": build_confusion_matrix(parsed_preds, references),
        "n_samples": n,
        "n_parse_errors": n_parse_errors,
    }