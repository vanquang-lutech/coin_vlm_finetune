"""Pydantic response schemas for the coin extraction API."""

import re
from pydantic import BaseModel, Field, StringConstraints, field_validator
from typing import Annotated
from enum import Enum

YearStr = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}$"),
]

class MintMark(str, Enum):
    D = "D"
    P = "P"
    S = "S"
    W = "W"
    O = "O"
    CC = "CC"
    Mo = "Mo"

# Case-insensitive lookup: "MO" -> Mo, "o" -> O, "cc" -> CC, etc.
_MINT_BY_UPPER = {m.value.upper(): m for m in MintMark}

class CoinPrediction(BaseModel):
    year: YearStr | None = Field(None, description="Extracted 4-digit year, or null.")
    mint_mark: MintMark | None = Field(
        None, description="Extracted mint mark (D/P/S/W/O/CC/Mo), or null."
    )
    raw: str = Field(..., description="Raw model output before parsing.")
    parse_ok: bool = Field(..., description="Whether the raw output parsed cleanly.")

    @field_validator("mint_mark", mode="before")
    @classmethod
    def _normalize_mint_mark(cls, v):
        """Normalise known case variants to the canonical enum, anything genuinely unknown -> None."""
        if v is None or isinstance(v, MintMark):
            return v
        return _MINT_BY_UPPER.get(str(v).strip().upper())  # unknown -> None

    @field_validator("year", mode="before")
    @classmethod
    def _normalize_year(cls, v):
        """Coerce a malformed year (not exactly 4 digits) to None"""
        if v is None:
            return None
        s = str(v).strip()
        return s if re.fullmatch(r"\d{4}", s) else None


class HealthResponse(BaseModel):
    status: str
    model_path: str
    quantization: str | None
