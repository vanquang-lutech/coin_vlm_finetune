"""Pydantic response schemas for the coin extraction API."""

from pydantic import BaseModel, Field


class CoinPrediction(BaseModel):
    year: str | None = Field(None, description="Extracted 4-digit year, or null.")
    mint_mark: str | None = Field(
        None, description="Extracted mint mark (D/P/S/W/O/CC/Mo), or null."
    )
    raw: str = Field(..., description="Raw model output before parsing.")
    parse_ok: bool = Field(..., description="Whether the raw output parsed cleanly.")


class HealthResponse(BaseModel):
    status: str
    model_path: str
    quantization: str | None
