"""FastAPI app serving the merged + quantized coin VLM via vLLM.

Endpoints:
  GET  /health            liveness + which checkpoint is loaded
  POST /predict           multipart image upload  -> {year, mint_mark, raw, parse_ok}

The vLLM engine is heavy to initialize, so it's built once during the FastAPI
lifespan startup and shared across requests.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.serving.engine import VLLMCoinEngine
from src.serving.schemas import CoinPrediction, HealthResponse
from src.utils import dated_log_path, get_logger

logger = get_logger(__name__)


class _PredictionLog:
    """Append one JSON line per prediction (filename, result, latency) so served
    requests are auditable and can be re-scored offline. The target file is
    date-partitioned (day/month) and recomputed per write, so it rolls over with
    the calendar automatically. No-op if disabled."""

    def __init__(self, log_dir, rotation: str = "daily", enabled: bool = True):
        self.log_dir = log_dir
        self.rotation = rotation
        self.enabled = enabled and log_dir is not None

    def write(self, record: dict) -> None:
        if not self.enabled:
            return
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        path = dated_log_path(self.log_dir, "predictions", "jsonl", self.rotation)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Failed to write prediction log line")


def create_app(config) -> FastAPI:
    state: dict = {}

    serving = config.serving
    log_dir = serving.get("log_dir", None)
    rotation = serving.get("log_rotation", "daily")
    pred_log = _PredictionLog(
        log_dir, rotation=rotation, enabled=serving.get("predictions_log", True)
    )

    # Request guards. max_upload_mb caps raw bytes BEFORE decode/enhance (the
    # min/max_pixels resize runs later, inside vLLM, so it does not protect the
    # decode + CLAHE step from a huge or decompression-bomb upload). 0 disables.
    max_upload_mb = serving.get("max_upload_mb", 20)
    max_upload_bytes = int(max_upload_mb * 1024 * 1024) if max_upload_mb else 0
    request_timeout_s = serving.get("request_timeout_s", 60)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting vLLM coin engine...")
        state["engine"] = VLLMCoinEngine(config)
        if pred_log.enabled:
            logger.info(
                "Prediction log (%s, partitioned): %s",
                rotation,
                dated_log_path(log_dir, "predictions", "jsonl", rotation),
            )
        logger.info("Engine started; API ready.")
        yield
        state.clear()

    app = FastAPI(
        title="Coin VLM Extraction API",
        description="Extract {year, mint_mark} from a US coin image using a "
        "fine-tuned, merged + quantized VLM served by vLLM.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        serving = config.serving
        return HealthResponse(
            status="ok" if state.get("engine") else "starting",
            model_path=serving.model_path,
            quantization=serving.get("quantization", None) or None,
        )

    @app.post("/predict", response_model=CoinPrediction)
    async def predict(file: UploadFile = File(..., description="Coin image.")):
        engine: VLLMCoinEngine = state.get("engine")
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not ready.")

        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Expected an image upload, got content-type={file.content_type}",
            )

        # Reject oversized uploads before reading the whole body when the client
        # sent a Content-Length (file.size).
        if max_upload_bytes and file.size and file.size > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file.size} bytes); limit is {max_upload_bytes} bytes.",
            )

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file.")
        # Second check in case Content-Length was absent/understated.
        if max_upload_bytes and len(data) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(data)} bytes); limit is {max_upload_bytes} bytes.",
            )

        start = time.perf_counter()
        try:
            if request_timeout_s:
                result = await asyncio.wait_for(
                    engine.predict(data), timeout=request_timeout_s
                )
            else:
                result = await engine.predict(data)
        except asyncio.TimeoutError:
            logger.warning("Prediction timed out (%ss) for %s", request_timeout_s, file.filename)
            pred_log.write({"file": file.filename, "error": f"timeout>{request_timeout_s}s"})
            raise HTTPException(
                status_code=504,
                detail=f"Prediction timed out after {request_timeout_s}s.",
            )
        except Exception as exc:  # noqa: BLE001 - surface as a clean 500
            logger.exception("Prediction failed for %s", file.filename)
            pred_log.write({"file": file.filename, "error": str(exc)})
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "predict file=%s year=%s mint_mark=%s parse_ok=%s latency_ms=%s",
            file.filename, result.get("year"), result.get("mint_mark"),
            result.get("parse_ok"), latency_ms,
        )
        pred_log.write({
            "file": file.filename,
            "year": result.get("year"),
            "mint_mark": result.get("mint_mark"),
            "parse_ok": result.get("parse_ok"),
            "raw": result.get("raw"),
            "latency_ms": latency_ms,
        })

        return CoinPrediction(**result)

    return app
