"""FastAPI app serving the merged + quantized coin VLM via vLLM.

Endpoints:
  GET  /health            liveness + which checkpoint is loaded
  GET  /metrics           Prometheus exposition (vllm:* engine metrics + coin_*
                          app metrics). Mounted only when serving.metrics.enabled.
  POST /predict           multipart upload of TWO images (obverse + reverse)
                          -> {year, mint_mark, raw, parse_ok}. The two images are
                          enhanced, resized to fit, and concatenated side-by-side
                          to match the training-time input.

The vLLM engine is heavy to initialize, so it's built once during the FastAPI
lifespan startup and shared across requests.
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.serving import metrics as M
from src.serving.engine import VLLMCoinEngine
from src.serving.schemas import CoinError, CoinPrediction, HealthResponse
from src.utils import dated_log_path, get_logger

logger = get_logger(__name__)


# Fallback code for a bare HTTPException (one we didn't raise as APIError).
_STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "file_too_large",
    422: "validation_error",
    500: "internal_error",
    503: "engine_unavailable",
    504: "prediction_timeout",
}


class APIError(HTTPException):
    """HTTPException that also carries a stable machine-readable `code` for the
    {code, detail, request_id} error envelope."""

    def __init__(self, status_code: int, code: str, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code


def _error_response(status_code: int, code: str, detail: str, request_id) -> JSONResponse:
    """Build the uniform error envelope; X-Request-ID header set on every error."""
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "detail": detail, "request_id": request_id},
        headers={"X-Request-ID": request_id or ""},
    )


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

    # Prometheus /metrics. The vLLM engine publishes vllm:* metrics to the
    # default registry; this just surfaces them (plus the coin_* app metrics).
    metrics_cfg = serving.get("metrics", {}) or {}
    metrics_enabled = metrics_cfg.get("enabled", True)
    metrics_path = metrics_cfg.get("path", "/metrics")

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
        # Warm up JIT kernels so the FIRST real request isn't a ~30s spike (Triton
        # compiles rotary/causal_conv1d/vision kernels inline otherwise). Toggle
        # with serving.warmup; serving.warmup_runs controls how many dummy passes.
        if serving.get("warmup", True):
            await state["engine"].warmup(runs=serving.get("warmup_runs", 1))
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

    if metrics_enabled:
        # Sub-app on the default registry: scrapes vllm:* (engine) + coin_* (app).
        app.mount(metrics_path, M.metrics_app())
        logger.info("Prometheus metrics exposed at %s", metrics_path)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        # One id per request: echoed as X-Request-ID and tagged into every log
        # line + error body, so a client can quote it and we grep the server log.
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError):
        return _error_response(
            exc.status_code, exc.code, exc.detail,
            getattr(request.state, "request_id", None),
        )

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(request: Request, exc: HTTPException):
        # Bare HTTPException we didn't raise as APIError (e.g. 404/405): map to a
        # code from the status. 4xx detail is safe to surface; 5xx stays generic.
        code = _STATUS_CODES.get(exc.status_code, "error")
        detail = str(exc.detail) if exc.status_code < 500 else "Internal server error"
        return _error_response(exc.status_code, code, detail,
                               getattr(request.state, "request_id", None))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError):
        # Missing/invalid multipart fields -> same envelope as everything else.
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
            for e in exc.errors()
        ) or "Validation error"
        return _error_response(422, "validation_error", detail,
                               getattr(request.state, "request_id", None))

    @app.exception_handler(Exception)
    async def _handle_unhandled(request: Request, exc: Exception):
        # Anything uncaught (incl. errors after predict): generic 500 to client,
        # full traceback to the log under the same request_id.
        request_id = getattr(request.state, "request_id", None)
        logger.exception("[req=%s] Unhandled server error", request_id)
        return _error_response(500, "internal_error", "Internal server error", request_id)

    @app.middleware("http")
    async def add_process_time_header(request, call_next):
        # Total SERVER-side handling time incl. reading the (uploaded) body.
        # Compare against the client's TTFB: if X-Process-Time-Ms is small but
        # the client sees >1s, the gap is network/upload (e.g. an SSH tunnel),
        # not server compute. The gap between this and the per-prediction
        # latency_ms log is the time spent waiting for the upload to arrive.
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health():
        serving = config.serving
        return HealthResponse(
            status="ok" if state.get("engine") else "starting",
            model_path=serving.model_path,
            quantization=serving.get("quantization", None) or None,
        )

    async def _read_image(file: UploadFile, label: str) -> bytes:
        """Validate content-type + size and return the bytes of one upload."""
        if not (file.content_type or "").startswith("image/"):
            raise APIError(
                400, "not_an_image",
                f"'{label}' must be an image; got content-type={file.content_type}",
            )
        if max_upload_bytes and file.size and file.size > max_upload_bytes:
            raise APIError(
                413, "file_too_large",
                f"'{label}' too large ({file.size} bytes); limit is {max_upload_bytes} bytes.",
            )
        data = await file.read()
        if not data:
            raise APIError(400, "empty_file", f"'{label}' is empty.")
        if max_upload_bytes and len(data) > max_upload_bytes:
            raise APIError(
                413, "file_too_large",
                f"'{label}' too large ({len(data)} bytes); limit is {max_upload_bytes} bytes.",
            )
        return data

    @app.post(
        "/predict",
        response_model=CoinPrediction,
        responses={
            400: {"model": CoinError}, 413: {"model": CoinError},
            422: {"model": CoinError}, 503: {"model": CoinError},
            504: {"model": CoinError}, 500: {"model": CoinError},
        },
    )
    async def predict(
        request: Request,
        obverse: UploadFile = File(..., description="Obverse (front) coin image."),
        reverse: UploadFile = File(..., description="Reverse (back) coin image."),
    ):
        req_start = time.perf_counter()
        request_id = getattr(request.state, "request_id", None)
        M.INFLIGHT.inc()
        try:
            engine: VLLMCoinEngine = state.get("engine")
            if engine is None:
                M.REQUESTS_TOTAL.labels(status="unavailable").inc()
                raise APIError(503, "engine_unavailable", "Engine not ready.")

            try:
                obv = await _read_image(obverse, "obverse")
                rev = await _read_image(reverse, "reverse")
            except APIError:
                M.REQUESTS_TOTAL.labels(status="bad_request").inc()
                raise
            M.UPLOAD_MB.observe(len(obv) / 1024 / 1024)
            M.UPLOAD_MB.observe(len(rev) / 1024 / 1024)
            names = f"{obverse.filename}|{reverse.filename}"

            start = time.perf_counter()
            try:
                if request_timeout_s:
                    result = await asyncio.wait_for(
                        engine.predict(obv, rev), timeout=request_timeout_s
                    )
                else:
                    result = await engine.predict(obv, rev)
            except asyncio.TimeoutError:
                M.REQUESTS_TOTAL.labels(status="timeout").inc()
                logger.warning("[req=%s] Prediction timed out (%ss) for %s",
                               request_id, request_timeout_s, names)
                pred_log.write({"files": names, "error": f"timeout>{request_timeout_s}s",
                                "request_id": request_id})
                raise APIError(
                    504, "prediction_timeout",
                    f"Prediction timed out after {request_timeout_s}s.",
                )
            except Exception as exc:  # noqa: BLE001 - real cause logged, client gets generic 500
                M.REQUESTS_TOTAL.labels(status="error").inc()
                logger.exception("[req=%s] Prediction failed for %s", request_id, names)
                pred_log.write({"files": names, "error": str(exc), "request_id": request_id})
                raise APIError(500, "prediction_failed", "Internal server error") from exc

            M.PREDICT_LATENCY.observe(time.perf_counter() - start)
            M.REQUESTS_TOTAL.labels(status="ok").inc()
            M.record_prediction(result)

            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.info(
                "[req=%s] predict files=%s year=%s mint_mark=%s parse_ok=%s latency_ms=%s",
                request_id, names, result.get("year"), result.get("mint_mark"),
                result.get("parse_ok"), latency_ms,
            )
            pred_log.write({
                "request_id": request_id,
                "obverse": obverse.filename,
                "reverse": reverse.filename,
                "year": result.get("year"),
                "mint_mark": result.get("mint_mark"),
                "parse_ok": result.get("parse_ok"),
                "raw": result.get("raw"),
                "latency_ms": latency_ms,
            })

            return CoinPrediction(**result)
        finally:
            M.INFLIGHT.dec()
            M.REQUEST_LATENCY.observe(time.perf_counter() - req_start)

    return app
