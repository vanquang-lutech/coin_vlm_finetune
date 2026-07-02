"""Prometheus metrics for the coin VLM serving API.

Two layers are exposed through a single `/metrics` endpoint:

  * vLLM engine metrics (`vllm:*`) — latency, throughput, KV-cache usage, queue
    depth, time-to-first-token. vLLM's AsyncLLMEngine registers these on the
    DEFAULT prometheus_client registry when stat logging is enabled
    (disable_log_stats=False), so we get them for free.
  * App-level metrics (`coin_*`) defined here — end-to-end request latency,
    error breakdown, in-flight count, and the drift signals the engine cannot
    see: parse-rate and per-field null-rate (output), predicted mint-mark /
    decade distribution (output), and upload resolution + contrast/sharpness
    (input — accuracy is resolution-capped, so input quality shifting down
    predicts an accuracy drop before any output metric moves).

Mounting `prometheus_client.make_asgi_app()` (default registry) on the FastAPI
app scrapes BOTH layers in one exposition.

The metric objects are module-level singletons. They register against the
default registry on import; the try/except guards a second import under reload
(e.g. uvicorn --reload), where re-registering the same name would raise.
"""

import cv2
import numpy as np
from prometheus_client import CONTENT_TYPE_LATEST  # noqa: F401 (re-exported)
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# Buckets tuned for a heavy multimodal model: a single coin pair prefill +
# decode is typically hundreds of ms to a few seconds, occasionally more.
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
# Upload size in MB; coin photos are usually < a few MB, capped by max_upload_mb.
_UPLOAD_MB_BUCKETS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
# Raw pixel count (w*h) per uploaded image. The known source distribution sits
# around ~250px squares (62.5k px), so buckets are dense there and stretch up
# to phone-camera sizes.
_PIXEL_BUCKETS = (
    16_384, 40_000, 62_500, 90_000, 160_000, 250_000,
    562_500, 1_000_000, 2_250_000, 4_000_000,
)
# Gray-std contrast; the smart-enhance "low contrast" threshold is 50.
_CONTRAST_BUCKETS = (10, 20, 30, 40, 50, 60, 75, 90, 110)
# Laplacian-variance sharpness; the smart-enhance "blurry" threshold is 600.
_SHARPNESS_BUCKETS = (50, 100, 200, 400, 600, 900, 1500, 3000, 6000)


def _counter(name, doc, labelnames=()):
    """Create a Counter, or return the already-registered collector if this
    module is imported twice (reload). Avoids 'Duplicated timeseries' errors."""
    try:
        return Counter(name, doc, labelnames)
    except ValueError:
        from prometheus_client import REGISTRY

        return REGISTRY._names_to_collectors[name]


def _histogram(name, doc, buckets, labelnames=()):
    try:
        return Histogram(name, doc, labelnames, buckets=buckets)
    except ValueError:
        from prometheus_client import REGISTRY

        return REGISTRY._names_to_collectors[name]


def _gauge(name, doc):
    try:
        return Gauge(name, doc)
    except ValueError:
        from prometheus_client import REGISTRY

        return REGISTRY._names_to_collectors[name]


# --- Operational -----------------------------------------------------------
# status label: ok | bad_request | timeout | error | unavailable
REQUESTS_TOTAL = _counter(
    "coin_requests_total",
    "Total /predict requests by outcome.",
    ("status",),
)
REQUEST_LATENCY = _histogram(
    "coin_request_latency_seconds",
    "End-to-end server-side /predict latency (read + enhance + concat + infer).",
    _LATENCY_BUCKETS,
)
PREDICT_LATENCY = _histogram(
    "coin_predict_latency_seconds",
    "Engine-only latency (VLLMCoinEngine.predict): enhance + concat + vLLM generate.",
    _LATENCY_BUCKETS,
)
INFLIGHT = _gauge(
    "coin_inflight_requests",
    "Number of /predict requests currently being handled.",
)
UPLOAD_MB = _histogram(
    "coin_upload_megabytes",
    "Decoded upload size per image (megabytes).",
    _UPLOAD_MB_BUCKETS,
)

# --- Input drift (upload quality) -------------------------------------------
IMAGE_PIXELS = _histogram(
    "coin_image_pixels",
    "Decoded upload resolution per image (raw pixel count, width*height).",
    _PIXEL_BUCKETS,
)
IMAGE_CONTRAST = _histogram(
    "coin_image_contrast",
    "Grayscale std of each upload, pre-enhancement (CoinEnhancer.analyze "
    "definition; smart-enhance low-contrast threshold is 50).",
    _CONTRAST_BUCKETS,
)
IMAGE_SHARPNESS = _histogram(
    "coin_image_sharpness",
    "Laplacian variance of each upload, pre-enhancement (CoinEnhancer.analyze "
    "definition; smart-enhance blurry threshold is 600).",
    _SHARPNESS_BUCKETS,
)

# --- Output drift (no ground truth at serve time) --------------------------
# result label: ok | fail
PARSE_TOTAL = _counter(
    "coin_parse_total",
    "Predictions by whether the raw model output parsed cleanly.",
    ("result",),
)
# field label: year | mint_mark — counts predictions where the field came back null.
PREDICTION_NULL_TOTAL = _counter(
    "coin_prediction_null_total",
    "Predictions where a field was null (drift signal).",
    ("field",),
)

# Prediction-distribution drift. Null-rate only catches degradation INTO null;
# these catch degradation into wrong-but-parseable values: a sudden shift in
# the predicted mint-mark or decade mix is a drift signal even at 0% null.
# Label sets are fixed (pre-created below) so cardinality stays bounded.
_MINT_MARKS = ("P", "D", "S", "W", "CC", "O")
_DECADES = ("none", "other", "pre-1900") + tuple(
    f"{d}s" for d in range(1900, 2030, 10)
)
PREDICTION_MINT_TOTAL = _counter(
    "coin_prediction_mint_total",
    "Predictions by normalized mint mark (US mint letters, none, or other).",
    ("mint_mark",),
)
PREDICTION_DECADE_TOTAL = _counter(
    "coin_prediction_decade_total",
    "Predictions by decade of the predicted year (none/other for null or "
    "out-of-range 1793..2029).",
    ("decade",),
)


def _mint_label(mint) -> str:
    if mint is None:
        return "none"
    mint = str(mint).strip().upper()
    return mint if mint in _MINT_MARKS else "other"


def _decade_label(year) -> str:
    if year is None:
        return "none"
    try:
        y = int(str(year).strip())
    except ValueError:
        return "other"
    if 1900 <= y <= 2029:
        return f"{y // 10 * 10}s"
    if 1793 <= y < 1900:
        return "pre-1900"
    return "other"


# Pre-create every label combination so all series exist from the first scrape.
# A labeled Counter child is only born on its first .labels(...).inc(); until
# then PromQL sees no series at all, so ratio panels (parse-fail, error-ratio,
# null-rate) render "No data" instead of 0 while everything is healthy.
for _status in ("ok", "bad_request", "timeout", "error", "unavailable"):
    REQUESTS_TOTAL.labels(status=_status)
for _result in ("ok", "fail"):
    PARSE_TOTAL.labels(result=_result)
for _field in ("year", "mint_mark"):
    PREDICTION_NULL_TOTAL.labels(field=_field)
for _mint in _MINT_MARKS + ("none", "other"):
    PREDICTION_MINT_TOTAL.labels(mint_mark=_mint)
for _decade in _DECADES:
    PREDICTION_DECADE_TOTAL.labels(decade=_decade)


def record_prediction(result: dict) -> None:
    """Update the output-drift counters from one parsed prediction result.
    `result` is the dict VLLMCoinEngine.predict returns
    ({year, mint_mark, parse_ok, raw}). Tolerate None (a parse failure that
    yielded no dict) so metrics recording never turns a bad parse into a 500."""
    result = result or {}
    PARSE_TOTAL.labels(result="ok" if result.get("parse_ok") else "fail").inc()
    if result.get("year") is None:
        PREDICTION_NULL_TOTAL.labels(field="year").inc()
    if result.get("mint_mark") is None:
        PREDICTION_NULL_TOTAL.labels(field="mint_mark").inc()
    PREDICTION_MINT_TOTAL.labels(mint_mark=_mint_label(result.get("mint_mark"))).inc()
    PREDICTION_DECADE_TOTAL.labels(decade=_decade_label(result.get("year"))).inc()


def observe_image(pil_image) -> None:
    """Record the input-drift stats of one decoded upload (RGB PIL image).
    Contrast/sharpness reuse the CoinEnhancer.analyze definitions (gray std /
    Laplacian variance) so the values are directly comparable to the
    smart-enhance thresholds (50 / 600). Runs on the pre-enhancement image —
    the point is to measure what clients send, not what we feed the model."""
    w, h = pil_image.size
    IMAGE_PIXELS.observe(w * h)
    gray = np.asarray(pil_image.convert("L"))
    IMAGE_CONTRAST.observe(float(gray.std()))
    IMAGE_SHARPNESS.observe(float(cv2.Laplacian(gray, cv2.CV_64F).var()))


def metrics_app():
    """ASGI app exposing the default registry (vllm:* + coin_*) in Prometheus
    text format. Mount at the configured path (default /metrics)."""
    return make_asgi_app()
