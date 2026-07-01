"""Prometheus metrics for the coin VLM serving API.

Two layers are exposed through a single `/metrics` endpoint:

  * vLLM engine metrics (`vllm:*`) — latency, throughput, KV-cache usage, queue
    depth, time-to-first-token. vLLM's AsyncLLMEngine registers these on the
    DEFAULT prometheus_client registry when stat logging is enabled
    (disable_log_stats=False), so we get them for free.
  * App-level metrics (`coin_*`) defined here — end-to-end request latency,
    error breakdown, in-flight count, and the two output-drift signals the
    engine cannot see: parse-rate and per-field null-rate.

Mounting `prometheus_client.make_asgi_app()` (default registry) on the FastAPI
app scrapes BOTH layers in one exposition.

The metric objects are module-level singletons. They register against the
default registry on import; the try/except guards a second import under reload
(e.g. uvicorn --reload), where re-registering the same name would raise.
"""

from prometheus_client import CONTENT_TYPE_LATEST  # noqa: F401 (re-exported)
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# Buckets tuned for a heavy multimodal model: a single coin pair prefill +
# decode is typically hundreds of ms to a few seconds, occasionally more.
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
# Upload size in MB; coin photos are usually < a few MB, capped by max_upload_mb.
_UPLOAD_MB_BUCKETS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


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


def metrics_app():
    """ASGI app exposing the default registry (vllm:* + coin_*) in Prometheus
    text format. Mount at the configured path (default /metrics)."""
    return make_asgi_app()
