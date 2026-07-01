# Serving monitoring — Prometheus + Grafana

Operational + output-drift monitoring for the coin VLM serving API
([scripts/serve.py](../../scripts/serve.py)). This is **Phụ lục B → Operational**
of [docs/mlops.md](../../docs/mlops.md), now implemented.

## What is monitored

The serving app exposes `/metrics` (Prometheus text format) combining two layers:

| Layer | Source | Example metrics |
|---|---|---|
| **Engine** | vLLM `AsyncLLMEngine` (`vllm:*`) | `vllm:e2e_request_latency_seconds`, `vllm:time_to_first_token_seconds`, `vllm:gpu_cache_usage_perc`, `vllm:num_requests_running/waiting` |
| **App** | `src/serving/metrics.py` (`coin_*`) | `coin_requests_total{status}`, `coin_request_latency_seconds`, `coin_predict_latency_seconds`, `coin_inflight_requests`, `coin_parse_total{result}`, `coin_prediction_null_total{field}`, `coin_upload_megabytes` |

`coin_parse_total` (parse-rate) and `coin_prediction_null_total` (year/mint
null-rate) are the **output-drift** signals — there is no ground truth at serve
time, so a rising null/parse-fail rate is the early warning that the input
distribution has shifted. The same data is also written per-request to the
predictions JSONL (`outputs/logs/serving/`) for offline drift analysis.

## Enable on the API

On by default. Controlled in `config/serving/serving.yaml`:

```yaml
serving:
  metrics:
    enabled: true
    path: "/metrics"
```

After starting `scripts/serve.py`, verify:

```bash
curl -s localhost:49710/metrics | grep -E '^(coin_|vllm:)'
```

## Run the monitoring stack

### Native (primary — matches this box's firewall constraints)

The serving box blocks Docker Hub and GitHub, so the script downloads
Prometheus through the `ghfast.top` proxy and runs both binaries natively
(same pattern as the native MLflow setup).

```bash
bash deploy/monitoring/run_monitoring.sh start    # download + launch
bash deploy/monitoring/run_monitoring.sh status
bash deploy/monitoring/run_monitoring.sh stop
```

- Prometheus → http://localhost:49713
- Grafana → http://localhost:49714 (admin / admin), dashboard **Coin VLM — Serving** auto-provisioned.

Ports use the project's dedicated 4971x range (serve 49710 / mlflow 49711 /
airflow 49712 / prometheus 49713 / grafana 49714) so they don't clash with the
default 9090/3000 that other projects on the shared box are likely using.

Override ports / versions / mirrors via env vars (`PROM_PORT`, `GRAFANA_PORT`,
`PROM_VERSION`, `GRAFANA_VERSION`, `PROM_URL`, `GRAFANA_URL`, `GH_PROXY`).

### Docker Compose

Images are pulled through the DaoCloud mirror (`docker.m.daocloud.io/...`), so
this works on the firewalled serving box too — same trick as the MLflow stack.

On the serving box (Compose V1):

```bash
PYTHONNOUSERSITE=1 docker-compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
```

On a machine with clean Docker Hub access + Compose V2 (the mirror prefix is
harmless):

```bash
docker compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
```

## Wiring notes

- **Scrape target** — [prometheus.yml](prometheus.yml) defaults to
  `host.docker.internal:49710` (Docker: inside the container `localhost` is the
  container itself, so it must reach the host). The native script rewrites this
  to `localhost:49710`. If serve runs on a different host/port, edit the target.
- **Grafana datasource URL** — `http://prometheus:9090` for compose (service
  name + the in-container port, which stays 9090); the native script rewrites it
  to `http://localhost:49713` (the native Prometheus host port).

## Files

```
deploy/monitoring/
├── prometheus.yml                       scrape config
├── run_monitoring.sh                    native launcher (start|stop|status)
├── docker-compose.monitoring.yml        compose fallback
├── grafana/
│   ├── provisioning/datasources/        Prometheus datasource
│   ├── provisioning/dashboards/         dashboard provider
│   └── dashboards/coin-serving.json     the dashboard
└── README.md
```
