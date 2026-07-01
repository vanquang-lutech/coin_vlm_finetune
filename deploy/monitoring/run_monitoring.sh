#!/usr/bin/env bash
# Native Prometheus + Grafana for the coin VLM serving API (PRIMARY path).
#
# Why native, not Docker: the serving box's firewall blocks Docker Hub and the
# installed Compose V1 is broken (same reason MLflow runs natively here). This
# script downloads the official release tarballs, rewrites the Grafana
# provisioning to point at localhost, and runs both as background processes.
#
# GitHub is also firewalled on this box, so the Prometheus download (hosted on
# github.com) is routed through the ghfast.top proxy by default. Grafana is
# hosted on dl.grafana.com (reachable) and is fetched directly. Override either
# with PROM_URL / GRAFANA_URL if your network differs.
#
# Usage:
#   bash deploy/monitoring/run_monitoring.sh start    # download (first run) + launch
#   bash deploy/monitoring/run_monitoring.sh stop
#   bash deploy/monitoring/run_monitoring.sh status
#
# Then: Prometheus http://localhost:49713 , Grafana http://localhost:49714
#       (admin/admin; the "Coin VLM — Serving" dashboard is auto-provisioned).
#       Override with PROM_PORT / GRAFANA_PORT env vars.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${HERE}/.run"
GH_PROXY="${GH_PROXY:-https://ghfast.top/}"

# Host ports in the project's dedicated 4971x range — 9090/3000 are too common
# and clash with other projects on the shared box.
PROM_PORT="${PROM_PORT:-49713}"
GRAFANA_PORT="${GRAFANA_PORT:-49714}"

PROM_VERSION="${PROM_VERSION:-2.53.0}"
GRAFANA_VERSION="${GRAFANA_VERSION:-11.1.0}"
PROM_URL="${PROM_URL:-${GH_PROXY}https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz}"
GRAFANA_URL="${GRAFANA_URL:-https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz}"

PROM_HOME="${RUN_DIR}/prometheus-${PROM_VERSION}.linux-amd64"
GRAFANA_HOME="${RUN_DIR}/grafana-v${GRAFANA_VERSION}"
PROM_PID="${RUN_DIR}/prometheus.pid"
GRAFANA_PID="${RUN_DIR}/grafana.pid"
PROM_CONFIG="${RUN_DIR}/prometheus.yml"

mkdir -p "${RUN_DIR}"

download() {
  echo ">> Downloading Prometheus ${PROM_VERSION} ..."
  [ -d "${PROM_HOME}" ] || {
    curl -fL "${PROM_URL}" -o "${RUN_DIR}/prometheus.tar.gz"
    tar -xzf "${RUN_DIR}/prometheus.tar.gz" -C "${RUN_DIR}"
  }
  echo ">> Downloading Grafana ${GRAFANA_VERSION} ..."
  [ -d "${GRAFANA_HOME}" ] || {
    curl -fL "${GRAFANA_URL}" -o "${RUN_DIR}/grafana.tar.gz"
    tar -xzf "${RUN_DIR}/grafana.tar.gz" -C "${RUN_DIR}"
  }
}

prepare_provisioning() {
  # Copy the provisioning tree and rewrite the two container-only paths so they
  # resolve for a native Grafana: datasource URL -> localhost, dashboard
  # provider path -> the absolute dashboards dir on this box.
  local prov="${RUN_DIR}/provisioning"
  rm -rf "${prov}"
  cp -r "${HERE}/grafana/provisioning" "${prov}"
  sed -i "s#http://prometheus:9090#http://localhost:${PROM_PORT}#g" \
    "${prov}/datasources/prometheus.yml"
  sed -i "s#/etc/grafana/dashboards#${HERE}/grafana/dashboards#g" \
    "${prov}/dashboards/dashboards.yml"

  # The committed prometheus.yml targets host.docker.internal (Docker default).
  # Natively the API is on the same box, so rewrite the scrape target to
  # localhost and run off this copy.
  cp "${HERE}/prometheus.yml" "${PROM_CONFIG}"
  sed -i "s#host.docker.internal:49710#localhost:49710#g" "${PROM_CONFIG}"
}

start() {
  download
  prepare_provisioning

  echo ">> Starting Prometheus on :${PROM_PORT} ..."
  nohup "${PROM_HOME}/prometheus" \
    --config.file="${PROM_CONFIG}" \
    --storage.tsdb.path="${RUN_DIR}/prometheus-data" \
    --storage.tsdb.retention.time=15d \
    --web.listen-address=":${PROM_PORT}" \
    >"${RUN_DIR}/prometheus.log" 2>&1 &
  echo $! > "${PROM_PID}"

  echo ">> Starting Grafana on :${GRAFANA_PORT} ..."
  GF_PATHS_PROVISIONING="${RUN_DIR}/provisioning" \
  GF_PATHS_DATA="${RUN_DIR}/grafana-data" \
  GF_PATHS_LOGS="${RUN_DIR}/grafana-logs" \
  GF_SERVER_HTTP_PORT="${GRAFANA_PORT}" \
  GF_SECURITY_ADMIN_USER=admin \
  GF_SECURITY_ADMIN_PASSWORD=admin \
  GF_USERS_ALLOW_SIGN_UP=false \
  nohup "${GRAFANA_HOME}/bin/grafana" server \
    --homepath "${GRAFANA_HOME}" \
    >"${RUN_DIR}/grafana.log" 2>&1 &
  echo $! > "${GRAFANA_PID}"

  echo ">> Up. Prometheus http://localhost:${PROM_PORT}  |  Grafana http://localhost:${GRAFANA_PORT} (admin/admin)"
  echo "   Logs: ${RUN_DIR}/{prometheus,grafana}.log"
}

stop() {
  for pidf in "${PROM_PID}" "${GRAFANA_PID}"; do
    if [ -f "${pidf}" ]; then
      kill "$(cat "${pidf}")" 2>/dev/null || true
      rm -f "${pidf}"
    fi
  done
  echo ">> Stopped."
}

status() {
  for name in prometheus grafana; do
    pidf="${RUN_DIR}/${name}.pid"
    if [ -f "${pidf}" ] && kill -0 "$(cat "${pidf}")" 2>/dev/null; then
      echo "${name}: running (pid $(cat "${pidf}"))"
    else
      echo "${name}: stopped"
    fi
  done
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
