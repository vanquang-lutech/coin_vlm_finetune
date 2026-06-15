"""Continuous Training (CT) DAG: train -> gate+register the coin VLM.

Airflow only ORCHESTRATES. Each task runs `make ...` inside the train image
(coin-vlm-train) via DockerOperator — no ML logic lives here. The train task
holds the single `gpu_pool` slot so concurrent runs can't fight over the shared
GPU. See docs/mlops.md Mục 8.

Config via Airflow Variables (set once — sub-step 4b):
  coin_project_dir     host repo path, bind-mounted into the container as /app
                       (default /data/coin/coin_vlm_finetune)
  coin_train_image     train image tag (default coin-vlm-train:latest)
  coin_gpu_device      GPU id, e.g. "3", or "all"  (default "all")
  coin_docker_network  docker network to join so the container reaches MLflow
                       (default: unset -> bridge; then mlflow_tracking_uri must
                       be host-reachable)
  mlflow_tracking_uri  e.g. http://mlflow:5000     (default)
  hf_token             HF token (secret; name contains "token" -> masked)
"""
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest, Mount

PROJECT_DIR = Variable.get("coin_project_dir", default_var="/data/coin/coin_vlm_finetune")
TRAIN_IMAGE = Variable.get("coin_train_image", default_var="coin-vlm-train:latest")
GPU_DEVICE = Variable.get("coin_gpu_device", default_var="all")
DOCKER_NETWORK = Variable.get("coin_docker_network", default_var=None)
MLFLOW_URI = Variable.get("mlflow_tracking_uri", default_var="http://mlflow:5000")
HF_TOKEN = Variable.get("hf_token", default_var="")

COMMON_ENV = {
    "MLFLOW_TRACKING_URI": MLFLOW_URI,
    "MLFLOW_EXPERIMENT_NAME": "coin-vlm-finetune",
    "HF_MLFLOW_LOG_ARTIFACTS": "0",
    "HF_TOKEN": HF_TOKEN,
}


def _gpu_requests():
    if GPU_DEVICE == "all":
        return [DeviceRequest(count=-1, capabilities=[["gpu"]])]
    return [DeviceRequest(device_ids=[GPU_DEVICE], capabilities=[["gpu"]])]


def make_task(task_id: str, make_target: str, use_gpu: bool) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image=TRAIN_IMAGE,
        command=["make", make_target],
        working_dir="/app",
        mounts=[Mount(source=PROJECT_DIR, target="/app", type="bind")],
        environment=COMMON_ENV,
        device_requests=_gpu_requests() if use_gpu else None,
        network_mode=DOCKER_NETWORK,
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
        # Shared GPU: only one GPU task at a time. CPU-only register stays out
        # of the pool so it doesn't block a queued train.
        pool="gpu_pool" if use_gpu else "default_pool",
    )


with DAG(
    dag_id="coin_retrain",
    description="CT: train + gate-register coin VLM (extract_match >= 0.90)",
    start_date=pendulum.datetime(2026, 6, 1, tz="Asia/Ho_Chi_Minh"),
    schedule="0 2 * * 1",        # 4d: 02:00 every Monday. Set None for manual-only.
    catchup=False,
    max_active_runs=1,           # never two retrains at once
    tags=["coin-vlm", "ct"],
) as dag:
    # `make train` trains AND runs post-train test-eval -> metrics.json +
    # best_checkpoint.txt under the mounted project dir, so `register` sees them.
    train = make_task("train", "train", use_gpu=True)
    # gate + register; CPU only (no model load) -> not in gpu_pool.
    register = make_task("register", "register", use_gpu=False)

    train >> register

# 4d — data-driven trigger (alternative to the weekly schedule above).
# `data/raw/` is partitioned by day; to retrain when a new day's folder lands,
# set schedule=None above and prepend a sensor that waits for new data, e.g.:
#
#   from airflow.sensors.filesystem import FileSensor
#   wait_new_batch = FileSensor(
#       task_id="wait_new_batch",
#       filepath="/data/coin/coin_vlm_finetune/data/raw/dataset",
#       poke_interval=3600, timeout=24 * 3600, mode="reschedule",
#   )
#   wait_new_batch >> train
