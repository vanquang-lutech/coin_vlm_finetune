"""Launch the FastAPI + vLLM backend for the coin VLM.

Loads the same layered config as the rest of the project (data/model/training
+ optional method + inference), merges the serving config on top, then starts
uvicorn. The serving model_path must point at a merged (+ optionally AWQ)
checkpoint produced by scripts/export.py.

Example:
    python scripts/serve.py \
        --data_config config/data/coin_dataset.yaml \
        --model_config config/model/qwen3_vl_8b.yaml \
        --training_config config/training/training.yaml \
        --serving_config config/serving/serving.yaml \
        --override serving.model_path=outputs/merged_models/coin-vlm-awq
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from omegaconf import OmegaConf

from src.serving import create_app
from src.utils import ConfigLoader, setup_logging, get_logger

logger = get_logger(__name__)


def start_ngrok(config, port: int):
    """Open an ngrok tunnel to the local port using the reserved domain, so the
    API on the A100 box is publicly reachable for testing. Returns the tunnel
    (or None if disabled). Failures are logged but do NOT stop the server."""
    ng = config.serving.get("ngrok", None)
    if not ng or not ng.get("enabled", False):
        return None

    try:
        from pyngrok import ngrok
    except ImportError:
        logger.error(
            "ngrok.enabled=true but pyngrok is not installed. "
            "Run `pip install pyngrok` or set serving.ngrok.enabled=false."
        )
        return None

    token = os.getenv("NGROK_AUTHTOKEN") or ng.get("authtoken", None)
    domain = ng.get("domain", None)
    if domain and not token:
        logger.error(
            "ngrok reserved domain '%s' requires an authtoken. "
            "Set NGROK_AUTHTOKEN env var (or serving.ngrok.authtoken). "
            "Continuing WITHOUT a tunnel.",
            domain,
        )
        return None

    # Use a pre-installed ngrok binary if provided, so pyngrok does NOT try to
    # download it (the A100 box may not reach bin.ngrok.com).
    binary_path = os.getenv("NGROK_PATH") or ng.get("binary_path", None)
    pyngrok_config = None
    if binary_path:
        from pyngrok.conf import PyngrokConfig
        pyngrok_config = PyngrokConfig(ngrok_path=binary_path)
        logger.info("Using ngrok binary at: %s", binary_path)

    try:
        if token:
            ngrok.set_auth_token(token, pyngrok_config=pyngrok_config)
        connect_kwargs = {"addr": str(port), "proto": "http"}
        if domain:
            connect_kwargs["domain"] = domain
        if pyngrok_config is not None:
            connect_kwargs["pyngrok_config"] = pyngrok_config
        tunnel = ngrok.connect(**connect_kwargs)
        logger.info("ngrok tunnel up: %s -> http://localhost:%d", tunnel.public_url, port)
        return tunnel
    except Exception:  # noqa: BLE001 - tunnel is best-effort, never fatal
        logger.exception(
            "Failed to start ngrok tunnel; continuing without it. "
            "If this is a download error, install the ngrok binary manually and "
            "set NGROK_PATH (or serving.ngrok.binary_path)."
        )
        return None


def stop_ngrok(tunnel) -> None:
    if tunnel is None:
        return
    try:
        from pyngrok import ngrok
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        logger.info("ngrok tunnel closed.")
    except Exception:  # noqa: BLE001
        logger.warning("Failed to cleanly close ngrok tunnel.")


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the coin VLM via FastAPI + vLLM.")
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--method_config", default=None)
    parser.add_argument(
        "--inference_config",
        default="config/inference/inference.yaml",
        help="Provides prompt + generation defaults (shared with scripts/inference.py).",
    )
    parser.add_argument(
        "--serving_config",
        default="config/serving/serving.yaml",
        help="vLLM engine + HTTP server settings.",
    )
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)  # bootstrap (stdout) until config is loaded

    config = ConfigLoader.load(
        data_config=args.data_config,
        model_config=args.model_config,
        training_config=args.training_config,
        method_config=args.method_config,
        inference_config=args.inference_config,
        overrides=args.override,
    )

    # Merge the serving config (ConfigLoader doesn't know about it). Config is
    # read-only after load, so reopen, merge, and re-freeze.
    OmegaConf.set_readonly(config, False)
    serving_cfg = OmegaConf.load(args.serving_config)
    config = OmegaConf.merge(config, serving_cfg)
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    OmegaConf.set_readonly(config, True)

    if "serving" not in config:
        raise ValueError("Missing `serving:` block; check --serving_config.")

    # Re-init logging now that we know serving.log_dir, adding a date-partitioned
    # file sink that rolls over by day/month.
    log_dir = config.serving.get("log_dir", None)
    setup_logging(
        args.log_level,
        log_dir=log_dir,
        log_stem="serve",
        log_rotation=config.serving.get("log_rotation", "daily"),
    )

    app = create_app(config)

    host = config.serving.get("host", "0.0.0.0")
    port = int(config.serving.get("port", 8000))
    logger.info("Serving on http://%s:%d (model=%s)", host, port, config.serving.model_path)

    tunnel = start_ngrok(config, port)
    try:
        # log_config=None: keep OUR logging setup. uvicorn's default logging
        # config runs dictConfig() which would close our file handler's stream.
        uvicorn.run(
            app, host=host, port=port,
            log_level=args.log_level.lower(),
            log_config=None,
        )
    finally:
        stop_ngrok(tunnel)


if __name__ == "__main__":
    main()
