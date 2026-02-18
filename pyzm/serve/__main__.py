"""CLI entry point: ``python -m pyzm.serve``."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="pyzm ML Detection Server",
        prog="python -m pyzm.serve",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["yolo11s"],
        help=(
            "Model names to load (space-separated, default: yolo11s). "
            "Use 'all' to auto-discover every model in --base-path "
            "(loaded lazily on first request)."
        ),
    )
    ap.add_argument("--base-path", default="/var/lib/zmeventnotification/models")
    ap.add_argument("--processor", default="cpu", choices=["cpu", "gpu", "tpu"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--auth", action="store_true", help="Enable JWT authentication")
    ap.add_argument("--auth-user", default="admin")
    ap.add_argument("--auth-password", default="")
    ap.add_argument("--token-secret", default="change-me")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    ap.add_argument(
        "--config",
        help="Path to a YAML config file (ServerConfig). Overrides CLI flags.",
    )
    args = ap.parse_args()

    import logging

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("pyzm").setLevel(level)

    from pyzm.models.config import Processor, ServerConfig

    if args.config:
        import yaml

        with open(args.config) as fh:
            raw = yaml.safe_load(fh) or {}
        config = ServerConfig.model_validate(raw)
    else:
        config = ServerConfig(
            host=args.host,
            port=args.port,
            models=args.models,
            base_path=args.base_path,
            processor=Processor(args.processor),
            auth_enabled=args.auth,
            auth_username=args.auth_user,
            auth_password=args.auth_password,
            token_secret=args.token_secret,
        )

    from pyzm.serve.app import create_app

    app = create_app(config)

    import uvicorn

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
