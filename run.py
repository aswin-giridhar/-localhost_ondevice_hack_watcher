"""Watcher entry point. Usage: python run.py [config.yaml]

Starts the offline agent + dashboard. Open http://<laptop-ip>:8000 on the laptop,
and http://<laptop-ip>:8000/phone on the phone (same LAN, no internet) to stream
the camera.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ package importable without installation.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn  # noqa: E402

from watcher.config import Config  # noqa: E402
from watcher.server import create_app  # noqa: E402


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = Config.load(cfg_path)
    app = create_app(cfg)

    ssl_args = {}
    scheme = "http"
    if cfg.server.https:
        from watcher.tls import ensure_cert, local_ips

        cert, key = ensure_cert(cfg.server.certfile, cfg.server.keyfile)
        ssl_args = {"ssl_certfile": cert, "ssl_keyfile": key}
        scheme = "https"
        ips = ", ".join(f"{scheme}://{ip}:{cfg.server.port}/phone" for ip in local_ips()
                        if ip != "127.0.0.1")
        print(f"Phone/tablet camera (same Wi-Fi): {ips or '<laptop-lan-ip>'}")
        print("  -> accept the self-signed certificate warning on the device.")

    print(f"Watcher dashboard:  {scheme}://localhost:{cfg.server.port}")
    print(f"Phone camera page:  {scheme}://<laptop-lan-ip>:{cfg.server.port}/phone")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info", **ssl_args)


if __name__ == "__main__":
    main()
