#!/usr/bin/env python3
"""
WebRTC Signaling Server Entry Point

Usage:
    python bin/run_signaling.py

Starts WebRTC signaling server using config from configs/face-recognition.yaml
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_config
from sinks.webrtc import SignalingServer


CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "face-recognition.yaml")


async def main():
    """Start signaling server"""
    config = load_config(CONFIG_PATH)
    sig = config.get("signaling", {})

    host = sig.get("host", "0.0.0.0")
    port = int(sig.get("port", 8555))

    print(f"Starting signaling server on ws://{host}:{port}")
    server = SignalingServer(host=host, port=port)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
