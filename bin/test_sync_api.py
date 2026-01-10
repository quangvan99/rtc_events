#!/usr/bin/env python3
"""
Simple synchronous HTTP server for GStreamer operations.
Uses threading.Lock() and time.sleep() to avoid asyncio conflicts.
"""

import sys
import os
import threading
import time
import http.server
import socketserver
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from core.tee_fanout_builder import TeeFanoutPipelineBuilder
from core.multibranch_camera_manager import MultibranchCameraManager
from core.config import load_config
from sinks.filesink_adapter import FilesinkAdapter

DEFAULT_CONFIG = "configs/multi-branch.yaml"
DEFAULT_OUTPUT_DIR = "/home/mq/disk2T/quangnv/face/data"


class GStreamerAPIHandler(http.server.BaseHTTPRequestHandler):
    server = None

    def log_message(self, format, *args):
        pass

    def send_json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/api/health":
            count = self.server.manager.count()
            branches = list(self.server.manager.branches.keys())
            self.send_json(200, {"status": "healthy", "cameras": count, "branches": branches})
        elif self.path == "/api/cameras":
            cameras = self.server.manager.list_cameras()
            self.send_json(200, {"cameras": cameras})
        elif self.path == "/api/branches":
            branches = {}
            for name, info in self.server.manager.branches.items():
                cameras_in_branch = [
                    cam_id for cam_id, cam in self.server.manager._cameras.items()
                    if name in cam.branch_queues
                ]
                branches[name] = {"max_cameras": info.max_cameras, "current_cameras": cameras_in_branch}
            self.send_json(200, {"branches": branches})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/cameras":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                camera_id = data["camera_id"]
                uri = data["uri"]
                branches = data.get("branches", [])
                with self.server.op_lock:
                    now = time.time()
                    elapsed = now - self.server.last_op
                    if elapsed < self.server.min_delay:
                        time.sleep(self.server.min_delay - elapsed)
                    self.server.last_op = time.time()
                    result = self.server.manager.add_camera(camera_id, uri, branches)
                if result:
                    self.send_json(200, {"status": "ok", "camera_id": camera_id})
                else:
                    self.send_json(400, {"error": "add failed"})
            except Exception as e:
                self.send_json(400, {"error": str(e)})
        elif self.path == "/api/pipeline/kill":
            with self.server.op_lock:
                now = time.time()
                elapsed = now - self.server.last_op
                if elapsed < self.server.min_delay:
                    time.sleep(self.server.min_delay - elapsed)
                self.server.last_op = time.time()
                count = self.server.manager.kill_all()
            self.send_json(200, {"status": "ok", "cameras_removed": count})
        elif self.path == "/api/pipeline/stop":
            with self.server.op_lock:
                now = time.time()
                elapsed = now - self.server.last_op
                if elapsed < self.server.min_delay:
                    time.sleep(self.server.min_delay - elapsed)
                self.server.last_op = time.time()
                self.server.manager.kill_all()
            if self.server.shutdown_event:
                self.server.shutdown_event.set()
            self.send_json(200, {"status": "ok", "message": "shutdown"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_DELETE(self):
        if self.path.startswith("/api/cameras/"):
            parts = self.path.split("/")
            if len(parts) >= 4:
                camera_id = parts[3]
                try:
                    with self.server.op_lock:
                        now = time.time()
                        elapsed = now - self.server.last_op
                        if elapsed < self.server.min_delay:
                            time.sleep(self.server.min_delay - elapsed)
                        self.server.last_op = time.time()
                        if len(parts) >= 6 and parts[4] == "branches":
                            branch_name = parts[5]
                            result = self.server.manager.remove_camera_from_branch(camera_id, branch_name)
                        else:
                            result = self.server.manager.remove_camera(camera_id)
                    self.send_json(200, {"status": "ok", "camera_id": camera_id})
                except Exception as e:
                    self.send_json(400, {"error": str(e)})
            else:
                self.send_json(400, {"error": "invalid path"})
        else:
            self.send_json(404, {"error": "not found"})


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    print("=" * 60)
    print("Multi-Branch Pipeline - Synchronous API Server")
    print("=" * 60)

    config = load_config(DEFAULT_CONFIG)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    branches_cfg = config.get("pipeline", {}).get("branches", {})
    sinks = {}
    for branch_name in branches_cfg.keys():
        sinks[branch_name] = FilesinkAdapter(location=f"{DEFAULT_OUTPUT_DIR}/output_{branch_name}.avi")

    print(f"\n[Setup] Building pipeline with {len(branches_cfg)} branches...")
    for branch_name, branch_cfg in branches_cfg.items():
        elements = branch_cfg.get("elements", [])
        element_types = [e.get("type", "") for e in elements if e.get("name")]
        print(f"  - {branch_name}: {', '.join(element_types) if element_types else 'configured in YAML'}")

    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    manager = MultibranchCameraManager(pipeline, builder.branches)

    stop_event = threading.Event()

    def on_shutdown(signum, frame):
        print(f"\n[Shutdown] Received signal...")
        stop_event.set()

    import signal
    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)

    for sink in sinks.values():
        sink.start()

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline!")
        return 1

    print(f"\n[Running] Pipeline started!")

    server = ThreadedHTTPServer(("0.0.0.0", 8083), GStreamerAPIHandler)
    server.manager = manager
    server.shutdown_event = stop_event
    server.op_lock = threading.Lock()
    server.min_delay = 2.0
    server.last_op = 0.0

    print(f"[API] Server at http://localhost:8083")

    print("\nCURL Commands:")
    print("1. Add cam1: curl -X POST http://localhost:8083/api/cameras -H 'Content-Type: application/json' -d '{\"camera_id\": \"cam1\", \"uri\": \"rtsp://192.168.6.14:8554/test\", \"branches\": [\"recognition\"]}'")
    print("2. Add cam2: curl -X POST http://localhost:8083/api/cameras -H 'Content-Type: application/json' -d '{\"camera_id\": \"cam2\", \"uri\": \"rtsp://192.168.6.14:8554/test\", \"branches\": [\"detection\"]}'")
    print("3. List: curl http://localhost:8083/api/cameras")
    print("4. Stop: curl -X POST http://localhost:8083/api/pipeline/stop")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

    print("\n[Shutdown] Stopping pipeline...")
    server.shutdown()
    pipeline.set_state(Gst.State.NULL)
    for sink in sinks.values():
        sink.stop()
    print("[Done]")


if __name__ == "__main__":
    main()
