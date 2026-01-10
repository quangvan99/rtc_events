#!/usr/bin/env python3
"""
Test with proper GLib main loop and idle_add for thread-safe GStreamer operations.
"""

import sys
import os
import threading
import time
import http.server
import socketserver
import json
import queue

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
# Note: FaceProbes requires full face recognition setup (config, database, tracker)
# Skipping probe registration - pipeline will work without face recognition probes

DEFAULT_CONFIG = "configs/multi-branch.yaml"
DEFAULT_OUTPUT_DIR = "/home/mq/disk2T/quangnv/face/data"


class GLibSafeAPI:
    def __init__(self, manager, port, shutdown_event):
        self.manager = manager
        self.port = port
        self.shutdown_event = shutdown_event
        self.op_queue = queue.Queue()
        self.op_results = {}
        self.op_lock = threading.Lock()
        self.min_delay = 2.0
        self.last_op = 0.0
        self.http_thread = None
        self.server = None

    def _schedule_op(self, op_id, func, args, kwargs):
        self.op_queue.put((op_id, func, args, kwargs))

    def _process_ops_idle(self):
        while True:
            try:
                op_id, func, args, kwargs = self.op_queue.get(timeout=0.1)
                with self.op_lock:
                    now = time.time()
                    elapsed = now - self.last_op
                    if elapsed < self.min_delay:
                        time.sleep(self.min_delay - elapsed)
                    self.last_op = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.op_results[op_id] = {"status": "ok", "result": result}
                except Exception as e:
                    self.op_results[op_id] = {"status": "error", "error": str(e)}
            except queue.Empty:
                break
            except Exception as e:
                pass
        return True

    def start(self):
        GLib.idle_add(self._process_ops_idle)
        self.http_thread = threading.Thread(target=self._run_http, daemon=True)
        self.http_thread.start()

    def _run_http(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            api = None

            def log_message(self, *args):
                pass

            def send_json(self, code, data):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def do_GET(self):
                if self.path == "/api/health":
                    count = Handler.api.manager.count()
                    branches = list(Handler.api.manager.branches.keys())
                    self.send_json(200, {"status": "healthy", "cameras": count, "branches": branches})
                elif self.path == "/api/cameras":
                    cameras = Handler.api.manager.list_cameras()
                    self.send_json(200, {"cameras": cameras})
                elif self.path == "/api/branches":
                    branches = {}
                    for name, info in Handler.api.manager.branches.items():
                        cams = [cid for cid, cam in Handler.api.manager._cameras.items() if name in cam.branch_queues]
                        branches[name] = {"max_cameras": info.max_cameras, "current_cameras": cams}
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
                        import uuid
                        op_id = str(uuid.uuid4())[:8]
                        Handler.api._schedule_op(op_id, Handler.api.manager.add_camera, (camera_id, uri, branches), {})
                        self.send_json(202, {"status": "accepted", "operation_id": op_id})
                    except Exception as e:
                        self.send_json(400, {"error": str(e)})
                elif self.path == "/api/pipeline/kill":
                    import uuid
                    op_id = str(uuid.uuid4())[:8]
                    Handler.api._schedule_op(op_id, Handler.api.manager.kill_all, (), {})
                    self.send_json(202, {"status": "accepted", "operation_id": op_id})
                elif self.path == "/api/pipeline/stop":
                    Handler.api._schedule_op("stop", Handler.api.manager.kill_all, (), {})
                    if Handler.api.shutdown_event:
                        Handler.api.shutdown_event.set()
                    self.send_json(200, {"status": "ok"})
                else:
                    self.send_json(404, {"error": "not found"})

            def do_DELETE(self):
                if self.path.startswith("/api/cameras/"):
                    parts = self.path.split("/")
                    if len(parts) >= 4:
                        camera_id = parts[3]
                        import uuid
                        if len(parts) >= 6 and parts[4] == "branches":
                            branch_name = parts[5]
                            op_id = str(uuid.uuid4())[:8]
                            Handler.api._schedule_op(op_id, Handler.api.manager.remove_camera_from_branch, (camera_id, branch_name), {})
                        else:
                            op_id = str(uuid.uuid4())[:8]
                            Handler.api._schedule_op(op_id, Handler.api.manager.remove_camera, (camera_id,), {})
                        self.send_json(202, {"status": "accepted", "operation_id": op_id})
                    else:
                        self.send_json(400, {"error": "invalid"})
                else:
                    self.send_json(404, {"error": "not found"})

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        Handler.api = self
        self.server = Server(("0.0.0.0", self.port), Handler)
        self.server.serve_forever()

    def stop(self):
        if self.server:
            self.server.shutdown()


def main():
    print("=" * 60)
    print("Multi-Branch Pipeline - GLib-safe API")
    print("=" * 60)

    config = load_config(DEFAULT_CONFIG)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    branches_cfg = config.get("pipeline", {}).get("branches", {})
    sinks = {}
    for branch_name in branches_cfg.keys():
        sinks[branch_name] = FilesinkAdapter(location=f"{DEFAULT_OUTPUT_DIR}/output_{branch_name}.avi")

    print(f"\n[Setup] Building pipeline with {len(branches_cfg)} branches...")

    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    manager = MultibranchCameraManager(pipeline, builder.branches)

    stop_event = threading.Event()

    def on_sig(signum, frame):
        print(f"\n[Shutdown] Signal received...")
        stop_event.set()

    import signal
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    for sink in sinks.values():
        sink.start()

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline!")
        return 1

    print(f"\n[Running] Pipeline started!")

    api = GLibSafeAPI(manager, 8083, stop_event)
    api.start()

    print(f"[API] Server at http://localhost:8083")

    print("\nCURL Commands:")
    print("1. Add cam1: curl -X POST http://localhost:8083/api/cameras -H 'Content-Type: application/json' -d '{\"camera_id\": \"cam1\", \"uri\": \"rtsp://192.168.6.14:8554/test\", \"branches\": [\"recognition\"]}'")
    print("2. List: curl http://localhost:8083/api/cameras")
    print("3. Stop: curl -X POST http://localhost:8083/api/pipeline/stop")

    main_loop = GLib.MainLoop()
    GLib.idle_add(lambda: not stop_event.is_set())

    try:
        main_loop.run()
    except KeyboardInterrupt:
        pass

    print("\n[Shutdown] Stopping...")
    pipeline.set_state(Gst.State.NULL)
    api.stop()
    for sink in sinks.values():
        sink.stop()
    print("[Done]")


if __name__ == "__main__":
    main()
