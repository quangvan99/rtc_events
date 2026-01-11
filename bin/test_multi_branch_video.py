#!/usr/bin/env python3
"""
Multi-Branch Pipeline Test with Face Recognition

Usage:
    python bin/test_multi_branch_video.py
"""

import argparse
import signal
import sys
import os
import threading
import time
import http.server
import socketserver
import json
import queue
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from core.tee_fanout_builder import TeeFanoutPipelineBuilder
from core.multibranch_camera_manager import MultibranchCameraManager
from core.config import load_config
from sinks.filesink_adapter import FilesinkAdapter
from apps.face.database import FaceDatabase
from apps.face.tracker import TrackerManager
from apps.face.probes import FaceProbes

DEFAULT_CONFIG = "configs/multi-branch.yaml"
DEFAULT_OUTPUT_DIR = "/home/mq/disk2T/quangnv/face/data"


class CameraAPIServer:
    def __init__(self, manager, port, shutdown_event):
        self.manager = manager
        self.port = port
        self.shutdown_event = shutdown_event
        self.op_queue = queue.Queue()
        self.op_results = {}
        self.op_lock = threading.Lock()
        self.min_delay = 3.0
        self.last_op = 0.0
        self._running = True
        self.rtsp_uri = "rtsp://192.168.6.14:8554/test"
        self.http_thread = None
        self._op_thread = None

    def _process_ops(self):
        while self._running:
            try:
                op_id, func, args, kwargs = self.op_queue.get(timeout=0.2)
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
                self.op_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def _run_http(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            server = None

            def log_message(self, *args):
                pass

            def send_json(self, code, data):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def do_GET(self):
                if self.path == "/api/health":
                    count = Handler.server.manager.count()
                    branches = list(Handler.server.manager.branches.keys())
                    self.send_json(200, {"status": "healthy", "cameras": count, "branches": branches})
                elif self.path == "/api/cameras":
                    cameras = Handler.server.manager.list_cameras()
                    self.send_json(200, {"cameras": cameras})
                elif self.path == "/api/branches":
                    branches = {}
                    for name, info in Handler.server.manager.branches.items():
                        cams = [cid for cid, cam in Handler.server.manager._cameras.items() if name in cam.branch_queues]
                        branches[name] = {"max_cameras": info.max_cameras, "current_cameras": cams}
                    self.send_json(200, {"branches": branches})
                elif self.path.startswith("/api/operations/"):
                    op_id = self.path.split("/")[-1]
                    result = Handler.server.op_results.get(op_id)
                    if result:
                        self.send_json(200, result)
                    else:
                        self.send_json(404, {"error": "operation not found"})
                else:
                    self.send_json(404, {"error": "not found"})

            def do_POST(self):
                if self.path == "/api/cameras":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        data = json.loads(body)
                        camera_id = data["camera_id"]
                        uri = data.get("uri", Handler.server.rtsp_uri)
                        branches = data.get("branches", [])
                        import uuid
                        op_id = str(uuid.uuid4())[:8]
                        Handler.server.op_queue.put((op_id, Handler.server.manager.add_camera, (camera_id, uri, branches), {}))
                        self.send_json(202, {"status": "accepted", "operation_id": op_id})
                    except Exception as e:
                        self.send_json(400, {"error": str(e)})
                elif self.path.startswith("/api/cameras/") and "/branches/" in self.path:
                    parts = self.path.split("/")
                    if len(parts) >= 7:
                        camera_id = parts[3]
                        branch_name = parts[6]
                        try:
                            import uuid
                            op_id = str(uuid.uuid4())[:8]
                            Handler.server.op_queue.put((op_id, Handler.server.manager.add_camera_to_branch, (camera_id, branch_name), {}))
                            self.send_json(202, {"status": "accepted", "operation_id": op_id})
                        except Exception as e:
                            self.send_json(400, {"error": str(e)})
                    else:
                        self.send_json(400, {"error": "invalid path"})
                elif self.path == "/api/pipeline/kill":
                    try:
                        import uuid
                        op_id = str(uuid.uuid4())[:8]
                        Handler.server.op_queue.put((op_id, Handler.server.manager.kill_all, (), {}))
                        self.send_json(202, {"status": "accepted", "operation_id": op_id})
                    except Exception as e:
                        self.send_json(400, {"error": str(e)})
                elif self.path == "/api/pipeline/stop":
                    Handler.server.op_queue.put(("stop", Handler.server.manager.kill_all, (), {}))
                    if Handler.server.shutdown_event:
                        Handler.server.shutdown_event.set()
                    self.send_json(200, {"status": "ok", "message": "shutdown"})
                else:
                    self.send_json(404, {"error": "not found"})

            def do_DELETE(self):
                if self.path.startswith("/api/cameras/"):
                    parts = self.path.split("/")
                    if len(parts) >= 4:
                        camera_id = parts[3]
                        try:
                            import uuid
                            if len(parts) >= 6 and parts[4] == "branches":
                                branch_name = parts[5]
                                op_id = str(uuid.uuid4())[:8]
                                Handler.server.op_queue.put((op_id, Handler.server.manager.remove_camera_from_branch, (camera_id, branch_name), {}))
                            else:
                                op_id = str(uuid.uuid4())[:8]
                                Handler.server.op_queue.put((op_id, Handler.server.manager.remove_camera, (camera_id,), {}))
                            self.send_json(202, {"status": "accepted", "operation_id": op_id})
                        except Exception as e:
                            self.send_json(400, {"error": str(e)})
                    else:
                        self.send_json(400, {"error": "invalid path"})
                else:
                    self.send_json(404, {"error": "not found"})

        class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        Handler.server = self
        httpd = ThreadedHTTPServer(("0.0.0.0", self.port), Handler)
        httpd.serve_forever()

    def start(self):
        self.http_thread = threading.Thread(target=self._run_http, daemon=True)
        self.http_thread.start()
        self._op_thread = threading.Thread(target=self._process_ops, daemon=True)
        self._op_thread.start()

    def stop(self):
        self._running = False


def main():
    print("=" * 60)
    print("Multi-Branch Pipeline with Face Recognition")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--rtsp", default="rtsp://192.168.6.14:8554/test")
    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    branches_cfg = config.get("pipeline", {}).get("branches", {})
    sinks = {}
    for branch_name in branches_cfg.keys():
        sinks[branch_name] = FilesinkAdapter(location=f"{DEFAULT_OUTPUT_DIR}/output_{branch_name}.avi")

    print(f"\n[Setup] Building pipeline with {len(branches_cfg)} branches...")

    rec_config = config.get("recognition", {})
    features_path = rec_config.get("features_json", "data/face/features.json")

    print(f"[FaceDB] Loading {features_path}...")
    db = FaceDatabase(features_path)
    print(f"[FaceDB] Loaded {len(db.names)} faces")

    print("[Tracker] Initializing tracker...")
    tracker_mgr = TrackerManager(rec_config)

    recognition_sink = sinks.get("recognition", sinks.get(list(sinks.keys())[0]))
    probes = FaceProbes(rec_config, db, tracker_mgr, recognition_sink)

    builder = TeeFanoutPipelineBuilder(config, sinks)
    builder.register_probe("tracker_probe", probes.tracker_probe)
    builder.register_probe("sgie_probe", probes.sgie_probe)
    builder.register_probe("fps_probe", probes.fps_probe)
    pipeline = builder.build()

    manager = MultibranchCameraManager(pipeline, builder.branches)

    stop_event = threading.Event()

    def on_shutdown(signum, frame):
        print(f"\n[Shutdown] Signal received...")
        stop_event.set()

    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)

    for sink in sinks.values():
        sink.start()

    print("\n[Pipeline] Setting to PLAYING...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline!")
        return 1

    print("[Pipeline] Started!")

    api = CameraAPIServer(manager, args.port, stop_event)
    api.rtsp_uri = args.rtsp
    api.start()

    print(f"\n" + "=" * 60)
    print(f"[API] Server at http://localhost:{args.port}")
    print(f"[Faces] {len(db.names)} registered")
    print(f"[RTSP] Default: {args.rtsp}")
    print("=" * 60)

    print("""
CURL Commands:

# Step 1: Add cam1 to both branches
curl -X POST http://localhost:8083/api/cameras \\
     -H "Content-Type: application/json" \\
     -d '{"camera_id": "cam1", "uri": "rtsp://192.168.6.14:8554/test", "branches": ["recognition", "detection"]}'

# Step 2: Add cam2 (wait 3s after step 1)
curl -X POST http://localhost:8083/api/cameras \\
     -H "Content-Type: application/json" \\
     -d '{"camera_id": "cam2", "uri": "rtsp://192.168.6.14:8554/test", "branches": ["recognition", "detection"]}'

# Step 3: Remove cam2 from detection (wait 3s after step 2)
curl -X DELETE http://localhost:8083/api/cameras/cam2/branches/detection

# Step 4: Add cam3
curl -X POST http://localhost:8083/api/cameras \\
     -H "Content-Type: application/json" \\
     -d '{"camera_id": "cam3", "uri": "rtsp://192.168.6.14:8554/test", "branches": ["recognition", "detection"]}'

# Check status
curl http://localhost:8083/api/cameras
curl http://localhost:8083/api/branches

Press Ctrl+C to stop...
    """)

    def check_pipeline_health():
        try:
            state = pipeline.get_state(0)
            if state == Gst.StateChangeReturn.FAILURE:
                return False
        except:
            return False
        return True

    try:
        restart_count = 0
        while not stop_event.is_set():
            time.sleep(1)
            if not check_pipeline_health():
                restart_count += 1
                print(f"[Warning] Pipeline unhealthy, attempt {restart_count}/3 to restart...")
                if restart_count <= 3:
                    try:
                        pipeline.set_state(Gst.State.NULL)
                        time.sleep(1)
                        pipeline.set_state(Gst.State.PLAYING)
                        print("[Pipeline] Restarted successfully")
                    except Exception as e:
                        print(f"[Error] Failed to restart pipeline: {e}")
                else:
                    print("[Error] Too many restart attempts, giving up")
                    break
    except KeyboardInterrupt:
        pass

    print("\n[Shutdown] Stopping...")
    api.stop()
    pipeline.set_state(Gst.State.NULL)
    for sink in sinks.values():
        sink.stop()
    print("[Done]")


if __name__ == "__main__":
    main()
