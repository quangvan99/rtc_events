"""
CameraAPIServer - REST API for MultibranchCameraManager using FastAPI

Endpoints:
- POST   /api/cameras                              - Add camera to branches
- DELETE /api/cameras/{camera_id}                  - Remove camera entirely
- POST   /api/cameras/{camera_id}/branches/{name}  - Add to branch
- DELETE /api/cameras/{camera_id}/branches/{name}  - Remove from branch
- GET    /api/cameras                              - List all cameras
- GET    /api/branches                             - List all branches
- GET    /api/health                               - Health check
- GET    /api/operations/{op_id}                   - Get operation status
- POST   /api/pipeline/kill                        - Remove all cameras
- POST   /api/pipeline/stop                        - Stop pipeline
"""

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

if TYPE_CHECKING:
    from core.multibranch_camera_manager import MultibranchCameraManager

logger = logging.getLogger(__name__)


class CameraAPIServer:
    MIN_OP_DELAY = 2.0

    def __init__(
        self,
        manager: "MultibranchCameraManager",
        host: str = "0.0.0.0",
        port: int = 8080,
        shutdown_event=None
    ):
        self.manager = manager
        self.host = host
        self.port = port
        self.shutdown_event = shutdown_event
        self.op_queue = queue.Queue()
        self.op_results = {}
        self.op_lock = threading.Lock()
        self._running = True
        self._app = None

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

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Camera API")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class AddCameraRequest(BaseModel):
            camera_id: str
            uri: str
            branches: List[str] = []

        @app.get("/api/health")
        async def health():
            count = self.manager.count()
            branches = list(self.manager.branches.keys())
            return {"status": "healthy", "cameras": count, "branches": branches}

        @app.get("/api/cameras")
        async def list_cameras():
            cameras = self.manager.list_cameras()
            return {"cameras": cameras}

        @app.get("/api/branches")
        async def list_branches():
            branches = {}
            for name, info in self.manager.branches.items():
                cams = [cid for cid, cam in self.manager._cameras.items() if name in cam.branch_queues]
                branches[name] = {"max_cameras": info.max_cameras, "current_cameras": cams}
            return {"branches": branches}

        @app.get("/api/operations/{op_id}")
        async def get_operation(op_id: str):
            result = self.op_results.get(op_id)
            if result:
                return result
            raise HTTPException(status_code=404, detail="operation not found")

        @app.post("/api/cameras")
        async def add_camera(request: AddCameraRequest):
            import uuid
            camera_id = request.camera_id
            uri = request.uri
            branches = request.branches
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, self.manager.add_camera, (camera_id, uri, branches), {}))
            return {"status": "accepted", "operation_id": op_id}

        @app.post("/api/cameras/{camera_id}/branches/{branch_name}")
        async def add_camera_to_branch(camera_id: str, branch_name: str):
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, self.manager.add_camera_to_branch, (camera_id, branch_name), {}))
            return {"status": "accepted", "operation_id": op_id}

        @app.post("/api/pipeline/kill")
        async def kill_pipeline():
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, self.manager.kill_all, (), {}))
            return {"status": "accepted", "operation_id": op_id}

        @app.post("/api/pipeline/stop")
        async def stop_pipeline():
            self.op_queue.put(("stop", self.manager.kill_all, (), {}))
            if self.shutdown_event:
                self.shutdown_event.set()
            return {"status": "ok", "message": "shutdown"}

        @app.delete("/api/cameras/{camera_id}")
        async def remove_camera(camera_id: str):
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, self.manager.remove_camera, (camera_id,), {}))
            return {"status": "accepted", "operation_id": op_id}

        @app.delete("/api/cameras/{camera_id}/branches/{branch_name}")
        async def remove_camera_from_branch(camera_id: str, branch_name: str):
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, self.manager.remove_camera_from_branch, (camera_id, branch_name), {}))
            return {"status": "accepted", "operation_id": op_id}

        return app

    def start(self):
        self._op_thread = threading.Thread(target=self._process_ops, daemon=True)
        self._op_thread.start()
        self._app = self._create_app()
        config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config=config)
        threading.Thread(target=self._server.run, daemon=True).start()
        logger.info(f"[CameraAPI] Server running at http://{self.host}:{self.port}")

    def stop(self):
        self._running = False

    @property
    def last_op(self):
        return self._last_op if hasattr(self, '_last_op') else 0.0

    @last_op.setter
    def last_op(self, value):
        self._last_op = value

    @property
    def min_delay(self):
        return self.MIN_OP_DELAY
