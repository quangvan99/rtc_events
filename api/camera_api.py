"""
CameraAPIServer - REST API for MultibranchCameraManager

Endpoints:
- POST   /api/cameras                              - Add camera to branches
- DELETE /api/cameras/{camera_id}                  - Remove camera entirely
- POST   /api/cameras/{camera_id}/branches/{name}  - Add to branch
- DELETE /api/cameras/{camera_id}/branches/{name}  - Remove from branch
- GET    /api/cameras                              - List all cameras
- GET    /api/branches                             - List all branches
- GET    /api/health                               - Health check
- POST   /api/pipeline/kill                        - Remove all cameras
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from core.multibranch_camera_manager import MultibranchCameraManager

logger = logging.getLogger(__name__)


class CameraAPIServer:
    """REST API server for camera management"""

    def __init__(
        self,
        manager: "MultibranchCameraManager",
        host: str = "0.0.0.0",
        port: int = 8080,
        shutdown_event: asyncio.Event | None = None
    ):
        self.manager = manager
        self.host = host
        self.port = port
        self.shutdown_event = shutdown_event
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""
        self.app.router.add_post("/api/cameras", self.add_camera)
        self.app.router.add_delete("/api/cameras/{camera_id}", self.remove_camera)
        self.app.router.add_post("/api/cameras/{camera_id}/branches/{branch_name}", self.add_to_branch)
        self.app.router.add_delete("/api/cameras/{camera_id}/branches/{branch_name}", self.remove_from_branch)
        self.app.router.add_get("/api/cameras", self.list_cameras)
        self.app.router.add_get("/api/branches", self.list_branches)
        self.app.router.add_get("/api/health", self.health_check)
        self.app.router.add_post("/api/pipeline/kill", self.kill_pipeline)
        self.app.router.add_post("/api/pipeline/stop", self.stop_pipeline)

    # === Camera CRUD ===

    async def add_camera(self, request: web.Request) -> web.Response:
        """POST /api/cameras - Add camera to branches

        Body: {"camera_id": "cam1", "uri": "rtsp://...", "branches": ["recognition"]}
        """
        try:
            data = await request.json()
            camera_id = data["camera_id"]
            uri = data["uri"]
            branches = data.get("branches", [])

            if not branches:
                return web.json_response({"error": "branches required"}, status=400)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.manager.add_camera(camera_id, uri, branches)
            )

            if result:
                logger.info(f"API: Camera added: {camera_id}")
                return web.json_response({"status": "ok", "camera_id": camera_id})
            else:
                return web.json_response({"error": "add failed"}, status=400)

        except KeyError as e:
            return web.json_response({"error": f"missing field: {e}"}, status=400)
        except Exception as e:
            logger.exception(f"API error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def remove_camera(self, request: web.Request) -> web.Response:
        """DELETE /api/cameras/{camera_id} - Remove camera entirely"""
        camera_id = request.match_info["camera_id"]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.manager.remove_camera(camera_id)
        )

        if result:
            logger.info(f"API: Camera removed: {camera_id}")
            return web.json_response({"status": "ok", "camera_id": camera_id})
        else:
            return web.json_response({"error": "camera not found"}, status=404)

    # === Branch Operations ===

    async def add_to_branch(self, request: web.Request) -> web.Response:
        """POST /api/cameras/{camera_id}/branches/{branch_name} - Add to branch"""
        camera_id = request.match_info["camera_id"]
        branch_name = request.match_info["branch_name"]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.manager.add_camera_to_branch(camera_id, branch_name)
        )

        if result:
            logger.info(f"API: {camera_id} added to branch {branch_name}")
            return web.json_response({"status": "ok", "camera_id": camera_id, "branch": branch_name})
        else:
            return web.json_response({"error": "operation failed"}, status=400)

    async def remove_from_branch(self, request: web.Request) -> web.Response:
        """DELETE /api/cameras/{camera_id}/branches/{branch_name} - Remove from branch"""
        camera_id = request.match_info["camera_id"]
        branch_name = request.match_info["branch_name"]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.manager.remove_camera_from_branch(camera_id, branch_name)
        )

        if result:
            logger.info(f"API: {camera_id} removed from branch {branch_name}")
            return web.json_response({"status": "ok", "camera_id": camera_id, "branch": branch_name})
        else:
            return web.json_response({"error": "operation failed"}, status=400)

    # === List Operations ===

    async def list_cameras(self, request: web.Request) -> web.Response:
        """GET /api/cameras - List all cameras"""
        cameras = self.manager.list_cameras()
        return web.json_response({"cameras": cameras})

    async def list_branches(self, request: web.Request) -> web.Response:
        """GET /api/branches - List all branches with cameras"""
        branches = {}
        for name, info in self.manager.branches.items():
            cameras_in_branch = [
                cam_id for cam_id, cam in self.manager._cameras.items()
                if name in cam.branch_queues
            ]
            branches[name] = {
                "max_cameras": info.max_cameras,
                "current_cameras": cameras_in_branch
            }
        return web.json_response({"branches": branches})

    async def health_check(self, request: web.Request) -> web.Response:
        """GET /api/health - Health check"""
        return web.json_response({
            "status": "healthy",
            "cameras": self.manager.count(),
            "branches": list(self.manager.branches.keys())
        })

    async def kill_pipeline(self, request: web.Request) -> web.Response:
        """POST /api/pipeline/kill - Remove all cameras from pipeline"""
        try:
            count = self.manager.kill_all()
            logger.info(f"API: Pipeline killed - {count} cameras removed")
            return web.json_response({"status": "ok", "cameras_removed": count})
        except Exception as e:
            logger.exception(f"API error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def stop_pipeline(self, request: web.Request) -> web.Response:
        """POST /api/pipeline/stop - Stop the entire pipeline (triggers shutdown)"""
        try:
            self.manager.kill_all()
            if self.shutdown_event:
                self.shutdown_event.set()
                logger.info("API: Pipeline stop signaled")
                return web.json_response({"status": "ok", "message": "Pipeline shutdown initiated"})
            else:
                logger.warning("API: No shutdown event configured")
                return web.json_response({"error": "Shutdown not configured", "cameras_removed": self.manager.count()}, status=400)
        except Exception as e:
            logger.exception(f"API error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    # === Server Control ===

    async def start(self):
        """Start the API server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"[CameraAPI] Server running at http://{self.host}:{self.port}")
        return runner

    def run_blocking(self):
        """Run server in blocking mode (for standalone use)"""
        web.run_app(self.app, host=self.host, port=self.port)
