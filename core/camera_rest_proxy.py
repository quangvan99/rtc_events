"""
Camera REST Proxy - Intercepts nvmultiurisrcbin REST API to sync SourceIDMapper

When cameras are added/removed via curl directly to nvmultiurisrcbin REST API,
SourceIDMapper doesn't get updated. This proxy intercepts requests, updates the
mapper, then forwards to nvmultiurisrcbin.

Usage:
    proxy = CameraRESTProxy(mapper, upstream_port=9000, proxy_port=9001)
    await proxy.start()

Then use proxy_port (9001) for curl commands:
    curl -X POST 'http://localhost:9001/api/v1/stream/add' \
      -H 'Content-Type: application/json' \
      -d '{"value":{"camera_id":"cam1","camera_url":"...","change":"camera_add"}}'
"""

import asyncio
import json
import logging
import urllib.request
import urllib.error
from typing import Optional
from aiohttp import web

from core.source_mapper import SourceIDMapper

logger = logging.getLogger(__name__)


class CameraRESTProxy:
    """HTTP proxy that intercepts camera add/remove and syncs SourceIDMapper"""

    def __init__(
        self,
        mapper: SourceIDMapper,
        upstream_host: str = "localhost",
        upstream_port: int = 9000,
        proxy_host: str = "0.0.0.0",
        proxy_port: int = 9001,
    ):
        """
        Initialize CameraRESTProxy

        Args:
            mapper: SourceIDMapper to keep in sync
            upstream_host: nvmultiurisrcbin REST API host
            upstream_port: nvmultiurisrcbin REST API port
            proxy_host: Host to bind proxy server
            proxy_port: Port to bind proxy server
        """
        self.mapper = mapper
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None

    def _forward_request(self, path: str, payload: dict) -> tuple[bool, str, int]:
        """Forward request to upstream nvmultiurisrcbin REST API"""
        url = f"http://{self.upstream_host}:{self.upstream_port}{path}"
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return True, resp.read().decode('utf-8'), resp.status
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code}: {e.reason}", e.code
        except urllib.error.URLError as e:
            return False, str(e.reason), 502
        except Exception as e:
            return False, str(e), 500

    async def handle_add(self, request: web.Request) -> web.Response:
        """Handle camera add request"""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        value = data.get("value", {})
        camera_id = value.get("camera_id")
        camera_url = value.get("camera_url")
        change = value.get("change")

        if change != "camera_add":
            return web.json_response({"error": "Invalid change type"}, status=400)

        if not camera_id or not camera_url:
            return web.json_response({"error": "Missing camera_id or camera_url"}, status=400)

        # Check if camera already exists
        if self.mapper.has_camera(camera_id):
            return web.json_response({"error": f"Camera {camera_id} already exists"}, status=409)

        # Add to mapper FIRST (to reserve source_id)
        try:
            source_id = self.mapper.add(camera_id, camera_url, "")
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=409)

        # Forward to upstream
        ok, msg, status = self._forward_request("/api/v1/stream/add", data)

        if ok:
            logger.info(f"[Proxy] Camera added: {camera_id} -> source_id={source_id}")
            return web.json_response({
                "success": True,
                "camera_id": camera_id,
                "source_id": source_id,
                "message": msg
            })
        else:
            # Rollback mapper on failure
            self.mapper.remove(camera_id)
            logger.error(f"[Proxy] Failed to add camera: {msg}")
            return web.json_response({"error": msg}, status=status)

    async def handle_remove(self, request: web.Request) -> web.Response:
        """Handle camera remove request"""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        value = data.get("value", {})
        camera_id = value.get("camera_id")
        change = value.get("change")

        if change != "camera_remove":
            return web.json_response({"error": "Invalid change type"}, status=400)

        if not camera_id:
            return web.json_response({"error": "Missing camera_id"}, status=400)

        # Get camera info before removal
        info = self.mapper.get_by_camera_id(camera_id)
        if not info:
            return web.json_response({"error": f"Camera {camera_id} not found"}, status=404)

        # Ensure camera_url is in the request (nvmultiurisrcbin requires it)
        if "camera_url" not in value:
            value["camera_url"] = info.url
            data["value"] = value

        # Forward to upstream FIRST
        ok, msg, status = self._forward_request("/api/v1/stream/remove", data)

        if ok:
            # Remove from mapper on success
            self.mapper.remove(camera_id)
            logger.info(f"[Proxy] Camera removed: {camera_id}")
            return web.json_response({
                "success": True,
                "camera_id": camera_id,
                "message": msg
            })
        else:
            logger.error(f"[Proxy] Failed to remove camera: {msg}")
            return web.json_response({"error": msg}, status=status)

    async def handle_list(self, request: web.Request) -> web.Response:
        """Handle camera list request"""
        cameras = self.mapper.list_cameras()
        return web.json_response({
            "cameras": [
                {
                    "camera_id": c.camera_id,
                    "source_id": c.source_id,
                    "url": c.url,
                    "name": c.name,
                    "added_at": c.added_at
                }
                for c in cameras
            ],
            "count": len(cameras)
        })

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "ok",
            "proxy_port": self.proxy_port,
            "upstream": f"{self.upstream_host}:{self.upstream_port}",
            "cameras": self.mapper.count()
        })

    async def start(self) -> None:
        """Start the proxy server"""
        self.app = web.Application()
        self.app.router.add_post("/api/v1/stream/add", self.handle_add)
        self.app.router.add_post("/api/v1/stream/remove", self.handle_remove)
        self.app.router.add_get("/api/v1/cameras", self.handle_list)
        self.app.router.add_get("/health", self.handle_health)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.proxy_host, self.proxy_port)
        await site.start()
        logger.info(f"[Proxy] Started on http://{self.proxy_host}:{self.proxy_port}")

    async def stop(self) -> None:
        """Stop the proxy server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("[Proxy] Stopped")
