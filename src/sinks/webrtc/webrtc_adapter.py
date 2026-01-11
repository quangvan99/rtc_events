#!/usr/bin/env python3
"""
WebRTC Sink Adapter - Implements BaseSink for WebRTC streaming
"""

import asyncio
import json
from typing import Callable, Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstWebRTC", "1.0")
from gi.repository import Gst, GstSdp, GstWebRTC

import websockets

from src.sinks.base_sink import BaseSink


class WebRTCAdapter(BaseSink):
    """WebRTC sink adapter for streaming video and sending events"""

    def __init__(
        self,
        ws_url: str,
        peer_id: str = "stream",
        stun_server: str = "",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.ws_url = ws_url
        self.peer_id = peer_id
        self.stun_server = stun_server
        self.loop = loop

        # WebSocket connection
        self.conn: Optional[websockets.WebSocketClientProtocol] = None

        # WebRTC elements
        self.webrtc: Optional[Gst.Element] = None
        self.data_channel = None
        self.data_channel_open = False

        # Negotiation state
        self.awaiting_answer = False

        # Event handling
        self.pending_events: list[dict] = []
        self.on_ready_callback: Optional[Callable] = None

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        """
        Create WebRTC bin with encoding pipeline.
        Returns: first_element (conv2) for linking from upstream
        """
        # Create webrtcbin
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")
        if not self.webrtc:
            raise RuntimeError("Cannot create webrtcbin")

        # Only set STUN server if provided (not needed for local network)
        if self.stun_server:
            self.webrtc.set_property("stun-server", self.stun_server)
        self.webrtc.set_property("latency", 200)
        self.webrtc.set_property("bundle-policy", 3)

        # Connect signals
        self.webrtc.connect("on-negotiation-needed", self._on_negotiation)
        self.webrtc.connect("on-ice-candidate", self._on_ice)
        self.webrtc.connect("on-data-channel", self._on_data_channel)
        self.webrtc.connect(
            "notify::ice-connection-state", self._on_ice_connection_state
        )
        self.webrtc.connect("notify::ice-gathering-state", self._on_ice_gathering_state)
        self.webrtc.connect("notify::connection-state", self._on_connection_state)

        pipeline.add(self.webrtc)

        # Create encoding pipeline
        def make_element(factory: str, name: str) -> Gst.Element:
            elem = Gst.ElementFactory.make(factory, name)
            if not elem:
                raise RuntimeError(f"Cannot create element: {factory}")
            pipeline.add(elem)
            return elem

        # Convert for encoding
        conv2 = make_element("nvvideoconvert", "conv2")
        conv2.set_property("compute-hw", 1)

        caps2 = make_element("capsfilter", "caps2")
        caps2.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420"))

        queue_enc = make_element("queue", "queue_enc")
        queue_enc.set_property("max-size-buffers", 3)
        queue_enc.set_property("leaky", 2)

        # H264 encoding
        enc = make_element("x264enc", "encoder")
        enc.set_property("tune", "zerolatency")
        enc.set_property("speed-preset", "ultrafast")
        enc.set_property("key-int-max", 30)
        enc.set_property("bitrate", 2000)

        h264parse = make_element("h264parse", "h264parse")
        h264parse.set_property("config-interval", -1)

        pay = make_element("rtph264pay", "pay")
        pay.set_property("config-interval", -1)

        rtpcaps = make_element("capsfilter", "rtpcaps")
        rtpcaps.set_property(
            "caps",
            Gst.Caps.from_string(
                "application/x-rtp,media=video,encoding-name=H264,payload=96"
            ),
        )

        # Link encoding chain
        elements = [conv2, caps2, queue_enc, enc, h264parse, pay, rtpcaps]

        for i in range(len(elements) - 1):
            if not elements[i].link(elements[i + 1]):
                raise RuntimeError(f"Failed to link {elements[i].get_name()}")

        # Link to WebRTC bin
        rtpcaps.link(self.webrtc)

        print("[WebRTC] Encoding pipeline created and linked")
        return conv2

    def start(self) -> None:
        """Called before pipeline starts - connection happens async."""
        pass

    def stop(self) -> None:
        """Called when pipeline stops."""
        if self.loop and self.conn:
            asyncio.run_coroutine_threadsafe(self.disconnect(), self.loop)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for async operations"""
        self.loop = loop

    def set_on_ready_callback(self, callback: Callable) -> None:
        """Set callback to be called when WebRTC is ready"""
        self.on_ready_callback = callback

    # =========================================================================
    # WebRTC Signal Handlers
    # =========================================================================

    def _send(self, msg: str) -> None:
        """Send message to signaling server"""
        if self.conn and self.loop:
            asyncio.run_coroutine_threadsafe(self.conn.send(msg), self.loop)

    def _on_negotiation(self, _) -> None:
        """Handle WebRTC negotiation needed"""
        print("[WebRTC] Negotiation needed")

        if not self.data_channel:
            self.data_channel = self.webrtc.emit("create-data-channel", "data", None)
            if self.data_channel:
                self.data_channel.connect("on-open", self._on_channel_open)
                self.data_channel.connect(
                    "on-close", lambda c: self._on_channel_close()
                )
                self.data_channel.connect("on-message-string", self._on_channel_message)
                print("[WebRTC] DataChannel created")

        promise = Gst.Promise.new_with_change_func(self._on_offer, None, None)
        self.webrtc.emit("create-offer", None, promise)

    def _on_offer(self, promise, _, __) -> None:
        """Handle offer creation"""
        reply = promise.get_reply()
        if not reply:
            return
        offer = reply["offer"]
        if not offer:
            return
        self.awaiting_answer = True
        self.webrtc.emit("set-local-description", offer, Gst.Promise.new())
        self._send(json.dumps({"sdp": {"type": "offer", "sdp": offer.sdp.as_text()}}))

    def _on_ice(self, _, idx: int, candidate: str) -> None:
        """Handle ICE candidate"""
        print(f"[ICE] Sending local candidate: {candidate[:70]}...")
        self._send(json.dumps({"ice": {"candidate": candidate, "sdpMLineIndex": idx}}))

    def _on_ice_connection_state(self, webrtc, pspec) -> None:
        """Handle ICE connection state changes"""
        state = webrtc.get_property("ice-connection-state")
        state_names = {
            0: "new",
            1: "checking",
            2: "connected",
            3: "completed",
            4: "failed",
            5: "disconnected",
            6: "closed",
        }
        state_name = state_names.get(state, f"unknown({state})")
        print(f"[ICE] Connection state: {state_name}")

        if state == 4:
            print("[ICE] Connection failed - check network/STUN server")
        elif state in (2, 3):
            print("[ICE] Connection established successfully!")

    def _on_ice_gathering_state(self, webrtc, pspec) -> None:
        """Handle ICE gathering state changes"""
        state = webrtc.get_property("ice-gathering-state")
        state_names = {0: "new", 1: "gathering", 2: "complete"}
        state_name = state_names.get(state, f"unknown({state})")
        print(f"[ICE] Gathering state: {state_name}")

    def _on_connection_state(self, webrtc, pspec) -> None:
        """Handle overall connection state changes"""
        state = webrtc.get_property("connection-state")
        state_names = {
            0: "new",
            1: "connecting",
            2: "connected",
            3: "disconnected",
            4: "failed",
            5: "closed",
        }
        state_name = state_names.get(state, f"unknown({state})")
        print(f"[WebRTC] Connection state: {state_name}")

        if state == 2 and self.on_ready_callback:
            self.on_ready_callback()

    # =========================================================================
    # SDP/ICE Handlers
    # =========================================================================

    def handle_answer(self, sdp: str) -> None:
        """Handle SDP answer from peer"""
        if not self.awaiting_answer:
            print(f"[SDP] Ignoring unexpected answer ({len(sdp)} bytes)")
            return

        self.awaiting_answer = False
        print(f"[SDP] Processing answer ({len(sdp)} bytes)")
        result, msg = GstSdp.SDPMessage.new_from_text(sdp)
        if result != GstSdp.SDPResult.OK:
            print(f"[SDP] Failed to parse answer: {result}")
            return
        answer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.ANSWER, msg
        )
        promise = Gst.Promise.new_with_change_func(self._on_answer_set, None, None)
        self.webrtc.emit("set-remote-description", answer, promise)

    def _on_answer_set(self, promise, _, __) -> None:
        """Callback when answer is set"""
        result = promise.wait()
        if result == Gst.PromiseResult.REPLIED:
            print("[SDP] Remote description set successfully")
        else:
            print(f"[SDP] Failed to set remote description: {result}")

    def handle_ice(self, ice: dict) -> None:
        """Handle ICE candidate from peer"""
        candidate = ice.get("candidate", "")
        sdp_mline_index = ice.get("sdpMLineIndex", 0)

        if not candidate:
            print("[ICE] Received end-of-candidates signal")
            return

        # Note: mDNS candidates may work in local network, so try to add them
        if ".local" in candidate:
            print(f"[ICE] Processing mDNS candidate: {candidate[:50]}...")

        print(f"[ICE] Adding remote candidate: {candidate[:70]}...")
        try:
            self.webrtc.emit("add-ice-candidate", sdp_mline_index, candidate)
            print("[ICE] Added candidate successfully")
        except Exception as e:
            print(f"[ICE] Failed to add candidate: {e}")

    # =========================================================================
    # DataChannel Handlers
    # =========================================================================

    def _on_data_channel(self, _, channel) -> None:
        """Handle incoming DataChannel"""
        print("[DataChannel] Received from peer")
        self.data_channel = channel
        channel.connect("on-open", self._on_channel_open)
        channel.connect("on-close", lambda c: self._on_channel_close())
        channel.connect("on-message-string", self._on_channel_message)

    def _on_channel_open(self, channel) -> None:
        """Handle DataChannel open"""
        print("[DataChannel] OPEN - ready to send events")
        self.data_channel_open = True

        # Send pending events
        for event in self.pending_events:
            self._do_send_event(event)
        self.pending_events.clear()

    def _on_channel_close(self) -> None:
        """Handle DataChannel close"""
        print("[DataChannel] Closed")
        self.data_channel = None
        self.data_channel_open = False

    def _on_channel_message(self, channel, msg: str) -> None:
        """Handle message from DataChannel"""
        print(f"[DataChannel] Received: {msg}")

    def _do_send_event(self, msg: dict) -> bool:
        """Actually send event via DataChannel"""
        if not self.data_channel or not self.data_channel_open:
            return False
        try:
            self.data_channel.emit("send-string", json.dumps(msg))
            print(f"[DataChannel] Sent: {msg.get('type', 'unknown')}")
            return True
        except Exception as e:
            print(f"[DataChannel] Failed to send: {e}")
            return False

    def send_event(self, event: dict) -> None:
        """Send event via DataChannel (queues if not ready)"""
        if self.data_channel_open:
            success = self._do_send_event(event)
            if not success:
                print(
                    f"[DataChannel] Failed to send, channel open={self.data_channel_open}"
                )
        else:
            self.pending_events.append(event)
            print(f"[DataChannel] Queued event (pending={len(self.pending_events)})")

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> None:
        """Connect to signaling server (local network, no SSL)"""
        self.conn = await websockets.connect(self.ws_url)
        print(f"[WebRTC] Connected to {self.ws_url}")
        await self.conn.send("HELLO sender")

    async def handle_signaling(self) -> None:
        """Handle signaling messages"""
        if not self.conn:
            raise RuntimeError("Not connected to signaling server")

        try:
            async for msg in self.conn:
                print(f"[WS] <<< {msg[:80]}" if len(msg) > 80 else f"[WS] <<< {msg}")

                if msg == "HELLO":
                    await self.conn.send(f"SESSION {self.peer_id}")
                elif msg == "SESSION_OK":
                    print("[WS] Session established, ready for pipeline")
                elif msg.startswith("ERROR"):
                    print(f"[WS] Server error: {msg}")
                    break
                else:
                    try:
                        data = json.loads(msg)
                        if "sdp" in data and data["sdp"]["type"] == "answer":
                            self.handle_answer(data["sdp"]["sdp"])
                        elif "ice" in data:
                            self.handle_ice(data["ice"])
                    except Exception as e:
                        print(f"[WS] Failed to parse message: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("[WS] Connection closed")

    async def disconnect(self) -> None:
        """Disconnect from signaling server"""
        if self.conn:
            await self.conn.close()
            self.conn = None
        print("[WebRTC] Disconnected")
