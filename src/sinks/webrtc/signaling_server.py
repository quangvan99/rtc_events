#!/usr/bin/env python3
"""
WebRTC Signaling Server - 1 sender, 1 receiver only
"""

import asyncio

import websockets

class SignalingServer:
    """WebRTC signaling server for 1 sender and 1 receiver"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8555):
        """
        Args:
            host: Host to bind to (default: 0.0.0.0)
            port: Port to listen on (default: 8555)
        """
        self.host = host
        self.port = port
        self.sender = None
        self.receiver = None

    async def handler(self, websocket):
        """Handle WebSocket connections"""
        client_type = None

        try:
            async for message in websocket:
                print(
                    f"[Server] << {message[:80]}..."
                    if len(message) > 80
                    else f"[Server] << {message}"
                )

                if message.startswith("HELLO"):
                    parts = message.split()
                    client_type = parts[1] if len(parts) > 1 else "unknown"

                    # Register as sender or receiver
                    if client_type == "sender":
                        self.sender = websocket
                    else:
                        self.receiver = websocket

                    await websocket.send("HELLO")
                    print(f"[Server] {client_type} connected")

                elif message.startswith("SESSION"):
                    await websocket.send("SESSION_OK")
                    print(f"[Server] Session OK for {client_type}")

                else:
                    # Forward to the other peer only
                    target = self.receiver if websocket == self.sender else self.sender
                    if target:
                        try:
                            await target.send(message)
                            print(
                                f"[Server] Forwarded to {'receiver' if target == self.receiver else 'sender'}"
                            )
                        except:
                            pass

        except websockets.exceptions.ConnectionClosed:
            print(f"[Server] {client_type} disconnected")
        except Exception as e:
            print(f"[Server] Error: {e}")
        finally:
            # Clear reference on disconnect
            if websocket == self.sender:
                self.sender = None
            elif websocket == self.receiver:
                self.receiver = None
            print(
                f"[Server] sender={'connected' if self.sender else 'none'}, "
                f"receiver={'connected' if self.receiver else 'none'}"
            )

    async def process_request(self, connection, request):
        """Process HTTP request before WebSocket upgrade

        Args:
            connection: WebSocket connection object
            request: Request object with path and headers attributes
        """
        # In websockets >= 13.0, the callback receives (connection, request)
        # request.headers is a Headers object (case-insensitive multidict)
        headers = request.headers
        conn_header = headers.get("Connection")
        if not conn_header:
            host = headers.get("Host", "unknown")
            print(f"[Server] Rejected non-WebSocket request from {host}")
            print(f"[Server] Path: {request.path}")
            return connection.respond(400, "WebSocket connection required\n")
        return None  # Accept the connection

    async def run(self):
        """Run the signaling server"""
        print("Mode: 1 sender, 1 receiver only")
        print("Press Ctrl+C to stop")

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            process_request=self.process_request
        ):
            await asyncio.Future()


async def main():
    """Entry point for standalone signaling server"""
    server = SignalingServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
