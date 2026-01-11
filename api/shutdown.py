"""Shutdown handler for graceful pipeline shutdown"""

import signal
import threading

stop_event = threading.Event()


def setup_signal_handlers():
    """Setup SIGINT/SIGTERM handlers"""
    def on_shutdown(signum, frame):
        print(f"\n[Shutdown] Signal {signum} received...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)


def wait_for_shutdown():
    """Block until stop_event is set"""
    while not stop_event.is_set():
        threading.Event().wait(1)
