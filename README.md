# FACE - Face Recognition Streaming System

Real-time face recognition with WebRTC streaming for NVIDIA Jetson platforms.

## Overview

FACE is a GPU-accelerated face recognition pipeline built on NVIDIA DeepStream 7.1. It detects, tracks, and identifies faces in real-time from video sources, streaming the results to web browsers via WebRTC.

**Key Features:**
- Real-time face detection using SCRFD 2.5G
- Face tracking with NvDCF visual tracker
- Face recognition using ArcFace ResNet-100
- Low-latency WebRTC streaming with H.264
- Browser-based viewer with recognition events

## Prerequisites

- NVIDIA Jetson (Orin/Xavier) with JetPack 6.x
- DeepStream SDK 7.1
- Python 3.10+
- CUDA 12.6 / TensorRT 10.x

## Quick Start

### 1. Setup Environment

```bash
# Run Jetson setup script
cd scripts
bash setup_7_1_jetson.sh

# Generate SSL certificates
bash genkey.sh
```

### 2. Configure

Edit `cfg.py` to set:
- `ws_server`: Your Jetson's IP address
- `input_uri`: Video source (file path or RTSP URL)

### 3. Run

```bash
# Terminal 1: Start signaling server
python3 signalling.py

# Terminal 2: Open viewer in browser
# Navigate to view.html (use HTTPS)

# Terminal 3: Start face recognition pipeline
python3 stream.py
```

## Project Structure

```
FACE/
├── cfg.py              # Pipeline configuration
├── db.py               # Face database with caching
├── track.py            # Face tracking and voting
├── stream.py           # Main DeepStream pipeline
├── signalling.py       # WebRTC signaling server
├── view.html           # Browser-based viewer
├── features.json       # Registered face features
├── models/
│   ├── arcface/        # Face recognition model
│   ├── scrfd640/       # Face detection model
│   └── NvDCF/          # Tracker configuration
├── scripts/
│   ├── setup_7_1_jetson.sh  # Environment setup
│   └── genkey.sh       # SSL certificate generation
└── docs/               # Documentation
    ├── project-overview-pdr.md
    ├── codebase-summary.md
    ├── code-standards.md
    └── system-architecture.md
```

## Architecture

```
Video Source ──▶ DeepStream Pipeline ──▶ WebRTC ──▶ Browser
                       │
                       ├── SCRFD (Detection)
                       ├── NvDCF (Tracking)
                       ├── ArcFace (Recognition)
                       └── DataChannel (Events)
```

## Configuration

Key parameters in `cfg.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ws_server` | 192.168.6.112 | WebSocket server address |
| `ws_port` | 8555 | WebSocket port |
| `l2_threshold` | 1.20 | Max L2 distance for matching |
| `vote_threshold` | 3 | Votes needed for confirmation |
| `vote_window_frames` | 90 | Sliding window size |

## Face Registration

Add faces to `features.json`:

```json
{
    "PersonName": {
        "feature": [0.1, 0.2, ...],
        "avatar": "<base64_encoded_image>"
    }
}
```

Features are 512-dimensional normalized vectors from ArcFace.

## Video Sources

Supported input formats:
- Local files: `file:///path/to/video.mp4`
- RTSP streams: `rtsp://user:pass@ip:port/path`
- HTTP streams: `http://example.com/stream`

## Browser Viewer

The `view.html` viewer provides:
- 70/30 split layout (video/sidebar)
- Real-time recognition event feed
- Auto-reconnection on disconnect
- Fullscreen support

Access via: `https://<jetson-ip>:8555` (after opening view.html)

## Documentation

See `docs/` for detailed documentation:
- [Project Overview & PDR](docs/project-overview-pdr.md)
- [Codebase Summary](docs/codebase-summary.md)
- [Code Standards](docs/code-standards.md)
- [System Architecture](docs/system-architecture.md)

## Troubleshooting

### WebRTC Connection Issues
- Ensure SSL certificates are generated
- Check firewall allows port 8555
- Verify STUN server is reachable

### Recognition Not Working
- Check `features.json` is properly formatted
- Delete `features.cache.npz` to force cache rebuild
- Enable `debug_voting: True` in config

### Low FPS
- Reduce input resolution via `muxer_width/height`
- Increase `skip_reid` to reduce SGIE calls
- Check GPU utilization with `tegrastats`

## License

Proprietary - Internal use only.
