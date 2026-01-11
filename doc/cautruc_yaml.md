# Multi-Branch Pipeline YAML Configuration

## Overview

Pipeline configuration file: `configs/multi-branch.yaml`

## Cấu trúc tổng quan

```yaml
# 1. Recognition settings (face recognition parameters)
recognition:
  features_json: "${FEATURES_JSON:/home/mq/disk2T/quangnv/face/data/face/features.json}"
  l2_threshold: 1.0
  min_streak: 3
  skip_reid: 3
  reid_interval: 30
  min_face_size: 50

# 2. Pipeline definition (branches, elements)
pipeline:
  name: multi-branch-tee-pipeline
  branches:
    recognition:
      max_cameras: 8
      sink: ...
      muxer: ...
      elements: ...
    detection:
      ...

# 3. Video output configuration
output:
  dir: "${OUTPUT_DIR:/home/mq/disk2T/quangnv/face/data}"
  prefix: "output"
  extension: "avi"
  type: "${OUTPUT_TYPE:filesink}"
  sync: false

# 4. REST API configuration
camera_api:
  host: "${API_HOST:0.0.0.0}"
  port: "${API_PORT:8083}"
```

---

## 1. Recognition Settings (`recognition`)

Cấu hình cho face recognition.

```yaml
recognition:
  # Path to face features database JSON
  features_json: "${FEATURES_JSON:/home/mq/disk2T/quangnv/face/data/face/features.json}"
  
  # Max L2 distance for valid face match (lower = stricter)
  l2_threshold: 1.0
  
  # Consecutive frames with same identity to confirm
  min_streak: 3
  
  # Skip SGIE inference for N frames after last inference
  skip_reid: 3
  
  # Re-identify confirmed faces every N frames (~1s at 30fps)
  reid_interval: 30
  
  # Skip SGIE for faces smaller than NxN pixels
  min_face_size: 50
```

### Environment Override

```bash
FEATURES_JSON=/path/to/features.json python bin/test_multi_branch_video.py
```

---

## 2. Pipeline Definition (`pipeline`)

### Cấu trúc cơ bản

```yaml
pipeline:
  name: multi-branch-tee-pipeline  # Tên pipeline
  
  branches:
    recognition: ...               # Branch 1
    detection: ...                 # Branch 2
```

### Branch Configuration

```yaml
pipeline:
  branches:
    recognition:                   # Tên branch
      max_cameras: 8               # Số camera tối đa
      
      # Override output cho branch này (optional)
      sink:
        type: filesink             # filesink | fakesink
        location: /path/to/file.avi  # Custom file path
      
      # nvstreammux configuration
      muxer:
        batch-size: 8
        batched-push-timeout: 40000
        width: 1920
        height: 1080
        live-source: 1
        nvbuf-memory-type: 0
      
      # Các element trong branch
      elements:
        - type: queue
          properties:
            max-size-buffers: 5
            leaky: 2
        
        - type: nvinfer
          name: pgie
          config_file: "${PGIE_CONFIG:data/face/models/scrfd640/infer.txt}"
```

### Muxer Properties (`muxer`)

| Property | Default | Description |
|----------|---------|-------------|
| `batch-size` | 8 | Số frame tối đa trong batch |
| `batched-push-timeout` | 40000 | Timeout (microseconds) |
| `width` | 1920 | Chiều rộng frame |
| `height` | 1080 | Chiều cao frame |
| `live-source` | 1 | Đánh dấu live stream |
| `nvbuf-memory-type` | 0 | Memory type (0=NVBUF_MEM_DEFAULT) |

### Element Configuration

```yaml
elements:
  - type: <gstreamer-element>
    name: <optional-name>
    config_file: <path>           # Cho nvinfer, nvtracker
    properties:                   # Element properties
      <property>: <value>
    caps: <caps-string>           # Cho capsfilter
    probes:                       # Pad probes
      <pad-name>: <probe-name>
```

#### Element Types

| Type | Config File | Purpose |
|------|-------------|---------|
| `queue` | - | Buffer isolation, backpressure |
| `nvinfer` | `${PGIE_CONFIG:...}` | Face detection (ScrFD) |
| `nvtracker` | `${TRACKER_CONFIG:...}` | Object tracking |
| `nvinfer` (SGIE) | `${SGIE_CONFIG:...}` | Face embedding (ArcFace) |
| `nvmultistreamtiler` | - | Multi-camera grid view |
| `nvdsosd` | - | On-screen display |

#### Probes

| Probe Name | Attached To | Purpose |
|------------|-------------|---------|
| `tracker_probe` | tracker src pad | Tracking events |
| `sgie_probe` | SGIE src pad | Face recognition results |
| `fps_probe` | OSD src pad | FPS monitoring |

#### Properties Coercion

```yaml
# String numbers -> int/float
batch-size: 8          # -> 8 (int)
threshold: 1.5         # -> 1.5 (float)

# Boolean strings -> bool
sync: false            # -> False
display-text: yes      # -> True
```

---

## 3. Output Configuration (`output`)

### Cấu hình mặc định

```yaml
output:
  # Thư mục lưu video
  dir: "${OUTPUT_DIR:/home/mq/disk2T/quangnv/face/data}"
  
  # Prefix cho filename
  prefix: "output"
  
  # File extension
  extension: "avi"
  
  # Loại sink mặc định
  type: "${OUTPUT_TYPE:filesink}"
  
  # Sync flag
  sync: false
```

### Output File Naming

```
{dir}/{prefix}_{branch_name}.{extension}
# Ví dụ: /home/mq/disk2T/quangnv/face/data/output_recognition.avi
```

### Sink Types

| Type | Description |
|------|-------------|
| `filesink` | Lưu video ra file |
| `fakesink` | Không lưu (discard frames) |

### Per-Branch Override

```yaml
pipeline:
  branches:
    recognition:
      # Sử dụng fakesink thay vì filesink mặc định
      sink:
        type: fakesink
    
    detection:
      # Custom file path
      sink:
        type: filesink
        location: /tmp/detection_output.avi
```

### Environment Override

```bash
# Sử dụng fakesink thay vì filesink
OUTPUT_TYPE=fakesink python bin/test_multi_branch_video.py

# Thư mục output tùy chỉnh
OUTPUT_DIR=/tmp python bin/test_multi_branch_video.py
```

---

## 4. Camera API Configuration (`camera_api`)

```yaml
camera_api:
  host: "${API_HOST:0.0.0.0}"  # Listen address
  port: "${API_PORT:8083}"     # Port
```

### Environment Override

```bash
API_HOST=127.0.0.1 API_PORT=9000 python bin/test_multi_branch_video.py
```

---

## 5. REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/cameras` | List cameras |
| GET | `/api/branches` | List branches |
| POST | `/api/cameras` | Add camera |
| DELETE | `/api/cameras/{id}` | Remove camera |
| POST | `/api/cameras/{id}/branches/{name}` | Add to branch |
| DELETE | `/api/cameras/{id}/branches/{name}` | Remove from branch |
| POST | `/api/pipeline/kill` | Remove all cameras |
| POST | `/api/pipeline/stop` | Stop pipeline |

### Add Camera Request

```bash
curl -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "cam1",
    "uri": "rtsp://192.168.6.14:8554/test",
    "branches": ["recognition", "detection"]
  }'
```

### Add Camera to Branch

```bash
curl -X POST http://localhost:8083/api/cameras/cam1/branches/recognition
```

---

## 6. All Environment Variables

| Config Path | Env Var | Default |
|-------------|---------|---------|
| `recognition.features_json` | `FEATURES_JSON` | `/home/mq/disk2T/quangnv/face/data/face/features.json` |
| `pipeline.branches.*.elements.*.config_file` (PGIE) | `PGIE_CONFIG` | `data/face/models/scrfd640/infer.txt` |
| `pipeline.branches.*.elements.*.config_file` (SGIE) | `SGIE_CONFIG` | `data/face/models/arcface/infer.txt` |
| `pipeline.branches.*.elements.*.config_file` (tracker) | `TRACKER_CONFIG` | `data/face/models/NvDCF/config_tracker.txt` |
| `output.dir` | `OUTPUT_DIR` | `/home/mq/disk2T/quangnv/face/data` |
| `output.type` | `OUTPUT_TYPE` | `filesink` |
| `camera_api.host` | `API_HOST` | `0.0.0.0` |
| `camera_api.port` | `API_PORT` | `8083` |

---

## 7. Ví dụ hoàn chỉnh

```yaml
# Multi-Branch Pipeline Configuration
# Architecture: Camera -> nvurisrcbin -> tee -> multiple branches

# =============================================================================
# Recognition Settings
# =============================================================================
recognition:
  features_json: "${FEATURES_JSON:/home/mq/disk2T/quangnv/face/data/face/features.json}"
  l2_threshold: 1.0
  min_streak: 3
  skip_reid: 3
  reid_interval: 30
  min_face_size: 50

# =============================================================================
# Pipeline Definition
# =============================================================================
pipeline:
  name: multi-branch-tee-pipeline
  branches:
    # Branch 1: Face Recognition với tracking + embedding
    recognition:
      max_cameras: 8
      # sink:  # Uncomment để override
      #   type: fakesink
      muxer:
        batch-size: 8
        batched-push-timeout: 40000
        width: 1920
        height: 1080
        live-source: 1
        nvbuf-memory-type: 0
      elements:
        - type: queue
          properties:
            max-size-buffers: 5
            leaky: 2
        - type: nvinfer
          name: pgie
          config_file: "${PGIE_CONFIG:data/face/models/scrfd640/infer.txt}"
        - type: nvtracker
          name: tracker
          config_file: "${TRACKER_CONFIG:data/face/models/NvDCF/config_tracker.txt}"
          probes:
            src: tracker_probe
        - type: nvinfer
          name: sgie
          config_file: "${SGIE_CONFIG:data/face/models/arcface/infer.txt}"
          probes:
            src: sgie_probe
        - type: queue
          properties:
            max-size-buffers: 3
            leaky: 2
        - type: nvmultistreamtiler
          name: tiler
          properties:
            rows: 2
            columns: 2
            width: 1920
            height: 1080
        - type: nvdsosd
          properties:
            process-mode: 0
            display-text: 1
          probes:
            src: fps_probe

    # Branch 2: Face Detection Only
    detection:
      max_cameras: 8
      muxer:
        batch-size: 8
        batched-push-timeout: 40000
        width: 1920
        height: 1080
        live-source: 1
        nvbuf-memory-type: 0
      elements:
        - type: queue
          properties:
            max-size-buffers: 5
            leaky: 2
        - type: nvinfer
          name: pgie
          config_file: "${PGIE_CONFIG:data/face/models/scrfd640/infer.txt}"
        - type: queue
          properties:
            max-size-buffers: 3
            leaky: 2
        - type: nvmultistreamtiler
          name: tiler
          properties:
            rows: 2
            columns: 2
            width: 1920
            height: 1080
        - type: nvdsosd
          properties:
            process-mode: 0
            display-text: 1

# =============================================================================
# Output Configuration
# =============================================================================
output:
  dir: "${OUTPUT_DIR:/home/mq/disk2T/quangnv/face/data}"
  prefix: "output"
  extension: "avi"
  type: "${OUTPUT_TYPE:filesink}"
  sync: false

# =============================================================================
# Camera API Configuration
# =============================================================================
camera_api:
  host: "${API_HOST:0.0.0.0}"
  port: "${API_PORT:8083}"
```

---

## 8. Sử dụng

### Chạy với cấu hình mặc định

```bash
python bin/test_multi_branch_video.py
```

### Chạy với fakesink (không lưu video)

```bash
OUTPUT_TYPE=fakesink python bin/test_multi_branch_video.py
```

### Custom output directory

```bash
OUTPUT_DIR=/tmp/recordings python bin/test_multi_branch_video.py
```

### Custom API port

```bash
API_PORT=9000 python bin/test_multi_branch_video.py
```

### Full customization

```bash
OUTPUT_DIR=/tmp OUTPUT_TYPE=fakesink API_PORT=9000 \
  PGIE_CONFIG=/custom/pgie.txt \
  python bin/test_multi_branch_video.py
```
