# Face Stream Pipeline - Project Guide

## Quick Reference

| Item | Value |
|------|-------|
| Container | `face-stream` |
| Host Path | `/home/mq/disk2T/quangnv/face` |
| Container Path | `/app` |
| API Port | `8083` |
| Python | `3.10.12` |
| OS | Ubuntu 22.04 |

## Docker Execution

### Enter Container (Interactive Shell)

```bash
docker exec -it -w /app face-stream bash
```

### Run Commands Inside Container

```bash
# Single command
docker exec -w /app face-stream <command>

# With bash
docker exec -w /app face-stream bash -c "<commands>"
```

## Debug & Test

### Run Pipeline

```bash
# Inside container
python3 entry/test_multi_branch_video.py

# With custom config
python3 entry/test_multi_branch_video.py --config configs/test-single-branch.yaml

# From host
docker exec -w /app face-stream python3 entry/test_multi_branch_video.py
```

### Background Run (with logs)

```bash
# Start in background
docker exec -d -w /app face-stream bash -c "python3 entry/test_multi_branch_video.py > /tmp/pipeline.log 2>&1"

# View logs
docker exec face-stream tail -f /tmp/pipeline.log
```

### Full Test Script

```bash
# From host (outside container)
./entry/full_test_video.sh
```

## API Endpoints

Base URL: `http://localhost:8083`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/cameras` | List cameras |
| GET | `/api/branches` | List branches |
| POST | `/api/cameras` | Add camera |
| DELETE | `/api/cameras/{id}` | Remove camera |
| POST | `/api/cameras/{id}/branches/{branch}` | Add camera to branch |
| DELETE | `/api/cameras/{id}/branches/{branch}` | Remove camera from branch |
| POST | `/api/pipeline/kill` | Remove all cameras |
| POST | `/api/pipeline/stop` | Stop pipeline |

### API Examples

```bash
# Health check
curl http://localhost:8083/api/health

# Add camera
curl -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "cam1", "uri": "rtsp://192.168.6.14:8554/testface", "branches": ["recognition", "detection"]}'

# List cameras
curl http://localhost:8083/api/cameras | python3 -m json.tool
```

## Project Structure

```
/app/
├── api/              # REST API server
├── apps/             # Processor apps (detection, face)
│   ├── detection/    # Detection processor
│   └── face/         # Face recognition processor
├── configs/          # Pipeline configurations
├── data/             # Data files, output videos
├── entry/            # Entry points
├── scripts/          # Docker management scripts
└── src/              # Core pipeline modules
    ├── camera_manager.py
    ├── common.py
    └── pipeline_builder.py
```

## Debugging Tips

### Check Container Status

```bash
docker ps -a --filter "name=face-stream"
```

### View Logs

```bash
# Container logs
docker logs face-stream --tail 100

# Pipeline logs
docker exec face-stream tail -f /tmp/pipeline.log
docker exec face-stream tail -f /tmp/full_test.log
```

### Kill Pipeline Processes

```bash
docker exec face-stream pkill -9 -f 'python.*test_multi'
```

### Check Running Processes

```bash
docker exec face-stream ps aux | grep python
```

### Container Restart

```bash
cd scripts && docker compose restart
# or
./scripts/stop.sh && ./scripts/start.sh
```

## Common Issues

1. **Container unhealthy**: Check `docker logs face-stream` for errors
2. **Port 8083 in use**: Run `lsof -i :8083` and kill the process
3. **Pipeline crash**: Check `/tmp/pipeline.log` inside container
4. **RTSP timeout**: Verify camera URI is reachable from container
