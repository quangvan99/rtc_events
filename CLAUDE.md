# Claude Code Instructions

## Docker Environment

**IMPORTANT:** Always use Docker container `face-stream` for debugging, development, and testing.

### Workflow

1. **Enter Docker container first:**
   ```bash
   docker exec -it face-stream bash
   ```

2. **Change to project directory:**
   ```bash
   cd /home/mq/disk2T/quangnv/face
   ```

3. **Then run commands:**
   ```bash
   python3 bin/test_multi_branch_video.py
   ```

### Quick One-liner
```bash
docker exec -it face-stream bash -c "cd /home/mq/disk2T/quangnv/face && python3 bin/test_multi_branch_video.py"
```
