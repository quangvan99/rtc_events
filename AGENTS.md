# AGENTS.md - Development Guidelines for AI Assistants

## Overview

This is a **DeepStream-based face recognition streaming system** for NVIDIA Jetson platforms. The codebase uses GStreamer/Python bindings with aiohttp for REST APIs and WebRTC for streaming.

---

## Available Agents

OpenCode supports the following agent types:

| Agent Type | Description | Use Case |
|------------|-------------|----------|
| `general` | General-purpose agent for multi-step tasks | Complex development tasks, refactoring, debugging |
| `explore` | Fast agent for codebase exploration | Finding files, searching code, understanding structure |

### Agent Usage

```python
from agents import general, explore

# Launch general agent for complex tasks
result = general.run(
    prompt="Refactor the camera manager to use async/await",
    description="Refactor camera manager"
)

# Launch explore agent for searching
files = explore.run(
    prompt="Find all files related to face recognition",
    description="Find face recognition files"
)
```

---

## Available Skills

The following skills are available for use with OpenCode:

### Core Development Skills

| Skill Name | Description |
|------------|-------------|
| `planning` | Plan technical solutions that are scalable, secure, and maintainable |
| `research` | Research and analyze technical solutions |
| `Debugging` | Systematic debugging framework with root cause investigation |
| `code-review` | Code review with technical rigor and verification gates |
| `backend-development` | Build robust backend systems with modern technologies |
| `frontend-dev-guidelines` | Frontend development guidelines for React/TypeScript |

### Specialized Skills

| Skill Name | Description |
|------------|-------------|
| `web-frameworks` | Build full-stack web applications with Next.js, Turborepo |
| `ui-styling` | Create beautiful UIs with shadcn/ui, Tailwind CSS |
| `ui-ux-pro-max` | UI/UX design intelligence with 50+ styles |
| `frontend-design` | Production-grade frontend interfaces |
| `frontend-design-pro` | High-quality interfaces with real images |
| `threejs` | Build 3D web experiences with Three.js |
| `mobile-development` | Build mobile apps with React Native, Flutter |
| `docs-seeker` | Search technical documentation |
| `media-processing` | Process multimedia with FFmpeg, ImageMagick |
| `databases` | Work with MongoDB and PostgreSQL |
| `devops` | Deploy to Cloudflare, Docker, GCP |
| `ai-multimodal` | Process and generate multimedia with Google Gemini |
| `chrome-devtools` | Browser automation and debugging |
| `dev-browser` | Browser automation with persistent state |
| `payment-integration` | Implement payment with SePay and Polar |
| `shopify` | Build Shopify applications |
| `better-auth` | Authentication with Better Auth framework |
| `mcp-builder` | Create MCP servers |
| `mcp-management` | Manage MCP servers |

### Productivity Skills

| Skill Name | Description |
|------------|-------------|
| `repomix` | Package codebases into AI-friendly files |
| `aesthetic` | Create aesthetically beautiful interfaces |
| `pptx` | Create and edit presentations |
| `xlsx` | Create and edit spreadsheets |
| `pdf` | Manipulate PDF documents |
| `docx` | Create and edit Word documents |

### Advanced Skills

| Skill Name | Description |
|------------|-------------|
| `sequential-thinking` | Structured problem-solving for complex tasks |
| `skill-creator` | Create custom skills |
| `template-skill` | Skill creation guide |
| `websearch` | Real-time web search |
| `codesearch` | Search code APIs and libraries |

### Skill Usage

```python
from skills import planning, Debugging, web_frameworks

# Use planning skill
plan = planning.run(
    query="Design a scalable REST API for camera management"
)

# Use debugging skill
Debugging.run(
    command="Investigate memory leak in camera pipeline",
    description="Debug memory leak"
)

# Use web-frameworks skill
result = web_frameworks.run(
    query="Next.js App Router with Server Components example"
)
```

---

## Configuration Examples

### Using Agents with Context

```python
from agents import general

general.run(
    prompt="""Analyze the face recognition pipeline and:
1. Identify performance bottlenecks
2. Suggest optimization strategies
3. Implement one optimization""",
    description="Optimize face recognition pipeline",
    context={
        "project_root": "/home/mq/disk2T/quangnv/face",
        "focus_areas": ["apps/face/", "core/"]
    }
)
```

### Chaining Skills

```python
from skills import research, planning, web_frameworks

# Research phase
docs = research.run(query="DeepStream Python bindings best practices")

# Planning phase
plan = planning.run(query="Design pattern for multi-camera GStreamer pipeline")

# Implementation phase
result = web_frameworks.run(query="Python asyncio with GStreamer integration")
```

---

## Quick Command Aliases

Create shell aliases for fast agent/skill access:

```bash
# Add to ~/.bashrc or ~/.zshrc

# Agent shortcuts
alias oa='python3 /home/mq/disk2T/quangnv/face/bin/opencode_agent.py'
alias oa-explore='python3 /home/mq/disk2T/quangnv/face/bin/opencode_agent.py --agent explore'
alias oa-general='python3 /home/mq/disk2T/quangnv/face/bin/opencode_agent.py --agent general'

# Skill shortcuts
alias os-plan='python3 /home/mq/disk2T/quangnv/face/bin/opencode_skill.py --skill planning'
alias os-debug='python3 /home/mq/disk2T/quangnv/face/bin/opencode_skill.py --skill debugging'
alias os-search='python3 /home/mq/disk2T/quangnv/face/bin/opencode_skill.py --skill codesearch'
alias os-websearch='python3 /home/mq/disk2T/quangnv/face/bin/opencode_skill.py --skill websearch'
```

### Quick Command Scripts

Create `/home/mq/disk2T/quangnv/face/bin/opencode_agent.py`:

```python
#!/usr/bin/env python3
"""
Quick agent launcher for OpenCode
Usage: python3 bin/opencode_agent.py --agent explore --prompt "Find face files"
"""

import argparse
import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')

from agents import general, explore

def main():
    parser = argparse.ArgumentParser(description='OpenCode Quick Agent')
    parser.add_argument('--agent', choices=['general', 'explore'], required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--description', default='Quick task')
    
    args = parser.parse_args()
    
    if args.agent == 'general':
        result = general.run(prompt=args.prompt, description=args.description)
    else:
        result = explore.run(prompt=args.prompt, description=args.description)
    
    print(result)

if __name__ == '__main__':
    main()
```

Create `/home/mq/disk2T/quangnv/face/bin/opencode_skill.py`:

```python
#!/usr/bin/env python3
"""
Quick skill launcher for OpenCode
Usage: python3 bin/opencode_skill.py --skill planning --query "Design API"
"""

import argparse
import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')

from skills import (
    planning, research, Debugging, code_review,
    websearch, codesearch, web_frameworks,
    media_processing, databases, devops
)

SKILL_MAP = {
    'planning': planning,
    'research': research,
    'debugging': Debugging,
    'code-review': code_review,
    'websearch': websearch,
    'codesearch': codesearch,
    'web-frameworks': web_frameworks,
    'media-processing': media_processing,
    'databases': databases,
    'devops': devops,
}

def main():
    parser = argparse.ArgumentParser(description='OpenCode Quick Skill')
    parser.add_argument('--skill', choices=SKILL_MAP.keys(), required=True)
    parser.add_argument('--query', required=True)
    
    args = parser.parse_args()
    
    skill = SKILL_MAP[args.skill]
    result = skill.run(query=args.query)
    print(result)

if __name__ == '__main__':
    main()
```

### Make scripts executable

```bash
chmod +x /home/mq/disk2T/quangnv/face/bin/opencode_agent.py /home/mq/disk2T/quangnv/face/bin/opencode_skill.py
```

### Usage Examples

```bash
# Quick explore
python3 /home/mq/disk2T/quangnv/face/bin/opencode_agent.py --agent explore --prompt "Find all Python files in apps/face/" --description="Find face files"

# Quick planning
python3 /home/mq/disk2T/quangnv/face/bin/opencode_skill.py --skill planning --query "Design REST API for camera management"

# Quick web search
python3 /home/mq/disk2T/quangnv/face/bin/opencode_skill.py --skill websearch --query "GStreamer Python bindings 2024"

# Quick code search
python3 /home/mq/disk2T/quangnv/face/bin/opencode_skill.py --skill codesearch --query "Python dataclass examples"
```

---

## Docker Environment

**Always use Docker container `face-stream` for development:**

```bash
docker exec -it face-stream bash
cd /home/mq/disk2T/quangnv/face
```

**Quick one-liner:**
```bash
docker exec -it face-stream bash -c "cd /home/mq/disk2T/quangnv/face && python3 bin/test_multi_branch_video.py"
```

---

## Build/Lint/Test Commands

### Running Tests

**Run all tests:**
```bash
cd /home/mq/disk2T/quangnv/face
python3 -m pytest tests/                    # All tests
python3 tests/test_multibranch_camera_manager.py  # Specific test file
```

**Run single test function:**
```bash
python3 -c "from tests.test_multibranch_camera_manager import test_add_camera_single_branch; test_add_camera_single_branch()"
python3 tests/test_camera_api.py && echo "PASS"
```

**Run specific test file:**
```bash
python3 bin/test_multi_branch_video.py                              # Main integration test
python3 bin/test_camera_crud_webrtc.py                               # WebRTC CRUD tests
python3 bin/test_multi_camera_e2e.py                                 # End-to-end tests
```

**Bash test scripts:**
```bash
bash bin/test_multi_branch_scenario.sh   # Full scenario with curl commands
```

### Running Applications

```bash
python3 bin/run_multi_branch.py --config configs/multi-branch.yaml   # Multi-branch pipeline
python3 bin/run_face_webrtc.py          # Face recognition + WebRTC
python3 bin/run_face_fakesink.py        # Face with fake sink (testing)
python3 bin/run_signaling.py            # WebRTC signaling server
```

### Code Quality

```bash
python3 -m py_compile <file.py          # Syntax check
python3 -m flake8 <file.py              # Style check (if installed)
python3 -c "import ast; ast.parse(open(f).read())"   # Parse validation
```

---

## Code Style Guidelines

### Imports

- **Python 3.10+ type hints** with `from __future__ import annotations`
- **Standard library first**, then third-party, then local:
  ```python
  from __future__ import annotations

  import asyncio
  import configparser
  import os
  import sys
  from dataclasses import dataclass, field
  from typing import Optional

  import gi
  gi.require_version("Gst", "1.0")
  from gi.repository import GLib, Gst

  sys.path.append("/opt/nvidia/deepstream/deepstream/lib")

  from sinks.base_sink import BaseSink
  from core.probe_registry import ProbeRegistry
  ```
- **Avoid wildcard imports** (`from X import *`)
- **Group imports** logically with blank lines between groups

### Formatting

- **4-space indentation** (no tabs)
- **Line length: 120 characters** (flexible for readability)
- **Blank lines:**
  - 2 blank lines between class definitions
  - 1 blank line between method definitions in classes
  - 1 blank line between logical code blocks in functions
- **No trailing whitespace**

### Type Annotations

- Use **Python 3.10+ union syntax** (`int | None` instead of `Optional[int]`)
- Use `dict[str, ...]` for typed dicts
- Use `list[...]` for typed lists
- Annotate function parameters and return values
- Use `TYPE_CHECKING` guard for imports that cause circular dependencies

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `CameraManager`, `TeeFanoutPipelineBuilder` |
| Functions/methods | snake_case | `add_camera()`, `get_camera_id()` |
| Variables | snake_case | `camera_id`, `source_id` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_CONFIG`, `TEST_VIDEO` |
| Private methods | _snake_case | `_create_element()`, `_configure_tracker()` |
| Module-level private | _prefix | `_coerce_property_value()` |
| Type variables | PascalCase | `T`, `K`, `V` |

### Data Classes

Use `@dataclass` for simple data structures:

```python
@dataclass
class CameraInfo:
    camera_id: str
    source_id: int
    url: str
    name: str = ""
    added_at: float = 0.0

    def __post_init__(self):
        if self.added_at == 0.0:
            self.added_at = time.time()
```

### Error Handling

- **Use specific exceptions**: `ValueError`, `RuntimeError`, `KeyError`, `TypeError`
- **Avoid bare `except:` clauses** - catch specific exceptions
- **Return (True, result) / (False, error)** patterns for operations that may fail:
  ```python
  def _post_json(self, url: str, payload: dict) -> tuple[bool, str]:
      try:
          with urllib.request.urlopen(req, timeout=self.timeout) as resp:
              return True, resp.read().decode('utf-8')
      except urllib.error.HTTPError as e:
          return False, f"HTTP {e.code}: {e.reason}"
      except urllib.error.URLError as e:
          return False, str(e.reason)
      except Exception as e:
          return False, str(e)
  ```
- **Logging**: Use `logger = logging.getLogger(__name__)` for library code
- **Print statements** acceptable for CLI tools and scripts (e.g., `[PipelineBuilder] Message`)

### Documentation

- **Docstrings** for all public classes and methods
- **Google-style docstrings**:
  ```python
  def add(self, camera: Camera) -> bool:
      """
      Add camera to pipeline

      Args:
          camera: Camera configuration

      Returns:
          True if successful, False otherwise
      """
  ```
- **Args/Returns** sections for complex functions
- **Raises** section for exceptions that may be raised

### Project Structure

```
/home/mq/disk2T/quangnv/face/
├── api/              # REST API endpoints (aiohttp)
├── apps/face/        # Face recognition application logic
│   ├── database.py   # Feature storage and L2 matching
│   ├── tracker.py    # Multi-track state machine
│   ├── events.py     # Recognition events
│   ├── display.py    # OSD rendering
│   └── probes.py     # GStreamer pad probes
├── core/             # Pipeline framework
│   ├── config.py     # YAML config loader
│   ├── tee_fanout_builder.py    # Multi-branch builder
│   ├── multibranch_camera_manager.py
│   ├── camera_bin.py
│   ├── probe_registry.py
│   └── source_mapper.py  # Thread-safe ID mapping
├── sinks/            # Output adapters
│   ├── base_sink.py  # Abstract interface
│   ├── fakesink_adapter.py
│   ├── filesink_adapter.py
│   └── webrtc/       # WebRTC streaming
├── bin/              # Entry points and test scripts
├── configs/          # YAML pipeline configs
└── tests/            # Test files
```

### GStreamer Patterns

- **Always call `Gst.init(None)`** before creating elements
- **Use `gi.require_version("Gst", "1.0")`** before imports
- **Pipeline states**: NULL → READY → PAUSED → PLAYING
- **Handle EOS/ERROR messages** via bus callbacks
- **Use `GLib.MainLoop()`** or `asyncio` for event loops

### Thread Safety

- Use `threading.Lock()` for shared mutable state
- Provide `_unsafe` variants for hot paths where lock overhead matters:
  ```python
  def get_camera_id(self, source_id: int) -> Optional[str]:
      with self._lock:
          return self._by_source_id.get(source_id)

  def get_camera_id_unsafe(self, source_id: int) -> Optional[str]:
      # No lock - use only in probe hot path
      return self._by_source_id.get(source_id)
  ```

### Configuration

- **YAML configs** with `${VAR:default}` environment variable expansion
- **Config-driven pipeline** construction - avoid hardcoding element types
- **Default parameters** in function signatures for optional arguments

### REST API Patterns

- Use **aiohttp** for async HTTP servers
- Routes return JSON responses
- Health check endpoint at `/api/health`
- Camera CRUD at `/api/cameras`
- Branch management at `/api/branches`

### Testing Patterns

- **Test files in `tests/`** with `_test_*.py` naming
- **Use `sys.path.insert(0, ...)`** to add project root to path
- **Test helper functions** at module level (`get_test_config()`, `create_test_pipeline()`)
- **Cleanup**: Set pipeline to `Gst.State.NULL` after tests
- **Assert-based validation** with descriptive messages
- **Print test progress** with `=== Test: Name ===` headers

### Code to Avoid

- **Malicious code**: Any code designed to harm systems or exfiltrate data
- **Secrets in code**: API keys, passwords, certificates
- **Blocking I/O** in async contexts
- **Magic numbers** - use named constants
- **Deep imports** from private modules
- **Modifying globals** from library code
