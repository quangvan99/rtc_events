"""YAML config loader with environment variable expansion"""

import os
import re
import yaml


def load_config(path: str) -> dict:
    """Load YAML with ${VAR:default} expansion"""
    with open(path) as f:
        text = f.read()

    # Expand ${VAR:default} before parsing
    def replace(m):
        return os.environ.get(m.group(1), m.group(2) or "")

    expanded = re.sub(r'\$\{(\w+):?([^}]*)\}', replace, text)
    return yaml.safe_load(expanded)