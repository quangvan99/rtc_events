"""Processor Registry - Auto-discovery for BranchProcessors."""

from abc import ABC, abstractmethod
from typing import Any, Callable
import importlib
import os

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink


class BranchProcessor(ABC):
    """Abstract base class for branch processors."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def setup(self, config: dict, sink: BaseSink) -> None:
        pass

    @abstractmethod
    def get_probes(self) -> dict[str, Callable]:
        pass

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        pass

    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass


class ProcessorRegistry:
    """Global registry for BranchProcessor classes."""

    _registry: dict[str, type] = {}
    _imported: list[str] = []

    @classmethod
    def register(cls, branch_name: str):
        """Decorator to register a processor class."""
        def decorator(proc_cls: type):
            cls._registry[branch_name] = proc_cls
            print(f"[ProcessorRegistry] Registered: {branch_name} -> {proc_cls.__name__}")
            return proc_cls
        return decorator

    @classmethod
    def create_for_config(cls, config: dict) -> list:
        """Create processors for configured branches."""
        processors = []
        for name in config.get("pipeline", {}).get("branches", {}):
            if name in cls._registry:
                proc = cls._registry[name]()
                processors.append(proc)
                print(f"[ProcessorRegistry] Created: {proc.__class__.__name__} for '{name}'")
        return processors

    @classmethod
    def auto_import(cls, apps_dir: str = "apps") -> None:
        """Auto-import processor modules from apps directory."""
        if not os.path.isdir(apps_dir):
            return

        for app in os.listdir(apps_dir):
            path = os.path.join(apps_dir, app, "processor.py")
            if os.path.isfile(path):
                module = f"apps.{app}.processor"
                if module not in cls._imported:
                    try:
                        importlib.import_module(module)
                        cls._imported.append(module)
                        print(f"[ProcessorRegistry] Auto-imported: {module}")
                    except Exception as e:
                        print(f"[ProcessorRegistry] Failed to import {module}: {e}")

    @classmethod
    def get(cls, name: str) -> type | None:
        return cls._registry.get(name)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._imported.clear()
