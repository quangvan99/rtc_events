"""
Processor Registry - Auto-discovery pattern for BranchProcessors

This module provides a global registry where processor classes can self-register
using a decorator. The TeeFanoutPipelineBuilder can then automatically discover
and instantiate processors based on the configuration.

Usage:
    # In processor module (e.g., apps/face/processor.py):
    from src.registry import ProcessorRegistry
    
    @ProcessorRegistry.register("recognition")
    class FaceRecognitionProcessor(BranchProcessor):
        ...
    
    # In entry point or builder:
    from src.registry import ProcessorRegistry
    
    # Just importing the processor module triggers registration
    import apps.face.processor
    import apps.detection.processor
    
    # Or use auto-import
    ProcessorRegistry.auto_import()
    
    # Create all processors for configured branches
    processors = ProcessorRegistry.create_for_config(config)
"""

from typing import Dict, Type, List, Optional, Any
import importlib
import os

from src.interfaces.branch_processor import BranchProcessor


class ProcessorRegistry:
    """
    Global registry for BranchProcessor classes.
    
    Allows processors to self-register using a decorator, eliminating the need
    for the entry point to explicitly import and instantiate them.
    
    Features:
    - Decorator-based registration
    - Auto-import of processor modules from apps/
    - Factory method to create processors based on config
    """
    
    # Class-level registry: branch_name -> processor_class
    _registry: Dict[str, Type[BranchProcessor]] = {}
    
    # Track which modules have been imported
    _imported_modules: List[str] = []
    
    @classmethod
    def register(cls, branch_name: str):
        """
        Decorator to register a processor class for a branch name.
        
        Args:
            branch_name: The branch name this processor handles
                        (must match config branch names like 'recognition', 'detection')
        
        Returns:
            Decorator function
            
        Example:
            @ProcessorRegistry.register("recognition")
            class FaceRecognitionProcessor(BranchProcessor):
                ...
        """
        def decorator(processor_cls: Type[BranchProcessor]):
            if not issubclass(processor_cls, BranchProcessor):
                raise TypeError(
                    f"Registered class must be a BranchProcessor subclass: {processor_cls}"
                )
            
            cls._registry[branch_name] = processor_cls
            print(f"[ProcessorRegistry] Registered: {branch_name} -> {processor_cls.__name__}")
            return processor_cls
        
        return decorator
    
    @classmethod
    def get(cls, branch_name: str) -> Optional[Type[BranchProcessor]]:
        """
        Get processor class by branch name.
        
        Args:
            branch_name: Branch name to look up
            
        Returns:
            Processor class if registered, None otherwise
        """
        return cls._registry.get(branch_name)
    
    @classmethod
    def create(cls, branch_name: str, **kwargs) -> Optional[BranchProcessor]:
        """
        Create a processor instance by branch name.
        
        Args:
            branch_name: Branch name to create processor for
            **kwargs: Arguments to pass to processor constructor
            
        Returns:
            Processor instance if registered, None otherwise
        """
        processor_cls = cls._registry.get(branch_name)
        if processor_cls:
            return processor_cls(**kwargs)
        return None
    
    @classmethod
    def create_all(cls, config: dict) -> List[BranchProcessor]:
        """
        Create processor instances for all configured branches.
        
        Args:
            config: Pipeline configuration dict with 'pipeline.branches'
            
        Returns:
            List of processor instances for configured branches
        """
        processors = []
        branches = config.get("pipeline", {}).get("branches", {})
        
        for branch_name in branches:
            processor_cls = cls._registry.get(branch_name)
            if processor_cls:
                processor = processor_cls()
                processors.append(processor)
                print(f"[ProcessorRegistry] Created: {processor_cls.__name__} for '{branch_name}'")
            else:
                print(f"[ProcessorRegistry] Warning: No processor registered for branch '{branch_name}'")
        
        return processors
    
    @classmethod
    def create_for_config(cls, config: dict) -> List[BranchProcessor]:
        """Alias for create_all() for clarity."""
        return cls.create_all(config)
    
    @classmethod
    def list_registered(cls) -> Dict[str, str]:
        """
        List all registered processors.
        
        Returns:
            Dict mapping branch_name -> processor class name
        """
        return {name: proc_cls.__name__ for name, proc_cls in cls._registry.items()}
    
    @classmethod
    def is_registered(cls, branch_name: str) -> bool:
        """Check if a processor is registered for a branch name."""
        return branch_name in cls._registry
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Mainly for testing."""
        cls._registry.clear()
        cls._imported_modules.clear()
    
    @classmethod
    def auto_import(cls, apps_dir: str = "apps") -> None:
        """
        Auto-import all processor modules from apps directory.
        
        This triggers the @register decorators in each processor module,
        populating the registry automatically.
        
        Args:
            apps_dir: Path to apps directory (default: "apps")
            
        Example directory structure:
            apps/
                face/
                    processor.py  # Contains @ProcessorRegistry.register("recognition")
                detection/
                    processor.py  # Contains @ProcessorRegistry.register("detection")
        """
        if not os.path.isdir(apps_dir):
            print(f"[ProcessorRegistry] Warning: Apps directory not found: {apps_dir}")
            return
        
        # Find all processor.py files in apps subdirectories
        for app_name in os.listdir(apps_dir):
            app_path = os.path.join(apps_dir, app_name)
            
            if not os.path.isdir(app_path):
                continue
            
            processor_file = os.path.join(app_path, "processor.py")
            
            if os.path.isfile(processor_file):
                module_name = f"apps.{app_name}.processor"
                
                if module_name in cls._imported_modules:
                    continue
                
                try:
                    importlib.import_module(module_name)
                    cls._imported_modules.append(module_name)
                    print(f"[ProcessorRegistry] Auto-imported: {module_name}")
                except ImportError as e:
                    print(f"[ProcessorRegistry] Warning: Failed to import {module_name}: {e}")
                except Exception as e:
                    print(f"[ProcessorRegistry] Error importing {module_name}: {e}")
    
    @classmethod
    def import_processor(cls, module_path: str) -> bool:
        """
        Import a specific processor module.
        
        Args:
            module_path: Full module path (e.g., "apps.face.processor")
            
        Returns:
            True if import succeeded, False otherwise
        """
        if module_path in cls._imported_modules:
            return True
        
        try:
            importlib.import_module(module_path)
            cls._imported_modules.append(module_path)
            print(f"[ProcessorRegistry] Imported: {module_path}")
            return True
        except ImportError as e:
            print(f"[ProcessorRegistry] Failed to import {module_path}: {e}")
            return False