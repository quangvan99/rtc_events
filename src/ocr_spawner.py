"""
OCR Worker Spawner

This module contains ONLY the spawn wrapper function.
Located outside of apps.plate to avoid ProcessorRegistry imports.
"""


def spawn_ocr_worker(input_queue, output_queue, engine_path, dict_path, ocr_threshold):
    """Wrapper function that loads ocr_worker.py without triggering apps.plate imports.

    This runs inside the spawned subprocess. It loads ocr_worker.py directly
    using importlib to avoid importing apps.plate.__init__.py which would
    trigger ProcessorRegistry and cause CUDA context conflicts.
    """
    import importlib.util

    # Load ocr_worker.py directly to avoid apps.plate.__init__.py
    worker_path = "/app/apps/plate/ocr_worker.py"
    spec = importlib.util.spec_from_file_location("ocr_worker_standalone", worker_path)
    worker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_module)

    # Call the actual worker function
    worker_module.ocr_worker_main(input_queue, output_queue, engine_path, dict_path, ocr_threshold)
