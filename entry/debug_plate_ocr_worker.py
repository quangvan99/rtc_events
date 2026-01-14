#!/usr/bin/env python3
"""
Debug script for OCR Worker Process

Tests the multiprocessing OCR worker in isolation.
Run inside container: python3 entry/debug_plate_ocr_worker.py
"""

import sys
import time
import numpy as np
import cv2

# Test imports first
print("=" * 60)
print("Testing OCR Worker Process")
print("=" * 60)

try:
    from apps.plate.ocr import OCRWorkerProcess, check_plate_square, check_format_plate
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_helper_functions():
    """Test helper functions without TensorRT."""
    print("\n--- Testing Helper Functions ---")

    # Test check_plate_square with 2-line plate (square)
    square_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    horizontal, img_list = check_plate_square(square_img)
    assert img_list is not None, "2-line plate should be detected"
    assert len(img_list) == 2, "Should have 2 images"
    print("✓ check_plate_square: 2-line plate detected correctly")

    # Test check_plate_square with 1-line plate (wide)
    wide_img = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
    horizontal, img_list = check_plate_square(wide_img)
    assert img_list is None, "1-line plate should not be split"
    print("✓ check_plate_square: 1-line plate handled correctly")

    # Test format validation
    assert check_format_plate("29A12345") == True, "Valid plate format"
    assert check_format_plate("INVALID") == False, "Invalid plate format"
    print("✓ check_format_plate: Validation works")

    print("✓ All helper function tests passed!")


def test_ocr_worker():
    """Test OCR worker process lifecycle."""
    print("\n--- Testing OCR Worker Process ---")

    engine_path = "data/license_plate/ocr/ppocr_dummy.engine"
    dict_path = "data/license_plate/ocr/licence_plate_dict.txt"

    # Check files exist
    import os
    if not os.path.exists(engine_path):
        print(f"⚠ Engine not found: {engine_path}")
        print("Skipping worker test (no TensorRT engine)")
        return False

    if not os.path.exists(dict_path):
        print(f"⚠ Dict not found: {dict_path}")
        print("Skipping worker test (no character dict)")
        return False

    print(f"Engine: {engine_path}")
    print(f"Dict: {dict_path}")

    # Initialize worker
    print("\n[Test] Starting OCR worker...")
    worker = OCRWorkerProcess(
        engine_path=engine_path,
        dict_path=dict_path,
        ocr_threshold=0.9,
        maxsize=10
    )

    # Wait for worker to initialize
    time.sleep(2)

    if not worker.is_alive():
        print("✗ Worker failed to start")
        return False
    print("✓ Worker started")

    # Create test plate image (dummy)
    test_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    object_id = 12345

    # Submit test
    print("\n[Test] Submitting test image...")
    success = worker.submit(object_id, test_img)
    if success:
        print(f"✓ Submit successful for object_id={object_id}")
    else:
        print("✗ Submit failed (queue full?)")
        worker.shutdown()
        return False

    # Poll results with timeout
    print("\n[Test] Polling results...")
    timeout = 10.0
    start = time.time()
    results = []

    while time.time() - start < timeout:
        results = worker.poll_results()
        if results:
            break
        time.sleep(0.2)

    if results:
        for oid, text in results:
            print(f"✓ OCR result: object_id={oid}, text={text}")
    else:
        print(f"⚠ No results after {timeout}s (this may be expected for random image)")

    # Shutdown
    print("\n[Test] Shutting down worker...")
    worker.shutdown()

    if not worker.is_alive():
        print("✓ Worker shutdown complete")
    else:
        print("✗ Worker still alive after shutdown")
        return False

    return True


def main():
    print("\n" + "=" * 60)

    # Test helper functions (no TensorRT needed)
    test_helper_functions()

    # Test OCR worker (needs TensorRT engine)
    print("\n" + "=" * 60)
    worker_ok = test_ocr_worker()

    print("\n" + "=" * 60)
    if worker_ok:
        print("✓ All tests passed!")
    else:
        print("⚠ Worker test skipped or failed (check TensorRT engine)")

    print("=" * 60)


if __name__ == "__main__":
    main()
