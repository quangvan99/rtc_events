"""
Standalone OCR Worker Process

This file contains ONLY the worker function to avoid import side effects.
The spawn subprocess imports this file directly without triggering
ProcessorRegistry or other CUDA-related imports.
"""

import gc
import math
import re
import signal
import traceback
from typing import List, Optional

import cv2
import numpy as np


# =============================================================================
# Constants (duplicated to avoid imports)
# =============================================================================

INPUT_HEIGHT = 48
INPUT_WIDTH = 640
INPUT_CHANNELS = 3
BATCH_SIZE = 2
OUTPUT_SHAPE = (2, 80, 42)


# =============================================================================
# CTC Label Decode (duplicated to avoid imports)
# =============================================================================

class CTCLabelDecode:
    """CTC Label Decode for PaddleOCR recognition models."""

    def __init__(self, character_dict_path: str = None, use_space_char: bool = False):
        self.character_str = []

        if character_dict_path is None:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
            dict_character = self.character_str
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        # Add CTC blank token at index 0
        dict_character = ["blank"] + dict_character
        self.character = dict_character

    def __call__(self, preds):
        """Decode CTC predictions to text."""
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        result_list = []
        for batch_idx in range(len(preds_idx)):
            selection = np.ones(len(preds_idx[batch_idx]), dtype=bool)
            selection[1:] = preds_idx[batch_idx][1:] != preds_idx[batch_idx][:-1]
            selection &= preds_idx[batch_idx] != 0  # Remove blank

            char_list = [self.character[idx] for idx in preds_idx[batch_idx][selection]]
            conf_list = preds_prob[batch_idx][selection]

            text = "".join(char_list)
            conf = float(np.mean(conf_list)) if len(conf_list) > 0 else 0.0
            result_list.append((text, conf))

        return result_list


# =============================================================================
# Helper Functions (duplicated to avoid imports)
# =============================================================================

def check_plate_square(plate_img: np.ndarray):
    """Detect 2-line plates based on aspect ratio."""
    height, width = plate_img.shape[:2]
    if width == 0:
        return None, None
    if width / height > 1:
        mid = height // 2
        top = plate_img[:mid, :]
        bottom = plate_img[mid:, :]
        bottom = cv2.resize(bottom, (top.shape[1], top.shape[0]))
        return cv2.hconcat([top, bottom]), [top, bottom]
    return None, None


def check_format_plate(plate_text: str) -> bool:
    """Check if license plate matches standard Vietnamese formats."""
    patterns = [
        r"^[0-9]{2}[A-Z]{1}[0-9]{4}$",
        r"^[0-9]{2}[A-Z]{1}[0-9]{5}$",
        r"^[0-9]{2}[A-Z]{1}[0-9]{6}$",
    ]
    return any(re.match(p, plate_text) for p in patterns)


def check_format_plate_append(plate_text: str) -> Optional[str]:
    """Check and append markers for non-standard formats."""
    patterns = [
        (r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$", "##"),
        (r"^[0-9]{4,5}[A-Z]{2}[0-9]{2,3}$", "#"),
        (r"^[A-Z]{0,1}[0-9]{6,7}$", ""),
    ]
    for pattern, suffix in patterns:
        if re.match(pattern, plate_text):
            return plate_text + suffix
    return None


# =============================================================================
# Worker Function
# =============================================================================

def ocr_worker_main(input_queue, output_queue,
                    engine_path: str, dict_path: str, ocr_threshold: float):
    """Worker process main loop. Runs TensorRT OCR engine.

    This function is designed to run in a completely isolated subprocess
    with no imports from apps.plate module.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        print(f"[OCRWorker] Importing pycuda...", flush=True)
        import pycuda.driver as cuda

        print(f"[OCRWorker] Importing tensorrt...", flush=True)
        import tensorrt as trt

        print(f"[OCRWorker] Initializing CUDA...", flush=True)
        cuda.init()
        ctx = cuda.Device(0).make_context()

        print(f"[OCRWorker] Loading TRT engine: {engine_path}", flush=True)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, "")

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine")

        exec_context = engine.create_execution_context()

        # Setup I/O buffers
        print(f"[OCRWorker] Setting up buffers...", flush=True)
        inputs = []
        outputs = []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            buffer = {"name": name, "host": host_mem, "device": device_mem, "shape": shape}

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(buffer)
            else:
                outputs.append(buffer)

        ctx.pop()

        # Create decoder
        decoder = CTCLabelDecode(character_dict_path=dict_path, use_space_char=True)

        print(f"[OCRWorker] Ready", flush=True)

        # Main loop
        while True:
            item = input_queue.get()
            if item is None:
                break

            object_id, plate_img = item
            print(f"[OCRWorker] Processing id={object_id}, img_shape={plate_img.shape}", flush=True)

            try:
                result = process_plate(
                    plate_img, ctx, exec_context, inputs, outputs, stream,
                    decoder, ocr_threshold
                )
                output_queue.put((object_id, result))
                print(f"[OCRWorker] Done id={object_id}, result={result}", flush=True)
            except Exception as e:
                print(f"[OCRWorker] Error: {e}", flush=True)
                output_queue.put((object_id, None))

        # Cleanup
        print(f"[OCRWorker] Shutting down", flush=True)
        ctx.push()
        try:
            del exec_context
            del engine
            del runtime
            for buf in inputs + outputs:
                buf["device"].free()
            gc.collect()
        finally:
            ctx.pop()
            ctx.detach()

    except Exception as e:
        print(f"[OCRWorker] Fatal error: {e}", flush=True)
        traceback.print_exc()


def process_plate(plate_img, ctx, exec_context, inputs, outputs, stream,
                  decoder, ocr_threshold) -> Optional[str]:
    """Process single plate image."""
    import pycuda.driver as cuda

    _, img_list = check_plate_square(plate_img)
    images = img_list if img_list else [plate_img]
    results = []
    for img in images:
        # Preprocess
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = INPUT_WIDTH if math.ceil(INPUT_HEIGHT * ratio) > INPUT_WIDTH else int(math.ceil(INPUT_HEIGHT * ratio))

        resized = cv2.resize(img, (resized_w, INPUT_HEIGHT))
        resized = resized.astype(np.float32).transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5

        single = np.zeros((INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
        single[:, :, :resized_w] = resized
        batch = np.stack([single, single], axis=0)

        # Inference
        ctx.push()
        try:
            np.copyto(inputs[0]["host"], batch.ravel())
            cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)

            for buf in inputs + outputs:
                exec_context.set_tensor_address(buf["name"], int(buf["device"]))

            exec_context.execute_async_v3(stream_handle=stream.handle)

            for out in outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

            stream.synchronize()
            result = outputs[0]["host"].copy().reshape(OUTPUT_SHAPE)[0:1]
        finally:
            ctx.pop()

        # Decode
        text_results = decoder(result)
        print(f"[OCRWorker] Decoded results: {text_results}", flush=True)
        if text_results and len(text_results) > 0:
            text, conf = text_results[0]
            if conf >= ocr_threshold:
                results.append(text)

    if not results:
        return None

    ocr_result = "".join(results)
    if not check_format_plate(ocr_result):
        fixed = check_format_plate_append(ocr_result)
        ocr_result = fixed if fixed else f"INVALID_{ocr_result}"

    return ocr_result
