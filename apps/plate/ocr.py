"""
OCR Module for License Plate Recognition

Multiprocessing-based OCR using TensorRT engine in separate process.
Uses spawn + Queue pattern to avoid CUDA context conflicts with GStreamer.

Usage:
    from apps.plate.ocr import OCRWorkerProcess

    worker = OCRWorkerProcess(engine_path, dict_path)
    worker.submit(object_id, plate_img)
    results = worker.poll_results()  # [(object_id, text), ...]
    worker.shutdown()
"""

import gc
import math
import queue
import re
import signal
import traceback
import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Note: TensorRT and PyCUDA imports happen inside worker process
# to ensure clean CUDA context initialization


# =============================================================================
# Constants
# =============================================================================

# Engine input dimensions (batch=2, channels=3, height=48, width=640)
INPUT_HEIGHT = 48
INPUT_WIDTH = 320
INPUT_CHANNELS = 3
BATCH_SIZE = 2

# Engine output shape (batch=2, time_steps=80, num_classes=42)
OUTPUT_SHAPE = (2, 80, 42)


# =============================================================================
# CTC Label Decode Classes
# =============================================================================

class BaseRecLabelDecode:
    """Convert between text-label and text-index."""

    def __init__(self, character_dict_path: str = None, use_space_char: bool = False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
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

        dict_character = self.add_special_char(dict_character)
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def get_ignored_tokens(self):
        return [0]  # CTC blank token

    def decode(self, text_index, text_prob=None, is_remove_duplicate: bool = False):
        """Convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)

            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]

            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)
            if self.reverse:
                text = self._pred_reverse(text)

            result_list.append((text, float(np.mean(conf_list))))

        return result_list

    def _pred_reverse(self, pred):
        """Reverse prediction for RTL languages."""
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)
        return "".join(pred_re[::-1])


class CTCLabelDecode(BaseRecLabelDecode):
    """CTC Label Decode for PaddleOCR recognition models."""

    def __init__(self, character_dict_path: str = None, use_space_char: bool = False, **kwargs):
        super().__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, **kwargs):
        """Decode CTC predictions to text."""
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]

        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)

        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        """Add CTC blank token at index 0."""
        return ["blank"] + dict_character


# =============================================================================
# TensorRT OCR Engine
# =============================================================================

class TRTOCREngine:
    """TensorRT inference engine for PaddleOCR text recognition.

    Designed to run in separate process with isolated CUDA context.
    No threading lock needed since each process has its own engine.
    """

    def __init__(self, engine_path: str):
        import pycuda.driver as cuda
        import tensorrt as trt

        self.engine_path = engine_path

        # Initialize CUDA
        cuda.init()
        self.ctx = cuda.Device(0).make_context()

        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, "")

        # Load engine
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Setup I/O buffers
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            buffer = {"name": name, "host": host_mem, "device": device_mem, "shape": shape}

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(buffer)
            else:
                self.outputs.append(buffer)

        self.ctx.pop()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR inference.

        Returns batch of 2 images (engine requires batch_size=2).
        Second image is duplicate of first.
        """
        h, w = image.shape[:2]
        ratio = w / float(h)

        if math.ceil(INPUT_HEIGHT * ratio) > INPUT_WIDTH:
            resized_w = INPUT_WIDTH
        else:
            resized_w = int(math.ceil(INPUT_HEIGHT * ratio))

        resized = cv2.resize(image, (resized_w, INPUT_HEIGHT))
        resized = resized.astype(np.float32)
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5

        # Create single image with padding
        single = np.zeros(
            (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
            dtype=np.float32
        )
        single[:, :, :resized_w] = resized

        # Create batch of 2 (engine requires batch_size=2)
        batch = np.stack([single, single], axis=0)  # Shape: (2, 3, 48, 640)
        return batch

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed image.

        Returns output for first batch element only: shape (1, 80, 42)
        """
        import pycuda.driver as cuda

        # Auto-preprocess if needed (HWC format)
        if image.ndim == 3 and image.shape[2] == 3:
            image = self.preprocess(image)

        self.ctx.push()
        try:
            np.copyto(self.inputs[0]["host"], image.ravel())
            cuda.memcpy_htod_async(
                self.inputs[0]["device"],
                self.inputs[0]["host"],
                self.stream,
            )

            for buf in self.inputs + self.outputs:
                self.context.set_tensor_address(buf["name"], int(buf["device"]))

            self.context.execute_async_v3(stream_handle=self.stream.handle)

            for out in self.outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

            self.stream.synchronize()

            result = self.outputs[0]["host"].copy()
            full_output = result.reshape(OUTPUT_SHAPE)  # (2, 80, 42)
            # Return only first batch element for single-image inference
            return full_output[0:1]  # (1, 80, 42)
        finally:
            self.ctx.pop()

    def destroy(self):
        """Clean up CUDA resources."""
        if hasattr(self, '_destroyed'):
            return
        self._destroyed = True

        try:
            self.ctx.push()
            try:
                if hasattr(self, 'context'):
                    del self.context
                if hasattr(self, 'engine'):
                    del self.engine
                if hasattr(self, 'runtime'):
                    del self.runtime
                if hasattr(self, 'stream'):
                    del self.stream
                for buf in self.inputs + self.outputs:
                    if 'device' in buf:
                        buf["device"].free()
                gc.collect()
            finally:
                self.ctx.pop()
                self.ctx.detach()
        except Exception:
            pass


# =============================================================================
# Post-Processing Functions
# =============================================================================

def check_plate_square(plate_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """Detect 2-line plates based on aspect ratio.

    Returns:
        (horizontal_concat, [top_plate, bottom_plate]) if 2-line plate
        (None, None) if single-line plate
    """
    height, width = plate_img.shape[:2]
    if width == 0:
        return None, None

    scale = width / height

    if scale < 2:  # 2-line plate (square-ish)
        mid_height = height // 2
        top_plate = plate_img[:mid_height, :]
        bottom_plate = plate_img[mid_height:, :]
        bottom_plate = cv2.resize(bottom_plate, (top_plate.shape[1], top_plate.shape[0]))
        horizontal = cv2.hconcat([top_plate, bottom_plate])
        return horizontal, [top_plate, bottom_plate]
    return None, None


def check_format_plate(plate_text: str) -> bool:
    """Check if license plate matches standard Vietnamese formats."""
    patterns = [
        r"^[0-9]{2}[A-Z]{1}[0-9]{4}$",
        r"^[0-9]{2}[A-Z]{1}[0-9]{5}$",
        r"^[0-9]{2}[A-Z]{1}[0-9]{6}$",
        r"^[0-9]{2}[A-Z]{1}[0-9]{3}\.[0-9]{2}$",
    ]
    return any(re.match(p, plate_text) for p in patterns)


def check_format_plate_append(plate_text: str) -> Optional[str]:
    """Check and append markers for non-standard formats."""
    patterns = [
        (r"^[0-9]{2}[ABCDEFGHIJKLMNOPQRSTUVWXYZĐ]{1,2}([0-9]{4,5}|[0-9]{6})$", "##"),
        (r"^[0-9]{4,5}[A-Z]{2}[0-9]{2,3}$", "#"),
        (r"^[A-Z]{0,1}[0-9]{6,7}$", ""),
    ]

    for pattern, suffix in patterns:
        if re.match(pattern, plate_text):
            return plate_text + suffix
    return None


# =============================================================================
# Multiprocessing OCR Worker
# =============================================================================

class OCRWorkerProcess:
    """Manages OCR inference in separate process.

    Uses spawn method for CUDA compatibility. Provides non-blocking
    interface for submitting plate images and polling OCR results.

    Args:
        engine_path: Path to TensorRT engine file
        dict_path: Path to character dictionary file
        ocr_threshold: Minimum confidence threshold for OCR results
        maxsize: Maximum queue size (drops frames if full)
    """

    def __init__(self, engine_path: str, dict_path: str,
                 ocr_threshold: float = 0.9, maxsize: int = 10):
        self.engine_path = engine_path
        self.dict_path = dict_path
        self.ocr_threshold = ocr_threshold

        # Use spawn context for CUDA compatibility
        ctx = mp.get_context('spawn')
        self.input_queue = ctx.Queue(maxsize=maxsize)
        self.output_queue = ctx.Queue()

        # Import spawner from src/ to avoid apps.plate imports in subprocess
        from src.ocr_spawner import spawn_ocr_worker

        self.worker = ctx.Process(
            target=spawn_ocr_worker,
            args=(self.input_queue, self.output_queue,
                  engine_path, dict_path, ocr_threshold),
            daemon=False  # Explicit shutdown needed
        )
        self.worker.start()
        print(f"[OCRWorker] Started process pid={self.worker.pid}")

    def submit(self, object_id: int, plate_img: np.ndarray) -> bool:
        """Submit plate image for OCR. Returns False if queue full."""
        try:
            self.input_queue.put_nowait((object_id, plate_img))
            return True
        except queue.Full:
            return False

    def poll_results(self) -> List[Tuple[int, Optional[str]]]:
        """Drain all available results. Returns [(object_id, text), ...]."""
        results = []
        try:
            while True:
                results.append(self.output_queue.get_nowait())
        except queue.Empty:
            pass
        return results

    def is_alive(self) -> bool:
        """Check if worker process is alive."""
        return self.worker.is_alive()

    def shutdown(self, timeout: float = 5.0):
        """Graceful shutdown with timeout."""
        try:
            self.input_queue.put(None, timeout=1.0)  # Poison pill
        except queue.Full:
            pass

        self.worker.join(timeout=timeout)

        if self.worker.is_alive():
            print(f"[OCRWorker] Timeout, terminating pid={self.worker.pid}")
            self.worker.terminate()
            self.worker.join(timeout=1.0)

        print(f"[OCRWorker] Shutdown complete")


def _ocr_worker_loop(input_queue: Queue, output_queue: Queue,
                     engine_path: str, dict_path: str, ocr_threshold: float):
    """Worker process main loop. Runs TensorRT OCR engine.

    This function runs in a separate process with isolated CUDA context.
    Receives plate images from input_queue, runs OCR, and puts results
    in output_queue.

    IMPORTANT: All heavy imports (pycuda, tensorrt) happen inside this function
    to avoid CUDA context conflicts with the parent process.
    """
    import sys
    # Ignore SIGINT (parent handles shutdown)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        # =====================================================================
        # Import TensorRT/PyCUDA inside worker (isolated CUDA context)
        # =====================================================================
        print(f"[OCRWorker] Step 1: Import pycuda...", flush=True)
        import pycuda.driver as cuda

        print(f"[OCRWorker] Step 2: Import tensorrt...", flush=True)
        import tensorrt as trt

        print(f"[OCRWorker] Initializing TRT engine: {engine_path}", flush=True)

        # Initialize CUDA
        print(f"[OCRWorker] Step 3: cuda.init()...", flush=True)
        cuda.init()

        print(f"[OCRWorker] Step 4: make_context()...", flush=True)
        ctx = cuda.Device(0).make_context()

        # Initialize TensorRT
        print(f"[OCRWorker] Step 5: TRT runtime...", flush=True)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, "")

        # Load engine
        print(f"[OCRWorker] Step 6: Read engine file...", flush=True)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        print(f"[OCRWorker] Engine file size: {len(engine_bytes)} bytes", flush=True)

        print(f"[OCRWorker] Step 7: Deserialize engine...", flush=True)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
        print(f"[OCRWorker] Engine: {engine}", flush=True)

        print(f"[OCRWorker] Step 8: Create exec context...", flush=True)
        exec_context = engine.create_execution_context()
        print(f"[OCRWorker] Exec context: {exec_context}", flush=True)

        # Setup I/O buffers (keep context active)
        print(f"[OCRWorker] Step 9: Setup buffers...", flush=True)
        inputs = []
        outputs = []
        stream = cuda.Stream()
        print(f"[OCRWorker] Stream created", flush=True)

        num_tensors = engine.num_io_tensors
        print(f"[OCRWorker] num_io_tensors = {num_tensors}", flush=True)

        for i in range(num_tensors):
            try:
                print(f"[OCRWorker] Getting tensor {i} info...", flush=True)
                name = engine.get_tensor_name(i)
                print(f"[OCRWorker] Tensor {i} name: {name}", flush=True)

                print(f"[OCRWorker] Getting shape...", flush=True)
                shape = engine.get_tensor_shape(name)
                print(f"[OCRWorker] Shape: {shape}", flush=True)

                print(f"[OCRWorker] Getting dtype...", flush=True)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                print(f"[OCRWorker] Dtype: {dtype}", flush=True)

                print(f"[OCRWorker] Computing volume...", flush=True)
                size = int(np.prod(shape))  # Use numpy instead of trt.volume
                print(f"[OCRWorker] Tensor {i}: {name} shape={shape} size={size}", flush=True)

                print(f"[OCRWorker] Allocating host mem...", flush=True)
                host_mem = cuda.pagelocked_empty(size, dtype)
                print(f"[OCRWorker] Host mem OK, allocating device mem...", flush=True)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                print(f"[OCRWorker] Device mem OK", flush=True)

                buffer = {"name": name, "host": host_mem, "device": device_mem, "shape": shape}

                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    inputs.append(buffer)
                else:
                    outputs.append(buffer)
                print(f"[OCRWorker] Buffer {i}: {name} {shape}", flush=True)
            except Exception as e:
                print(f"[OCRWorker] Error in tensor {i} setup: {e}", flush=True)
                import traceback as tb
                tb.print_exc()
                raise

        print(f"[OCRWorker] Step 10: Pop context...", flush=True)
        ctx.pop()

        # Create decoder
        print(f"[OCRWorker] Step 11: Create decoder...", flush=True)
        decoder = CTCLabelDecode(
            character_dict_path=dict_path,
            use_space_char=True
        )

        print(f"[OCRWorker] Ready", flush=True)

        # =====================================================================
        # Main processing loop
        # =====================================================================
        while True:
            item = input_queue.get()

            if item is None:  # Poison pill
                break

            object_id, plate_img = item

            try:
                # Process plate with inline engine inference
                ocr_result = _process_plate_inline(
                    plate_img, ctx, exec_context, inputs, outputs, stream,
                    decoder, ocr_threshold
                )
                output_queue.put((object_id, ocr_result))

            except Exception as e:
                print(f"[OCRWorker] Error processing object {object_id}: {e}", flush=True)
                output_queue.put((object_id, None))

        # Cleanup
        print(f"[OCRWorker] Shutting down", flush=True)
        ctx.push()
        try:
            del exec_context
            del engine
            del runtime
            del stream
            for buf in inputs + outputs:
                if 'device' in buf:
                    buf["device"].free()
            gc.collect()
        finally:
            ctx.pop()
            ctx.detach()

    except Exception as e:
        print(f"[OCRWorker] Fatal error: {e}", flush=True)
        traceback.print_exc()


def _process_plate_inline(plate_img: np.ndarray, ctx, exec_context,
                          inputs, outputs, stream,
                          decoder: CTCLabelDecode, ocr_threshold: float) -> Optional[str]:
    """Process single plate image using inline TRT inference."""
    import pycuda.driver as cuda

    # Check if 2-line plate
    _, img_list = check_plate_square(plate_img)
    images_to_ocr = img_list if img_list is not None else [plate_img]

    # Run OCR on each region
    results = []
    for img in images_to_ocr:
        # Preprocess
        h, w = img.shape[:2]
        ratio = w / float(h)

        if math.ceil(INPUT_HEIGHT * ratio) > INPUT_WIDTH:
            resized_w = INPUT_WIDTH
        else:
            resized_w = int(math.ceil(INPUT_HEIGHT * ratio))

        resized = cv2.resize(img, (resized_w, INPUT_HEIGHT))
        resized = resized.astype(np.float32)
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5

        single = np.zeros((INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
        single[:, :, :resized_w] = resized
        batch = np.stack([single, single], axis=0)  # Shape: (2, 3, 48, 640)

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

            result = outputs[0]["host"].copy()
            output = result.reshape(OUTPUT_SHAPE)[0:1]  # (1, 80, 42)
        finally:
            ctx.pop()

        # Decode
        text_results = decoder(output)
        if text_results and len(text_results) > 0:
            text, conf = text_results[0]
            if conf >= ocr_threshold:
                results.append(text)

    if not results:
        return None

    ocr_result = "".join(results)

    # Validate format
    if not check_format_plate(ocr_result):
        fixed = check_format_plate_append(ocr_result)
        if fixed is not None:
            ocr_result = fixed
        else:
            ocr_result = f"INVALID_{ocr_result}"

    return ocr_result
