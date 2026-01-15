"""
License Plate OCR with Multiprocessing Support
Wraps PlateOCREngine with multiprocessing for async inference
"""

import multiprocessing as mp
import numpy as np
import cv2
import os
import time
from typing import Optional, Tuple, List, Dict, Any
from queue import Empty


class CTCLabelDecode:
    """CTC Label Decoder for OCR postprocessing"""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.character_str = []

        if character_dict_path is None:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")

        self.character = ["blank"] + self.character_str
        self.dict = {char: i for i, char in enumerate(self.character)}

    def __call__(self, preds):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        result_list = []
        batch_size = len(preds_idx)

        for batch_idx in range(batch_size):
            selection = np.ones(len(preds_idx[batch_idx]), dtype=bool)
            selection[1:] = preds_idx[batch_idx][1:] != preds_idx[batch_idx][:-1]
            selection &= preds_idx[batch_idx] != 0

            char_list = [
                self.character[text_id]
                for text_id in preds_idx[batch_idx][selection]
            ]
            conf_list = preds_prob[batch_idx][selection]

            text = "".join(char_list)
            confidence = float(np.mean(conf_list)) if len(conf_list) > 0 else 0.0

            result_list.append((text, confidence))

        return result_list


class PlateOCREngine:
    """License Plate OCR Engine using TensorRT"""

    def __init__(self, engine_path, dict_path, input_shape=(2, 3, 48, 640)):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        self.img_c = input_shape[1]
        self.img_h = input_shape[2]
        self.img_w = input_shape[3]

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

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

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape})

        self.decoder = CTCLabelDecode(dict_path, use_space_char=False)
        self.cuda = cuda

        print(f"[PID {os.getpid()}] Engine loaded: {engine_path}")

    def preprocess(self, image):
        h, w = image.shape[:2]
        max_wh_ratio = w / h

        resize_w = int(self.img_h * max_wh_ratio)
        if resize_w > self.img_w:
            resize_w = self.img_w

        resized = cv2.resize(image, (resize_w, self.img_h))
        resized = resized.astype(np.float32)
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5

        padding = np.zeros((self.img_c, self.img_h, self.img_w), dtype=np.float32)
        padding[:, :, :resize_w] = resized

        return padding

    def infer(self, images):
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]

        batch = np.zeros((self.batch_size, self.img_c, self.img_h, self.img_w), dtype=np.float32)
        num_images = min(len(images), self.batch_size)
        for i in range(num_images):
            batch[i] = images[i]

        np.copyto(self.inputs[0]["host"], batch.ravel())
        self.cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )

        for buf in self.inputs + self.outputs:
            self.context.set_tensor_address(buf["name"], int(buf["device"]))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        output = self.outputs[0]["host"].copy()
        output_shape = self.outputs[0]["shape"]
        output = output.reshape(output_shape)

        return output[:num_images]

    def recognize(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot read image: {image}")

        processed = self.preprocess(image)
        output = self.infer(processed)
        results = self.decoder(output)

        return results[0]

    def recognize_batch(self, images):
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = cv2.imread(img)
            processed_images.append(self.preprocess(img))

        results = []
        for i in range(0, len(processed_images), self.batch_size):
            batch = processed_images[i:i + self.batch_size]
            output = self.infer(batch)
            batch_results = self.decoder(output)
            results.extend(batch_results[:len(batch)])

        return results

    def destroy(self):
        del self.context
        del self.engine
        del self.runtime


class PlateOCRWorker:
    """
    Multiprocessing worker for Plate OCR

    Usage:
        input_queue = mp.Queue()
        output_queue = mp.Queue()

        worker = PlateOCRWorker(
            input_queue=input_queue,
            output_queue=output_queue,
            engine_path="path/to/engine",
            dict_path="path/to/dict"
        )
        worker.start()

        # Send image
        input_queue.put({"id": "001", "image": img})

        # Get result
        result = output_queue.get()
        print(result)  # {"id": "001", "text": "ABC123", "confidence": 0.98}

        # Stop worker
        input_queue.put(None)
        worker.join()
    """

    def __init__(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        engine_path: str,
        dict_path: str,
        input_shape: Tuple[int, int, int, int] = (2, 3, 48, 640),
        poll_interval: float = 0.01,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.engine_path = engine_path
        self.dict_path = dict_path
        self.input_shape = input_shape
        self.poll_interval = poll_interval
        self.process: Optional[mp.Process] = None
        self._stop_event = mp.Event()

    def _run(self):
        """Worker process main loop"""
        print(f"[PID {os.getpid()}] OCR Worker starting...")

        # Initialize engine inside process (CUDA context must be created in same process)
        engine = PlateOCREngine(
            engine_path=self.engine_path,
            dict_path=self.dict_path,
            input_shape=self.input_shape
        )

        print(f"[PID {os.getpid()}] OCR Worker ready, waiting for images...")

        while not self._stop_event.is_set():
            try:
                # Non-blocking check with timeout
                if self.input_queue.empty():
                    time.sleep(self.poll_interval)
                    continue

                # Get data from queue
                data = self.input_queue.get(timeout=self.poll_interval)

                # None signals shutdown
                if data is None:
                    print(f"[PID {os.getpid()}] Received shutdown signal")
                    break

                # Process the request
                request_id = data.get("id", "unknown")
                image = data.get("image")

                if image is None:
                    self.output_queue.put({
                        "id": request_id,
                        "text": "",
                        "confidence": 0.0,
                        "error": "No image provided"
                    })
                    continue

                # Run OCR
                try:
                    text, confidence = engine.recognize(image)
                    self.output_queue.put({
                        "id": request_id,
                        "text": text,
                        "confidence": confidence,
                        "error": None
                    })
                    print(f"[PID {os.getpid()}] Processed {request_id}: {text} ({confidence:.2%})")

                except Exception as e:
                    self.output_queue.put({
                        "id": request_id,
                        "text": "",
                        "confidence": 0.0,
                        "error": str(e)
                    })

            except Empty:
                continue
            except Exception as e:
                print(f"[PID {os.getpid()}] Error: {e}")
                continue

        # Cleanup
        engine.destroy()
        print(f"[PID {os.getpid()}] OCR Worker stopped")

    def start(self):
        """Start the worker process"""
        self._stop_event.clear()
        self.process = mp.Process(target=self._run)
        self.process.start()
        return self

    def stop(self):
        """Stop the worker process gracefully"""
        self._stop_event.set()
        # Send None to unblock queue.get()
        try:
            self.input_queue.put(None, timeout=1.0)
        except:
            pass

    def join(self, timeout: float = None):
        """Wait for worker process to finish"""
        if self.process:
            self.process.join(timeout=timeout)

    def is_alive(self) -> bool:
        """Check if worker process is running"""
        return self.process is not None and self.process.is_alive()


class PlateOCRPool:
    """
    Pool of OCR workers for parallel processing

    Usage:
        pool = PlateOCRPool(
            num_workers=2,
            engine_path="path/to/engine",
            dict_path="path/to/dict"
        )
        pool.start()

        # Submit multiple images
        pool.submit("001", img1)
        pool.submit("002", img2)

        # Get results
        results = pool.get_all_results(timeout=5.0)

        pool.stop()
    """

    def __init__(
        self,
        num_workers: int,
        engine_path: str,
        dict_path: str,
        input_shape: Tuple[int, int, int, int] = (2, 3, 48, 640),
        queue_size: int = 100,
    ):
        self.num_workers = num_workers
        self.engine_path = engine_path
        self.dict_path = dict_path
        self.input_shape = input_shape

        self.input_queue = mp.Queue(maxsize=queue_size)
        self.output_queue = mp.Queue(maxsize=queue_size)
        self.workers: List[PlateOCRWorker] = []

    def start(self):
        """Start all workers"""
        for i in range(self.num_workers):
            worker = PlateOCRWorker(
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                engine_path=self.engine_path,
                dict_path=self.dict_path,
                input_shape=self.input_shape
            )
            worker.start()
            self.workers.append(worker)
        return self

    def submit(self, request_id: str, image: np.ndarray):
        """Submit an image for OCR processing"""
        self.input_queue.put({"id": request_id, "image": image})

    def get_result(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Get a single result from output queue"""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_all_results(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """Get all available results"""
        results = []
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                result = self.output_queue.get(timeout=0.1)
                results.append(result)
            except Empty:
                if self.input_queue.empty():
                    break
                continue

        return results

    def stop(self):
        """Stop all workers"""
        for worker in self.workers:
            worker.stop()

        for worker in self.workers:
            worker.join(timeout=5.0)

        self.workers.clear()


def main():
    """Test multiprocessing OCR"""
    mp.set_start_method("spawn", force=True)

    # Paths for testing
    data_dir = "/home/mq/disk2T/quangnv/face/data/license_plate/ocr"
    engine_path = os.path.join(data_dir, "bz2_640", "ppocr_dummy.engine")
    dict_path = os.path.join(data_dir, "bz2_640", "licence_plate_dict.txt")
    image_path = os.path.join(data_dir, "plate_debug", "plate_87637_part1.jpg")

    # Check files
    for path in [engine_path, dict_path, image_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

    # Create queues
    input_queue = mp.Queue(maxsize=10)
    output_queue = mp.Queue(maxsize=10)

    # Create and start worker
    worker = PlateOCRWorker(
        input_queue=input_queue,
        output_queue=output_queue,
        engine_path=engine_path,
        dict_path=dict_path,
        input_shape=(2, 3, 48, 640)
    )
    worker.start()

    # Wait for worker to initialize
    time.sleep(2)

    # Load test image
    image = cv2.imread(image_path)
    print(f"\n[Main] Sending image: {image_path}")
    print(f"[Main] Image shape: {image.shape}")

    # Send image to worker
    input_queue.put({"id": "test_001", "image": image})

    # Wait for result
    print("[Main] Waiting for result...")
    try:
        result = output_queue.get(timeout=10.0)
        print(f"\n=== OCR Result ===")
        print(f"ID: {result['id']}")
        print(f"Text: {result['text']}")
        print(f"Confidence: {result['confidence']:.4f}")
        if result.get('error'):
            print(f"Error: {result['error']}")
    except Empty:
        print("[Main] Timeout waiting for result")

    # Stop worker
    print("\n[Main] Stopping worker...")
    worker.stop()
    worker.join(timeout=5.0)
    print("[Main] Done")


if __name__ == "__main__":
    main()
