"""
An example that uses TensorRT's Python api to make inferences.
"""
import os
import time
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
# import torch.nn.functional as F
path_cur=os.path.dirname(os.path.abspath(__file__))


class AdaFace(object):
    def __init__(self, engine_path = "/home/project/faceVMS/test/arcface/arcface_r18.engine"):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        # Deserialize the engine from file
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

        self.input_size = [112, 112]

    def predict(self, img_raw) :
        # Restore
        t1 = time.time()
        self.ctx.push()
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        resized_img = self.preprocess_image(img_raw)
        np.copyto(host_inputs[0], resized_img.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)

        # Run inference.
        context.execute_async(batch_size = self.batch_size, bindings = bindings, stream_handle = stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
        # Synchronize the stream
        stream.synchronize()
        self.ctx.pop()
        output = host_outputs
        
        feature = self.post_process(output)
        # return result_boxes, result_classid, result_scores
        return feature
    
    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, img):
        img = img.astype(np.float)
        img = (img / 255 - 0.5) / 0.5
        img = np.transpose(img, [2, 0, 1])
        # CHW to NCHW format
        if len(img.shape) != 3 :
            img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        return img

    def post_process(self, output):
        norm = np.linalg.norm(output[1])
        feature = output[1]/norm
        return feature
