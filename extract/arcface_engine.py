import os
import time
import cv2
import numpy as np
import onnxruntime as ort

class IRES(object):
    def __init__(self, onnx_path):
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # Get model details from the ONNX model
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:  # NCHW format
            self.batch_size = input_shape[0]
            self.input_h = input_shape[2]
            self.input_w = input_shape[3]
        else:
            self.batch_size = 1
            self.input_h = input_shape[0]
            self.input_w = input_shape[1]
            
        self.input_size = [self.input_h, self.input_w]
        
        print(f"Model loaded with input shape: {input_shape}")
        print(f"Input name: {self.input_name}, Output name: {self.output_name}")

    def predict(self, img_raw):
        t1 = time.time()
        
        # Preprocess image
        resized_img = self.preprocess_image(img_raw)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: resized_img})
        
        # Postprocess results
        feature = self.post_process(outputs)
        
        inference_time = time.time() - t1
        # print(f"Inference time: {inference_time:.4f} seconds")
        
        return feature
    
    def destroy(self):
        # Clean up resources if needed
        pass
        
    def preprocess_image(self, img):
        # Resize if needed
        if img.shape[0] != self.input_h or img.shape[1] != self.input_w:
            img = cv2.resize(img, (self.input_w, self.input_h))
        
        img = img.astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5
        
        # Check if the model expects NCHW format (common in many models)
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4 and input_shape[1] == 3:  # NCHW format
            img = np.transpose(img, [2, 0, 1])  # HWC to CHW
            img = np.expand_dims(img, axis=0)  # CHW to NCHW
        
        # Ensure the array is contiguous in memory
        img = np.ascontiguousarray(img)
        return img

    def post_process(self, output):
        # Normalize feature vector
        norm = np.linalg.norm(output[0])
        feature = output[0] / norm
        return feature