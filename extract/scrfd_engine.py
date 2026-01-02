"""
An example that uses TensorRT's Python api to make inferences.
"""

import os
import time
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from utils import *
import pycuda.autoinit

path_cur=os.path.dirname(os.path.abspath(__file__))

class SCRFDTRT(object):
    def __init__(self, engine_path = "/home/project/faceVMS/face-engine-jetson/models/scrfd640/scrfd_2.5g_bnkps_dynamic.engine"):
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

        self.input_size = [320, 320]
        self.thresh = 0.6
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.max_num=0

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

        resized_img, det_scale = self.preprocess_image(img_raw)
        cuda.memcpy_htod_async(cuda_inputs[0], resized_img, stream)

        # Run inference.
        context.execute_async(batch_size = self.batch_size, bindings = bindings, stream_handle = stream.handle)
        # Transfer predictions back from the GPU.
        for i in range(len(self.bindings) - 1 ):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
        # Synchronize the stream
        stream.synchronize()
        self.ctx.pop()
        output = host_outputs
        
        bboxes, kpss = self.post_process(output, det_scale)
        # return result_boxes, result_classid, result_scores
        return bboxes, kpss
    
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
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        if im_ratio > model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))

        det_imgs = np.zeros((1, self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_imgs[0][:new_height, :new_width, :] = resized_img

        input_size = (self.input_size[0],self.input_size[1])
        
        blobs = cv2.dnn.blobFromImages(
           det_imgs, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)

        return blobs, det_scale

    def post_process(self, output, det_scale):
        score_8s = output[0].reshape(12800, 1)
        bbox_8s = output[1].reshape(12800, 4)
        kps_8s = output[2].reshape(12800, 10)
        score_16s = output[3].reshape(3200, 1)
        bbox_16s = output[4].reshape(3200, 4)
        kps_16s = output[5].reshape(3200, 10)
        score_32s = output[6].reshape(800, 1)
        bbox_32s = output[7].reshape(800, 4)
        kps_32s = output[8].reshape(800, 10)

        net_outs=[score_8s, score_16s, score_32s, bbox_8s, bbox_16s, bbox_32s, kps_8s, kps_16s,kps_32s]
        scores_list, bboxes_list, kpss_list = self.forward(net_outs)
        bboxes, kpss = self.postprocess(scores_list, bboxes_list, kpss_list, det_scale)
        return bboxes, kpss

    def forward(self, net_outs):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
               kps_preds = net_outs[idx + fmc * 2] * stride
            height = self.input_size[0] // stride
            width = self.input_size[1] // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
                    
            pos_inds = np.where(scores >= self.thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list
    
    def postprocess(self, scores_list, bboxes_list, kpss_list,det_scale):
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det,self.nms_thresh)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        return det, kpss