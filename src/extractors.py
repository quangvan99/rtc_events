"""DeepStream tensor extraction utilities for face recognition"""

import ctypes
from typing import Optional

import numpy as np
import pyds
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


SKIP_SGIE_COMPONENT_ID = 100


def extract_embedding(obj_meta) -> Optional[np.ndarray]:
    """Extract 512-dim L2-normalized embedding from SGIE tensor"""
    l_user = obj_meta.obj_user_meta_list
    while l_user:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tensor = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                layer = pyds.get_nvds_LayerInfo(tensor, 0)
                if layer:
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    emb = np.ctypeslib.as_array(ptr, shape=(512,)).copy()
                    norm = np.linalg.norm(emb)
                    return emb / norm if norm > 0 else emb
            l_user = l_user.next
        except StopIteration:
            break
    return None


def get_source_id(frame_meta) -> int:
    """Extract source_id from frame metadata"""
    return getattr(frame_meta, 'source_id', 0)


def get_frame_num(frame_meta) -> int:
    """Extract frame number from frame metadata"""
    return getattr(frame_meta, 'frame_num', 0)


def iterate_frame_metas(batch) -> list:
    """Iterate over all frame metas in a batch"""
    frame_metas = []
    l_frame = batch.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_metas.append(frame_meta)
            l_frame = l_frame.next
        except StopIteration:
            break
    return frame_metas


def iterate_object_metas(frame_meta) -> list:
    """Iterate over all object metas in a frame"""
    object_metas = []
    l_obj = frame_meta.obj_meta_list
    while l_obj:
        try:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            object_metas.append(obj)
            l_obj = l_obj.next
        except StopIteration:
            break
    return object_metas


def create_tracker_probe_callback(min_face_size: int = 50):
    """Create a tracker probe callback with configured min_face_size"""
    def tracker_probe(pad, info, user_data) -> Gst.PadProbeReturn:
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch:
            return Gst.PadProbeReturn.OK

        l_frame = batch.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                source_id = get_source_id(frame_meta)
                frame = get_frame_num(frame_meta)

                l_obj = frame_meta.obj_meta_list
                while l_obj:
                    try:
                        obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                        rect = obj.rect_params
                        skip = rect.width < min_face or rect.height < min_face
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK
    
    return tracker_probe


def create_sgie_probe_callback(process_face_func, source_mapper=None, log_interval: int = 300):
    """Create an SGIE probe callback for processing face recognition results"""
    def sgie_probe(pad, info, user_data) -> Gst.PadProbeReturn:
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch:
            return Gst.PadProbeReturn.OK

        l_frame = batch.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                source_id = get_source_id(frame_meta)

                if source_mapper:
                    camera_id = source_mapper.get_camera_id(source_id)
                    if frame_meta.frame_num % log_interval == 0:
                        print(f"[DEBUG] Frame from source_id={source_id} -> camera_id={camera_id}")

                l_obj = frame_meta.obj_meta_list
                while l_obj:
                    try:
                        obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                        process_face_func(source_id, obj, frame_meta.frame_num)
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK
    
    return sgie_probe


def create_fps_probe_callback(get_stats_func, cleanup_func):
    """Create an FPS probe callback for monitoring and cleanup"""
    import time
    
    fps_start = time.time()
    fps_count = 0
    stats_last = time.time()
    
    def fps_probe(pad, info, user_data) -> Gst.PadProbeReturn:
        nonlocal fps_start, fps_count, stats_last
        
        now = time.time()
        fps_count += 1

        if now - fps_start >= 1.0:
            print(f"[FPS] {fps_count / (now - fps_start):.1f}")
            fps_start, fps_count = now, 0

        if now - stats_last >= 10.0:
            if get_stats_func:
                total, confirmed, pending = get_stats_func()
                print(f"[STATS] total={total}, confirmed={confirmed}, pending={pending}")
            stats_last = now

        if cleanup_func:
            removed = cleanup_func()
        
        return Gst.PadProbeReturn.OK
    
    return fps_probe
