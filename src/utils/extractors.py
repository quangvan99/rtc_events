"""
DeepStream Metadata Extractors

Generic utilities for working with DeepStream batch metadata.
Uses generators for memory-efficient iteration.

These are GENERIC utilities that work with any DeepStream pipeline.
Application-specific logic (face, detection, etc.) belongs in the app modules.

Usage:
    from src.utils.extractors import BatchIterator, extract_embedding, get_batch_meta
    
    # Simple iteration
    for frame, obj in BatchIterator(batch):
        emb = extract_embedding(obj)
        
    # Or use helper methods
    batch_iter = BatchIterator(batch)
    for frame in batch_iter.frames():
        for obj in batch_iter.objects(frame):
            ...
"""

import ctypes
from typing import Iterator, Tuple, Optional

import numpy as np


def _get_pyds():
    """Lazy import pyds module"""
    import pyds
    return pyds


# =============================================================================
# Batch Iterator - Main interface for DeepStream metadata
# =============================================================================

class BatchIterator:
    """
    Simple iterator for DeepStream batch metadata.
    
    Examples:
        # Iterate all objects
        for frame, obj in BatchIterator(batch):
            source_id = frame.source_id
            object_id = obj.object_id
            
        # Iterate frames only
        for frame in BatchIterator(batch).frames():
            print(f"Frame {frame.frame_num} from source {frame.source_id}")
            
        # Nested iteration
        it = BatchIterator(batch)
        for frame in it.frames():
            for obj in it.objects(frame):
                process(obj)
    """
    
    def __init__(self, batch):
        self.batch = batch
    
    def __iter__(self) -> Iterator[Tuple]:
        """Iterate all (frame_meta, obj_meta) pairs"""
        for frame in self.frames():
            for obj in self.objects(frame):
                yield frame, obj
    
    def frames(self) -> Iterator:
        """Iterate frame metas only"""
        pyds = _get_pyds()
        l_frame = self.batch.frame_meta_list
        while l_frame:
            try:
                yield pyds.NvDsFrameMeta.cast(l_frame.data)
                l_frame = l_frame.next
            except StopIteration:
                break
    
    def objects(self, frame_meta) -> Iterator:
        """Iterate object metas in a frame"""
        pyds = _get_pyds()
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                yield pyds.NvDsObjectMeta.cast(l_obj.data)
                l_obj = l_obj.next
            except StopIteration:
                break


# =============================================================================
# Extraction Functions
# =============================================================================

def extract_embedding(obj_meta, dim: int = 512) -> Optional[np.ndarray]:
    """
    Extract L2-normalized embedding from SGIE tensor output.
    
    Args:
        obj_meta: NvDsObjectMeta with tensor output
        dim: Embedding dimension (default: 512)
        
    Returns:
        Normalized numpy array or None if no tensor found
    """
    pyds = _get_pyds()
    l_user = obj_meta.obj_user_meta_list
    while l_user:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tensor = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                layer = pyds.get_nvds_LayerInfo(tensor, 0)
                if layer:
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    emb = np.ctypeslib.as_array(ptr, shape=(dim,)).copy()
                    norm = np.linalg.norm(emb)
                    return emb / norm if norm > 0 else emb
            l_user = l_user.next
        except StopIteration:
            break
    return None


def get_batch_meta(buffer):
    """
    Get NvDsBatchMeta from GStreamer buffer.
    
    Args:
        buffer: GstBuffer from probe info
        
    Returns:
        NvDsBatchMeta or None
    """
    if not buffer:
        return None
    pyds = _get_pyds()
    return pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
