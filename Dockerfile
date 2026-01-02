# DeepStream Face Recognition WebRTC Stream
FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gst-1.0 \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-nice \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    websockets \
    numpy

# Set environment variables for GStreamer and DeepStream
ENV GST_DEBUG=2
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/nvidia/deepstream/deepstream/lib:$PYTHONPATH

# Expose ports for WebRTC signalling
EXPOSE 8555

# Default command (can be overridden)
CMD ["python3", "signalling.py"]
