"""
Pipeline modules for video processing
"""

from .gstreamer_pipeline import GStreamerVideoPipeline
from .webcam_pipeline import WebcamPipeline
from .frame_queue import ThreadSafeFrameQueue

__all__ = ['GStreamerVideoPipeline', 'WebcamPipeline', 'ThreadSafeFrameQueue']