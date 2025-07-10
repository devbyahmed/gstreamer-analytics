"""
Object detection modules
"""

from .yolo_detector import YOLODetector, DetectionResult
from .frame_annotator import FrameAnnotator

__all__ = ['YOLODetector', 'DetectionResult', 'FrameAnnotator']