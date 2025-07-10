"""
YOLO-based object detection module
"""

import time
import numpy as np
import torch
from ultralytics import YOLO
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Structure for detection results"""
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    frame_id: int
    timestamp: float


class YOLODetector:
    """YOLO-based object detector with optimizations"""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        device: str = "cpu"
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        # Optimize model for inference
        self._optimize_model()
        
        # Performance tracking
        self.inference_times = []
        self.total_detections = 0
        
    def _optimize_model(self) -> None:
        """Optimize model for inference"""
        try:
            # Fuse model layers for faster inference
            self.model.fuse()
            
            # Set model to evaluation mode
            if hasattr(self.model, 'model'):
                self.model.model.eval()
            
            # Move to specified device
            if torch.cuda.is_available() and self.device == 'cuda':
                self.model.to('cuda')
            
            print(f"✅ YOLO model optimized for {self.device}")
            
        except Exception as e:
            print(f"⚠️ Model optimization warning: {e}")

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """
        Perform object detection on a frame
        
        Args:
            frame: Input frame (RGB format)
            frame_id: Frame identifier
            
        Returns:
            DetectionResult containing detection information
        """
        start_time = time.time()
        
        try:
            # Run inference with optimized settings
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
                half=False,  # Use FP32 for better compatibility
                augment=False,  # Disable augmentation for speed
                save=False,  # Don't save intermediate results
                show=False   # Don't show results
            )[0]

            # Extract results
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
            else:
                boxes = np.array([])
                scores = np.array([])
                classes = np.array([])

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_detections += len(boxes)
            
            # Keep only recent inference times (last 100)
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]

            return DetectionResult(
                boxes=boxes,
                scores=scores,
                classes=classes,
                frame_id=frame_id,
                timestamp=time.time()
            )

        except Exception as e:
            print(f"Detection error: {e}")
            return DetectionResult(
                boxes=np.array([]),
                scores=np.array([]),
                classes=np.array([]),
                frame_id=frame_id,
                timestamp=time.time()
            )

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get detector performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {
                'avg_inference_time': 0.0,
                'avg_fps': 0.0,
                'total_detections': 0
            }
        
        avg_inference_time = np.mean(self.inference_times)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        return {
            'avg_inference_time': avg_inference_time,
            'avg_fps': avg_fps,
            'total_detections': self.total_detections,
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times)
        }

    def warmup(self, input_shape: tuple = (640, 640, 3)) -> None:
        """
        Warm up the model with dummy input
        
        Args:
            input_shape: Input shape for warmup (height, width, channels)
        """
        try:
            dummy_frame = np.random.randint(0, 255, input_shape, dtype=np.uint8)
            
            # Run a few warmup inferences
            for _ in range(3):
                self.detect(dummy_frame, frame_id=-1)
            
            print("✅ Model warmed up successfully")
            
        except Exception as e:
            print(f"⚠️ Model warmup warning: {e}")

    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.inference_times = []
        self.total_detections = 0