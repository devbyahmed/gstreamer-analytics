"""
Frame annotation module for visualizing detection results
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any
from .yolo_detector import DetectionResult


class FrameAnnotator:
    """Frame annotation utility for drawing detection results"""

    def __init__(self, class_names: Dict[int, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize frame annotator
        
        Args:
            class_names: Dictionary mapping class IDs to names
            config: Configuration dictionary for visualization settings
        """
        self.class_names = class_names
        
        # Default configuration
        default_config = {
            'show_confidence': True,
            'show_class_names': True,
            'box_thickness': 2,
            'text_size': 0.6,
            'colors_seed': 42
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Generate colors for each class
        self.colors = self._generate_colors()
        
    def _generate_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Generate random colors for each class
        
        Returns:
            Dictionary mapping class IDs to BGR colors
        """
        np.random.seed(self.config['colors_seed'])
        colors = {}
        
        for class_id in self.class_names.keys():
            # Generate random BGR color
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors[class_id] = color
            
        return colors

    def annotate(self, frame: np.ndarray, detection: DetectionResult) -> np.ndarray:
        """
        Annotate frame with detection results
        
        Args:
            frame: Input frame (RGB or BGR format)
            detection: Detection results
            
        Returns:
            Annotated frame
        """
        if len(detection.boxes) == 0:
            return frame

        annotated_frame = frame.copy()
        
        for box, score, class_id in zip(detection.boxes, detection.scores, detection.classes):
            try:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box)
                class_id = int(class_id)
                
                # Get class information
                class_name = self.class_names.get(class_id, "Unknown")
                color = self.colors.get(class_id, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    self.config['box_thickness']
                )
                
                # Create label
                label_parts = []
                if self.config['show_class_names']:
                    label_parts.append(class_name)
                if self.config['show_confidence']:
                    label_parts.append(f"{score:.2f}")
                
                label = ": ".join(label_parts) if label_parts else ""
                
                if label:
                    # Calculate label size and position
                    label_size = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config['text_size'],
                        2
                    )[0]
                    
                    # Position label above the box
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 30
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_frame,
                        (x1, label_y - label_size[1] - 5),
                        (x1 + label_size[0], label_y + 5),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config['text_size'],
                        (255, 255, 255),
                        2
                    )

            except Exception as e:
                # Skip problematic detections
                continue

        return annotated_frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS information on frame
        
        Args:
            frame: Input frame
            fps: Current FPS value
            
        Returns:
            Frame with FPS overlay
        """
        fps_text = f"FPS: {fps:.1f}"
        
        # Get text size
        text_size = cv2.getTextSize(
            fps_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )[0]
        
        # Position in top-right corner
        x = frame.shape[1] - text_size[0] - 10
        y = 30
        
        # Draw background
        cv2.rectangle(
            frame,
            (x - 5, y - text_size[1] - 5),
            (x + text_size[0] + 5, y + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            fps_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return frame

    def draw_detection_count(self, frame: np.ndarray, count: int) -> np.ndarray:
        """
        Draw detection count on frame
        
        Args:
            frame: Input frame
            count: Number of detections
            
        Returns:
            Frame with detection count overlay
        """
        count_text = f"Detections: {count}"
        
        # Get text size
        text_size = cv2.getTextSize(
            count_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )[0]
        
        # Position in top-left corner
        x = 10
        y = 30
        
        # Draw background
        cv2.rectangle(
            frame,
            (x - 5, y - text_size[1] - 5),
            (x + text_size[0] + 5, y + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            count_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return frame

    def add_info_overlay(
        self,
        frame: np.ndarray,
        fps: Optional[float] = None,
        detection_count: Optional[int] = None,
        frame_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Add information overlay to frame
        
        Args:
            frame: Input frame
            fps: Current FPS (optional)
            detection_count: Number of detections (optional)
            frame_id: Current frame ID (optional)
            
        Returns:
            Frame with information overlay
        """
        annotated_frame = frame.copy()
        
        # Draw FPS
        if fps is not None:
            annotated_frame = self.draw_fps(annotated_frame, fps)
        
        # Draw detection count
        if detection_count is not None:
            annotated_frame = self.draw_detection_count(annotated_frame, detection_count)
        
        # Draw frame ID
        if frame_id is not None:
            frame_text = f"Frame: {frame_id}"
            
            # Position in bottom-left corner
            x = 10
            y = frame.shape[0] - 10
            
            cv2.putText(
                annotated_frame,
                frame_text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated_frame

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration settings
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        
        # Regenerate colors if seed changed
        if 'colors_seed' in new_config:
            self.colors = self._generate_colors()