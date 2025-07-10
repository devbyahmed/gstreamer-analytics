"""
Main Video Analytics System
Handles both webcam and video file processing with GStreamer and YOLO
"""

import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

# Import modules
from .utils.logger import setup_logger, ProgressLogger, TimedLogger, log_system_info
from .utils.config import get_default_config
from .detection.yolo_detector import YOLODetector
from .detection.frame_annotator import FrameAnnotator
from .analytics.analytics_engine import AnalyticsEngine
from .pipeline.gstreamer_pipeline import GStreamerVideoPipeline
from .pipeline.webcam_pipeline import WebcamPipeline


class VideoAnalyticsSystem:
    """
    Main video analytics system supporting both webcam and video file processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the video analytics system
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        self.config = config or get_default_config()
        
        # Setup logging
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            "VideoAnalyticsSystem",
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file') if log_config.get('file_output') else None,
            console_output=log_config.get('console_output', True)
        )
        
        # Log system information
        log_system_info(self.logger)
        
        # Initialize components
        self.detector = None
        self.annotator = None
        self.analytics = None
        self.pipeline = None
        
        # Results storage
        self.last_analytics_report = None
        self.last_output_path = None
        
        self.logger.info("Video Analytics System initialized")
    
    def _initialize_components(self) -> bool:
        """Initialize detection and analytics components"""
        try:
            # Initialize YOLO detector
            model_config = self.config.get('model', {})
            self.detector = YOLODetector(
                model_path=model_config.get('path', 'yolov8n.pt'),
                confidence_threshold=model_config.get('confidence_threshold', 0.5),
                device=model_config.get('device', 'cpu')
            )
            
            # Initialize frame annotator
            viz_config = self.config.get('visualization', {})
            self.annotator = FrameAnnotator(
                class_names=self.detector.class_names,
                config=viz_config
            )
            
            # Initialize analytics engine
            analytics_config = self.config.get('analytics', {})
            self.analytics = AnalyticsEngine(
                class_names=self.detector.class_names,
                config=analytics_config
            )
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False
    
    def run_webcam_detection(
        self,
        device_id: int = 0,
        output_path: Optional[str] = None,
        duration: Optional[int] = None
    ) -> bool:
        """
        Run real-time object detection on webcam feed
        
        Args:
            device_id: Webcam device ID
            output_path: Output video file path (optional)
            duration: Duration in seconds (optional)
            
        Returns:
            Success status
        """
        if not self._initialize_components():
            return False
        
        with TimedLogger(self.logger, "webcam detection"):
            try:
                # Get webcam configuration
                webcam_config = self.config.get('webcam', {})
                pipeline_config = self.config.get('pipeline', {})
                
                # Create webcam pipeline
                self.pipeline = WebcamPipeline(
                    detector=self.detector,
                    annotator=self.annotator,
                    analytics=self.analytics,
                    config={
                        **webcam_config,
                        **pipeline_config,
                        'device_id': device_id
                    },
                    logger=self.logger
                )
                
                # Start processing
                success = self.pipeline.start(
                    output_path=output_path,
                    duration=duration
                )
                
                if success:
                    # Store results
                    self.last_analytics_report = self.analytics.get_report()
                    self.last_output_path = output_path
                    
                    self.logger.info("Webcam detection completed successfully")
                    return True
                else:
                    self.logger.error("Webcam detection failed")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Webcam detection error: {e}")
                return False
            finally:
                if self.pipeline:
                    self.pipeline.stop()
    
    def run_file_processing(
        self,
        input_path: str,
        output_path: str = "processed_output.mp4",
        duration: Optional[int] = None
    ) -> bool:
        """
        Process video file with object detection
        
        Args:
            input_path: Input video file path
            output_path: Output video file path
            duration: Duration limit in seconds (optional)
            
        Returns:
            Success status
        """
        if not os.path.exists(input_path):
            self.logger.error(f"Input file not found: {input_path}")
            return False
        
        if not self._initialize_components():
            return False
        
        with TimedLogger(self.logger, f"video file processing: {input_path}"):
            try:
                # Get video information
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    self.logger.error(f"Cannot open video: {input_path}")
                    return False
                
                input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                input_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                if input_fps <= 0 or input_fps > 120:
                    input_fps = 25
                
                self.logger.info(
                    f"Input video: {input_width}x{input_height} @ {input_fps}fps, "
                    f"{total_frames} frames"
                )
                
                # Get pipeline configuration
                pipeline_config = self.config.get('pipeline', {})
                video_config = self.config.get('video', {})
                
                # Create GStreamer pipeline
                self.pipeline = GStreamerVideoPipeline(
                    detector=self.detector,
                    annotator=self.annotator,
                    analytics=self.analytics,
                    config={
                        **pipeline_config,
                        **video_config,
                        'output_width': input_width,
                        'output_height': input_height,
                        'output_fps': int(input_fps)
                    },
                    logger=self.logger
                )
                
                # Start processing
                success = self.pipeline.start(
                    input_source=input_path,
                    output_path=output_path,
                    duration=duration
                )
                
                if success:
                    # Store results
                    self.last_analytics_report = self.analytics.get_report()
                    self.last_output_path = output_path
                    
                    self.logger.info("Video file processing completed successfully")
                    return True
                else:
                    self.logger.error("Video file processing failed")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Video processing error: {e}")
                return False
            finally:
                if self.pipeline:
                    self.pipeline.stop()
    
    def display_results(self) -> None:
        """Display processing results"""
        if not self.last_analytics_report:
            self.logger.warning("No analytics report available")
            return
        
        report = self.last_analytics_report
        
        print("\n" + "="*70)
        print("VIDEO ANALYTICS RESULTS")
        print("="*70)
        print(f"Total frames processed: {report['total_frames']}")
        print(f"Total detections: {report['total_detections']}")
        print(f"Processing FPS: {report['processing_fps']:.2f}")
        print(f"Average detections per frame: {report['avg_detections_per_frame']:.2f}")
        print(f"Processing time: {report['elapsed_time']:.2f} seconds")
        print(f"Dropped frames: {report.get('dropped_frames', 0)}")
        print(f"Skipped frames: {report.get('skipped_frames', 0)}")
        
        most_detected = report.get('most_detected_object', ('None', 0))
        print(f"Most detected object: {most_detected[0]} ({most_detected[1]} times)")
        
        print("\nObject Detection Summary:")
        object_counts = report.get('object_counts', {})
        if object_counts:
            for obj_class, count in sorted(object_counts.items(), 
                                         key=lambda x: x[1], reverse=True):
                percentage = (count / report['total_detections'] * 100) if report['total_detections'] > 0 else 0
                print(f"   {obj_class}: {count} detections ({percentage:.1f}%)")
        else:
            print("   No objects detected")
        
        # Save analytics visualization
        try:
            output_dir = self.config.get('output', {}).get('output_directory', 'outputs')
            viz_path = os.path.join(output_dir, "analytics_results.png")
            self.analytics.create_visualization(viz_path)
            print(f"\nAnalytics visualization saved to: {viz_path}")
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
        
        # Output file information
        if self.last_output_path and os.path.exists(self.last_output_path):
            try:
                file_size = os.path.getsize(self.last_output_path) / (1024 * 1024)
                print(f"\nProcessed video saved to: {self.last_output_path}")
                print(f"File size: {file_size:.2f} MB")
                
                # Verify output video
                cap = cv2.VideoCapture(self.last_output_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"Output video: {frame_count} frames @ {fps:.2f}fps")
                    cap.release()
                    print("Output video verified successfully")
                else:
                    print("Warning: Output video may be corrupted")
            except Exception as e:
                self.logger.warning(f"Could not verify output video: {e}")
        
        print("="*70)
    
    def get_analytics_report(self) -> Optional[Dict[str, Any]]:
        """Get the last analytics report"""
        return self.last_analytics_report
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.pipeline:
            self.pipeline.stop()
        
        self.logger.info("System cleanup completed")