"""
Webcam pipeline for real-time object detection
"""

import cv2
import time
import threading
import logging
from typing import Optional, Dict, Any
import numpy as np

from .frame_queue import ThreadSafeFrameQueue
from ..detection.yolo_detector import YOLODetector
from ..detection.frame_annotator import FrameAnnotator
from ..analytics.analytics_engine import AnalyticsEngine
from ..utils.logger import ProgressLogger


class WebcamPipeline:
    """Real-time webcam processing pipeline"""

    def __init__(
        self,
        detector: YOLODetector,
        annotator: FrameAnnotator,
        analytics: AnalyticsEngine,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize webcam pipeline
        
        Args:
            detector: YOLO detector instance
            annotator: Frame annotator instance
            analytics: Analytics engine instance
            config: Configuration dictionary
            logger: Logger instance
        """
        self.detector = detector
        self.annotator = annotator
        self.analytics = analytics
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.device_id = config.get('device_id', 0)
        self.width = config.get('width', 1280)
        self.height = config.get('height', 720)
        self.fps = config.get('fps', 30)
        self.max_queue_size = config.get('max_queue_size', 100)
        self.processing_threads = config.get('processing_threads', 1)
        
        # Pipeline components
        self.cap = None
        self.video_writer = None
        self.frame_queue = ThreadSafeFrameQueue(self.max_queue_size)
        self.output_queue = ThreadSafeFrameQueue(self.max_queue_size)
        
        # Threading
        self.capture_thread = None
        self.processing_threads_list = []
        self.display_thread = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # State tracking
        self.frame_count = 0
        self.processed_count = 0
        self.displayed_count = 0
        self.start_time = None
        
        # Performance tracking
        self.last_fps_update = 0
        self.fps_counter = 0
        self.current_fps = 0

    def _initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            # Check if device exists
            test_cap = cv2.VideoCapture(self.device_id)
            if not test_cap.isOpened():
                self.logger.error(f"Cannot open camera device {self.device_id}")
                return False
            test_cap.release()
            
            # Initialize capture
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.device_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(
                f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}fps"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False

    def _capture_worker(self) -> None:
        """Camera capture worker thread"""
        self.logger.info("Camera capture thread started")
        
        while not self.shutdown_event.is_set() and self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    continue
                
                # Convert BGR to RGB for YOLO
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add to processing queue
                if not self.frame_queue.put(rgb_frame, timeout=0.1):
                    # Frame dropped due to full queue
                    continue
                
                self.frame_count += 1
                
                # Limit capture rate if needed
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                break
        
        self.logger.info("Camera capture thread stopped")

    def _processing_worker(self, worker_id: int) -> None:
        """Frame processing worker thread"""
        self.logger.info(f"Processing worker {worker_id} started")
        
        while not self.shutdown_event.is_set() and self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.5)
                if frame is None:
                    continue
                
                # Run detection
                detection = self.detector.detect(frame, self.processed_count)
                
                # Update analytics
                self.analytics.update(detection)
                
                # Annotate frame
                annotated_frame = self.annotator.annotate(frame, detection)
                
                # Add FPS and detection count overlay
                annotated_frame = self.annotator.add_info_overlay(
                    annotated_frame,
                    fps=self.current_fps,
                    detection_count=len(detection.boxes),
                    frame_id=self.processed_count
                )
                
                # Convert back to BGR for display/recording
                bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Add to output queue
                self.output_queue.put(bgr_frame, timeout=0.1)
                
                self.processed_count += 1
                self.frame_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                continue
        
        self.logger.info(f"Processing worker {worker_id} stopped")

    def _display_worker(self, output_path: Optional[str] = None) -> None:
        """Display and recording worker thread"""
        self.logger.info("Display worker started")
        
        # Initialize video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, (self.width, self.height)
            )
            self.logger.info(f"Recording to: {output_path}")
        
        while not self.shutdown_event.is_set() and self.is_running:
            try:
                # Get processed frame
                frame = self.output_queue.get(timeout=0.5)
                if frame is None:
                    continue
                
                # Display frame
                cv2.imshow('Object Detection', frame)
                
                # Record frame
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                
                self.displayed_count += 1
                
                # Update FPS counter
                self._update_fps()
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Quit key pressed")
                    self.shutdown_event.set()
                    break
                
                self.output_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Display error: {e}")
                continue
        
        # Cleanup
        cv2.destroyAllWindows()
        if self.video_writer is not None:
            self.video_writer.release()
            self.logger.info("Video recording saved")
        
        self.logger.info("Display worker stopped")

    def _update_fps(self) -> None:
        """Update FPS calculation"""
        current_time = time.time()
        self.fps_counter += 1
        
        # Update FPS every second
        if current_time - self.last_fps_update >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_update)
            self.fps_counter = 0
            self.last_fps_update = current_time

    def start(
        self,
        output_path: Optional[str] = None,
        duration: Optional[int] = None
    ) -> bool:
        """
        Start webcam processing
        
        Args:
            output_path: Output video file path (optional)
            duration: Duration in seconds (optional)
            
        Returns:
            Success status
        """
        try:
            # Initialize camera
            if not self._initialize_camera():
                return False
            
            # Warm up detector
            dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.detector.warmup((self.height, self.width, 3))
            
            # Reset state
            self.is_running = True
            self.shutdown_event.clear()
            self.frame_count = 0
            self.processed_count = 0
            self.displayed_count = 0
            self.start_time = time.time()
            self.last_fps_update = time.time()
            self.fps_counter = 0
            
            # Start worker threads
            self.capture_thread = threading.Thread(
                target=self._capture_worker, daemon=True
            )
            self.capture_thread.start()
            
            # Start processing threads
            for i in range(self.processing_threads):
                thread = threading.Thread(
                    target=self._processing_worker, args=(i,), daemon=True
                )
                thread.start()
                self.processing_threads_list.append(thread)
            
            # Start display thread
            self.display_thread = threading.Thread(
                target=self._display_worker, args=(output_path,), daemon=True
            )
            self.display_thread.start()
            
            self.logger.info("Webcam pipeline started")
            self.logger.info("Press 'q' to quit")
            
            # Handle duration limit
            if duration:
                def duration_handler():
                    time.sleep(duration)
                    self.logger.info(f"Duration limit reached ({duration}s)")
                    self.shutdown_event.set()
                
                duration_thread = threading.Thread(target=duration_handler, daemon=True)
                duration_thread.start()
            
            # Wait for completion
            try:
                while not self.shutdown_event.is_set():
                    time.sleep(0.1)
                    
                    # Log progress periodically
                    if self.processed_count > 0 and self.processed_count % 100 == 0:
                        stats = self.analytics.get_real_time_stats()
                        self.logger.info(
                            f"Processed: {self.processed_count} frames, "
                            f"FPS: {stats.get('fps_average', 0):.1f}, "
                            f"Detections: {stats.get('total_detections', 0)}"
                        )
                        
            except KeyboardInterrupt:
                self.logger.info("Interrupted by user")
                self.shutdown_event.set()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline start error: {e}")
            return False
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop webcam processing"""
        self.logger.info("Stopping webcam pipeline...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        for thread in self.processing_threads_list:
            if thread.is_alive():
                thread.join(timeout=2)
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2)
        
        # Cleanup
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Final statistics
        elapsed_time = time.time() - (self.start_time or time.time())
        self.logger.info(f"Pipeline stopped after {elapsed_time:.2f}s")
        self.logger.info(f"Processed {self.processed_count} frames")
        
        queue_stats = self.frame_queue.get_statistics()
        if queue_stats['drop_rate'] > 0:
            self.logger.warning(f"Frame drop rate: {queue_stats['drop_rate']:.2f}%")