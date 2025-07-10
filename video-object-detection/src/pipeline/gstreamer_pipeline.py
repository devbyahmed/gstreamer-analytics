"""
GStreamer pipeline for video file processing
"""

import os
import sys
import time
import threading
import logging
from typing import Optional, Dict, Any
import numpy as np
import cv2

# Try to import GStreamer components
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstApp', '1.0')
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst, GstApp, GstVideo, GObject, GLib
    
    # Initialize GStreamer
    Gst.init(None)
    GST_AVAILABLE = True
except ImportError as e:
    print(f"GStreamer not available: {e}")
    print("Falling back to OpenCV-based processing")
    GST_AVAILABLE = False

from .frame_queue import ThreadSafeFrameQueue
from ..detection.yolo_detector import YOLODetector
from ..detection.frame_annotator import FrameAnnotator
from ..analytics.analytics_engine import AnalyticsEngine
from ..utils.logger import ProgressLogger


class GStreamerVideoPipeline:
    """GStreamer-based video processing pipeline with OpenCV fallback"""

    def __init__(
        self,
        detector: YOLODetector,
        annotator: FrameAnnotator,
        analytics: AnalyticsEngine,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize GStreamer pipeline
        
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
        self.output_width = config.get('output_width', 1280)
        self.output_height = config.get('output_height', 720)
        self.output_fps = config.get('output_fps', 25)
        self.max_queue_size = config.get('max_queue_size', 1000)
        self.processing_threads = config.get('processing_threads', 1)
        self.buffer_size = config.get('buffer_size', 500)
        
        # Pipeline components
        self.pipeline = None
        self.appsink = None
        self.bus = None
        self.loop = None
        
        # Processing components
        self.frame_queue = ThreadSafeFrameQueue(self.max_queue_size)
        self.processing_threads_list = []
        self.is_processing = False
        self.shutdown_event = threading.Event()
        
        # State tracking
        self.frame_count = 0
        self.processed_count = 0
        self.error_count = 0
        self.video_writer = None
        
        # Fallback mode
        self.use_opencv_fallback = not GST_AVAILABLE
        
        if self.use_opencv_fallback:
            self.logger.warning("Using OpenCV fallback instead of GStreamer")

    def _create_gstreamer_pipeline(self, input_source: str) -> bool:
        """Create GStreamer pipeline"""
        if not GST_AVAILABLE:
            return False
            
        try:
            # Determine source type and create appropriate pipeline
            if input_source.startswith(('rtsp://', 'http://', 'https://')):
                source_element = f"souphttpsrc location={input_source} is-live=true"
            elif input_source.isdigit():
                source_element = f"v4l2src device=/dev/video{input_source}"
            elif os.path.exists(input_source):
                source_element = f"filesrc location={input_source}"
            else:
                raise ValueError(f"Invalid input source: {input_source}")

            # Build pipeline with large buffers
            pipeline_elements = [
                source_element,
                f"queue max-size-buffers={self.buffer_size} max-size-time=0 max-size-bytes=0",
                "decodebin",
                "videoconvert",
                "videoscale method=bilinear",
                f"video/x-raw,format=RGB,width={self.output_width},height={self.output_height}",
                f"queue max-size-buffers={self.buffer_size} max-size-time=0 max-size-bytes=0",
                f"appsink name=sink emit-signals=true sync=false max-buffers={self.buffer_size} drop=false"
            ]

            pipeline_string = " ! ".join(pipeline_elements)
            self.logger.info(f"GStreamer pipeline: {pipeline_string}")

            # Create pipeline
            self.pipeline = Gst.parse_launch(pipeline_string)
            self.appsink = self.pipeline.get_by_name('sink')

            # Configure appsink
            caps = Gst.Caps.from_string(
                f"video/x-raw,format=RGB,width={self.output_width},height={self.output_height}"
            )
            self.appsink.set_property('caps', caps)

            # Connect signals
            self.appsink.connect('new-sample', self._on_new_sample)

            # Setup bus
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect('message::error', self._on_error)
            self.bus.connect('message::eos', self._on_eos)

            # Create main loop
            self.loop = GLib.MainLoop()

            self.logger.info("GStreamer pipeline created successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create GStreamer pipeline: {e}")
            return False

    def _create_opencv_pipeline(self, input_source: str) -> bool:
        """Create OpenCV-based pipeline as fallback"""
        try:
            # Verify input source
            if not os.path.exists(input_source):
                self.logger.error(f"Input file not found: {input_source}")
                return False
            
            self.opencv_cap = cv2.VideoCapture(input_source)
            if not self.opencv_cap.isOpened():
                self.logger.error(f"Cannot open video: {input_source}")
                return False
            
            # Get video properties
            total_frames = int(self.opencv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.opencv_cap.get(cv2.CAP_PROP_FPS)
            width = int(self.opencv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.opencv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"OpenCV input: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create OpenCV pipeline: {e}")
            return False

    def _opencv_processing_loop(self, output_path: Optional[str], duration: Optional[int]) -> None:
        """OpenCV-based processing loop"""
        start_time = time.time()
        
        # Initialize video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.output_fps,
                (self.output_width, self.output_height)
            )
        
        frame_id = 0
        
        while not self.shutdown_event.is_set():
            ret, frame = self.opencv_cap.read()
            
            if not ret:
                self.logger.info("End of video reached")
                break
            
            # Check duration limit
            if duration and (time.time() - start_time) > duration:
                self.logger.info(f"Duration limit reached ({duration}s)")
                break
            
            try:
                # Resize frame
                if frame.shape[:2] != (self.output_height, self.output_width):
                    frame = cv2.resize(frame, (self.output_width, self.output_height))
                
                # Convert BGR to RGB for YOLO
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection
                detection = self.detector.detect(rgb_frame, frame_id)
                
                # Update analytics
                self.analytics.update(detection)
                
                # Annotate frame
                annotated_frame = self.annotator.annotate(rgb_frame, detection)
                
                # Add info overlay
                annotated_frame = self.annotator.add_info_overlay(
                    annotated_frame,
                    detection_count=len(detection.boxes),
                    frame_id=frame_id
                )
                
                # Convert back to BGR for output
                bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Write output
                if self.video_writer is not None:
                    self.video_writer.write(bgr_frame)
                
                frame_id += 1
                self.processed_count = frame_id
                
                # Progress reporting
                if frame_id % 25 == 0:
                    report = self.analytics.get_report()
                    self.logger.info(
                        f"Processed: {frame_id} frames - "
                        f"Detections: {report['total_detections']} - "
                        f"FPS: {report['processing_fps']:.2f}"
                    )
                
            except Exception as e:
                self.logger.error(f"Processing error at frame {frame_id}: {e}")
                self.error_count += 1
                continue
        
        # Cleanup
        self.opencv_cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            self.logger.info("Video output saved")

    def _on_new_sample(self, sink):
        """Handle new frame from GStreamer"""
        try:
            sample = sink.emit('pull-sample')
            if sample is None:
                return Gst.FlowReturn.ERROR

            buffer = sample.get_buffer()
            caps = sample.get_caps()

            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')

            success, mapinfo = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            try:
                frame_data = np.frombuffer(mapinfo.data, dtype=np.uint8)
                frame = frame_data.reshape((height, width, 3))

                # Queue frame for processing
                if not self.frame_queue.put(frame.copy(), timeout=0.1):
                    self.logger.warning("Frame dropped due to full queue")
                
                self.frame_count += 1
                
                if self.frame_count % 50 == 0:
                    self.logger.info(f"Frames received: {self.frame_count}")

            finally:
                buffer.unmap(mapinfo)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Error in _on_new_sample: {e}")
            return Gst.FlowReturn.ERROR

    def _processing_worker(self, output_path: Optional[str]) -> None:
        """Processing worker thread for GStreamer"""
        thread_id = threading.current_thread().ident
        self.logger.info(f"Processing worker {thread_id} started")
        
        # Initialize video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.output_fps,
                (self.output_width, self.output_height)
            )

        while not self.shutdown_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
                if frame is None:
                    continue

                # Run detection
                detection = self.detector.detect(frame, self.processed_count)
                
                # Update analytics
                self.analytics.update(detection)
                
                # Annotate frame
                annotated_frame = self.annotator.annotate(frame, detection)
                
                # Add info overlay
                annotated_frame = self.annotator.add_info_overlay(
                    annotated_frame,
                    detection_count=len(detection.boxes),
                    frame_id=self.processed_count
                )
                
                # Write output
                if self.video_writer is not None:
                    bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(bgr_frame)

                self.processed_count += 1

                # Progress reporting
                if self.processed_count % 25 == 0:
                    report = self.analytics.get_report()
                    self.logger.info(
                        f"Processed: {self.processed_count} frames - "
                        f"Queue: {self.frame_queue.size()} - "
                        f"FPS: {report['processing_fps']:.2f}"
                    )

                self.frame_queue.task_done()

            except Exception as e:
                self.logger.error(f"Processing worker error: {e}")
                continue

        # Cleanup
        if self.video_writer is not None:
            self.video_writer.release()
            self.logger.info("Video output saved")
        
        self.logger.info(f"Processing worker {thread_id} stopped")

    def _on_error(self, bus, msg):
        """Handle pipeline errors"""
        err, debug = msg.parse_error()
        self.logger.error(f"Pipeline error: {err}")
        self.logger.error(f"Debug info: {debug}")
        self.shutdown_event.set()
        if self.loop:
            self.loop.quit()

    def _on_eos(self, bus, msg):
        """Handle end of stream"""
        self.logger.info("End of stream reached")
        
        # Wait for processing to complete
        while self.frame_queue.size() > 0:
            time.sleep(0.1)
        
        self.shutdown_event.set()
        if self.loop:
            self.loop.quit()

    def start(
        self,
        input_source: str,
        output_path: Optional[str] = None,
        duration: Optional[int] = None
    ) -> bool:
        """
        Start video processing
        
        Args:
            input_source: Input video file path
            output_path: Output video file path
            duration: Duration limit in seconds
            
        Returns:
            Success status
        """
        try:
            self.is_processing = True
            self.shutdown_event.clear()
            self.frame_count = 0
            self.processed_count = 0
            self.error_count = 0
            
            # Choose processing method
            if self.use_opencv_fallback:
                success = self._create_opencv_pipeline(input_source)
                if not success:
                    return False
                
                # Start OpenCV processing
                self._opencv_processing_loop(output_path, duration)
                
            else:
                # Try GStreamer pipeline
                success = self._create_gstreamer_pipeline(input_source)
                if not success:
                    self.logger.warning("GStreamer failed, falling back to OpenCV")
                    self.use_opencv_fallback = True
                    return self.start(input_source, output_path, duration)
                
                # Start processing worker
                worker_thread = threading.Thread(
                    target=self._processing_worker, 
                    args=(output_path,), 
                    daemon=True
                )
                worker_thread.start()
                self.processing_threads_list.append(worker_thread)
                
                # Start pipeline
                ret = self.pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    self.logger.error("Failed to start GStreamer pipeline")
                    return False
                
                self.logger.info("GStreamer pipeline started")
                
                # Handle duration limit
                if duration:
                    def timeout_handler():
                        time.sleep(duration)
                        self.logger.info(f"Duration limit reached ({duration}s)")
                        self.shutdown_event.set()
                        if self.loop:
                            self.loop.quit()
                    
                    timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
                    timeout_thread.start()
                
                # Run main loop
                try:
                    self.loop.run()
                except KeyboardInterrupt:
                    self.logger.info("Interrupted by user")
                    self.shutdown_event.set()

            self.logger.info("Video processing completed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            return False
        finally:
            self.stop()

    def stop(self):
        """Stop the pipeline"""
        self.logger.info("Stopping pipeline...")
        
        self.shutdown_event.set()
        self.is_processing = False
        
        # Stop GStreamer pipeline
        if self.pipeline and not self.use_opencv_fallback:
            self.pipeline.set_state(Gst.State.NULL)
        
        # Stop main loop
        if self.loop and self.loop.is_running():
            self.loop.quit()
        
        # Wait for processing threads
        for thread in self.processing_threads_list:
            thread.join(timeout=5)
        
        # Clean up bus
        if self.bus:
            self.bus.remove_signal_watch()
        
        # Clean up OpenCV components
        if self.use_opencv_fallback:
            if hasattr(self, 'opencv_cap'):
                self.opencv_cap.release()
        
        self.logger.info("Pipeline stopped")