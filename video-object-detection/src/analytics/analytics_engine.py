"""
Analytics engine for video object detection
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from ..detection.yolo_detector import DetectionResult


@dataclass
class AnalyticsData:
    """Data structure for analytics information"""
    total_frames: int = 0
    total_detections: int = 0
    object_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    processing_fps: float = 0.0
    start_time: float = field(default_factory=time.time)
    frame_times: List[float] = field(default_factory=list)
    dropped_frames: int = 0
    skipped_frames: int = 0
    detection_history: List[int] = field(default_factory=list)
    fps_history: List[float] = field(default_factory=list)


class AnalyticsEngine:
    """Analytics processing engine for object detection"""

    def __init__(self, class_names: Dict[int, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize analytics engine
        
        Args:
            class_names: Dictionary mapping class IDs to names
            config: Configuration dictionary
        """
        self.class_names = class_names
        self.config = config or {}
        self.analytics = AnalyticsData()
        self.lock = threading.Lock()
        
        # Configuration
        self.enable_tracking = self.config.get('enable_tracking', True)
        self.enable_visualization = self.config.get('enable_visualization', True)
        self.save_analytics = self.config.get('save_analytics', True)

    def update(
        self,
        detection: DetectionResult,
        dropped_frames: int = 0,
        skipped_frames: int = 0
    ) -> None:
        """
        Update analytics with new detection
        
        Args:
            detection: Detection results
            dropped_frames: Number of dropped frames
            skipped_frames: Number of skipped frames
        """
        if not self.enable_tracking:
            return

        with self.lock:
            self.analytics.total_frames += 1
            self.analytics.total_detections += len(detection.boxes)
            self.analytics.frame_times.append(detection.timestamp)
            self.analytics.dropped_frames += dropped_frames
            self.analytics.skipped_frames += skipped_frames
            
            # Track detections per frame
            self.analytics.detection_history.append(len(detection.boxes))

            # Update class counts
            for class_id in detection.classes:
                class_name = self.class_names.get(int(class_id), "Unknown")
                self.analytics.object_counts[class_name] += 1

            # Calculate processing FPS
            if self.analytics.total_frames > 1:
                elapsed = time.time() - self.analytics.start_time
                self.analytics.processing_fps = self.analytics.total_frames / elapsed
                
                # Track FPS history (keep last 100 samples)
                self.analytics.fps_history.append(self.analytics.processing_fps)
                if len(self.analytics.fps_history) > 100:
                    self.analytics.fps_history = self.analytics.fps_history[-100:]

    def get_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report
        
        Returns:
            Dictionary containing analytics data
        """
        with self.lock:
            elapsed_time = time.time() - self.analytics.start_time
            
            # Calculate statistics
            avg_detections_per_frame = (
                self.analytics.total_detections / self.analytics.total_frames
                if self.analytics.total_frames > 0 else 0
            )
            
            most_detected = max(
                self.analytics.object_counts.items(),
                key=lambda x: x[1],
                default=("None", 0)
            )
            
            # Frame rate statistics
            fps_stats = {}
            if self.analytics.fps_history:
                fps_stats = {
                    'avg_fps': np.mean(self.analytics.fps_history),
                    'min_fps': np.min(self.analytics.fps_history),
                    'max_fps': np.max(self.analytics.fps_history),
                    'std_fps': np.std(self.analytics.fps_history)
                }
            
            # Detection statistics
            detection_stats = {}
            if self.analytics.detection_history:
                detection_stats = {
                    'avg_detections': np.mean(self.analytics.detection_history),
                    'min_detections': np.min(self.analytics.detection_history),
                    'max_detections': np.max(self.analytics.detection_history),
                    'std_detections': np.std(self.analytics.detection_history)
                }

            return {
                "total_frames": self.analytics.total_frames,
                "total_detections": self.analytics.total_detections,
                "processing_fps": self.analytics.processing_fps,
                "elapsed_time": elapsed_time,
                "dropped_frames": self.analytics.dropped_frames,
                "skipped_frames": self.analytics.skipped_frames,
                "avg_detections_per_frame": avg_detections_per_frame,
                "object_counts": dict(self.analytics.object_counts),
                "most_detected_object": most_detected,
                "fps_stats": fps_stats,
                "detection_stats": detection_stats,
                "efficiency": {
                    "drop_rate": (self.analytics.dropped_frames / max(1, self.analytics.total_frames)) * 100,
                    "skip_rate": (self.analytics.skipped_frames / max(1, self.analytics.total_frames)) * 100
                }
            }

    def create_visualization(self, save_path: str = "analytics_chart.png") -> None:
        """
        Create comprehensive analytics visualization
        
        Args:
            save_path: Path to save the visualization
        """
        if not self.enable_visualization:
            return

        with self.lock:
            if not self.analytics.object_counts:
                print("No data available for visualization")
                return

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Video Object Detection Analytics', fontsize=16, fontweight='bold')

            # 1. Object Detection Bar Chart
            ax1 = axes[0, 0]
            objects = list(self.analytics.object_counts.keys())
            counts = list(self.analytics.object_counts.values())

            bars = ax1.bar(objects, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            ax1.set_title('Object Detection Summary', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Object Class')
            ax1.set_ylabel('Number of Detections')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')

            # 2. FPS Over Time
            ax2 = axes[0, 1]
            if len(self.analytics.fps_history) > 1:
                ax2.plot(self.analytics.fps_history, color='green', alpha=0.7, linewidth=2)
                ax2.fill_between(range(len(self.analytics.fps_history)), 
                               self.analytics.fps_history, alpha=0.3, color='green')
                ax2.set_title('Processing FPS Over Time')
                ax2.set_xlabel('Sample Number')
                ax2.set_ylabel('FPS')
                ax2.grid(True, alpha=0.3)
                
                # Add average line
                avg_fps = np.mean(self.analytics.fps_history)
                ax2.axhline(y=avg_fps, color='red', linestyle='--', 
                           label=f'Average: {avg_fps:.1f} FPS')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'Insufficient FPS data', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Processing FPS Over Time')

            # 3. Object Distribution Pie Chart
            ax3 = axes[1, 0]
            if len(objects) > 0:
                # Create pie chart with custom colors
                colors = plt.cm.Set3(np.linspace(0, 1, len(objects)))
                wedges, texts, autotexts = ax3.pie(counts, labels=objects, autopct='%1.1f%%', 
                                                  startangle=90, colors=colors)
                ax3.set_title('Object Distribution')
                
                # Enhance text appearance
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                ax3.text(0.5, 0.5, 'No objects detected', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Object Distribution')

            # 4. Detection Statistics and Summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            report = self.get_report()
            
            # Create summary text
            summary_text = f"""
ANALYTICS SUMMARY
================
Total Frames: {report['total_frames']:,}
Total Detections: {report['total_detections']:,}
Dropped Frames: {report['dropped_frames']:,}
Skipped Frames: {report['skipped_frames']:,}

PERFORMANCE METRICS
==================
Processing FPS: {report['processing_fps']:.2f}
Avg Detections/Frame: {report['avg_detections_per_frame']:.2f}
Processing Time: {report['elapsed_time']:.2f}s

EFFICIENCY
==========
Drop Rate: {report['efficiency']['drop_rate']:.2f}%
Skip Rate: {report['efficiency']['skip_rate']:.2f}%

TOP DETECTION
=============
Most Detected: {report['most_detected_object'][0]}
Count: {report['most_detected_object'][1]:,}
            """

            # Add FPS statistics if available
            if 'fps_stats' in report and report['fps_stats']:
                fps_stats = report['fps_stats']
                summary_text += f"""

FPS STATISTICS
==============
Average: {fps_stats['avg_fps']:.2f}
Min: {fps_stats['min_fps']:.2f}
Max: {fps_stats['max_fps']:.2f}
Std Dev: {fps_stats['std_fps']:.2f}
"""

            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

            # Adjust layout and save
            plt.tight_layout()
            
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.show()

    def save_report(self, save_path: str = "analytics_report.json") -> None:
        """
        Save analytics report to JSON file
        
        Args:
            save_path: Path to save the report
        """
        if not self.save_analytics:
            return
            
        import json
        
        report = self.get_report()
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def reset(self) -> None:
        """Reset all analytics data"""
        with self.lock:
            self.analytics = AnalyticsData()

    def get_real_time_stats(self) -> Dict[str, Any]:
        """
        Get real-time statistics for live monitoring
        
        Returns:
            Dictionary with current statistics
        """
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.analytics.start_time
            
            return {
                'frames_processed': self.analytics.total_frames,
                'detections_current': len(self.analytics.detection_history[-1:]),
                'fps_current': self.analytics.fps_history[-1] if self.analytics.fps_history else 0,
                'fps_average': np.mean(self.analytics.fps_history) if self.analytics.fps_history else 0,
                'elapsed_time': elapsed,
                'total_detections': self.analytics.total_detections
            }