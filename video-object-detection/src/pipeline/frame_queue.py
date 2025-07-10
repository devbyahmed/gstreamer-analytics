"""
Thread-safe frame queue for video processing
"""

import queue
import threading
import time
from typing import Optional, Any
import numpy as np


class ThreadSafeFrameQueue:
    """Thread-safe frame queue with performance tracking"""

    def __init__(self, max_size: int = 1000):
        """
        Initialize frame queue
        
        Args:
            max_size: Maximum queue size
        """
        self.queue = queue.Queue(maxsize=max_size)
        self.max_size = max_size
        self.lock = threading.Lock()
        
        # Statistics
        self.dropped_count = 0
        self.total_put_attempts = 0
        self.total_get_attempts = 0
        self.total_wait_time = 0.0

    def put(self, frame: np.ndarray, timeout: Optional[float] = None) -> bool:
        """
        Add frame to queue
        
        Args:
            frame: Frame to add
            timeout: Timeout in seconds (None for blocking)
            
        Returns:
            True if frame was added, False if dropped
        """
        with self.lock:
            self.total_put_attempts += 1
        
        start_time = time.time()
        
        try:
            if timeout is None:
                self.queue.put(frame)  # Blocking put
            else:
                self.queue.put(frame, timeout=timeout)
            
            # Track wait time
            wait_time = time.time() - start_time
            with self.lock:
                self.total_wait_time += wait_time
            
            return True
            
        except queue.Full:
            with self.lock:
                self.dropped_count += 1
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get frame from queue
        
        Args:
            timeout: Timeout in seconds (None for blocking)
            
        Returns:
            Frame or None if timeout/empty
        """
        with self.lock:
            self.total_get_attempts += 1
        
        start_time = time.time()
        
        try:
            if timeout is None:
                frame = self.queue.get()  # Blocking get
            else:
                frame = self.queue.get(timeout=timeout)
            
            # Track wait time
            wait_time = time.time() - start_time
            with self.lock:
                self.total_wait_time += wait_time
            
            return frame
            
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """Mark task as done"""
        try:
            self.queue.task_done()
        except ValueError:
            # Task done called more times than items put
            pass

    def size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()

    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.queue.full()

    def get_dropped_count(self) -> int:
        """Get number of dropped frames"""
        with self.lock:
            return self.dropped_count

    def get_drop_rate(self) -> float:
        """Get frame drop rate as percentage"""
        with self.lock:
            if self.total_put_attempts == 0:
                return 0.0
            return (self.dropped_count / self.total_put_attempts) * 100.0

    def get_statistics(self) -> dict:
        """
        Get detailed queue statistics
        
        Returns:
            Dictionary with queue statistics
        """
        with self.lock:
            avg_wait_time = (
                self.total_wait_time / max(1, self.total_put_attempts + self.total_get_attempts)
            )
            
            return {
                'current_size': self.size(),
                'max_size': self.max_size,
                'utilization': (self.size() / self.max_size) * 100,
                'dropped_frames': self.dropped_count,
                'total_put_attempts': self.total_put_attempts,
                'total_get_attempts': self.total_get_attempts,
                'drop_rate': self.get_drop_rate(),
                'avg_wait_time': avg_wait_time,
                'total_wait_time': self.total_wait_time
            }

    def clear(self) -> int:
        """
        Clear all frames from queue
        
        Returns:
            Number of frames cleared
        """
        cleared_count = 0
        
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        
        return cleared_count

    def reset_statistics(self) -> None:
        """Reset all statistics"""
        with self.lock:
            self.dropped_count = 0
            self.total_put_attempts = 0
            self.total_get_attempts = 0
            self.total_wait_time = 0.0

    def wait_until_empty(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until queue is empty
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if queue became empty, False if timeout
        """
        start_time = time.time()
        
        while not self.is_empty():
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
        
        return True

    def __len__(self) -> int:
        """Get queue size"""
        return self.size()

    def __bool__(self) -> bool:
        """Check if queue has items"""
        return not self.is_empty()