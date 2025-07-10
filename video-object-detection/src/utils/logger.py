"""
Logging utilities for video object detection system
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "VideoDetection",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logger with console and file output
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        console_output: Enable console output
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ProgressLogger:
    """Progress logging utility"""
    
    def __init__(self, logger: logging.Logger, total: int, interval: int = 25):
        self.logger = logger
        self.total = total
        self.interval = interval
        self.current = 0
    
    def update(self, count: int = 1) -> None:
        """Update progress counter"""
        self.current += count
        
        if self.current % self.interval == 0 or self.current == self.total:
            percentage = (self.current / self.total) * 100 if self.total > 0 else 0
            self.logger.info(f"Progress: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self) -> None:
        """Mark progress as complete"""
        if self.current != self.total:
            self.current = self.total
            self.logger.info(f"Progress: {self.current}/{self.total} (100.0%) - Complete!")


class TimedLogger:
    """Timer-based logging utility"""
    
    def __init__(self, logger: logging.Logger, name: str):
        self.logger = logger
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            if exc_type is None:
                self.logger.info(f"Completed {self.name} in {elapsed:.2f}s")
            else:
                self.logger.error(f"Failed {self.name} after {elapsed:.2f}s")


def log_system_info(logger: logging.Logger) -> None:
    """Log system information"""
    import platform
    import psutil
    import torch
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  CPU: {platform.processor()}")
    logger.info(f"  CPU Cores: {psutil.cpu_count()}")
    logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA Device: {torch.cuda.get_device_name()}")
        logger.info(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")