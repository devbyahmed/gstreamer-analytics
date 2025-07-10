"""
Configuration management utilities
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories if they don't exist
    for dir_key in ['output_directory', 'temp_directory', 'log_directory']:
        if 'output' in config and dir_key in config['output']:
            os.makedirs(config['output'][dir_key], exist_ok=True)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        "model": {
            "path": "yolov8n.pt",
            "confidence_threshold": 0.5,
            "device": "cpu",
            "half_precision": False
        },
        "video": {
            "output_width": 1280,
            "output_height": 720,
            "output_fps": 25,
            "codec": "mp4v"
        },
        "webcam": {
            "device_id": 0,
            "width": 1280,
            "height": 720,
            "fps": 30
        },
        "pipeline": {
            "max_queue_size": 1000,
            "processing_threads": 1,
            "buffer_size": 500,
            "drop_frames": False,
            "frame_skip": 1,
            "max_processing_fps": 1000.0
        },
        "analytics": {
            "enable_tracking": True,
            "enable_visualization": True,
            "save_analytics": True,
            "analytics_interval": 25
        },
        "output": {
            "save_video": True,
            "save_analytics": True,
            "output_directory": "outputs",
            "temp_directory": "temp",
            "log_directory": "logs"
        },
        "visualization": {
            "show_confidence": True,
            "show_class_names": True,
            "box_thickness": 2,
            "text_size": 0.6,
            "colors_seed": 42
        },
        "performance": {
            "enable_gpu_acceleration": False,
            "optimize_model": True,
            "benchmark_mode": False
        },
        "logging": {
            "level": "INFO",
            "console_output": True,
            "file_output": True,
            "log_file": "logs/detection.log"
        }
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)