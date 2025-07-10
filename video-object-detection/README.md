# Video Object Detection with GStreamer

A real-time video object detection system using GStreamer, YOLO, and OpenCV. This project supports both video file processing and live webcam detection with comprehensive analytics.

## Features

- 🎥 **Video File Processing**: Process video files with complete frame analysis
- 📹 **Live Webcam Detection**: Real-time object detection from webcam feed
- 📊 **Analytics Dashboard**: Comprehensive detection statistics and visualizations
- 🎯 **YOLO Integration**: State-of-the-art object detection using YOLOv8
- 🔄 **GStreamer Pipeline**: Optimized video processing pipeline
- 📈 **Performance Metrics**: FPS tracking, detection counts, and processing statistics

## Quick Start

### 1. Install Dependencies
```bash
# For GitHub Codespaces (automatically handled)
./scripts/install_dependencies.sh

# For local development
pip install -r requirements.txt
```

### 2. Run Webcam Detection
```bash
python main.py --source webcam --duration 30
```

### 3. Process Video File
```bash
python main.py --source video --input path/to/video.mp4 --output processed_video.mp4
```

## Project Structure

```
video-object-detection/
├── src/                    # Source code
│   ├── __init__.py
│   ├── pipeline/          # GStreamer pipeline components
│   ├── detection/         # YOLO detection modules
│   ├── analytics/         # Analytics and visualization
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── scripts/               # Setup and utility scripts
├── examples/              # Example usage
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── .devcontainer/         # Codespaces configuration
└── README.md
```

## Usage Examples

### Basic Webcam Detection
```python
from src.main import VideoAnalyticsSystem

system = VideoAnalyticsSystem()
system.run_webcam_detection(duration=60)
```

### Video File Processing
```python
system = VideoAnalyticsSystem()
system.run_file_processing("input.mp4", "output.mp4")
```

## Configuration

Edit `config/config.yaml` to customize:
- Model settings (YOLO model, confidence threshold)
- Video output settings (resolution, FPS)
- Processing parameters (buffer sizes, threading)
- Analytics options

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- GStreamer 1.0+
- YOLOv8 (ultralytics)

## Compatibility

- ✅ Linux (Ubuntu 20.04+)
- ✅ GitHub Codespaces
- ✅ Google Colab
- ⚠️ Windows (requires GStreamer installation)
- ⚠️ macOS (requires GStreamer installation)

## License

MIT License - see LICENSE file for details.