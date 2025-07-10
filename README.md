# GStreamer Analytics

A comprehensive collection of video analytics projects using GStreamer, YOLO, and OpenCV for real-time video processing and object detection.

## Projects

### 🎥 Video Object Detection System

A professional-grade video object detection system that supports both real-time webcam processing and video file analysis using GStreamer and YOLO.

**Location:** `video-object-detection/`

**Features:**
- 📹 **Real-time Webcam Detection**: Live object detection from webcam feeds
- 🎬 **Video File Processing**: Complete analysis of video files with frame-by-frame processing
- 🎯 **YOLO Integration**: State-of-the-art object detection using YOLOv8
- 📊 **Comprehensive Analytics**: Detailed statistics, visualizations, and performance metrics
- 🔄 **GStreamer Pipeline**: Optimized video processing with OpenCV fallback
- ⚙️ **Configurable**: Extensive configuration options via YAML files
- 🚀 **GitHub Codespaces Ready**: Optimized for cloud development environments

**Quick Start:**
```bash
cd video-object-detection
./scripts/install_dependencies.sh
python main.py --source webcam --duration 30
python main.py --source video --input path/to/video.mp4
```

**Key Components:**
- Modular architecture with separate detection, analytics, and pipeline modules
- Thread-safe frame processing with configurable buffer sizes
- Comprehensive logging and error handling
- Professional visualization and reporting system

## Development Environment

This repository is optimized for GitHub Codespaces with automatic dependency installation and environment setup.

### GitHub Codespaces Setup

1. Open this repository in GitHub Codespaces
2. The environment will automatically install all dependencies
3. Navigate to `video-object-detection/` and start using the system

### Local Development

For local development, ensure you have:
- Python 3.8+
- GStreamer 1.0+ (with development packages)
- OpenCV
- CUDA (optional, for GPU acceleration)

## System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), Windows (with GStreamer), macOS (with GStreamer)
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB+ recommended
- **Storage**: 2GB for dependencies and models

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [GStreamer](https://gstreamer.freedesktop.org/) for video processing
- [OpenCV](https://opencv.org/) for computer vision utilities

## Support

For questions and support:
- Open an issue in this repository
- Check the documentation in each project folder
- Review the examples in `video-object-detection/examples/`