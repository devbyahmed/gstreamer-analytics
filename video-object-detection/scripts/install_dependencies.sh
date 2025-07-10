#!/bin/bash

echo "🚀 Installing Video Object Detection Dependencies"
echo "================================================"

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update

# Install GStreamer and related packages
echo "🎥 Installing GStreamer..."
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev

# Install Python GObject bindings
echo "🐍 Installing Python GObject bindings..."
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0

# Install system dependencies for OpenCV and other packages
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    python3-dev \
    python3-pip \
    build-essential

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install PyGObject manually (sometimes needed)
pip install PyGObject

# Create directories
echo "📁 Creating directories..."
mkdir -p outputs
mkdir -p temp
mkdir -p logs
mkdir -p examples/videos

# Test installations
echo "🧪 Testing installations..."

# Test GStreamer
if command -v gst-inspect-1.0 &> /dev/null; then
    echo "✅ GStreamer installed successfully"
    gst-inspect-1.0 --version
else
    echo "❌ GStreamer installation failed"
fi

# Test Python packages
python3 -c "
import sys
failed = []
packages = ['cv2', 'torch', 'ultralytics', 'numpy', 'matplotlib', 'yaml']

for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        failed.append(pkg)

if failed:
    print(f'Failed packages: {failed}')
    sys.exit(1)
else:
    print('✅ All Python packages installed successfully')
"

# Test GStreamer Python bindings
python3 -c "
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    print('✅ GStreamer Python bindings working')
except Exception as e:
    print(f'⚠️ GStreamer Python bindings issue: {e}')
    print('Will use OpenCV fallback')
"

# Check camera devices
echo "📹 Checking camera devices..."
if ls /dev/video* 2>/dev/null; then
    echo "✅ Camera devices found"
else
    echo "ℹ️ No camera devices found (normal in cloud environments)"
fi

# Download sample model if needed
echo "🤖 Checking YOLO model..."
python3 -c "
from ultralytics import YOLO
try:
    model = YOLO('yolov8n.pt')
    print('✅ YOLO model ready')
except Exception as e:
    print(f'⚠️ YOLO model issue: {e}')
"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Quick test commands:"
echo "  python main.py --source webcam --duration 10"
echo "  python main.py --source video --input path/to/video.mp4"
echo ""
echo "Note: In GitHub Codespaces, webcam may not be available."
echo "Use video files for testing in cloud environments."