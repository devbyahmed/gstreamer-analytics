#!/bin/bash

echo "🚀 Setting up Video Object Detection Environment..."

# Update system packages
sudo apt-get update

# Install GStreamer and related packages
echo "📦 Installing GStreamer..."
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
    libx264-dev

# Install Python dependencies
echo "📚 Installing Python packages..."
cd /workspaces/gstreamer-analytics/video-object-detection
pip install --upgrade pip
pip install -r requirements.txt

# Install PyGObject manually (sometimes needed)
pip install PyGObject

# Make scripts executable
chmod +x scripts/*.sh

# Test GStreamer installation
echo "🧪 Testing GStreamer installation..."
gst-inspect-1.0 --version

# Test camera access (if available)
echo "📹 Checking camera devices..."
ls -la /dev/video* 2>/dev/null || echo "No camera devices found (normal in Codespaces)"

# Create necessary directories
mkdir -p outputs
mkdir -p temp
mkdir -p logs

echo "✅ Setup complete! Ready for video object detection."
echo ""
echo "Quick start:"
echo "  python main.py --source webcam --duration 30"
echo "  python main.py --source video --input sample.mp4"