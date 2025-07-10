#!/usr/bin/env python3
"""
System Test Script for Video Object Detection
Tests all components and dependencies
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append("src")

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    failed_imports = []
    
    # Test standard libraries
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        failed_imports.append(f"OpenCV: {e}")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        failed_imports.append(f"PyTorch: {e}")
    
    try:
        import ultralytics
        print(f"✅ Ultralytics")
    except ImportError as e:
        failed_imports.append(f"Ultralytics: {e}")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        failed_imports.append(f"NumPy: {e}")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        failed_imports.append(f"Matplotlib: {e}")
    
    try:
        import yaml
        print(f"✅ PyYAML")
    except ImportError as e:
        failed_imports.append(f"PyYAML: {e}")
    
    # Test GStreamer (optional)
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
        print("✅ GStreamer bindings")
    except Exception as e:
        print(f"⚠️ GStreamer: {e} (will use OpenCV fallback)")
    
    return failed_imports

def test_components():
    """Test system components"""
    print("\n🔧 Testing components...")
    
    try:
        # Test configuration
        from src.utils.config import get_default_config, load_config
        config = get_default_config()
        print("✅ Configuration system")
    except Exception as e:
        print(f"❌ Configuration: {e}")
        return False
    
    try:
        # Test logger
        from src.utils.logger import setup_logger
        logger = setup_logger("test")
        print("✅ Logging system")
    except Exception as e:
        print(f"❌ Logging: {e}")
        return False
    
    try:
        # Test YOLO detector
        from src.detection.yolo_detector import YOLODetector
        detector = YOLODetector("yolov8n.pt", 0.5)
        print("✅ YOLO detector")
    except Exception as e:
        print(f"❌ YOLO detector: {e}")
        return False
    
    try:
        # Test frame annotator
        from src.detection.frame_annotator import FrameAnnotator
        annotator = FrameAnnotator(detector.class_names)
        print("✅ Frame annotator")
    except Exception as e:
        print(f"❌ Frame annotator: {e}")
        return False
    
    try:
        # Test analytics engine
        from src.analytics.analytics_engine import AnalyticsEngine
        analytics = AnalyticsEngine(detector.class_names)
        print("✅ Analytics engine")
    except Exception as e:
        print(f"❌ Analytics engine: {e}")
        return False
    
    try:
        # Test frame queue
        from src.pipeline.frame_queue import ThreadSafeFrameQueue
        queue = ThreadSafeFrameQueue(10)
        print("✅ Frame queue")
    except Exception as e:
        print(f"❌ Frame queue: {e}")
        return False
    
    return True

def test_detection():
    """Test object detection on dummy frame"""
    print("\n🎯 Testing detection...")
    
    try:
        from src.detection.yolo_detector import YOLODetector
        from src.detection.frame_annotator import FrameAnnotator
        
        # Create detector
        detector = YOLODetector("yolov8n.pt", 0.5)
        annotator = FrameAnnotator(detector.class_names)
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run detection
        detection = detector.detect(dummy_frame, 0)
        
        # Annotate frame
        annotated = annotator.annotate(dummy_frame, detection)
        
        print(f"✅ Detection test completed")
        print(f"   Input shape: {dummy_frame.shape}")
        print(f"   Output shape: {annotated.shape}")
        print(f"   Detections: {len(detection.boxes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def test_analytics():
    """Test analytics system"""
    print("\n📊 Testing analytics...")
    
    try:
        from src.analytics.analytics_engine import AnalyticsEngine
        from src.detection.yolo_detector import YOLODetector, DetectionResult
        
        # Create components
        detector = YOLODetector("yolov8n.pt", 0.5)
        analytics = AnalyticsEngine(detector.class_names)
        
        # Create dummy detection result
        detection = DetectionResult(
            boxes=np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
            scores=np.array([0.8, 0.9]),
            classes=np.array([0, 2]),
            frame_id=0,
            timestamp=1234567890.0
        )
        
        # Update analytics
        analytics.update(detection)
        
        # Get report
        report = analytics.get_report()
        
        print(f"✅ Analytics test completed")
        print(f"   Total frames: {report['total_frames']}")
        print(f"   Total detections: {report['total_detections']}")
        print(f"   Object counts: {report['object_counts']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analytics test failed: {e}")
        return False

def test_system_integration():
    """Test full system integration"""
    print("\n🔗 Testing system integration...")
    
    try:
        from src.main import VideoAnalyticsSystem
        from src.utils.config import get_default_config
        
        # Create system
        config = get_default_config()
        system = VideoAnalyticsSystem(config)
        
        print("✅ System integration test completed")
        print("   System created successfully")
        print("   All components initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def test_environment():
    """Test environment setup"""
    print("\n🌍 Testing environment...")
    
    # Check directories
    required_dirs = ["outputs", "temp", "logs", "config"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory: {dir_name}")
        else:
            print(f"⚠️ Directory missing: {dir_name} (will be created)")
            os.makedirs(dir_name, exist_ok=True)
    
    # Check configuration file
    if os.path.exists("config/config.yaml"):
        print("✅ Configuration file")
    else:
        print("⚠️ Configuration file missing")
    
    # Check camera devices (optional)
    camera_devices = list(Path("/dev").glob("video*"))
    if camera_devices:
        print(f"✅ Camera devices found: {len(camera_devices)}")
    else:
        print("ℹ️ No camera devices found (normal in cloud environments)")
    
    return True

def main():
    """Main test function"""
    print("🚀 Video Object Detection System - Comprehensive Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test imports
    failed_imports = test_imports()
    if failed_imports:
        print("\n❌ Failed imports:")
        for fail in failed_imports:
            print(f"   {fail}")
        all_tests_passed = False
    else:
        print("\n✅ All imports successful")
    
    # Test environment
    if not test_environment():
        all_tests_passed = False
    
    # Test components
    if not test_components():
        all_tests_passed = False
    
    # Test detection
    if not test_detection():
        all_tests_passed = False
    
    # Test analytics
    if not test_analytics():
        all_tests_passed = False
    
    # Test system integration
    if not test_system_integration():
        all_tests_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe system is ready to use:")
        print("  python main.py --source webcam --duration 10")
        print("  python main.py --source video --input path/to/video.mp4")
        print("  python examples/basic_usage.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease fix the issues before using the system.")
        print("Run: ./scripts/install_dependencies.sh")
    
    print("\n📚 Documentation:")
    print("  - README.md for general information")
    print("  - config/config.yaml for configuration options")
    print("  - examples/ for usage examples")

if __name__ == "__main__":
    main()