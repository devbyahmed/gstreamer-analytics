#!/usr/bin/env python3
"""
Basic usage examples for Video Object Detection System
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.main import VideoAnalyticsSystem
from src.utils.config import load_config, get_default_config


def example_webcam_detection():
    """Example: Real-time webcam detection"""
    print("🎥 Example: Webcam Detection")
    print("=" * 40)
    
    try:
        # Load configuration
        config = get_default_config()
        
        # Customize for webcam
        config['webcam']['width'] = 640
        config['webcam']['height'] = 480
        config['webcam']['fps'] = 30
        
        # Create system
        system = VideoAnalyticsSystem(config)
        
        # Run webcam detection for 30 seconds
        print("Starting webcam detection for 30 seconds...")
        print("Press 'q' in the video window to quit early")
        
        success = system.run_webcam_detection(
            device_id=0,
            output_path="outputs/webcam_output.mp4",
            duration=30
        )
        
        if success:
            print("✅ Webcam detection completed!")
            system.display_results()
        else:
            print("❌ Webcam detection failed")
            
    except Exception as e:
        print(f"Error: {e}")


def example_video_processing():
    """Example: Video file processing"""
    print("\n🎬 Example: Video File Processing")
    print("=" * 40)
    
    # Create a sample video file path
    # In real usage, replace with your actual video file
    sample_video = "examples/videos/sample.mp4"
    
    if not os.path.exists(sample_video):
        print(f"⚠️ Sample video not found: {sample_video}")
        print("Please add a video file to examples/videos/ directory")
        return
    
    try:
        # Load configuration
        config = get_default_config()
        
        # Customize for video processing
        config['model']['confidence_threshold'] = 0.6
        config['video']['output_fps'] = 25
        
        # Create system
        system = VideoAnalyticsSystem(config)
        
        # Process video file
        print(f"Processing video: {sample_video}")
        
        success = system.run_file_processing(
            input_path=sample_video,
            output_path="outputs/processed_video.mp4"
        )
        
        if success:
            print("✅ Video processing completed!")
            system.display_results()
        else:
            print("❌ Video processing failed")
            
    except Exception as e:
        print(f"Error: {e}")


def example_custom_configuration():
    """Example: Using custom configuration"""
    print("\n⚙️ Example: Custom Configuration")
    print("=" * 40)
    
    try:
        # Create custom configuration
        custom_config = {
            "model": {
                "path": "yolov8s.pt",  # Use larger model
                "confidence_threshold": 0.7,
                "device": "cpu"
            },
            "video": {
                "output_width": 1920,
                "output_height": 1080,
                "output_fps": 30
            },
            "analytics": {
                "enable_tracking": True,
                "enable_visualization": True,
                "analytics_interval": 50
            },
            "visualization": {
                "show_confidence": True,
                "show_class_names": True,
                "box_thickness": 3,
                "text_size": 0.8
            }
        }
        
        print("Custom configuration created:")
        print(f"  Model: {custom_config['model']['path']}")
        print(f"  Confidence: {custom_config['model']['confidence_threshold']}")
        print(f"  Resolution: {custom_config['video']['output_width']}x{custom_config['video']['output_height']}")
        
        # You can use this config with the system
        # system = VideoAnalyticsSystem(custom_config)
        
        print("✅ Custom configuration ready for use")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example: Batch processing multiple videos"""
    print("\n📁 Example: Batch Processing")
    print("=" * 40)
    
    try:
        # List of video files to process
        video_files = [
            "examples/videos/video1.mp4",
            "examples/videos/video2.mp4",
            "examples/videos/video3.mp4"
        ]
        
        # Filter existing files
        existing_files = [f for f in video_files if os.path.exists(f)]
        
        if not existing_files:
            print("⚠️ No video files found in examples/videos/")
            print("Add some video files to test batch processing")
            return
        
        # Load configuration
        config = get_default_config()
        config['model']['confidence_threshold'] = 0.5
        
        # Process each video
        for i, video_path in enumerate(existing_files):
            print(f"\nProcessing video {i+1}/{len(existing_files)}: {video_path}")
            
            # Create system for each video (fresh analytics)
            system = VideoAnalyticsSystem(config)
            
            output_path = f"outputs/batch_output_{i+1}.mp4"
            
            success = system.run_file_processing(
                input_path=video_path,
                output_path=output_path
            )
            
            if success:
                print(f"✅ Video {i+1} processed successfully")
                
                # Get analytics for this video
                report = system.get_analytics_report()
                print(f"   Frames: {report['total_frames']}")
                print(f"   Detections: {report['total_detections']}")
                print(f"   FPS: {report['processing_fps']:.2f}")
            else:
                print(f"❌ Video {i+1} failed")
        
        print("\n✅ Batch processing completed!")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function to run examples"""
    print("🚀 Video Object Detection - Usage Examples")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("examples/videos", exist_ok=True)
    
    # Run examples
    print("\nAvailable examples:")
    print("1. Webcam Detection")
    print("2. Video File Processing")
    print("3. Custom Configuration")
    print("4. Batch Processing")
    
    choice = input("\nSelect example (1-4) or 'all' to run all: ").strip().lower()
    
    if choice == '1' or choice == 'all':
        example_webcam_detection()
    
    if choice == '2' or choice == 'all':
        example_video_processing()
    
    if choice == '3' or choice == 'all':
        example_custom_configuration()
    
    if choice == '4' or choice == 'all':
        example_batch_processing()
    
    print("\n🎉 Examples completed!")
    print("\nNext steps:")
    print("- Add your own video files to examples/videos/")
    print("- Customize configuration in config/config.yaml")
    print("- Run: python main.py --help for command-line options")


if __name__ == "__main__":
    main()