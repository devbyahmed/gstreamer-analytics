#!/usr/bin/env python3
"""
Video Object Detection System
Main entry point for webcam and video file processing
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.main import VideoAnalyticsSystem
from src.utils.logger import setup_logger
from src.utils.config import load_config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Video Object Detection System with GStreamer and YOLO"
    )
    
    # Source type
    parser.add_argument(
        "--source",
        choices=["webcam", "video"],
        required=True,
        help="Input source type"
    )
    
    # Video file input
    parser.add_argument(
        "--input",
        type=str,
        help="Input video file path (required for video source)"
    )
    
    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        help="Output video file path"
    )
    
    # Duration
    parser.add_argument(
        "--duration",
        type=int,
        help="Processing duration in seconds (optional)"
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration file path"
    )
    
    # Webcam device
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Webcam device ID (default: 0)"
    )
    
    # Model path
    parser.add_argument(
        "--model",
        type=str,
        help="YOLO model path (overrides config)"
    )
    
    # Confidence threshold
    parser.add_argument(
        "--confidence",
        type=float,
        help="Confidence threshold (overrides config)"
    )
    
    # Verbose mode
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source == "video" and not args.input:
        parser.error("--input is required when using video source")
    
    if args.input and not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    # Setup logging
    logger = setup_logger(
        "VideoDetection",
        level="DEBUG" if args.verbose else "INFO"
    )

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Override config with command line arguments
        if args.model:
            config["model"]["path"] = args.model
        if args.confidence:
            config["model"]["confidence_threshold"] = args.confidence

        # Create system
        system = VideoAnalyticsSystem(config)

        # Run based on source type
        if args.source == "webcam":
            logger.info("Starting webcam detection...")
            success = system.run_webcam_detection(
                device_id=args.device,
                output_path=args.output,
                duration=args.duration
            )
        else:  # video
            logger.info(f"Starting video file processing: {args.input}")
            success = system.run_file_processing(
                input_path=args.input,
                output_path=args.output or "output_processed.mp4",
                duration=args.duration
            )

        if success:
            logger.info("✅ Processing completed successfully!")
            
            # Display results
            system.display_results()
            
        else:
            logger.error("❌ Processing failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()