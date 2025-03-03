"""
Command-line interface for the vehicle tracker.
"""
import argparse
from vehicle_tracker.config.settings import DEFAULT_SETTINGS

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Vehicle detection, tracking, counting, and speed estimation")
    
    parser.add_argument(
        "--source_video_path", 
        required=True, 
        help="Path to the source video file", 
        type=str
    )
    
    parser.add_argument(
        "--output_video_path", 
        help="Path to save the output video (optional)", 
        type=str
    )
    
    parser.add_argument(
        "--vehicle_classes", 
        nargs='+', 
        type=int, 
        default=DEFAULT_SETTINGS["vehicle_classes"],
        help="List of vehicle class IDs to detect (2:car, 3:motorcycle, 5:bus, 7:truck)"
    )
    
    parser.add_argument(
        "--line_position", 
        type=float, 
        default=DEFAULT_SETTINGS["line_position"],
        help="Position of the counting line as a fraction of frame height (0-1)"
    )
    
    parser.add_argument(
        "--display_scale", 
        type=float, 
        default=DEFAULT_SETTINGS["display_scale"],
        help="Scale factor for display window (0-1)"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=DEFAULT_SETTINGS["model_path"],
        help="Path to the YOLOv8 model file"
    )
    
    parser.add_argument(
        "--save_video", 
        action="store_true",
        help="Save the processed video"
    )
    
    parser.add_argument(
        "--confidence_threshold", 
        type=float, 
        default=DEFAULT_SETTINGS.get("confidence_threshold", 0.3),
        help="Confidence threshold for detections (0-1)"
    )
    
    parser.add_argument(
        "--enable_traces", 
        action="store_true",
        help="Enable movement traces visualization"
    )
    
    return parser.parse_args() 