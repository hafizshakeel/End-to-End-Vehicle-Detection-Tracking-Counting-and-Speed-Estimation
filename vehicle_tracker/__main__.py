"""
Main entry point for the vehicle tracker application.
"""
import cv2
import supervision as sv
import numpy as np

from vehicle_tracker.core.detector import VehicleDetector
from vehicle_tracker.core.tracker import VehicleTracker
from vehicle_tracker.core.counter import VehicleCounter
from vehicle_tracker.core.speed_estimator import ViewTransformer, calculate_speed
from vehicle_tracker.visualization.annotator import FrameAnnotator
from vehicle_tracker.utils.video import get_video_info, get_frame_generator, get_first_frame, create_video_writer
from vehicle_tracker.utils.cli import parse_arguments
from vehicle_tracker.config.settings import SOURCE, TARGET

def main():
    """Main function to run the vehicle detection and tracking pipeline."""
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize video info
    video_info = get_video_info(args.source_video_path)
    
    # Get first frame to determine dimensions
    first_frame = get_first_frame(args.source_video_path)
    frame_height, frame_width = first_frame.shape[:2]
    
    # Initialize components
    detector = VehicleDetector(model_path=args.model_path, vehicle_classes=args.vehicle_classes)
    tracker = VehicleTracker(fps=video_info.fps)
    counter = VehicleCounter(frame_width, frame_height, line_position=args.line_position)
    annotator = FrameAnnotator(video_info)
    
    # Set up zones
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    
    # Initialize video writer if saving is enabled
    video_writer = None
    if args.save_video:
        output_path = args.output_video_path if args.output_video_path else "output.mp4"
        video_writer = create_video_writer(output_path, frame_width, frame_height, video_info.fps)
    
    # Process each frame
    frame_generator = get_frame_generator(args.source_video_path)
    for frame in frame_generator:
        # Detect vehicles with confidence threshold
        result = detector.model(frame, conf=args.confidence_threshold)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detector.filter_by_class(detections)
        
        # Filter by region of interest
        detections_in_zone = detections[polygon_zone.trigger(detections)]
        
        if len(detections_in_zone) == 0:
            # No detections, just annotate the counting line
            annotated_frame = frame.copy()
            annotated_frame = annotator.annotate_frame(
                frame=annotated_frame,
                detections=[],
                line_counter=counter.line_zone
            )
        else:
            # Track detections
            tracked_detections = tracker.update(detections_in_zone)
            
            # Update the counter with the tracked detections
            counter.update(tracked_detections)
            
            # Transform points for speed estimation
            try:
                points = tracked_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                transformed_points = view_transformer.transform_points(points=points).astype(int)
                
                # Create labels with class and speed information
                labels = []
                for idx, (tracker_id, class_id) in enumerate(zip(tracked_detections.tracker_id, tracked_detections.class_id)):
                    # Create label
                    class_name = detector.get_class_name(class_id)
                    
                    # Store coordinates for speed calculation
                    y = transformed_points[idx][1]
                    tracker.store_coordinates(tracker_id, y)
                    
                    # Calculate speed
                    speed = calculate_speed(tracker.get_coordinates(tracker_id), video_info.fps)
                    
                    if speed is None:
                        labels.append(f"#{tracker_id} {class_name}")
                    else:
                        labels.append(f"#{tracker_id} {class_name}: {speed} km/h")
            except Exception as e:
                # If there's an error in point transformation, just use class names as labels
                labels = [f"#{tracker_id} {detector.get_class_name(class_id)}" 
                          for tracker_id, class_id in zip(tracked_detections.tracker_id, tracked_detections.class_id)]
                print(f"Error in point transformation: {str(e)}")
            
            # Annotate the frame
            annotated_frame = annotator.annotate_frame(
                frame=frame,
                detections=tracked_detections,
                labels=labels,
                line_counter=counter.line_zone,
                enable_traces=args.enable_traces
            )
        
        # Display the frame
        if args.display_scale != 1.0:
            display_width = int(frame_width * args.display_scale)
            display_height = int(frame_height * args.display_scale)
            display_frame = cv2.resize(annotated_frame, (display_width, display_height))
        else:
            display_frame = annotated_frame
            
        cv2.imshow("Vehicle Detection and Tracking", display_frame)
        
        # Write the frame to the output video if saving is enabled
        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\nVehicle Detection and Tracking Summary:")
    print(f"Total vehicles in: {counter.in_count}")
    print(f"Total vehicles out: {counter.out_count}")
    print(f"Total vehicles detected: {counter.in_count + counter.out_count}")

if __name__ == "__main__":
    main() 