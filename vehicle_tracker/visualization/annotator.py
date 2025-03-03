"""
Visualization module for annotating video frames.
"""
import supervision as sv
import cv2

class FrameAnnotator:
    """
    Handles annotation of video frames with detections, tracking, and counting information.
    """
    def __init__(self, video_info):
        """
        Initialize the frame annotator.
        
        Args:
            video_info: Supervision VideoInfo object
        """
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
        
        self.box_annotator = sv.BoxAnnotator(
            thickness=thickness,
            color_lookup=sv.ColorLookup.TRACK
        )

        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK
        )

        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK
        )
        
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=thickness,
            text_thickness=thickness,
            text_scale=text_scale
        )
    
    def annotate_frame(self, frame, detections, labels=None, line_counter=None):
        """
        Annotate a frame with detections, tracking, and counting information.
        
        Args:
            frame: Input image frame
            detections: Supervision detections object
            labels: List of label strings for each detection
            line_counter: LineZone object for counting
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Annotate with traces, boxes, and labels
        if len(detections) > 0:
            annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
            
            if labels is not None:
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame, 
                    detections=detections, 
                    labels=labels
                )
        
        # Annotate with line counter
        if line_counter is not None:
            self.line_zone_annotator.annotate(annotated_frame, line_counter=line_counter)
        
        return annotated_frame
    
    def resize_frame(self, frame, scale):
        """
        Resize a frame by a scale factor.
        
        Args:
            frame: Input image frame
            scale: Scale factor (0-1)
            
        Returns:
            np.ndarray: Resized frame
        """
        height, width = frame.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height)) 