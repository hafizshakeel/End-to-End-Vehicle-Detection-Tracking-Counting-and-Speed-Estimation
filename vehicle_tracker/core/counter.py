"""
Vehicle counting module using LineZone.
"""
import supervision as sv

class VehicleCounter:
    """
    Handles vehicle counting using LineZone.
    """
    def __init__(self, frame_width, frame_height, line_position=0.5):
        """
        Initialize the vehicle counter.
        
        Args:
            frame_width (int): Width of the video frame
            frame_height (int): Height of the video frame
            line_position (float): Position of the counting line as a fraction of frame height (0-1)
        """
        # Create a horizontal line for counting
        line_y = int(frame_height * line_position)
        line_start = sv.Point(0, line_y)
        line_end = sv.Point(frame_width, line_y)
        self.line_zone = sv.LineZone(start=line_start, end=line_end)
    
    def update(self, detections):
        """
        Update counter with new detections.
        
        Args:
            detections: Supervision detections object
            
        Returns:
            bool: True if the line was triggered
        """
        return self.line_zone.trigger(detections=detections)
    
    @property
    def in_count(self):
        """Get the count of vehicles entering."""
        return self.line_zone.in_count
    
    @property
    def out_count(self):
        """Get the count of vehicles exiting."""
        return self.line_zone.out_count
    
    @property
    def total_count(self):
        """Get the total count of vehicles."""
        return self.in_count + self.out_count 