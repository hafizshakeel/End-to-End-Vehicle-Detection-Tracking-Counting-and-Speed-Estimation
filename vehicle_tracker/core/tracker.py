"""
Vehicle tracking module using ByteTrack.
"""
import supervision as sv
from collections import defaultdict, deque

class VehicleTracker:
    """
    Handles vehicle tracking using ByteTrack algorithm.
    """
    def __init__(self, fps):
        """
        Initialize the vehicle tracker.
        
        Args:
            fps (float): Frames per second of the video
        """
        self.tracker = sv.ByteTrack(frame_rate=fps)
        self.coordinates = defaultdict(lambda: deque(maxlen=int(fps)))
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: Supervision detections object
            
        Returns:
            sv.Detections: Tracked detections
        """
        return self.tracker.update_with_detections(detections=detections)
    
    def store_coordinates(self, tracker_id, y_coordinate):
        """
        Store y-coordinates for a tracked object for speed calculation.
        
        Args:
            tracker_id: ID of the tracked object
            y_coordinate: Y-coordinate to store
        """
        self.coordinates[tracker_id].append(y_coordinate)
    
    def get_coordinates(self, tracker_id):
        """
        Get stored coordinates for a tracked object.
        
        Args:
            tracker_id: ID of the tracked object
            
        Returns:
            deque: Stored coordinates
        """
        return self.coordinates[tracker_id] 