"""
Vehicle detection module using YOLOv8.
"""
from ultralytics import YOLO
import supervision as sv
import numpy as np

# YOLO class IDs for vehicles
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Class names mapping for display
CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

class VehicleDetector:
    """
    Handles vehicle detection using YOLOv8 model.
    """
    def __init__(self, model_path="yolov8n.pt", vehicle_classes=None):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path (str): Path to the YOLOv8 model file
            vehicle_classes (list): List of vehicle class IDs to detect
        """
        self.model = YOLO(model_path)
        self.vehicle_classes = vehicle_classes if vehicle_classes is not None else VEHICLE_CLASSES
    
    def detect(self, frame):
        """
        Detect vehicles in a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            sv.Detections: Supervision detections object
        """
        # Run inference
        result = self.model(frame)[0]
        
        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter by vehicle classes
        return self.filter_by_class(detections)
    
    def filter_by_class(self, detections):
        """
        Filter detections to only include specified vehicle classes.
        
        Args:
            detections: Supervision detections object
            
        Returns:
            sv.Detections: Filtered detections
        """
        if len(detections) == 0:
            return detections
            
        mask = np.array([class_id in self.vehicle_classes for class_id in detections.class_id], dtype=bool)
        return detections[mask]
    
    @staticmethod
    def get_class_name(class_id):
        """
        Get the name of a vehicle class.
        
        Args:
            class_id: YOLO class ID
            
        Returns:
            str: Class name
        """
        return CLASS_NAMES.get(class_id, "Vehicle") 