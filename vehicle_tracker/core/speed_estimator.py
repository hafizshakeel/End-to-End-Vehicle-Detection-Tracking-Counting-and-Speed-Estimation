"""
Vehicle speed estimation module.
"""
import cv2
import numpy as np
import logging

class ViewTransformer:
    """
    Handles perspective transformation between source and target coordinates.
    Used for accurate speed estimation by transforming to a bird's eye view.
    """
    def __init__(self, source:np.ndarray, target:np.array):
        """
        Initialize the view transformer.
        
        Args:
            source (np.ndarray): Source polygon coordinates
            target (np.ndarray): Target polygon coordinates
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
    
    def transform_points(self, points:np.array) -> np.ndarray:
        """
        Transform points from source to target perspective.
        
        Args:
            points (np.array): Points to transform
            
        Returns:
            np.ndarray: Transformed points
        """
        try:
            if points is None or len(points) == 0:
                logging.warning("No points to transform")
                return np.array([])
                
            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
            
            if transformed_points is None:
                logging.warning("Perspective transformation failed")
                return np.array([])
                
            return transformed_points.reshape(-1, 2)
        except Exception as e:
            logging.error(f"Error in transform_points: {str(e)}")
            return np.array([])

def calculate_speed(coordinates, fps):
    """
    Calculate speed in km/h based on coordinate changes over time.
    
    Args:
        coordinates: List of coordinates over time
        fps: Frames per second
        
    Returns:
        int or None: Calculated speed in km/h, or None if not enough data
    """
    if len(coordinates) < fps / 2:
        return None
        
    coordinate_start = coordinates[-1]
    coordinate_end = coordinates[0]
    
    # Calculate distance in pixels
    distance = abs(coordinate_start - coordinate_end)
    
    # Convert time from frames to seconds
    time = len(coordinates) / fps
    
    # Apply a scaling factor to convert pixel distance to meters
    # This is an approximation - in a real application, you would calibrate this
    # based on known distances in your scene
    # pixels_per_meter = 30.0  # Adjust this value based on your specific scene
    # distance_meters = distance / pixels_per_meter
    
    # Convert to km/h (3.6 is the conversion factor from m/s to km/h)
    speed = (distance / time) * 3.6
    
    # Apply a sanity check to avoid unrealistic values
    if speed > 200:  # Most vehicles don't go faster than 200 km/h
        return 200
        
    return int(speed) 