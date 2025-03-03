"""
Configuration settings for the vehicle tracker.
"""
import numpy as np

# Define the region of interest polygon coordinates
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

# Define target dimensions for perspective transformation
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],                            # Top-left corner
    [TARGET_WIDTH - 1, 0],             # Top-right corner
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],  # Bottom-right corner
    [0, TARGET_HEIGHT - 1]             # Bottom-left corner
])

# Default settings
DEFAULT_SETTINGS = {
    "model_path": "yolov8n.pt",
    "vehicle_classes": [2, 3, 5, 7],  # 2: car, 3: motorcycle, 5: bus, 7: truck
    "line_position": 0.5,
    "display_scale": 0.3,
    "confidence_threshold": 0.3
} 