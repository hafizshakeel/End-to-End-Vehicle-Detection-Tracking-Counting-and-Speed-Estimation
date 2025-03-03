import streamlit as st
import cv2
import numpy as np
import supervision as sv
import tempfile
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, deque
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

from vehicle_tracker.core.detector import VehicleDetector
from vehicle_tracker.core.tracker import VehicleTracker
from vehicle_tracker.core.counter import VehicleCounter
from vehicle_tracker.core.speed_estimator import ViewTransformer, calculate_speed
from vehicle_tracker.config.settings import SOURCE, TARGET, DEFAULT_SETTINGS
from vehicle_tracker.utils.video import get_video_info, get_first_frame

# Set page configuration
st.set_page_config(
    page_title="Vehicle Detection, Tracking, Counting, and Speed Estimation",
    page_icon="🚗",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 1rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .stat-label {
        font-size: 1rem;
        color: #424242;
    }
    .canvas-container {
        position: relative;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# Function to convert OpenCV image to base64 for display
def get_image_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Function to draw ROI on image
def draw_roi(img, points):
    img_copy = img.copy()
    
    # Draw points and lines
    for i, point in enumerate(points):
        cv2.circle(img_copy, tuple(point), 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(img_copy, tuple(points[i-1]), tuple(point), (0, 255, 0), 2)
    
    # Close the polygon if we have at least 3 points
    if len(points) >= 3:
        cv2.line(img_copy, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
        
        # Draw filled polygon with transparency
        overlay = img_copy.copy()
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, img_copy, 0.7, 0, img_copy)
    
    return img_copy

# Initialize session state for ROI points
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []
if 'custom_roi' not in st.session_state:
    st.session_state.custom_roi = False
if 'first_frame' not in st.session_state:
    st.session_state.first_frame = None
if 'frame_width' not in st.session_state:
    st.session_state.frame_width = 0
if 'frame_height' not in st.session_state:
    st.session_state.frame_height = 0

# Title and description
st.markdown('<div class="main-header">Vehicle Detection, Tracking, Counting, and Speed Estimation</div>', unsafe_allow_html=True)
st.markdown("""
This application uses computer vision and deep learning to detect, track, count, and estimate the speed of vehicles in video footage.
Upload a video file to get started.
""")

# Sidebar for configuration
st.sidebar.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)

# Model selection
model_options = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt",
    "YOLOv8x": "yolov8x.pt",
}
selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()), index=0)
model_path = model_options[selected_model]

# Vehicle classes selection
vehicle_classes = {
    "Car": 2,
    "Motorcycle": 3,
    "Bus": 5,
    "Truck": 7,
}
selected_classes = st.sidebar.multiselect(
    "Select Vehicle Classes to Detect",
    list(vehicle_classes.keys()),
    default=[]
)
selected_class_ids = [vehicle_classes[cls] for cls in selected_classes]

# Line position for counting
line_position = st.sidebar.slider(
    "Counting Line Position",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1,
    help="Position of the counting line as a fraction of frame height"
)

# Display scale
display_scale = st.sidebar.slider(
    "Display Scale",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.1,
    help="Scale factor for display"
)

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Minimum confidence score for detections"
)

# Advanced settings expander
with st.sidebar.expander("Advanced Settings"):
    enable_speed_estimation = st.checkbox("Enable Speed Estimation", value=True)
    enable_traces = st.checkbox("Show Movement Traces", value=True)
    trace_length = st.slider("Trace Length (seconds)", 1, 5, 2, help="Length of movement traces in seconds")

# Region of Interest settings in its own expander for prominence
with st.sidebar.expander("Region of Interest (ROI) Settings", expanded=True):
    st.markdown("### Region of Interest")
    use_custom_roi = st.checkbox("Use Custom ROI", value=st.session_state.custom_roi)
    st.session_state.custom_roi = use_custom_roi
    
    if use_custom_roi:
        st.info("You can define a custom Region of Interest on the video frame.")
        if len(st.session_state.roi_points) >= 3:
            st.success(f"Custom ROI defined with {len(st.session_state.roi_points)} points.")
            
            # Show a small preview of the ROI if we have a frame
            if st.session_state.first_frame is not None:
                # Create a smaller version of the frame for the sidebar
                small_frame = cv2.resize(st.session_state.first_frame, (0, 0), fx=0.3, fy=0.3)
                roi_preview = draw_roi(small_frame.copy(), 
                                      [[int(p[0]*0.3), int(p[1]*0.3)] for p in st.session_state.roi_points])
                roi_preview_rgb = cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB)
                st.image(roi_preview_rgb, caption="ROI Preview", use_container_width=True)
        else:
            st.warning("No ROI points defined yet. Click 'Define ROI' below the video to set up your custom ROI.")
    else:
        st.info("Default ROI from configuration will be used.")
        
        # Show a preview of the default ROI if we have a frame
        if st.session_state.first_frame is not None:
            # Import the default SOURCE points
            from vehicle_tracker.config.settings import SOURCE
            
            # Create a smaller version of the frame for the sidebar
            small_frame = cv2.resize(st.session_state.first_frame, (0, 0), fx=0.3, fy=0.3)
            
            # Scale the SOURCE points to match the smaller frame
            scaled_source = [[int(p[0]*0.3), int(p[1]*0.3)] for p in SOURCE]
            
            # Draw the default ROI on the small frame
            roi_preview = draw_roi(small_frame.copy(), scaled_source)
            roi_preview_rgb = cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB)
            st.image(roi_preview_rgb, caption="Default ROI Preview", use_container_width=True)

# Function to handle ROI selection
def roi_selection():
    if st.session_state.first_frame is None:
        st.warning("Please upload a video first to select ROI.")
        return
    
    # Display instructions
    st.markdown("### Custom ROI Selection")
    st.write("Define the Region of Interest (ROI) polygon by adding points. Add at least 3 points.")
    
    # Display the frame with current ROI points
    frame_with_roi = draw_roi(st.session_state.first_frame.copy(), st.session_state.roi_points)
    frame_rgb = cv2.cvtColor(frame_with_roi, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption="Current ROI", use_container_width=True)
    
    # Display frame dimensions for reference
    st.info(f"Frame dimensions: Width = {st.session_state.frame_width}, Height = {st.session_state.frame_height}")
    
    # Create two columns for the controls
    col1, col2 = st.columns(2)
    
    # Column 1: Add new points
    with col1:
        st.subheader("Add New Point")
        x_col, y_col, btn_col = st.columns([2, 2, 1])
        
        with x_col:
            new_x = st.number_input("X coordinate", min_value=0, max_value=st.session_state.frame_width-1, 
                                   value=int(st.session_state.frame_width/2), step=1, key="new_x")
        
        with y_col:
            new_y = st.number_input("Y coordinate", min_value=0, max_value=st.session_state.frame_height-1, 
                                   value=int(st.session_state.frame_height/2), step=1, key="new_y")
        
        with btn_col:
            st.write("")  # Add spacing
            st.write("")  # Add spacing
            if st.button("Add", key="add_point", type="primary"):
                st.session_state.roi_points.append([new_x, new_y])
                st.rerun()
    
    # Column 2: ROI Controls
    with col2:
        st.subheader("ROI Controls")
        
        if st.button("Clear All Points", key="clear_points"):
            st.session_state.roi_points = []
            st.rerun()
        
        if st.button("Remove Last Point", key="remove_last"):
            if len(st.session_state.roi_points) > 0:
                st.session_state.roi_points.pop()
                st.rerun()
        
        if len(st.session_state.roi_points) >= 3:
            if st.button("Confirm ROI", key="confirm_roi", type="primary"):
                st.session_state.custom_roi = True
                st.success("Custom ROI has been set!")
                st.rerun()
        else:
            st.warning(f"Need at least 3 points. Currently have {len(st.session_state.roi_points)}.")
    
    # Display current points in a table
    if st.session_state.roi_points:
        st.subheader("Current ROI Points")
        
        # Create a DataFrame for the points
        points_df = pd.DataFrame(st.session_state.roi_points, columns=["X", "Y"])
        points_df.index = points_df.index + 1  # 1-based indexing
        st.dataframe(points_df, use_container_width=True)
        
        # Point editing and deletion
        st.subheader("Edit or Delete Points")
        
        # Select a point to edit or delete
        point_idx = st.number_input("Select Point #", min_value=1, max_value=len(st.session_state.roi_points), 
                                   value=1, step=1, key="point_idx")
        
        # Create columns for edit and delete buttons
        edit_col, delete_col = st.columns(2)
        
        with edit_col:
            if st.button("Edit Selected Point", key="edit_point"):
                # Store the selected point index and values for editing
                st.session_state.editing_point = point_idx - 1
                st.session_state.edit_x = st.session_state.roi_points[point_idx - 1][0]
                st.session_state.edit_y = st.session_state.roi_points[point_idx - 1][1]
                st.rerun()
        
        with delete_col:
            if st.button("Delete Selected Point", key="delete_point"):
                st.session_state.roi_points.pop(point_idx - 1)
                st.rerun()
        
        # If a point is being edited, show the edit form
        if 'editing_point' in st.session_state:
            st.subheader(f"Editing Point #{st.session_state.editing_point + 1}")
            
            edit_x_col, edit_y_col = st.columns(2)
            
            with edit_x_col:
                edit_x = st.number_input("New X", min_value=0, max_value=st.session_state.frame_width-1, 
                                        value=st.session_state.roi_points[st.session_state.editing_point][0], 
                                        step=1, key="edit_x")
            
            with edit_y_col:
                edit_y = st.number_input("New Y", min_value=0, max_value=st.session_state.frame_height-1, 
                                        value=st.session_state.roi_points[st.session_state.editing_point][1], 
                                        step=1, key="edit_y")
            
            save_col, cancel_col = st.columns(2)
            
            with save_col:
                if st.button("Save Changes", key="save_changes"):
                    st.session_state.roi_points[st.session_state.editing_point] = [edit_x, edit_y]
                    del st.session_state.editing_point
                    st.rerun()
            
            with cancel_col:
                if st.button("Cancel", key="cancel_edit"):
                    del st.session_state.editing_point
                    st.rerun()
    
    # Display the current ROI as a numpy array
    if st.session_state.roi_points:
        st.subheader("ROI Coordinates (for reference)")
        roi_array = f"SOURCE = np.array({st.session_state.roi_points})"
        st.code(roi_array, language="python")

# Function to process video
def process_video(video_path, progress_bar, status_text):
    # Get video info
    video_info = get_video_info(video_path)
    
    # Get first frame to determine dimensions
    first_frame = get_first_frame(video_path)
    frame_height, frame_width = first_frame.shape[:2]
    
    # Initialize components
    detector = VehicleDetector(model_path=model_path, vehicle_classes=selected_class_ids)
    tracker = VehicleTracker(fps=video_info.fps)
    counter = VehicleCounter(frame_width, frame_height, line_position=line_position)
    
    # Set up zones
    if st.session_state.custom_roi and len(st.session_state.roi_points) >= 3:
        roi_polygon = np.array(st.session_state.roi_points)
        st.info(f"Using custom ROI with {len(roi_polygon)} points")
    else:
        # Use default SOURCE from settings
        from vehicle_tracker.config.settings import SOURCE
        roi_polygon = SOURCE
        st.info("Using default ROI from configuration")
        
    # Create polygon zone for detection filtering
    polygon_zone = sv.PolygonZone(polygon=roi_polygon)
    
    # Set up view transformer for speed estimation
    # For custom ROI, we'll use the same points for source and a scaled version for target
    if st.session_state.custom_roi and len(st.session_state.roi_points) >= 3:
        # Create a scaled version of the ROI for the target (bird's eye view)
        # This is a simple scaling approach - you might want to adjust this based on your needs
        source_points = np.array(st.session_state.roi_points)
        
        # Calculate bounding box of the ROI
        min_x = np.min(source_points[:, 0])
        max_x = np.max(source_points[:, 0])
        min_y = np.min(source_points[:, 1])
        max_y = np.max(source_points[:, 1])
        
        # Create a rectangular target with the same aspect ratio
        width = max_x - min_x
        height = max_y - min_y
        scale = 10  # Scale factor for the bird's eye view
        
        # Create target points (rectangular bird's eye view)
        target_points = np.array([
            [0, 0],
            [width * scale, 0],
            [width * scale, height * scale],
            [0, height * scale]
        ])
        
        # If we have more than 4 points in the source, we need to adjust
        if len(source_points) > 4:
            # Use the 4 corner points of the convex hull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(source_points)
            hull_points = source_points[hull.vertices]
            
            # Get the 4 extreme points (top-left, top-right, bottom-right, bottom-left)
            # Sort by y first, then by x
            top_points = hull_points[hull_points[:, 1].argsort()][:2]
            bottom_points = hull_points[hull_points[:, 1].argsort()][-2:]
            
            # Sort top points by x
            top_points = top_points[top_points[:, 0].argsort()]
            # Sort bottom points by x (reversed)
            bottom_points = bottom_points[bottom_points[:, 0].argsort()[::-1]]
            
            # Combine points in the order: top-left, top-right, bottom-right, bottom-left
            source_points = np.vstack([top_points, bottom_points])
        elif len(source_points) < 4:
            # If we have 3 points, add a 4th point to make a quadrilateral
            if len(source_points) == 3:
                # Calculate the 4th point to form a parallelogram
                fourth_point = source_points[0] + (source_points[2] - source_points[1])
                source_points = np.vstack([source_points, [fourth_point]])
        
        view_transformer = ViewTransformer(source=source_points, target=target_points)
    else:
        # Use default SOURCE and TARGET from settings
        from vehicle_tracker.config.settings import TARGET
        view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary file for processed video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_filename = temp_file.name
    temp_file.close()
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        temp_filename, 
        fourcc, 
        video_info.fps, 
        (frame_width, frame_height)
    )
    
    # Initialize data collection
    vehicle_data = []
    frame_count = 0
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # Detect vehicles with confidence threshold
        result = detector.model(frame, conf=confidence_threshold)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detector.filter_by_class(detections)
        
        # Filter by region of interest
        detections_in_zone = detections[polygon_zone.trigger(detections)]
        
        if len(detections_in_zone) == 0:
            # No detections, just annotate the counting line
            annotated_frame = frame.copy()
            annotated_frame = sv.LineZoneAnnotator(
                thickness=sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height)),
                text_thickness=2,  # Use a fixed value instead of calculate_optimal_text_thickness
                text_scale=sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))
            ).annotate(annotated_frame, line_counter=counter.line_zone)
            
            # Draw ROI polygon
            if st.session_state.custom_roi and len(st.session_state.roi_points) >= 3:
                pts = np.array(st.session_state.roi_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, (0, 255, 0), 2)
        else:
            # Track detections
            tracked_detections = tracker.update(detections_in_zone)
            
            # Update the counter with the tracked detections
            counter.update(tracked_detections)
            
            # Transform points for speed estimation
            try:
                points = tracked_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                
                # Check if points is empty or None
                if points is None or len(points) == 0:
                    # Skip speed estimation if no points are available
                    transformed_points = []
                else:
                    transformed_points = view_transformer.transform_points(points=points).astype(int)
                    
                # Create labels with class and speed information
                labels = []
                for idx, (tracker_id, class_id) in enumerate(zip(tracked_detections.tracker_id, tracked_detections.class_id)):
                    # Create label
                    class_name = detector.get_class_name(class_id)
                    
                    # Only calculate speed if we have transformed points
                    if idx < len(transformed_points) and enable_speed_estimation:
                        # Store coordinates for speed calculation
                        y = transformed_points[idx][1]
                        tracker.store_coordinates(tracker_id, y)
                        
                        # Calculate speed
                        speed = calculate_speed(tracker.get_coordinates(tracker_id), video_info.fps)
                        
                        # Add a sanity check for unrealistic speeds (e.g., > 200 km/h)
                        if speed is None:
                            labels.append(f"#{tracker_id} {class_name}")
                        elif speed > 200:  # Cap unrealistic speeds
                            labels.append(f"#{tracker_id} {class_name}: ? km/h")
                        else:
                            labels.append(f"#{tracker_id} {class_name}: {speed} km/h")
                            
                            # Store data for analysis
                            vehicle_data.append({
                                'frame': frame_count,
                                'timestamp': frame_count / video_info.fps,
                                'tracker_id': tracker_id,
                                'class_id': class_id,
                                'class_name': class_name,
                                'speed': speed,
                                'y_position': y
                            })
                    else:
                        labels.append(f"#{tracker_id} {class_name}")
            except Exception as e:
                # If there's an error in point transformation, just use class names as labels
                labels = [f"#{tracker_id} {detector.get_class_name(class_id)}" 
                          for tracker_id, class_id in zip(tracked_detections.tracker_id, tracked_detections.class_id)]
                st.error(f"Error in point transformation: {str(e)}")

            # Annotate the frame
            annotated_frame = frame.copy()
            
            # Draw ROI polygon
            if st.session_state.custom_roi and len(st.session_state.roi_points) >= 3:
                pts = np.array(st.session_state.roi_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, (0, 255, 0), 2)
            
            # Add traces if enabled
            if enable_traces:
                trace_annotator = sv.TraceAnnotator(
                    thickness=sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height)),
                    trace_length=int(video_info.fps * trace_length),
                    position=sv.Position.BOTTOM_CENTER,
                    color_lookup=sv.ColorLookup.TRACK
                )
                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            
            # Add bounding boxes
            box_annotator = sv.BoxAnnotator(
                thickness=sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height)),
                color_lookup=sv.ColorLookup.TRACK
            )
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            
            # Add labels
            label_annotator = sv.LabelAnnotator(
                text_scale=sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height)),
                text_thickness=2,  # Use a fixed value instead of calculate_optimal_text_thickness
                text_position=sv.Position.BOTTOM_CENTER,
                color_lookup=sv.ColorLookup.TRACK
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, 
                detections=tracked_detections, 
                labels=labels
            )
            
            # Add counting line
            annotated_frame = sv.LineZoneAnnotator(
                thickness=sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height)),
                text_thickness=2,  # Use a fixed value instead of calculate_optimal_text_thickness
                text_scale=sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))
            ).annotate(annotated_frame, line_counter=counter.line_zone)
        
        # Write the frame to the output video
        video_writer.write(annotated_frame)
    
    # Release resources
    cap.release()
    video_writer.release()
    
    # Create DataFrame from collected data
    df = pd.DataFrame(vehicle_data) if vehicle_data else pd.DataFrame()
    
    return temp_filename, counter.in_count, counter.out_count, df

# Function to display statistics
def display_statistics(in_count, out_count, vehicle_data):
    # Create columns for statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{in_count}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Vehicles In</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{out_count}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Vehicles Out</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{in_count + out_count}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Total Vehicles</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if we have any vehicle data
    if vehicle_data is None or (isinstance(vehicle_data, pd.DataFrame) and vehicle_data.empty):
        st.info("No detailed vehicle data available for visualization. Try adjusting detection parameters or selecting different vehicle classes.")
        return
    
    # Display additional statistics if data is available
    st.markdown('<div class="sub-header">Vehicle Statistics</div>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Vehicle Types", "Speed Distribution", "Traffic Flow"])
    
    with tab1:
        # Vehicle type distribution
        if 'class_name' in vehicle_data.columns and not vehicle_data['class_name'].empty:
            vehicle_counts = vehicle_data['class_name'].value_counts().reset_index()
            vehicle_counts.columns = ['Vehicle Type', 'Count']
            
            fig = px.pie(
                vehicle_counts, 
                values='Count', 
                names='Vehicle Type',
                title='Vehicle Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No vehicle type data available for visualization.")
    
    with tab2:
        # Speed distribution
        if 'speed' in vehicle_data.columns and not vehicle_data['speed'].empty:
            # Filter out None values
            speed_data = vehicle_data[vehicle_data['speed'].notna()]
            
            if not speed_data.empty:
                fig = px.histogram(
                    speed_data, 
                    x='speed',
                    color='class_name',
                    nbins=20,
                    title='Speed Distribution',
                    labels={'speed': 'Speed (km/h)', 'count': 'Number of Vehicles', 'class_name': 'Vehicle Type'},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Average speed by vehicle type
                avg_speed = speed_data.groupby('class_name')['speed'].mean().reset_index()
                avg_speed.columns = ['Vehicle Type', 'Average Speed (km/h)']
                
                fig = px.bar(
                    avg_speed,
                    x='Vehicle Type',
                    y='Average Speed (km/h)',
                    title='Average Speed by Vehicle Type',
                    color='Vehicle Type',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid speed data available for visualization.")
        else:
            st.info("No speed data available for visualization.")
    
    with tab3:
        # Traffic flow over time
        if 'timestamp' in vehicle_data.columns and len(vehicle_data) > 0:
            # Group by minute for better visualization
            vehicle_data['minute'] = (vehicle_data['timestamp'] // 60).astype(int)
            flow_data = vehicle_data.groupby('minute').size().reset_index()
            flow_data.columns = ['Minute', 'Vehicle Count']
            
            fig = px.line(
                flow_data,
                x='Minute',
                y='Vehicle Count',
                title='Traffic Flow Over Time',
                labels={'Minute': 'Time (minutes)', 'Vehicle Count': 'Number of Vehicles'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time-based data available for visualization.")
    
    # Display raw data in an expander
    with st.expander("View Raw Data"):
        st.dataframe(vehicle_data)
        
        # Download button for CSV
        csv = vehicle_data.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"vehicle_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# Main application logic
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name
    temp_file.close()
    
    # Get first frame for ROI selection
    if st.session_state.first_frame is None:
        first_frame = get_first_frame(video_path)
        st.session_state.first_frame = first_frame
        st.session_state.frame_height, st.session_state.frame_width = first_frame.shape[:2]
    
    # Display a preview of the video
    st.video(video_path)
    
    # ROI Selection section
    if st.session_state.custom_roi or use_custom_roi:
        roi_selection()
    
    # Process button
    if st.button("Process Video"):
        # Check if any vehicle classes are selected
        if not selected_classes:
            st.error("Please select at least one vehicle class to detect.")
        else:
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process the video
            with st.spinner("Processing video..."):
                processed_video_path, in_count, out_count, vehicle_data = process_video(
                    video_path, progress_bar, status_text
                )
            
            # Display success message
            st.success("Video processing complete!")
            
            # Convert the processed video to a web-compatible format
            try:
                # Create a web-compatible version for browser playback
                web_compatible_path = processed_video_path.replace('.mp4', '_web.mp4')
                
                # Check if ffmpeg is available
                import shutil
                ffmpeg_available = shutil.which('ffmpeg') is not None
                
                if ffmpeg_available:
                    # Use ffmpeg if available
                    st.info("Converting video for web playback using ffmpeg...")
                    ffmpeg_cmd = f'ffmpeg -y -i "{processed_video_path}" -vcodec libx264 -preset fast -crf 28 -acodec aac -strict experimental "{web_compatible_path}"'
                    import subprocess
                    result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
                    
                    if os.path.exists(web_compatible_path) and os.path.getsize(web_compatible_path) > 0:
                        # Display the processed video
                        st.markdown('<div class="sub-header">Processed Video</div>', unsafe_allow_html=True)
                        st.video(web_compatible_path)
                    else:
                        raise Exception("ffmpeg conversion failed")
                else:
                    # If ffmpeg is not available, use the original video file
                    st.info("ffmpeg not found. Using original processed video for playback...")
                    st.markdown('<div class="sub-header">Processed Video</div>', unsafe_allow_html=True)
                    
                    # Try to display the video directly
                    try:
                        st.video(processed_video_path)
                    except Exception as e:
                        st.warning(f"Could not play video in browser: {str(e)}. Please download the video to view it.")
                        st.info("You can download the processed video using the button below.")
            except Exception as e:
                st.warning(f"Could not convert video for web playback: {str(e)}. You can still download the processed video below.")
            
            # Always provide download option
            with open(processed_video_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name=f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4",
                    key="download_processed_video_1",
                )
            
            # Display statistics
            st.markdown('<div class="sub-header">Results</div>', unsafe_allow_html=True)
            display_statistics(in_count, out_count, vehicle_data)
            
            # Clean up temporary files
            os.unlink(video_path)
            # Keep processed_video_path for download
            
            # Download button for processed video
            with open(processed_video_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name=f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4",
                    key="download_processed_video_2",
                )
            
            # Clean up processed video after download is available
            # os.unlink(processed_video_path)
else:
    # Display sample images when no video is uploaded
    st.markdown('<div class="sub-header">Sample Output</div>', unsafe_allow_html=True)
    st.info("Upload a video to see the vehicle detection, tracking, counting, and speed estimation in action.")
    
    # Display columns with sample images (these would need to be replaced with actual sample images)
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.markdown("#### Vehicle Detection and Tracking")
    #     st.image("https://ultralytics.com/images/yolov8-track-vehicles.jpg", use_container_width=True)
    # with col2:
    #     st.markdown("#### Speed Estimation")
    #     st.image("https://ultralytics.com/images/yolov8-speed-estimation.jpg", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Developed with ❤️ using YOLOv8, Supervision, and Streamlit</p>
    <p>© 2025 Vehicle Detection, Tracking, Counting, and Speed Estimation</p>
</div>
""", unsafe_allow_html=True)
