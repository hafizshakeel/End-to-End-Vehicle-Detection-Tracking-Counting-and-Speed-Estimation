# Add these imports at the very top of the file, before any other imports
import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONASYNCIODEBUG"] = "1"

# Import torch before streamlit to avoid conflicts
import torch
import asyncio
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    pass

# Disable PyTorch multiprocessing to avoid conflicts with Streamlit
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Now continue with the rest of the imports
import streamlit as st
import cv2
import numpy as np
import supervision as sv
import tempfile
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
from ultralytics import YOLO
from collections import defaultdict, deque
import threading
import queue

from vehicle_tracker.core.detector import VehicleDetector
from vehicle_tracker.core.tracker import VehicleTracker
from vehicle_tracker.core.counter import VehicleCounter
from vehicle_tracker.core.speed_estimator import ViewTransformer, calculate_speed
from vehicle_tracker.config.settings import SOURCE, TARGET, DEFAULT_SETTINGS

# Set page configuration
st.set_page_config(
    page_title="Real-time Vehicle Detection and Tracking",
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

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'stop_signal' not in st.session_state:
    st.session_state.stop_signal = False
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=10)
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'in_count': 0,
        'out_count': 0,
        'vehicle_data': []
    }
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'status' not in st.session_state:
    st.session_state.status = "Ready"
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""
# Add session state variables for IP camera configuration
if 'ip_address' not in st.session_state:
    st.session_state.ip_address = "192.168.1.100"
if 'port' not in st.session_state:
    st.session_state.port = "554"
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'password' not in st.session_state:
    st.session_state.password = ""
if 'path' not in st.session_state:
    st.session_state.path = "/stream"
if 'profile_applied' not in st.session_state:
    st.session_state.profile_applied = False
# Add session state variables for external camera configuration
if 'camera_index' not in st.session_state:
    st.session_state.camera_index = 1
if 'camera_resolution' not in st.session_state:
    st.session_state.camera_resolution = "640x480"
if 'camera_api' not in st.session_state:
    st.session_state.camera_api = None
# Add session state variables for ROI selection
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []
if 'custom_roi' not in st.session_state:
    st.session_state.custom_roi = False
if 'test_frame' not in st.session_state:
    st.session_state.test_frame = None

# Title and description
st.markdown('<div class="main-header">Real-time Vehicle Detection and Tracking</div>', unsafe_allow_html=True)
st.markdown("""
This application uses computer vision and deep learning to detect, track, count, and estimate the speed of vehicles in real-time from a camera feed.
""")

# Sidebar for configuration
st.sidebar.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)

# Camera selection - REMOVED WEBCAM, KEEPING ONLY EXTERNAL AND IP CAMERA
camera_options = {
    "External Camera": 1,
    "IP Camera": "rtsp://username:password@ip_address:port/path"
}
selected_camera = st.sidebar.selectbox("Select Camera", list(camera_options.keys()), index=0)
camera_source = camera_options[selected_camera]

# External Camera Configuration
if selected_camera == "External Camera":
    external_camera_config = st.sidebar.expander("External Camera Configuration", expanded=True)
    
    with external_camera_config:
        # Camera index selection
        camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=1, step=1,
                                      help="The index of the external camera (usually 0 for built-in, 1 for first external)")
        camera_source = int(camera_index)
        
        # Camera resolution
        resolution_options = ["640x480", "1280x720", "1920x1080"]
        selected_resolution = st.selectbox("Camera Resolution", resolution_options, index=0)
        width, height = map(int, selected_resolution.split("x"))
        
        st.info(f"Selected external camera at index {camera_index} with resolution {selected_resolution}")
        
        # Advanced settings - using checkbox instead of nested expander
        show_advanced = st.checkbox("Show Advanced Settings", value=False)
        
        if show_advanced:
            st.markdown("### Advanced Settings")
            # Camera API preference
            api_options = {
                "Default": None,
                "DirectShow (Windows)": cv2.CAP_DSHOW,
                "V4L2 (Linux)": cv2.CAP_V4L2,
                "AVFOUNDATION (macOS)": cv2.CAP_AVFOUNDATION
            }
            selected_api = st.selectbox("Camera API", list(api_options.keys()), index=0)
            camera_api = api_options[selected_api]
            
            # Store in session state
            st.session_state.camera_index = camera_index
            st.session_state.camera_resolution = selected_resolution
            st.session_state.camera_api = camera_api
            
            st.info("These settings will be applied when you test or start the camera")

# Custom RTSP URL
elif selected_camera == "IP Camera":
    # Add a more user-friendly interface for IP camera configuration
    ip_camera_config = st.sidebar.expander("IP Camera Configuration", expanded=True)
    
    with ip_camera_config:
        # Advanced mode with direct RTSP URL input
        use_direct_url = st.checkbox("Use Direct RTSP URL", value=False)
        
        if use_direct_url:
            camera_source = st.text_input("RTSP URL", value=camera_source)
            st.info("Example: rtsp://username:password@192.168.1.100:554/stream")
        else:
            # User-friendly form for building the RTSP URL
            col1, col2 = st.columns(2)
            with col1:
                ip_address = st.text_input("IP Address", value=st.session_state.ip_address, key="ip_address_input")
                username = st.text_input("Username (optional)", value=st.session_state.username, key="username_input")
            with col2:
                port = st.text_input("Port", value=st.session_state.port, key="port_input")
                password = st.text_input("Password (optional)", value=st.session_state.password, type="password", key="password_input")
            
            path = st.text_input("Stream Path", value=st.session_state.path, key="path_input")
            
            # Update session state
            st.session_state.ip_address = ip_address
            st.session_state.port = port
            st.session_state.username = username
            st.session_state.password = password
            st.session_state.path = path
            
            # Build the RTSP URL
            if username and password:
                camera_source = f"rtsp://{username}:{password}@{ip_address}:{port}{path}"
            else:
                camera_source = f"rtsp://{ip_address}:{port}{path}"
            
            st.info(f"Generated RTSP URL: {camera_source}")
            
        # Add common camera profiles for quick setup
        st.markdown("#### Common Camera Profiles")
        camera_profiles = {
            "Select a profile": "",
            "Hikvision": "rtsp://username:password@ip_address:554/Streaming/Channels/101",
            "Dahua": "rtsp://username:password@ip_address:554/cam/realmonitor?channel=1&subtype=0",
            "Axis": "rtsp://username:password@ip_address:554/axis-media/media.amp",
            "Amcrest": "rtsp://username:password@ip_address:554/cam/realmonitor?channel=1&subtype=0",
            "Generic ONVIF": "rtsp://username:password@ip_address:554/onvif/profile0"
        }
        
        selected_profile = st.selectbox("Quick Setup Profiles", list(camera_profiles.keys()))
        if selected_profile != "Select a profile":
            profile_url = camera_profiles[selected_profile]
            st.info(f"Profile template: {profile_url}")
            
            # Function to parse RTSP URL
            def parse_rtsp_url(url):
                # Default values
                parsed = {
                    "username": "",
                    "password": "",
                    "ip_address": "192.168.1.100",
                    "port": "554",
                    "path": "/stream"
                }
                
                try:
                    # Remove rtsp:// prefix
                    if url.startswith("rtsp://"):
                        url = url[7:]
                    
                    # Extract credentials if present
                    if "@" in url:
                        creds, rest = url.split("@", 1)
                        if ":" in creds:
                            parsed["username"], parsed["password"] = creds.split(":", 1)
                    else:
                        rest = url
                    
                    # Extract host and path
                    if "/" in rest:
                        host, parsed["path"] = rest.split("/", 1)
                        parsed["path"] = "/" + parsed["path"]
                    else:
                        host = rest
                    
                    # Extract IP and port
                    if ":" in host:
                        parsed["ip_address"], parsed["port"] = host.split(":", 1)
                    else:
                        parsed["ip_address"] = host
                    
                    return parsed
                except Exception as e:
                    st.warning(f"Error parsing URL: {str(e)}")
                    return parsed
            
            # Apply profile button
            if st.button("Apply Profile", help="Apply this profile template to the RTSP URL fields"):
                # Parse the profile URL
                parsed_profile = parse_rtsp_url(profile_url)
                
                # Update session state with the parsed values
                st.session_state.ip_address = parsed_profile["ip_address"]
                st.session_state.port = parsed_profile["port"]
                st.session_state.username = parsed_profile["username"]
                st.session_state.password = parsed_profile["password"]
                st.session_state.path = parsed_profile["path"]
                st.session_state.profile_applied = True
                
                # Force a rerun to update the UI
                st.rerun()

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
    default=list(vehicle_classes.keys())
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
    
    processing_resolution = st.selectbox(
        "Processing Resolution",
        ["640x480", "1280x720", "1920x1080"],
        index=0,
        help="Resolution for processing (higher resolutions require more computational power)"
    )
    
    display_fps = st.checkbox("Display FPS", value=True)
    
    # Add custom ROI selection
    st.markdown("### Region of Interest")
    use_custom_roi = st.checkbox("Use Custom ROI", value=st.session_state.custom_roi)
    st.session_state.custom_roi = use_custom_roi
    
    if use_custom_roi:
        st.info("You can define a custom Region of Interest on the camera view.")
    else:
        st.info("Default ROI will be used.")
    
    record_video = st.checkbox("Record Video", value=False)
    if record_video:
        record_path = st.text_input("Save Path", value="recordings")
        if not os.path.exists(record_path):
            os.makedirs(record_path, exist_ok=True)

# Add this function after the imports and before the main code
def test_camera_access(camera_source):
    """Test if the camera can be accessed and return a sample frame."""
    try:
        # Show a progress indicator
        progress_text = "Testing camera connection..."
        progress_bar = st.progress(0)
        
        # Update progress
        progress_bar.progress(10)
        st.info(f"Attempting to connect to: {camera_source}")
        
        # For Windows, try to use DirectShow backend first
        try:
            # Store in session state to prevent garbage collection
            progress_bar.progress(20)
            
            # Use the appropriate API based on camera type
            if selected_camera == "External Camera" and st.session_state.camera_api is not None:
                st.session_state.test_cap = cv2.VideoCapture(camera_source, st.session_state.camera_api)
            else:
                # Try DirectShow for Windows as default for external cameras
                st.session_state.test_cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
                
            cap = st.session_state.test_cap
            # Wait a moment for the camera to initialize
            progress_bar.progress(30)
            time.sleep(1.0)
        except Exception:
            # Fallback to default backend if DirectShow fails
            progress_bar.progress(40)
            st.info("DirectShow backend failed, trying default backend...")
            st.session_state.test_cap = cv2.VideoCapture(camera_source)
            cap = st.session_state.test_cap
            progress_bar.progress(50)
            time.sleep(1.0)
            
        if not cap.isOpened():
            progress_bar.progress(100)
            if hasattr(st.session_state, 'test_cap'):
                st.session_state.test_cap.release()
                st.session_state.test_cap = None
            return False, None, "Failed to open camera"
        
        # Set resolution for external camera
        if selected_camera == "External Camera":
            width, height = map(int, st.session_state.camera_resolution.split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Try to read a frame multiple times (sometimes the first frame fails)
        progress_bar.progress(60)
        st.info("Camera opened successfully. Attempting to read frames...")
        for i in range(5):  # Try up to 5 times
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                progress_bar.progress(70 + i*5)
                break
            st.info(f"Attempt {i+1}/5 to read frame failed. Retrying...")
            time.sleep(0.5)  # Longer wait between attempts
        
        if not ret or frame is None or frame.size == 0:
            progress_bar.progress(100)
            cap.release()
            if hasattr(st.session_state, 'test_cap'):
                st.session_state.test_cap = None
            return False, frame, "Failed to capture frame after multiple attempts"
        
        # Get camera properties
        progress_bar.progress(85)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 FPS if not available
        
        # Keep the camera open for a moment to ensure it's stable
        progress_bar.progress(90)
        time.sleep(1.0)
        
        # Read one more frame to ensure stability
        progress_bar.progress(95)
        ret, second_frame = cap.read()
        
        # Release the camera
        cap.release()
        if hasattr(st.session_state, 'test_cap'):
            st.session_state.test_cap = None
        
        progress_bar.progress(100)
        
        if ret and second_frame is not None and second_frame.size > 0:
            # Store the test frame for ROI selection
            st.session_state.test_frame = frame
            return True, frame, f"Camera accessed successfully. Resolution: {width}x{height}, FPS: {fps}"
        else:
            return False, frame, "Camera opened but unstable - could not read second frame"
    except Exception as e:
        if hasattr(st.session_state, 'test_cap') and st.session_state.test_cap is not None:
            st.session_state.test_cap.release()
            st.session_state.test_cap = None
        return False, None, f"Error accessing camera: {str(e)}"

# Function to draw ROI on image
def draw_roi(img, points):
    """Draw ROI points and polygon on the image."""
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

# Function to handle ROI selection
def roi_selection():
    """Handle ROI selection on the camera frame."""
    if st.session_state.test_frame is None:
        st.warning("Please test the camera first to capture a frame for ROI selection.")
        return
    
    # Display instructions
    st.markdown("### Custom ROI Selection")
    st.write("Click on the image to define the Region of Interest (ROI) polygon. Add at least 3 points.")
    
    # Get frame dimensions
    frame_height, frame_width = st.session_state.test_frame.shape[:2]
    
    # Create columns for the image and controls
    col1, col2 = st.columns([3, 1])
    
    # Display the image with current ROI points
    with col1:
        frame_with_roi = draw_roi(st.session_state.test_frame.copy(), st.session_state.roi_points)
        frame_rgb = cv2.cvtColor(frame_with_roi, cv2.COLOR_BGR2RGB)
        
        # Create a clickable image
        st.image(frame_rgb, use_container_width=True, caption="Click to add points")
        
        # Create HTML for click handling without using key parameter
        canvas_html = f"""
        <div style="position: relative; width: 100%;">
            <div id="roi-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10;"
                 onclick="handleCanvasClick(event)"></div>
        </div>
        <script>
            function handleCanvasClick(e) {{
                const overlay = document.getElementById('roi-overlay');
                const rect = overlay.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                // Calculate the scaling factor
                const scaleX = {frame_width} / rect.width;
                const scaleY = {frame_height} / rect.height;
                
                // Scale the coordinates
                const scaledX = Math.round(x * scaleX);
                const scaledY = Math.round(y * scaleY);
                
                // Redirect with query parameters
                window.location.href = window.location.pathname + '?roi_x=' + scaledX + '&roi_y=' + scaledY;
            }}
            
            // Add the overlay on top of the image
            document.addEventListener('DOMContentLoaded', function() {{
                const images = document.querySelectorAll('img');
                const lastImage = images[images.length - 1]; // Get the last image (our ROI image)
                if (lastImage) {{
                    const overlay = document.getElementById('roi-overlay');
                    overlay.style.height = lastImage.offsetHeight + 'px';
                }}
            }});
        </script>
        """
        
        # Use html component without the key parameter
        st.components.v1.html(canvas_html, height=100)
    
    # Controls for ROI selection
    with col2:
        if st.button("Clear Points", key="clear_roi"):
            st.session_state.roi_points = []
            st.rerun()
        
        if st.button("Use Default ROI", key="default_roi"):
            st.session_state.custom_roi = False
            st.session_state.roi_points = []
            st.rerun()
        
        if len(st.session_state.roi_points) >= 3:
            if st.button("Confirm ROI", key="confirm_roi"):
                st.session_state.custom_roi = True
                st.success("Custom ROI has been set!")
                st.rerun()
    
    # Display current points
    st.write("Current ROI Points:")
    st.write(st.session_state.roi_points)
    
    # Handle the click event using query parameters
    if "roi_x" in st.query_params and "roi_y" in st.query_params:
        try:
            x = int(st.query_params["roi_x"])
            y = int(st.query_params["roi_y"])
            st.session_state.roi_points.append([x, y])
            # Clear the query parameters
            st.query_params.clear()
            # Rerun to update the display
            st.rerun()
        except Exception as e:
            st.error(f"Error processing point: {str(e)}")

# Function to process camera feed
def process_camera_feed():
    """Process the camera feed in a separate thread."""
    try:
        # Set status
        st.session_state.status = "Initializing camera..."
        st.session_state.error_message = ""
        
        # Reset stats
        st.session_state.stats = {
            'in_count': 0,
            'out_count': 0,
            'vehicle_data': []
        }
        
        # Parse processing resolution
        if selected_camera == "External Camera":
            width, height = map(int, st.session_state.camera_resolution.split('x'))
        else:
            width, height = map(int, processing_resolution.split('x'))
        
        # Initialize camera - use cv2.CAP_DSHOW on Windows to avoid freezing
        try:
            # For external camera or IP camera
            if selected_camera == "External Camera":
                # Use the selected API if specified
                if st.session_state.camera_api is not None:
                    st.session_state.cap = cv2.VideoCapture(camera_source, st.session_state.camera_api)
                else:
                    st.session_state.cap = cv2.VideoCapture(camera_source)
                cap = st.session_state.cap
                time.sleep(1.0)  # Increased wait time
            else:  # IP Camera
                st.session_state.cap = cv2.VideoCapture(camera_source)
                cap = st.session_state.cap
                time.sleep(1.0)  # Give IP camera time to connect
                
            if not cap.isOpened():
                error_msg = f"Error: Could not open camera source {camera_source}"
                st.session_state.error_message = error_msg
                st.session_state.status = "Error"
                st.error(error_msg)
                st.session_state.processing = False
                return
                
            # Print camera info for debugging
            st.info(f"Camera opened successfully. Requested resolution: {width}x{height}")
        except Exception as e:
            error_msg = f"Error initializing camera: {str(e)}"
            st.session_state.error_message = error_msg
            st.session_state.status = "Error"
            st.error(error_msg)
            st.session_state.processing = False
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
        
        # Get actual frame dimensions (may differ from requested)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 FPS if not available
            
        st.info(f"Actual camera resolution: {actual_width}x{actual_height}, FPS: {fps}")
        
        # Test if we can actually read frames
        for _ in range(5):  # Try up to 5 times
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                # Successfully read a frame
                break
            time.sleep(0.1)
        else:
            # Could not read frames after multiple attempts
            error_msg = "Could not read frames from camera after multiple attempts"
            st.session_state.error_message = error_msg
            st.session_state.status = "Error"
            st.error(error_msg)
            cap.release()
            st.session_state.processing = False
            return
        
        # Create dynamic ROI based on frame dimensions
        # This ensures the ROI works with any camera resolution
        dynamic_source = np.array([
            [int(actual_width * 0.25), int(actual_height * 0.4)],
            [int(actual_width * 0.75), int(actual_height * 0.4)],
            [int(actual_width * 0.75), int(actual_height * 0.9)],
            [int(actual_width * 0.25), int(actual_height * 0.9)]
        ])
        
        # Use custom ROI if defined
        if st.session_state.custom_roi and len(st.session_state.roi_points) >= 3:
            dynamic_source = np.array(st.session_state.roi_points)
            st.info("Using custom ROI for detection and tracking")
        else:
            st.info("Using default ROI for detection and tracking")
        
        # Initialize components - load model only once to avoid threading issues
        if 'detector' not in st.session_state:
            st.session_state.detector = VehicleDetector(model_path=model_path, vehicle_classes=selected_class_ids)
        else:
            # Update vehicle classes if needed
            st.session_state.detector.vehicle_classes = selected_class_ids
            
        detector = st.session_state.detector
        tracker = VehicleTracker(fps=fps)
        counter = VehicleCounter(actual_width, actual_height, line_position=line_position)
        
        # Set up zones with dynamic ROI
        polygon_zone = sv.PolygonZone(polygon=dynamic_source)
        view_transformer = ViewTransformer(source=dynamic_source, target=TARGET)
        
        # Draw the ROI on a sample frame to show the user
        roi_frame = test_frame.copy()
        pts = dynamic_source.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(roi_frame, [pts], True, (0, 255, 0), 2)
        
        # Add counting line
        line_y = int(actual_height * line_position)
        cv2.line(roi_frame, (0, line_y), (actual_width, line_y), (0, 0, 255), 2)
        
        # Display the ROI frame
        roi_frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        with st.session_state.video_placeholder.container():
            st.image(roi_frame_rgb, channels="RGB", use_container_width=True, caption="Region of Interest")
        
        # Initialize video writer if recording
        video_writer = None
        if record_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(record_path, f"vehicle_tracking_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                fps, 
                (actual_width, actual_height)
            )
        
        # Initialize FPS calculation
        frame_count = 0
        start_time = time.time()
        fps_update_interval = 10  # Update FPS every 10 frames
        last_successful_read_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        # Set status to running
        st.session_state.status = "Running"
        
        # Store these in session state for the main thread to access
        st.session_state.actual_width = actual_width
        st.session_state.actual_height = actual_height
        st.session_state.dynamic_source = dynamic_source
        st.session_state.polygon_zone = polygon_zone
        st.session_state.view_transformer = view_transformer
        st.session_state.tracker = tracker
        st.session_state.counter = counter
        st.session_state.detector = detector
        st.session_state.video_writer = video_writer
        st.session_state.fps = fps
        
        # Signal that initialization is complete
        st.session_state.camera_initialized = True
        
    except Exception as e:
        error_msg = f"Fatal error in camera initialization: {str(e)}"
        st.session_state.error_message = str(e)
        st.session_state.status = "Error"
        st.error(error_msg)
        st.session_state.processing = False
        if hasattr(st.session_state, 'cap') and st.session_state.cap is not None:
            st.session_state.cap.release()

# Function to display statistics
def display_statistics():
    # Create columns for statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{st.session_state.stats["in_count"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Vehicles In</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{st.session_state.stats["out_count"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Vehicles Out</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        total = st.session_state.stats["in_count"] + st.session_state.stats["out_count"]
        st.markdown(f'<div class="stat-value">{total}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Total Vehicles</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display additional statistics if data is available
    vehicle_data = st.session_state.stats['vehicle_data']
    if vehicle_data:
        # Convert to DataFrame
        df = pd.DataFrame(vehicle_data)
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Vehicle Types", "Speed Distribution"])
        
        with tab1:
            # Vehicle type distribution
            vehicle_counts = df['class_name'].value_counts().reset_index()
            vehicle_counts.columns = ['Vehicle Type', 'Count']
            
            fig = px.pie(
                vehicle_counts, 
                values='Count', 
                names='Vehicle Type',
                title='Vehicle Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Speed distribution
            if 'speed' in df.columns:
                fig = px.histogram(
                    df, 
                    x='speed',
                    color='class_name',
                    nbins=20,
                    title='Speed Distribution',
                    labels={'speed': 'Speed (km/h)', 'count': 'Number of Vehicles', 'class_name': 'Vehicle Type'},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Display raw data in an expander
        with st.expander("View Raw Data"):
            st.dataframe(df)
            
            # Download button for CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"vehicle_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# Function to update display
def update_display():
    """Update the video display and statistics."""
    if st.session_state.processing:
        try:
            # Check if we have a new frame to display
            if hasattr(st.session_state, 'current_frame') and st.session_state.current_frame is not None:
                frame = st.session_state.current_frame
                if frame is not None and frame.size > 0:
                    # Convert to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Use a container to prevent flickering
                    with st.session_state.video_placeholder.container():
                        st.image(frame_rgb, channels="RGB", use_container_width=True)
                    st.session_state.placeholder_shown = True
                    # Reset the frame_updated flag
                    st.session_state.frame_updated = False
            elif not st.session_state.placeholder_shown:
                # Show a message if no frames are available yet
                with st.session_state.video_placeholder.container():
                    st.info(f"Waiting for camera frames... Status: {st.session_state.status}")
                st.session_state.placeholder_shown = True
            
            # Update statistics
            with st.session_state.stats_placeholder.container():
                display_statistics()
        except Exception as e:
            st.error(f"Error in display update: {str(e)}")
            st.session_state.error_message = str(e)

# Initialize additional session state variables
if 'video_placeholder' not in st.session_state:
    st.session_state.video_placeholder = None
if 'stats_placeholder' not in st.session_state:
    st.session_state.stats_placeholder = None
if 'placeholder_shown' not in st.session_state:
    st.session_state.placeholder_shown = False
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'frame_updated' not in st.session_state:
    st.session_state.frame_updated = False
if 'detector' not in st.session_state:
    st.session_state.detector = None

# Main application layout
col1, col2 = st.columns([3, 1])

with col1:
    # Status indicator
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        if st.session_state.processing:
            if "Error" in st.session_state.status:
                st.error(f"Status: {st.session_state.status}")
            else:
                st.success(f"Status: {st.session_state.status}")
        elif st.session_state.error_message:
            st.error(f"Status: Error - {st.session_state.error_message}")
        else:
            st.info("Status: Ready")
    with status_col2:
        if st.session_state.processing:
            st.write(f"FPS: {st.session_state.fps:.1f}")
    
    # Video display area
    st.session_state.video_placeholder = st.empty()
    
    # Start/Stop buttons
    test_col, start_col, stop_col = st.columns(3)
    with test_col:
        test_button = st.button("Test Camera", disabled=st.session_state.processing)
    with start_col:
        start_button = st.button("Start Processing", disabled=st.session_state.processing)
    with stop_col:
        stop_button = st.button("Stop Processing", disabled=not st.session_state.processing)

with col2:
    # Statistics area
    st.session_state.stats_placeholder = st.empty()

# Handle button clicks
if test_button:
    # Test camera access
    with st.spinner("Testing camera connection..."):
        success, test_frame, message = test_camera_access(camera_source)
    
    if success:
        st.success(message)
        if test_frame is not None:
            # Show a sample frame
            test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            st.session_state.video_placeholder.image(test_frame_rgb, channels="RGB", use_container_width=True, caption="Camera Test Frame")
            
            # Show camera details
            with st.expander("Camera Details", expanded=True):
                st.write(f"**Connection URL:** {camera_source}")
                st.write(f"**Resolution:** {test_frame.shape[1]}x{test_frame.shape[0]}")
                st.write(f"**Frame Format:** {'Color' if len(test_frame.shape) == 3 else 'Grayscale'}")
                
                # Add a button to save the test frame
                test_frame_path = f"test_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(test_frame_path, test_frame)
                with open(test_frame_path, "rb") as file:
                    st.download_button(
                        label="Download Test Frame",
                        data=file,
                        file_name=test_frame_path,
                        mime="image/jpeg",
                    )
            
            # Add ROI selection button if custom ROI is enabled
            if st.session_state.custom_roi:
                if st.button("Define ROI", key="define_roi"):
                    # Show ROI selection UI
                    roi_selection()
    else:
        st.error(f"Camera test failed: {message}")
        
        # Show troubleshooting tips
        with st.expander("Troubleshooting Tips", expanded=True):
            st.markdown("""
            ### Common Issues and Solutions:
            
            1. **Connection Refused**
               - Verify the IP address and port are correct
               - Check if the camera is on the same network
               - Ensure no firewall is blocking the connection
            
            2. **Authentication Failed**
               - Double-check username and password
               - Some cameras have default credentials (admin/admin)
               - Check if the camera requires special authentication
            
            3. **Stream Path Issues**
               - Different camera brands use different stream paths
               - Try using one of the predefined camera profiles
               - Check your camera's documentation for the correct RTSP path
            
            4. **For Testing Without a Physical Camera**
               - Use the "Sample Stream for Testing" option
               - Try using a public RTSP stream (search online for "public RTSP streams")
               - Use a virtual camera software to simulate a camera feed
            """)
            
            # Add a link to find public RTSP streams
            st.markdown("[Find Public RTSP Streams](https://www.google.com/search?q=public+rtsp+streams+for+testing)")
            
            # Add a button to copy a diagnostic report
            diagnostic_info = f"""
            Camera Test Diagnostic Report
            ----------------------------
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Camera Source: {camera_source}
            Error Message: {message}
            Operating System: {os.name}
            OpenCV Version: {cv2.__version__}
            """
            st.download_button(
                label="Download Diagnostic Report",
                data=diagnostic_info,
                file_name=f"camera_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

if start_button:
    # Test camera access first
    success, test_frame, message = test_camera_access(camera_source)
    
    if success:
        st.success(message)
        if test_frame is not None:
            # Show a sample frame
            test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            st.session_state.video_placeholder.image(test_frame_rgb, channels="RGB", use_container_width=True, caption="Camera Test Frame")
        
        # Reset placeholder flag
        st.session_state.placeholder_shown = False
        
        # Start processing
        st.session_state.processing = True
        st.session_state.stop_signal = False
        
        # Start processing in a separate thread
        processing_thread = threading.Thread(target=process_camera_feed)
        processing_thread.daemon = True
        processing_thread.start()
        
        st.rerun()
    else:
        st.error(f"Camera test failed: {message}")
        st.info("Please check your camera connection and permissions.")

if stop_button:
    st.session_state.stop_signal = True
    st.rerun()

# Display initial state or update display
if not st.session_state.processing:
    # Display placeholder when not processing
    with st.session_state.video_placeholder.container():
        st.info("Click 'Start Processing' to begin real-time vehicle detection and tracking.")
    
    # Display empty statistics
    with st.session_state.stats_placeholder.container():
        display_statistics()
else:
    # Update display periodically using a callback
    update_display()
    
    # Force a rerun every second to update the display
    current_time = time.time()
    if current_time - st.session_state.last_update_time > 0.3:  # Update more frequently (every 0.3 seconds)
        st.session_state.last_update_time = current_time
        st.rerun()

# Ensure camera is properly released when the app is stopped
def on_stop():
    if hasattr(st.session_state, 'cap') and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        
# Register the on_stop function to be called when the app is stopped
try:
    st.on_script_stop(on_stop)
except:
    # Fallback for older Streamlit versions
    pass

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Developed with ❤️ using YOLOv8, Supervision, and Streamlit</p>
    <p>© 2025 Vehicle Detection, Tracking, Counting, and Speed Estimation</p>
</div>
""", unsafe_allow_html=True) 

