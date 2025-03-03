"""
Utility functions for video handling.
"""
import supervision as sv
import cv2

def get_video_info(video_path):
    """
    Get information about a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        sv.VideoInfo: Video information object
    """
    return sv.VideoInfo.from_video_path(video_path)

def get_frame_generator(video_path):
    """
    Create a generator that yields frames from a video.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        generator: Generator that yields frames
    """
    return sv.get_video_frames_generator(video_path)

def get_first_frame(video_path):
    """
    Get the first frame of a video.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        np.ndarray: First frame of the video
    """
    frame_generator = get_frame_generator(video_path)
    return next(frame_generator)

def create_video_writer(output_path, frame_width, frame_height, fps):
    """
    Create a video writer for saving processed frames.
    
    Args:
        output_path (str): Path to save the output video
        frame_width (int): Width of the video frames
        frame_height (int): Height of the video frames
        fps (float): Frames per second
        
    Returns:
        cv2.VideoWriter: Video writer object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height)) 