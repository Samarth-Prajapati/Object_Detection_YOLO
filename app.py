import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import uuid
import av

# Set page configuration with vibrant theme
st.set_page_config(page_title="YOLOv11 Vision Hub 🌟", page_icon="🎥", layout="wide")

# Custom CSS for vibrant and colorful UI
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 8px; }
    .stSelectbox, .stSlider { background-color: #e6f3ff; border-radius: 8px; }
    .sidebar .sidebar-content { background-color: #d9e6ff; }
    h1, h2, h3 { color: #2c3e50; }
    .stAlert { background-color: #ffeaa7; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for file management
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Create directories for saving files (only images and videos)
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/videos", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# Sidebar configuration
st.sidebar.title("YOLOv11 Vision Hub ⚙️")
st.sidebar.markdown("Configure your object detection settings below! 🚀")

# Task selection
task = st.sidebar.selectbox("Select Task 🎯", ["Detection", "Segmentation", "Pose"], help="Choose the YOLOv11 task to perform.")

# Confidence threshold (1-100 scale)
confidence = st.sidebar.slider("Confidence Threshold (%) 🔍", 1, 100, 50, 1, help="Set the confidence threshold (1-100%) for detections.")
confidence = confidence / 100.0  # Convert to 0.01-1.0 for YOLO

# Source selection
source = st.sidebar.selectbox("Select Source 📸", ["Image", "Video", "Webcam"], help="Choose the input source for processing.")

# File uploader for image or video
uploaded_file = None
if source in ["Image", "Video"]:
    uploaded_file = st.sidebar.file_uploader(f"Upload {source} 📤", type=["jpg", "png", "jpeg"] if source == "Image" else ["mp4", "avi", "mov"], help=f"Upload a {source.lower()} file.")

# Detect button
detect_button = st.sidebar.button("Detect Objects! 🕵️‍♂️", help="Start the detection process.")

# Main screen
st.title("YOLOv11 Vision Hub 🌈")
st.markdown("Welcome to your one-stop solution for real-time object detection, segmentation, and pose estimation! 🎉 Upload an image/video or use your webcam to see YOLOv11 in action! ✨")

# Load YOLOv11 model based on task
model_map = {
    "Detection": "yolo11n.pt",
    "Segmentation": "yolo11n-seg.pt",
    "Pose": "yolo11n-pose.pt"
}
model_path = os.path.join("weights", model_map[task])
if not os.path.exists(model_path):
    st.warning(f"Downloading {model_map[task]}... Please wait! ⏳")
    try:
        model = YOLO(model_map[task])  # This will download the model if not present
        model.save(model_path)
    except Exception as e:
        st.error(f"Error downloading model: {e} 😞")
        st.stop()
else:
    model = YOLO(model_path)

# Function to process image
def process_image(image, model, task, confidence):
    try:
        results = model.predict(image, conf=confidence, verbose=False)
        if results and len(results) > 0:
            annotated_image = results[0].plot()
            return annotated_image
        else:
            st.warning("No detections found in image! 😕")
            return image
    except Exception as e:
        st.error(f"Image processing error: {e} 😞")
        return image

# Function to process video
def process_video(video_path, model, task, confidence):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file! 😞")
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or width <= 0 or height <= 0:
            st.error("Invalid video properties! 😞")
            cap.release()
            return None
        
        output_path = f"uploads/videos/processed_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.error("Error creating output video file! 😞")
            cap.release()
            return None
        
        progress = st.progress(0)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=confidence, verbose=False)
            annotated_frame = results[0].plot() if results and len(results) > 0 else frame
            out.write(annotated_frame)
            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))
        
        cap.release()
        out.release()
        if frame_count == 0:
            st.error("No frames processed in video! 😞")
            return None
        return output_path
    except Exception as e:
        st.error(f"Video processing error: {e} 😞")
        return None

# Webcam processor class
class YOLOv11WebcamProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.task = task
        self.confidence = confidence
    
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(img, conf=self.confidence, verbose=False)
            annotated_frame = results[0].plot() if results and len(results) > 0 else img
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception as e:
            st.error(f"Webcam processing error: {e} 😞")
            return frame

# Handle detection logic
if detect_button and not st.session_state.webcam_active:
    if source == "Webcam":
        st.session_state.webcam_active = True
    elif uploaded_file:
        unique_id = str(uuid.uuid4())
        file_ext = uploaded_file.name.split(".")[-1]
        save_path = f"uploads/{'images' if source == 'Image' else 'videos'}/{unique_id}.{file_ext}"
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_files.append(save_path)
        
        if source == "Image":
            st.subheader("Original Image 🖼️")
            st.image(save_path, use_column_width=True)
            image = Image.open(save_path)
            image_np = np.array(image)
            annotated_image = process_image(image_np, model, task, confidence)
            st.subheader("Detected Image 🎨")
            st.image(annotated_image, use_column_width=True)
        
        elif source == "Video":
            st.subheader("Original Video 🎬")
            st.video(save_path)
            with st.spinner("Processing video... Please wait! ⏳"):
                processed_video = process_video(save_path, model, task, confidence)
            if processed_video:
                st.subheader("Detected Video 🎥")
                st.video(processed_video)

# Webcam handling
if source == "Webcam" and st.session_state.webcam_active:
    st.subheader("Live Webcam Feed 📹")
    st.info("Please ensure your webcam is enabled and browser permissions allow camera access. If the feed doesn't load, try refreshing the page or using a different browser. 🛠️")
    webrtc_ctx = webrtc_streamer(
        key="yolov11-webcam",
        video_processor_factory=YOLOv11WebcamProcessor,
        rtc_configuration=RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        }),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    if st.button("Stop Webcam 🛑"):
        st.session_state.webcam_active = False
        st.experimental_rerun()

# Display saved files
if st.session_state.uploaded_files:
    st.subheader("Saved Files 📂")
    for file in st.session_state.uploaded_files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            st.image(file, caption=os.path.basename(file), width=200)
        elif file.endswith((".mp4", ".avi", ".mov")):
            st.video(file)