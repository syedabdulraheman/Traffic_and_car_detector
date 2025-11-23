import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic design
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0;
    }
    .subtitle {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 30px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .upload-box {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .detection-stats {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stat-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
    .stat-label {
        font-weight: 600;
        color: #333;
    }
    .stat-value {
        color: #667eea;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1>🎯 AI Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image or video to detect objects with advanced AI</p>", unsafe_allow_html=True)

# Load YOLO model with caching
@st.cache_resource
def load_model():
    try:
        # Using YOLOv8s (small) for better accuracy on vehicles
        model = YOLO('yolov8s.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to draw boxes and labels
def draw_detections(image, results, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    """Draw bounding boxes and labels on the image"""
    img = image.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{result.names[cls]} {conf:.2f}"
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw label background
            cv2.rectangle(
                img,
                (x1, y1 - label_height - 10),
                (x1 + label_width + 10, y1),
                box_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )
    
    return img

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help="Lower value detects more objects (recommended: 0.20 for vehicles)"
    )
    
    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Controls overlapping detections"
    )
    
    st.markdown("---")
    st.markdown("### 🚗 Vehicle Detection")
    st.success("""
    **Optimized for detecting:**
    - 🚗 Cars
    - 🏍️ Motorcycles  
    - 🚌 Buses
    - 🚚 Trucks
    - 🚲 Bicycles
    
    Using YOLOv8s for high accuracy!
    """)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This app uses **YOLOv8 Small** model for better vehicle detection accuracy.
    
    **80+ detectable objects** including all types of vehicles and common objects.
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Media")
    
    upload_type = st.radio("Select input type:", ["Image", "Video"], horizontal=True)
    
    if upload_type == "Image":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect objects"
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video to detect objects"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        # Load model
        model = load_model()
        
        if model is not None:
            if upload_type == "Image":
                # Process image
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Run detection
                with st.spinner("🔍 Detecting objects..."):
                    results = model(img_array, conf=confidence_threshold, iou=iou_threshold, imgsz=640)
                    
                    # Draw detections
                    detected_img = draw_detections(img_array, results)
                    
                    # Display results
                    st.markdown("### 🎨 Detection Results")
                    st.image(detected_img, use_container_width=True)
                    
                    # Show statistics
                    st.markdown("<div class='detection-stats'>", unsafe_allow_html=True)
                    st.markdown("### 📈 Detection Statistics")
                    
                    # Count detected objects
                    detected_classes = {}
                    for result in results:
                        for box in result.boxes:
                            cls_name = result.names[int(box.cls[0])]
                            detected_classes[cls_name] = detected_classes.get(cls_name, 0) + 1
                    
                    st.markdown(f"<div class='stat-item'><span class='stat-label'>Total Objects:</span><span class='stat-value'>{sum(detected_classes.values())}</span></div>", unsafe_allow_html=True)
                    
                    for cls_name, count in detected_classes.items():
                        st.markdown(f"<div class='stat-item'><span class='stat-label'>{cls_name}:</span><span class='stat-value'>{count}</span></div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            else:  # Video processing
                st.markdown("### 🎬 Processing Video")
                
                # Save uploaded video to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                
                # Create output video file
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                
                # Process video
                cap = cv2.VideoCapture(video_path)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Create video writer with H.264 codec for better quality
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                stframe = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_num = 0
                
                status_text.info(f"🎥 Processing video at {fps} FPS... Please wait.")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run detection on EVERY frame with optimized settings for vehicles
                    results = model(frame, conf=confidence_threshold, iou=iou_threshold, imgsz=640, verbose=False)
                    frame_detected = draw_detections(frame, results)
                    
                    # Write frame to output video
                    out.write(frame_detected)
                    
                    # Display preview (every 2nd frame to keep UI responsive)
                    if frame_num % 2 == 0:
                        frame_rgb = cv2.cvtColor(frame_detected, cv2.COLOR_BGR2RGB)
                        stframe.image(frame_rgb, use_container_width=True)
                    
                    # Update progress
                    frame_num += 1
                    progress_bar.progress(min(frame_num / frame_count, 1.0))
                
                cap.release()
                out.release()
                
                status_text.empty()
                st.success(f"✅ Video processing complete! Processed {frame_num} frames.")
                
                # Read the output video for download
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Create two columns for buttons and info
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    # Download button
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name="detected_objects.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
                with col_btn2:
                    st.info(f"🎬 Output: {fps} FPS | {width}x{height}")
                
                # Display the processed video
                st.markdown("### 🎥 Processed Video Preview")
                st.markdown("*The video below includes detections on every frame for smooth playback*")
                st.video(output_path)
                
                # Clean up temp files
                os.unlink(video_path)
                os.unlink(output_path)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: white; opacity: 0.8;'>Powered by YOLOv8 & Streamlit | Real-time Object Detection</p>",
    unsafe_allow_html=True
)