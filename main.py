"""
Streamlit web app with:
 - Simple login page
 - Realtime webcam object detection using ultralytics YOLO/RT-DETR model

Notes:
 - This demo uses simple, in-memory credentials (NOT secure). For production use a proper auth system.
 - Make sure yolov8 model file (yolov8n.pt) is available or the ultralytics package can download it.
"""

import streamlit as st
import time
import cv2
import numpy as np
from threading import Thread

# -----------------------
# Configuration & helper
# -----------------------
st.set_page_config(page_title="YOLOv8 Realtime Demo", layout="wide")

# Simple "users" for demo
VALID_USERS = {
    "admin": "admin123",
    "user": "password"
}

# -----------------------
# Authentication
# -----------------------
def login_page():
    st.title("Group 15 Project — Smart Surveillance")
    st.markdown("Please login to continue.")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Login"):
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Logged in as {username}")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid username or password")
    with col2:
        if st.button("Clear"):
            st.rerun()


def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    # Stop camera if running
    if "camera_running" in st.session_state:
        st.session_state["camera_running"] = False
    st.rerun()


# -----------------------
# YOLO Model Loading
# -----------------------
@st.cache_resource
def load_model(model_name="yolov8n.pt"):
    """Load YOLO/RT-DETR model - only called after login"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Make sure ultralytics is installed: pip install ultralytics")
        return None


# -----------------------
# Detection Page
# -----------------------
def detection_page():
    st.title("🎥 Realtime Object Detection")
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Detection Controls")
    
    model_choice = st.sidebar.selectbox(
        "Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "rtdetr-l.pt"],
        index=0
    )
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.05, 0.95, 0.35, 0.05
    )
    
    camera_source = st.sidebar.selectbox(
        "Camera Source",
        [0, 1, 2],
        format_func=lambda x: f"Camera {x}",
        index=0
    )
    
    show_fps = st.sidebar.checkbox("Show FPS", value=True)
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    
    # Account info
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 👤 Account")
    st.sidebar.write(f"**User:** {st.session_state.get('username', 'Unknown')}")
    if st.sidebar.button("Logout", use_container_width=True):
        logout()
    
    # Main area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Live Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.markdown("### Statistics")
        stats_placeholder = st.empty()
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        start_btn = st.button("▶️ Start Camera", use_container_width=True, type="primary")
    
    with col_btn2:
        stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)
    
    # Initialize camera running state
    if "camera_running" not in st.session_state:
        st.session_state["camera_running"] = False
    
    # Start camera
    if start_btn:
        st.session_state["camera_running"] = True
    
    # Stop camera
    if stop_btn:
        st.session_state["camera_running"] = False
    
    # Run detection loop
    if st.session_state["camera_running"]:
        run_detection(
            frame_placeholder=frame_placeholder,
            stats_placeholder=stats_placeholder,
            model_name=model_choice,
            conf_threshold=conf_threshold,
            camera_source=camera_source,
            show_fps=show_fps,
            show_labels=show_labels
        )
    else:
        frame_placeholder.info("📷 Click 'Start Camera' to begin detection")
        stats_placeholder.markdown("""
        **Ready to start**
        
        - Select model and settings
        - Click Start Camera
        - Allow camera access
        """)


def run_detection(frame_placeholder, stats_placeholder, model_name, conf_threshold, 
                  camera_source, show_fps, show_labels):
    """Run the detection loop with live camera feed"""
    
    # Load model
    with st.spinner(f"Loading {model_name}..."):
        model = load_model(model_name)
    
    if model is None:
        st.error("Failed to load model. Check terminal for errors.")
        st.session_state["camera_running"] = False
        return
    
    # Open camera
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        st.error(f"⚠️ Failed to open camera {camera_source}")
        st.session_state["camera_running"] = False
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Detection stats
    frame_count = 0
    start_time = time.time()
    fps = 0
    total_detections = 0
    
    st.success("✅ Camera started successfully!")
    
    # Detection loop
    while st.session_state.get("camera_running", False):
        ret, frame = cap.read()
        
        if not ret:
            st.warning("⚠️ Failed to grab frame")
            break
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        
        # Run model prediction
        try:
            results = model.predict(
                source=frame, 
                conf=conf_threshold,
                verbose=False,
                stream=False
            )
            
            # Get annotated frame
            if show_labels:
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame.copy()
            
            # Count detections
            num_detections = len(results[0].boxes)
            total_detections += num_detections
            
            # Add FPS overlay if enabled
            if show_fps:
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
            
            # Convert BGR to RGB for Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # Update statistics
            detected_classes = []
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    detected_classes.append(cls_name)
            
            # Count unique classes
            class_counts = {}
            for cls in detected_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            stats_placeholder.markdown(f"""
            **📊 Live Stats**
            
            **FPS:** {fps:.1f}  
            **Frames:** {frame_count}  
            **Current Detections:** {num_detections}  
            **Total Detections:** {total_detections}
            
            **Detected Objects:**
            """)
            
            if class_counts:
                for cls, count in class_counts.items():
                    stats_placeholder.markdown(f"- **{cls}**: {count}")
            else:
                stats_placeholder.markdown("_No objects detected_")
        
        except Exception as e:
            st.error(f"Detection error: {e}")
            break
        
        # Small delay to prevent overwhelming the UI
        time.sleep(0.01)
    
    # Release resources
    cap.release()
    st.info("Camera stopped")


# -----------------------
# Main
# -----------------------
def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None

    # Route to appropriate page
    if not st.session_state["logged_in"]:
        login_page()
    else:
        detection_page()


if __name__ == "__main__":
    main()