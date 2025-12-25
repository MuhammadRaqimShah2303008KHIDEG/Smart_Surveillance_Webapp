"""
Streamlit web app with:
 - Simple login page
 - Realtime webcam detection using ultralytics YOLO models
 - Support for: Object Detection, Pose Estimation, Instance Segmentation

Notes:
 - This demo uses simple, in-memory credentials (NOT secure).
 - Models are downloaded automatically by ultralytics if not present.
"""

import streamlit as st
import time
import cv2

# -----------------------
# Configuration
# -----------------------
st.set_page_config(page_title="YOLO Multi-Task Demo", layout="wide")

VALID_USERS = {"admin": "admin123", "user": "password"}

# Model configurations for each task
MODEL_CONFIG = {
    "Object Detection": {
        "models": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "rtdetr-l.pt"],
        "description": "Detect and classify objects in the scene",
        "icon": "🎯"
    },
    "Pose Estimation": {
        "models": ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt"],
        "description": "Detect human body keypoints and skeleton",
        "icon": "🏃"
    },
    "Instance Segmentation": {
        "models": ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt"],
        "description": "Segment individual object instances with masks",
        "icon": "🎨"
    }
}

# -----------------------
# Authentication
# -----------------------
def login_page():
    st.title("🎥 Smart Surveillance System")
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
    if "camera_running" in st.session_state:
        st.session_state["camera_running"] = False
    st.rerun()


# -----------------------
# Model Loading
# -----------------------
@st.cache_resource
def load_model(model_name):
    """Load YOLO model - cached for performance"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Install ultralytics: pip install ultralytics")
        return None


# -----------------------
# Detection Page
# -----------------------
def detection_page():
    st.title("🎥 Realtime Computer Vision")
    
    # Sidebar - Task Selection
    st.sidebar.markdown("## 🧠 Task Selection")
    
    task = st.sidebar.radio(
        "Choose Task",
        list(MODEL_CONFIG.keys()),
        format_func=lambda x: f"{MODEL_CONFIG[x]['icon']} {x}"
    )
    
    st.sidebar.info(MODEL_CONFIG[task]["description"])
    
    # Sidebar - Model & Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎛️ Model Settings")
    
    model_choice = st.sidebar.selectbox(
        "Model Size",
        MODEL_CONFIG[task]["models"],
        index=0,
        help="Larger models (l) are more accurate but slower"
    )
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.05, 0.95, 0.35, 0.05
    )
    
    # Task-specific settings
    if task == "Instance Segmentation":
        show_masks = st.sidebar.checkbox("Show Segmentation Masks", value=True)
        mask_alpha = st.sidebar.slider("Mask Opacity", 0.1, 1.0, 0.5, 0.1)
    else:
        show_masks = False
        mask_alpha = 0.5
    
    if task == "Pose Estimation":
        show_skeleton = st.sidebar.checkbox("Show Skeleton Lines", value=True)
        show_keypoints = st.sidebar.checkbox("Show Keypoints", value=True)
    else:
        show_skeleton = True
        show_keypoints = True
    
    # Camera settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📷 Camera Settings")
    
    camera_source = st.sidebar.selectbox(
        "Camera Source",
        [0, 1, 2],
        format_func=lambda x: f"Camera {x}",
        index=0
    )
    
    show_fps = st.sidebar.checkbox("Show FPS", value=True)
    show_labels = st.sidebar.checkbox("Show Labels/Boxes", value=True)
    
    # Account info
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 👤 Account")
    st.sidebar.write(f"**User:** {st.session_state.get('username', 'Unknown')}")
    if st.sidebar.button("Logout", use_container_width=True):
        logout()
    
    # Main area - Task info header
    st.markdown(f"### {MODEL_CONFIG[task]['icon']} {task}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### Live Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.markdown("#### Statistics")
        stats_placeholder = st.empty()
    
    # Control buttons
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    
    with col_btn1:
        start_btn = st.button("▶️ Start", use_container_width=True, type="primary")
    
    with col_btn2:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)
    
    if "camera_running" not in st.session_state:
        st.session_state["camera_running"] = False
    
    if start_btn:
        st.session_state["camera_running"] = True
    
    if stop_btn:
        st.session_state["camera_running"] = False
    
    # Run detection
    if st.session_state["camera_running"]:
        run_detection(
            frame_placeholder=frame_placeholder,
            stats_placeholder=stats_placeholder,
            model_name=model_choice,
            task=task,
            conf_threshold=conf_threshold,
            camera_source=camera_source,
            show_fps=show_fps,
            show_labels=show_labels,
            show_masks=show_masks,
            mask_alpha=mask_alpha,
            show_skeleton=show_skeleton,
            show_keypoints=show_keypoints
        )
    else:
        frame_placeholder.info("📷 Click 'Start' to begin")
        stats_placeholder.markdown(f"""
        **Ready for {task}**
        
        - Model: {model_choice}
        - Confidence: {conf_threshold}
        - Camera: {camera_source}
        """)


def run_detection(frame_placeholder, stats_placeholder, model_name, task,
                  conf_threshold, camera_source, show_fps, show_labels,
                  show_masks, mask_alpha, show_skeleton, show_keypoints):
    """Run detection loop with support for all task types"""
    
    with st.spinner(f"Loading {model_name}..."):
        model = load_model(model_name)
    
    if model is None:
        st.session_state["camera_running"] = False
        return
    
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        st.error(f"⚠️ Failed to open camera {camera_source}")
        st.session_state["camera_running"] = False
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    total_detections = 0
    
    st.success(f"✅ {task} started!")
    
    while st.session_state.get("camera_running", False):
        ret, frame = cap.read()
        
        if not ret:
            st.warning("⚠️ Failed to grab frame")
            break
        
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
        
        try:
            # Run prediction
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False,
                stream=False
            )
            
            # Get annotated frame based on task
            if show_labels:
                if task == "Instance Segmentation" and show_masks:
                    annotated = results[0].plot(masks=True)
                elif task == "Pose Estimation":
                    annotated = results[0].plot(
                        boxes=show_labels,
                        kpt_line=show_skeleton,
                        kpt_radius=3 if show_keypoints else 0
                    )
                else:
                    annotated = results[0].plot()
            else:
                annotated = frame.copy()
            
            # Count detections
            num_det = len(results[0].boxes) if results[0].boxes is not None else 0
            total_detections += num_det
            
            # FPS overlay
            if show_fps:
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(annotated, f"Task: {task}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Convert to RGB
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
            
            # Build statistics based on task
            stats_md = f"""
            **📊 Live Stats**
            
            **Task:** {task}  
            **FPS:** {fps:.1f}  
            **Frames:** {frame_count}  
            **Current:** {num_det}  
            **Total:** {total_detections}
            
            ---
            """
            
            if task == "Object Detection":
                stats_md += "\n**Detected Objects:**\n"
                if results[0].boxes is not None:
                    class_counts = {}
                    for box in results[0].boxes:
                        cls_name = model.names[int(box.cls[0])]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
                        stats_md += f"- **{cls}**: {cnt}\n"
                else:
                    stats_md += "_No objects detected_"
            
            elif task == "Pose Estimation":
                stats_md += "\n**Pose Info:**\n"
                if results[0].keypoints is not None:
                    num_people = len(results[0].keypoints)
                    stats_md += f"- **People detected:** {num_people}\n"
                    stats_md += f"- **Keypoints/person:** 17\n"
                else:
                    stats_md += "_No poses detected_"
            
            elif task == "Instance Segmentation":
                stats_md += "\n**Segmentation Info:**\n"
                if results[0].masks is not None:
                    num_masks = len(results[0].masks)
                    stats_md += f"- **Segments:** {num_masks}\n"
                    class_counts = {}
                    for box in results[0].boxes:
                        cls_name = model.names[int(box.cls[0])]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
                        stats_md += f"- **{cls}**: {cnt}\n"
                else:
                    stats_md += "_No segments detected_"
            
            stats_placeholder.markdown(stats_md)
        
        except Exception as e:
            st.error(f"Error: {e}")
            break
        
        time.sleep(0.01)
    
    cap.release()
    st.info("Camera stopped")


# -----------------------
# Main
# -----------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None

    if not st.session_state["logged_in"]:
        login_page()
    else:
        detection_page()


if __name__ == "__main__":
    main()
