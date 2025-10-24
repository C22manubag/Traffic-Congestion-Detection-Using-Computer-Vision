import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
from collections import deque
import os, requests, time
import plotly.graph_objects as go

# ----------------------------------------------------
# 🌐 STREAMLIT PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Smart Traffic Flow Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Smart Traffic Flow Analyzer")
st.write(
    "This system analyzes live or recorded traffic footage using computer vision. "
    "It detects vehicle motion and density to classify the road condition as **Free Flow**, **Moderate**, or **Stuck**."
)

# ----------------------------------------------------
# 📥 MODEL SETUP
# ----------------------------------------------------
MODEL_PATH = "yolov8s.pt"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading YOLOv8s model... please wait ⏳")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        with open(MODEL_PATH, "wb") as f:
            f.write(requests.get(url).content)
        st.success("✅ Model downloaded successfully!")

ensure_model()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ----------------------------------------------------
# ⚙️ SIDEBAR CONFIGURATION
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    input_type = st.radio("Input Source", ["📷 Live Camera", "🎞️ Upload Video", "🖼️ Upload Image"])
    conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.4)
    area_threshold = st.slider("Area Threshold (%)", 1, 20, 10)
    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.write("YOLOv8s - pre-trained on COCO dataset")
    st.write("Detects: car, motorbike, bus, truck")

vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# ----------------------------------------------------
# 📊 ANALYTICS SETUP (RIGHT SIDEBAR)
# ----------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    div[data-testid="stVerticalBlock"] div:first-child {flex: 3;}
    div[data-testid="stVerticalBlock"] div:last-child {flex: 1; background-color: #f8f9fa; padding: 20px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Placeholder containers
main_col, analytics_col = st.columns([3, 1])

# ----------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------
ratios = deque(maxlen=10)
vehicle_counts = deque(maxlen=10)
motions = deque(maxlen=10)
prev_positions = {}

# ----------------------------------------------------
# 🧠 CORE ANALYSIS FUNCTION
# ----------------------------------------------------
def analyze_flow(frame):
    """Estimate traffic congestion using vehicle count, area, and motion."""
    global prev_positions

    results = model.track(frame, conf=conf_threshold, persist=True, verbose=False, classes=[2, 3, 5, 7])
    height, width, _ = frame.shape

    boxes = results[0].boxes
    vehicle_count = 0
    vehicle_area = 0
    movements = []
    current_positions = {}

    # Loop through detected vehicles
    for box in boxes:
        if box.id is None:
            continue

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in vehicle_classes and conf > conf_threshold:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # center
            vid = int(box.id.cpu().numpy())

            current_positions[vid] = (cx, cy)
            vehicle_count += 1
            vehicle_area += (x2 - x1) * (y2 - y1)

            # Movement distance
            if vid in prev_positions:
                px, py = prev_positions[vid]
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                movements.append(dist)

    prev_positions = current_positions

    avg_move = np.mean(movements) if movements else 0
    area_ratio = vehicle_area / (height * width)
    ratios.append(area_ratio)
    vehicle_counts.append(vehicle_count)
    motions.append(avg_move)

    avg_ratio = np.mean(ratios)
    avg_count = np.mean(vehicle_counts)
    avg_motion = np.mean(motions)

    # Combine density + motion logic
    if avg_motion > 8 and avg_ratio < 0.08:
        status = "🟢 FREE FLOW"
        color = (0, 255, 0)
    elif avg_motion > 3 or avg_ratio < 0.15:
        status = "🟡 MODERATE"
        color = (0, 255, 255)
    else:
        status = "🔴 STUCK / HEAVY"
        color = (0, 0, 255)

    # Annotate
    annotated = results[0].plot()
    cv2.putText(
        annotated,
        f"{status} | Move: {avg_motion:.1f} | Veh: {int(avg_count)} | {avg_ratio*100:.1f}% area",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA
    )

    return annotated, status, avg_ratio * 100, avg_count, avg_motion


# ----------------------------------------------------
# 🖼️ INPUT HANDLING
# ----------------------------------------------------
if input_type == "🖼️ Upload Image":
    uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        frame = np.array(img)
        annotated, status, ratio, count, motion = analyze_flow(frame)
        with main_col:
            st.image(annotated, use_column_width=True)
        with analytics_col:
            st.metric("Traffic Status", status)
            st.metric("Avg Vehicles", int(count))
            st.metric("Road Coverage (%)", f"{ratio:.1f}")
            st.metric("Avg Motion", f"{motion:.2f}")
            st.info("📷 Image mode: static frame analysis only")

elif input_type == "🎞️ Upload Video":
    video = st.file_uploader("Upload road video", type=["mp4", "avi", "mov"])
    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = main_col.empty()

        # Initialize analytics data
        timeline = []
        motion_data, density_data, count_data = [], [], []

        start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated, status, ratio, count, motion = analyze_flow(frame)

            frame_placeholder.image(annotated, channels="RGB", use_column_width=True)
            timeline.append(time.time() - start)
            motion_data.append(motion)
            density_data.append(ratio)
            count_data.append(count)

            with analytics_col:
                st.metric("Current Status", status)
                st.metric("Vehicles", int(count))
                st.metric("Area (%)", f"{ratio:.2f}")
                st.metric("Motion", f"{motion:.2f}")

                # Live updating chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=timeline, y=density_data, mode="lines", name="Area %"))
                fig.add_trace(go.Scatter(x=timeline, y=motion_data, mode="lines", name="Motion"))
                fig.add_trace(go.Scatter(x=timeline, y=count_data, mode="lines", name="Vehicles"))
                fig.update_layout(
                    title="📈 Live Traffic Metrics",
                    xaxis_title="Time (s)",
                    yaxis_title="Value",
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        cap.release()
        with analytics_col:
            st.success(f"Final Condition: {status}")

elif input_type == "📷 Live Camera":
    st.write("### Live Camera Mode")
    camera_input = st.camera_input("Capture a frame")
    if camera_input:
        img = Image.open(camera_input)
        frame = np.array(img)
        annotated, status, ratio, count, motion = analyze_flow(frame)
        with main_col:
            st.image(annotated, use_column_width=True)
        with analytics_col:
            st.metric("Traffic Status", status)
            st.metric("Vehicles", int(count))
            st.metric("Coverage (%)", f"{ratio:.2f}")
            st.metric("Motion", f"{motion:.2f}")
            st.info("📸 Snapshot analysis (no live feed)")
