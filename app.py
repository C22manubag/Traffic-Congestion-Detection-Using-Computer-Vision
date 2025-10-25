import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import time
from collections import deque
import plotly.graph_objects as go

# -----------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------
st.set_page_config(page_title="🚦 Live Traffic Monitor", layout="wide")
st.title("🚦 Smart Live Traffic Flow Analyzer")
st.caption("Analyze real-time vehicle traffic using your browser webcam feed.")

# -----------------------------------------------
# MODEL LOADING
# -----------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# -----------------------------------------------
# CONFIGURATION PANEL
# -----------------------------------------------
with st.sidebar:
    st.header("⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.2, 0.9, 0.5, 0.05)
    area_threshold = st.slider("Area Threshold (%)", 1, 15, 8, 1)
    st.info("Using YOLOv8s pretrained model on COCO dataset.")
    st.caption("Detects: car, motorcycle, bus, truck")

vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# -----------------------------------------------
# DATA BUFFERS
# -----------------------------------------------
ratios = deque(maxlen=10)
vehicle_counts = deque(maxlen=10)
motions = deque(maxlen=10)
prev_positions = {}

# -----------------------------------------------
# ANALYSIS FUNCTION
# -----------------------------------------------
def analyze_flow(frame):
    global prev_positions
    results = model.track(frame, conf=conf_threshold, persist=True, verbose=False, classes=list(vehicle_classes.keys()))
    height, width, _ = frame.shape

    boxes = results[0].boxes
    vehicle_count, vehicle_area = 0, 0
    movements = []
    current_positions = {}

    for box in boxes:
        if box.id is None:
            continue
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in vehicle_classes and conf > conf_threshold:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            vid = int(box.id.cpu().numpy())
            current_positions[vid] = (cx, cy)
            vehicle_count += 1
            vehicle_area += (x2 - x1) * (y2 - y1)
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

    # Determine traffic condition
    if avg_motion > 8 and avg_ratio < 0.08:
        status, color = "🟢 FREE FLOW", (0, 255, 0)
    elif avg_motion > 3 or avg_ratio < 0.15:
        status, color = "🟡 MODERATE", (0, 255, 255)
    else:
        status, color = "🔴 STUCK / HEAVY", (0, 0, 255)

    annotated = results[0].plot()
    cv2.putText(
        annotated,
        f"{status} | Move:{avg_motion:.1f} | Veh:{int(avg_count)} | {avg_ratio*100:.1f}%",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated, status, avg_ratio * 100, avg_count, avg_motion

# -----------------------------------------------
# MAIN CAMERA STREAM
# -----------------------------------------------
main_col, analytics_col = st.columns([3, 1])

st.markdown("### 📷 Capture a Frame or Stream from Your Webcam")

camera_image = st.camera_input("Click below to capture a frame")

if camera_image is not None:
    img = Image.open(camera_image)
    frame = np.array(img)
    annotated, status, ratio, count, motion = analyze_flow(frame)

    with main_col:
        st.image(annotated, use_column_width=True)

    with analytics_col:
        st.metric("Traffic", status)
        st.metric("Vehicles", int(count))
        st.metric("Coverage (%)", f"{ratio:.2f}")
        st.metric("Motion", f"{motion:.2f}")
