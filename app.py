import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import plotly.graph_objects as go

# -----------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------
st.set_page_config(page_title="🚦 Live Traffic Monitor", layout="wide")
st.title("🚦 Smart Live Traffic Flow Analyzer")
st.caption("Real-time YOLOv8-based vehicle detection using your webcam feed.")

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
# MAIN STREAM HANDLER (WORKING LOCAL CAMERA)
# -----------------------------------------------
main_col, analytics_col = st.columns([3, 1])
start_btn = st.button("▶️ Start Live Stream")

if start_btn:
    st.info("Starting live webcam... press **Stop** to end.")
    frame_placeholder = main_col.empty()
    stop_btn = st.button("⏹ Stop Stream")

    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        st.error("🚫 Cannot access camera. Try a different index (1, 2) or check permissions.")
    else:
        timeline, motion_data, density_data, count_data = [], [], [], []
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ No camera frame detected.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated, status, ratio, count, motion = analyze_flow(frame)

            # Display live frame
            frame_placeholder.image(annotated, channels="RGB", use_column_width=True)

            # Record metrics
            timeline.append(time.time() - start_time)
            motion_data.append(motion)
            density_data.append(ratio)
            count_data.append(count)

            with analytics_col:
                st.metric("Status", status)
                st.metric("Vehicles", int(count))
                st.metric("Area (%)", f"{ratio:.2f}")
                st.metric("Motion", f"{motion:.2f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=timeline, y=density_data, mode="lines", name="Density"))
                fig.add_trace(go.Scatter(x=timeline, y=motion_data, mode="lines", name="Motion"))
                fig.add_trace(go.Scatter(x=timeline, y=count_data, mode="lines", name="Vehicles"))
                fig.update_layout(
                    title="📈 Live Traffic Metrics",
                    xaxis_title="Time (s)",
                    yaxis_title="Value",
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Break loop if Stop button pressed
            if stop_btn:
                st.success("✅ Stream stopped.")
                break

        cap.release()


