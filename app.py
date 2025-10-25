import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import tempfile
import time
from collections import deque

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(page_title="🚦 Traffic Flow Analyzer", layout="wide")
st.title("🚦 Smart Traffic Flow Analyzer")
st.caption("Detect and analyze vehicle traffic using YOLOv8 in real-time or via video upload.")

# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.2, 0.9, 0.5, 0.05)
    mode = st.radio("Select Input Mode", ["📹 Upload Video", "📷 Webcam Capture"])
    st.info("YOLOv8s pretrained model on COCO dataset — detects car, motorcycle, bus, truck.")

vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# --------------------------------------------------
# BUFFERS FOR ANALYSIS
# --------------------------------------------------
confidences = deque(maxlen=30)
ratios = deque(maxlen=30)
vehicle_counts = deque(maxlen=30)
motions = deque(maxlen=30)
prev_positions = {}

# --------------------------------------------------
# ANALYSIS FUNCTION
# --------------------------------------------------
def analyze_flow(frame):
    global prev_positions
    results = model.track(frame, conf=conf_threshold, persist=True, verbose=False, classes=list(vehicle_classes.keys()))
    height, width, _ = frame.shape

    boxes = results[0].boxes
    total_conf, vehicle_area, vehicle_count = 0, 0, 0
    movements, current_positions = [], {}

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
            total_conf += conf
            vehicle_area += (x2 - x1) * (y2 - y1)
            if vid in prev_positions:
                px, py = prev_positions[vid]
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                movements.append(dist)

    prev_positions = current_positions
    avg_conf = total_conf / vehicle_count if vehicle_count > 0 else 0
    avg_move = np.mean(movements) if movements else 0
    area_ratio = vehicle_area / (height * width)

    confidences.append(avg_conf)
    ratios.append(area_ratio)
    vehicle_counts.append(vehicle_count)
    motions.append(avg_move)

    avg_ratio = np.mean(ratios)
    avg_motion = np.mean(motions)

    # Determine Traffic Flow
    if avg_motion > 5 and avg_ratio < 0.12:
        status, color = "🟢 Free Flow", (0, 255, 0)
        closeness = min(100, (avg_motion / 5) * 100)
    else:
        status, color = "🔴 Traffic", (0, 0, 255)
        closeness = min(100, (avg_ratio / 0.25) * 100)

    annotated = results[0].plot()
    cv2.putText(
        annotated,
        f"{status} | Conf:{avg_conf:.2f} | Veh:{vehicle_count} | Move:{avg_motion:.1f}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated, status, avg_conf, avg_ratio * 100, vehicle_count, avg_motion, closeness


# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------
st.subheader("🎛 Select Input Mode")
st.write("Choose a method to analyze traffic:")

col1, col2 = st.columns([3, 1])
results_placeholder = st.empty()
stats_placeholder = st.empty()

# --------------------------------------------------
# WEBCAM MODE
# --------------------------------------------------
if mode == "📷 Webcam Capture":
    camera_feed = st.camera_input("Capture a Frame for Analysis")

    if camera_feed is not None:
        img = Image.open(camera_feed)
        frame = np.array(img)
        annotated, status, conf, ratio, count, motion, closeness = analyze_flow(frame)

        with col1:
            st.image(annotated, use_container_width=True)

        results_placeholder.markdown(f"### Current Result: {status}")
        stats_placeholder.write(
            f"**Confidence:** {conf*100:.2f}% | **Vehicle Count:** {count} | **Motion:** {motion:.2f} | **Closeness:** {closeness:.1f}%"
        )

        if st.button("📊 Generate Full Statistics"):
            st.subheader("📈 Summary Statistics")
            st.metric("Traffic Status", status)
            st.metric("Average Confidence", f"{np.mean(confidences)*100:.2f}%")
            st.metric("Average Flow Closeness", f"{np.mean(closeness):.1f}%")
            st.metric("Vehicle Count", int(np.mean(vehicle_counts)))


# --------------------------------------------------
# VIDEO UPLOAD MODE
# --------------------------------------------------
else:
    uploaded_video = st.file_uploader("📹 Upload a traffic video (mp4, mov, avi)", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = col1.empty()
        progress = st.progress(0)
        start_time = time.time()
        timeline, conf_data, closeness_data = [], [], []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, status, conf, ratio, count, motion, closeness = analyze_flow(frame)
            frame_placeholder.image(annotated, channels="RGB", use_column_width=True)

            timeline.append(time.time() - start_time)
            conf_data.append(conf)
            closeness_data.append(closeness)
            processed_frames += 1
            progress.progress(processed_frames / frame_count)

        cap.release()
        progress.empty()

        results_placeholder.markdown(f"### Final Result: {status}")

        if st.button("📊 Generate Full Statistics"):
            st.subheader("📈 Statistical Summary")
            col2.metric("Final Status", status)
            col2.metric("Average Confidence", f"{np.mean(conf_data)*100:.2f}%")
            col2.metric("Average Flow Closeness", f"{np.mean(closeness_data):.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timeline, y=conf_data, mode="lines", name="Confidence"))
            fig.add_trace(go.Scatter(x=timeline, y=closeness_data, mode="lines", name="Flow Closeness"))
            fig.update_layout(
                title="📊 Confidence & Flow Closeness Over Time",
                xaxis_title="Time (s)",
                yaxis_title="Value",
                height=350,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

