import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import time
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="🚦 YOLOv8 Traffic Flow Analyzer", layout="wide")
st.title("🚦 YOLOv8 Traffic Flow Analyzer (Smooth Playback)")
st.caption("Upload a traffic video to analyze flow and confidence in near real-time.")

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 0.9, 0.5)
st.sidebar.info("YOLOv8n pretrained model detects: car, motorcycle, bus, truck")

uploaded_video = st.file_uploader("📹 Upload a Traffic Video", type=["mp4", "mov", "avi"])
FRAME_WINDOW = st.empty()
status_box = st.empty()

if uploaded_video:
    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1 / fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    progress = st.progress(0)

    # Tracking metrics
    conf_data, closeness_data, count_data, timeline = [], [], [], []
    start_time = time.time()
    frame_idx = 0
    st.info("⏳ Running analysis — please wait...")

    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = model(frame, conf=conf_threshold)
        boxes = results[0].boxes
        annotated = results[0].plot()

        # --- Analysis logic ---
        vehicle_classes = [2, 3, 5, 7]
        count, total_conf = 0, 0
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in vehicle_classes and conf > conf_threshold:
                count += 1
                total_conf += conf

        avg_conf = (total_conf / count) if count > 0 else 0
        closeness = min(100, count * 15 + (1 - avg_conf) * 50)
        status = "🟢 Free Flow" if closeness < 50 else "🔴 Traffic"
        color = (0, 255, 0) if status == "🟢 Free Flow" else (0, 0, 255)

        cv2.putText(
            annotated,
            f"{status} | Veh:{count} | Conf:{avg_conf:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

        # --- Stream frame smoothly ---
        FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        status_box.markdown(f"**{status}** — Vehicles: `{count}`, Confidence: `{avg_conf:.2f}`, Closeness: `{closeness:.1f}%`")

        # Collect data
        frame_idx += 1
        conf_data.append(avg_conf)
        closeness_data.append(closeness)
        count_data.append(count)
        timeline.append(time.time() - start_time)
        progress.progress(min(frame_idx / frame_count, 1.0))

        # Sleep to maintain playback FPS
        time.sleep(frame_delay)

    cap.release()
    progress.empty()
    st.success("✅ Video analysis complete!")

    # ------------------------------------------------------------
    # SUMMARY + VISUALS
    # ------------------------------------------------------------
    st.subheader("📈 Analysis Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Confidence", f"{np.mean(conf_data)*100:.2f}%")
    col2.metric("Average Closeness", f"{np.mean(closeness_data):.1f}%")
    col3.metric("Average Vehicle Count", f"{np.mean(count_data):.1f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline, y=[c*100 for c in conf_data], mode="lines", name="Confidence (%)"))
    fig.add_trace(go.Scatter(x=timeline, y=closeness_data, mode="lines", name="Closeness (%)"))
    fig.update_layout(
        title="📊 Confidence & Traffic Closeness Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Percentage / Value",
        height=400,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("📂 Please upload a video to start analysis.")
