import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import plotly.graph_objects as go
from ultralytics import YOLO
import torch

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="🚦 Traffic Flow Intelligence", layout="centered")
st.title("🚗 Traffic Flow Intelligence using YOLO + Motion Detection")
st.caption("Analyzes vehicle movement, confidence rate, and congestion level in uploaded traffic videos.")

# ------------------------------------------------------------
# UPLOAD VIDEO
# ------------------------------------------------------------
uploaded_video = st.file_uploader("📹 Upload a Traffic Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    # Save temp
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    st.video(tfile.name)
    st.info("⏳ Initializing YOLOv8 model... Please wait.")

    # Load YOLOv8 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt").to(device)
    st.success("✅ Model loaded successfully.")

    # ------------------------------------------------------------
    # VIDEO PROCESSING
    # ------------------------------------------------------------
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps / 2)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    frame_placeholder = st.empty()

    # Motion detection reference
    ret, prev_frame = cap.read()
    if not ret:
        st.error("❌ Could not read video. Try another file.")
        st.stop()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Tracking data
    timestamps, vehicle_counts, avg_confidences, motion_scores = [], [], [], []
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.sum(diff > 30) / diff.size * 100
            motion_scores.append(motion_score)
            prev_gray = gray

            # --- YOLO detection ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(frame_rgb, conf=0.4, verbose=False, device=device)
            boxes = results[0].boxes

            vehicle_count, total_conf = 0, 0.0
            vehicle_labels = ["car", "bus", "truck", "motorbike", "jeep"]

            for box in boxes:
                cls = int(box.cls[0])
                label = model.names.get(cls, "obj")
                conf = float(box.conf[0])
                if label in vehicle_labels:
                    vehicle_count += 1
                    total_conf += conf
                    (x1, y1, x2, y2) = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            avg_conf = total_conf / max(1, vehicle_count)
            avg_confidences.append(avg_conf)
            vehicle_counts.append(vehicle_count)
            timestamps.append(time.time() - start_time)

            # --- Traffic Classification Logic ---
            if vehicle_count == 0:
                status = "🟢 Free Flow"
                color = (0, 255, 0)
            elif motion_score < 5 and vehicle_count > 5:
                status = "🔴 Traffic"
                color = (0, 0, 255)
            elif motion_score < 15:
                status = "🟡 Slow Flow"
                color = (0, 255, 255)
            else:
                status = "🟢 Free Flow"
                color = (0, 255, 0)

            # --- Realness Metric ---
            realness = np.clip((avg_conf * 50 + motion_score) / 1.5, 0, 100)

            text = f"{status} | Vehicles: {vehicle_count} | Motion: {motion_score:.1f}% | Conf: {avg_conf:.2f} | Realness: {realness:.1f}%"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            resized = cv2.resize(frame, (480, 360))
            frame_placeholder.image(resized, channels="BGR", use_container_width=False)
            progress.progress(frame_idx / total_frames)

        frame_idx += 1

    cap.release()
    progress.empty()
    st.success("✅ Video Analysis Complete!")

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------
    st.subheader("📊 Summary Report")

    avg_vehicle = np.mean(vehicle_counts) if vehicle_counts else 0
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    avg_conf = np.mean(avg_confidences) if avg_confidences else 0

    if avg_motion < 8:
        overall_status = "🔴 Heavy Traffic"
    elif avg_motion < 20:
        overall_status = "🟡 Moderate Flow"
    else:
        overall_status = "🟢 Free Flow"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Vehicles", f"{avg_vehicle:.1f}")
    col2.metric("Avg Motion (%)", f"{avg_motion:.1f}")
    col3.metric("Avg Confidence", f"{avg_conf:.2f}")
    col4.metric("Overall Status", overall_status)

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------
    if timestamps:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=timestamps, y=vehicle_counts, mode="lines+markers", name="Vehicles"))
        fig1.add_trace(go.Scatter(x=timestamps, y=motion_scores, mode="lines+markers", name="Motion %"))
        fig1.update_layout(title="🚦 Vehicle Count & Motion Over Time", xaxis_title="Time (s)", height=350)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=timestamps, y=avg_confidences, mode="lines+markers", name="Confidence"))
        fig2.update_layout(title="🧠 Confidence Rate Over Time", xaxis_title="Time (s)", yaxis_title="Confidence", height=350)
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("📂 Upload a traffic video to start analysis.")
