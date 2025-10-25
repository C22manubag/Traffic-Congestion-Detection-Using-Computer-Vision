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
st.set_page_config(page_title="🚦 Traffic Motion Analyzer", layout="centered")
st.title("🚗 Traffic Congestion Detection using YOLO + CV")
st.caption("Analyze uploaded traffic videos using YOLO-based vehicle detection and real-time analytics.")

# ------------------------------------------------------------
# VIDEO UPLOAD
# ------------------------------------------------------------
uploaded_video = st.file_uploader("📹 Upload a Traffic Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    # Save to temp
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Display compact preview
    st.video(tfile.name, format="video/mp4", start_time=0)
    st.info("⏳ Initializing YOLOv8 and processing video...")
    time.sleep(1)

    # Load YOLOv8 model (CPU-safe)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt").to(device)

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps / 2)  # process every half second

    FRAME_WINDOW = st.empty()
    progress = st.progress(0)

    # --- Data trackers ---
    motion_levels, confidences, timestamps = [], [], []
    frame_idx = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Process every Nth frame
        if frame_idx % frame_interval == 0:
            # --- Fix: ensure frame is valid uint8 array ---
            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            # YOLO inference (CPU-safe)
            results = model.predict(source=frame, verbose=False, device=device)
            boxes = results[0].boxes

            vehicle_count = 0
            total_conf = 0.0

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                # Only detect vehicles
                if label in ["car", "bus", "truck", "motorbike"]:
                    vehicle_count += 1
                    total_conf += conf
                    (x1, y1, x2, y2) = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            avg_conf = total_conf / max(1, vehicle_count)
            confidences.append(avg_conf)
            motion_levels.append(vehicle_count)
            timestamps.append(time.time() - start_time)

            # Traffic classification
            if vehicle_count > 25:
                traffic_status = "🔴 Traffic"
                color = (0, 0, 255)
            elif vehicle_count > 10:
                traffic_status = "🟡 Moderate"
                color = (0, 255, 255)
            else:
                traffic_status = "🟢 Free Flow"
                color = (0, 255, 0)

            # Confidence closeness
            closeness = min(100, avg_conf * 100)
            status_text = f"{traffic_status} | Vehicles: {vehicle_count} | Confidence: {avg_conf:.2f} ({closeness:.0f}%)"
            cv2.putText(frame, status_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            resized = cv2.resize(frame, (480, 360))
            FRAME_WINDOW.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), channels="RGB")

        progress.progress(min(frame_idx / total_frames, 1.0))
        frame_idx += 1

    cap.release()
    progress.empty()
    st.success("✅ Analysis Complete!")

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------
    st.subheader("📊 Analysis Summary")

    avg_motion = np.mean(motion_levels) if motion_levels else 0
    avg_conf = np.mean(confidences) if confidences else 0
    if avg_motion < 10:
        final_status = "🟢 Mostly Free Flow"
    elif avg_motion > 25:
        final_status = "🔴 Heavy Traffic"
    else:
        final_status = "🟡 Moderate Flow"

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Vehicle Count", f"{avg_motion:.1f}")
    col2.metric("Avg Confidence", f"{avg_conf:.2f}")
    col3.metric("Status", final_status)

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------
    if motion_levels:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=timestamps, y=motion_levels, mode="lines+markers", name="Vehicle Count"))
        fig1.update_layout(title="🚗 Vehicle Count Over Time", xaxis_title="Time (s)", yaxis_title="Count", height=350)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=timestamps, y=confidences, mode="lines+markers", name="Avg Confidence"))
        fig2.update_layout(title="🧠 Confidence Rate Over Time", xaxis_title="Time (s)", yaxis_title="Confidence", height=350)
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("📂 Upload a traffic video to begin analysis.")
