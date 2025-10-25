import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import pandas as pd
import plotly.express as px
import time
from collections import defaultdict

st.set_page_config(page_title="Traffic Flow Analyzer", layout="wide")
st.title("🚦 Smart Traffic Congestion Detection using YOLOv8")
st.write("Upload a traffic video to detect and classify vehicles, track motion, and analyze flow accuracy.")

uploaded_file = st.file_uploader("📤 Upload a Traffic Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Load YOLO model (use n or m)
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1  # analyze every frame

    # Data trackers
    total_objects_detected = 0
    class_counts = defaultdict(int)
    moving_objects = 0
    stationary_objects = 0
    all_confidences = []
    frame_confidences = []
    frame_times = []
    prev_positions = {}

    progress = st.progress(0)
    stframe = st.empty()
    frame_idx = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = model(frame, verbose=False)[0]
            detections = results.boxes
            frame_conf = []
            current_positions = {}

            for i, box in enumerate(detections):
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]

                if label not in ["car", "truck", "bus", "motorbike"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Save for motion comparison
                current_positions[i] = (cx, cy)

                all_confidences.append(conf)
                frame_conf.append(conf)
                total_objects_detected += 1
                class_counts[label] += 1

                # Draw bounding box
                color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Motion detection: compare with previous frame
            for obj_id, (cx, cy) in current_positions.items():
                if obj_id in prev_positions:
                    prev_cx, prev_cy = prev_positions[obj_id]
                    dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    if dist < 5:
                        stationary_objects += 1
                    else:
                        moving_objects += 1

            prev_positions = current_positions

            frame_confidences.append(np.mean(frame_conf) if frame_conf else 0)
            frame_times.append(frame_idx / fps if fps > 0 else frame_idx)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        frame_idx += 1
        progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()

    # Compute summary
    avg_conf = np.mean(all_confidences) if all_confidences else 0
    total_movement = moving_objects + stationary_objects
    movement_ratio = moving_objects / total_movement if total_movement > 0 else 0

    if movement_ratio < 0.45:
        status = "🚨 Traffic Detected"
    else:
        status = "✅ Free Flow"

    st.subheader("📊 Traffic Analysis Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧾 Total Vehicles", total_objects_detected)
    col2.metric("🚗 Moving Vehicles", moving_objects)
    col3.metric("🚙 Stationary Vehicles", stationary_objects)

    st.write(f"**Average Confidence:** {avg_conf * 100:.2f}%")
    st.success(f"### Final Result: {status}")

    # Charts
    if class_counts:
        st.plotly_chart(px.bar(
            x=list(class_counts.keys()),
            y=list(class_counts.values()),
            color=list(class_counts.keys()),
            title="Vehicle Type Distribution"
        ))

    conf_df = pd.DataFrame({"Time": frame_times, "Confidence": frame_confidences})
    st.plotly_chart(px.line(conf_df, x="Time", y="Confidence", title="Confidence Over Time"))

else:
    st.info("👆 Upload a video file to start analyzing traffic flow.")
