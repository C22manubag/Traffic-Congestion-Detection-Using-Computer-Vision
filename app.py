import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import time

st.set_page_config(page_title="Traffic Flow Analyzer", layout="wide")
st.title("🚗 Traffic Congestion Detection using YOLOv8")

st.write("Upload a traffic video to analyze flow, detect vehicles, and determine if the scene is **Traffic** or **Free Flow.**")

uploaded_file = st.file_uploader("📤 Upload a Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save the uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // 3)

    moving_objects = 0
    stationary_objects = 0
    all_confidences = []
    total_objects_detected = 0
    prev_positions = {}

    progress = st.progress(0)
    start_time = time.time()

    # Placeholder for displaying the video
    stframe = st.empty()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every few frames for speed
        if frame_idx % frame_interval == 0:
            results = model(frame, verbose=False)
            detections = results[0].boxes.data.cpu().numpy()

            frame_objects = 0
            for det in detections:
                x1, y1, x2, y2, conf, cls = det[:6]
                label = model.names[int(cls)]
                if label in ["car", "bus", "truck", "motorbike", "bicycle"]:
                    frame_objects += 1
                    all_confidences.append(float(conf))
                    total_objects_detected += 1
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    prev = prev_positions.get(label, (cx, cy))
                    movement = np.linalg.norm(np.array([cx, cy]) - np.array(prev))
                    prev_positions[label] = (cx, cy)

                    if movement > 5:
                        moving_objects += 1
                    else:
                        stationary_objects += 1

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Update video preview
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        frame_idx += 1
        progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    end_time = time.time()
    duration = end_time - start_time

    # --- Analysis Section ---
    st.subheader("📊 Traffic Analysis Summary")

    avg_conf = np.mean(all_confidences) if all_confidences else 0
    total_movement = moving_objects + stationary_objects
    movement_ratio = moving_objects / total_movement if total_movement > 0 else 0

    # Determine traffic status
    if movement_ratio < 0.4:
        status = "🚨 Traffic"
        traffic_closeness = round((1 - movement_ratio) * 100, 2)
        freeflow_closeness = 100 - traffic_closeness
    else:
        status = "✅ Free Flow"
        freeflow_closeness = round(movement_ratio * 100, 2)
        traffic_closeness = 100 - freeflow_closeness

    col1, col2, col3 = st.columns(3)
    col1.metric("🧾 Total Vehicles Detected", total_objects_detected)
    col2.metric("🚗 Moving Vehicles", moving_objects)
    col3.metric("🚙 Stationary Vehicles", stationary_objects)

    st.write(f"**Overall Confidence Level:** {avg_conf*100:.2f}%")
    st.write(f"**Traffic Closeness:** {traffic_closeness:.2f}%")
    st.write(f"**Free Flow Closeness:** {freeflow_closeness:.2f}%")
    st.write(f"**Analysis Duration:** {duration:.2f} seconds")
    st.success(f"### Final Result: {status}")

else:
    st.info("👆 Please upload a video file to start the analysis.")
