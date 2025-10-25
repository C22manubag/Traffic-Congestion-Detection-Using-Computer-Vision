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
    # --- Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # --- Load YOLOv8 Medium model (better accuracy)
    model = YOLO("yolov8m.pt")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // 3)

    # --- Data trackers
    total_objects_detected = 0
    class_counts = defaultdict(int)
    object_tracks = {}   # {id: [(x, y, time), ...]}
    moving_objects = 0
    stationary_objects = 0
    all_confidences = []
    frame_confidences = []
    frame_times = []

    progress = st.progress(0)
    stframe = st.empty()

    frame_idx = 0
    start_time = time.time()

    # --- Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = model.track(frame, persist=True, verbose=False)
            if results[0].boxes.id is None:
                frame_idx += 1
                continue

            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            detections = boxes.data.cpu().numpy()
            current_time = time.time() - start_time

            frame_conf = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls = det[:6]
                conf = float(conf)
                label = model.names[int(cls)]
                obj_id = ids[i]

                # Filter: only road vehicles
                if label not in ["car", "truck", "bus", "motorbike"]:
                    continue

                all_confidences.append(conf)
                frame_conf.append(conf)
                total_objects_detected += 1
                class_counts[label] += 1

                # --- Track movement
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track = object_tracks.get(obj_id, [])
                track.append((cx, cy, current_time))
                object_tracks[obj_id] = track[-20:]  # store only last few for speed

                # Movement distance between last 2 frames
                if len(track) > 2:
                    dx = track[-1][0] - track[-2][0]
                    dy = track[-1][1] - track[-2][1]
                    dist = np.sqrt(dx**2 + dy**2)

                    # Vehicle still for >3s or very small movement
                    if dist < 3:
                        stationary_objects += 1
                    else:
                        moving_objects += 1

                # Draw bounding box
                color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            frame_confidences.append(np.mean(frame_conf) if frame_conf else 0)
            frame_times.append(frame_idx / fps)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        frame_idx += 1
        progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    end_time = time.time()
    duration = end_time - start_time

    # --- Summaries ---
    st.subheader("📊 Traffic Analysis Summary")

    avg_conf = np.mean(all_confidences) if all_confidences else 0
    total_movement = moving_objects + stationary_objects
    movement_ratio = moving_objects / total_movement if total_movement > 0 else 0

    # Traffic logic
    if movement_ratio < 0.45:
        status = "🚨 Traffic Detected"
        traffic_closeness = round((1 - movement_ratio) * 100, 2)
        freeflow_closeness = 100 - traffic_closeness
    else:
        status = "✅ Free Flow"
        freeflow_closeness = round(movement_ratio * 100, 2)
        traffic_closeness = 100 - freeflow_closeness

    col1, col2, col3 = st.columns(3)
    col1.metric("🧾 Total Vehicles", total_objects_detected)
    col2.metric("🚗 Moving Vehicles", moving_objects)
    col3.metric("🚙 Stationary Vehicles", stationary_objects)

    st.write(f"**Average Confidence:** {avg_conf * 100:.2f}%")
    st.write(f"**Traffic Closeness:** {traffic_closeness:.2f}%")
    st.write(f"**Free Flow Closeness:** {freeflow_closeness:.2f}%")
    st.write(f"**Analysis Duration:** {duration:.2f}s")
    st.success(f"### Final Result: {status}")

    # --- Visualization Section ---
    st.subheader("📈 Visual Analytics")

    # Vehicle class distribution
    st.write("**Detected Vehicle Types**")
    class_df = pd.DataFrame(list(class_counts.items()), columns=["Vehicle Type", "Count"])
    if not class_df.empty:
        st.plotly_chart(px.bar(class_df, x="Vehicle Type", y="Count", color="Vehicle Type", text="Count"))

    # Confidence trend
    st.write("**Confidence Level Over Time**")
    conf_df = pd.DataFrame({"Time (s)": frame_times, "Confidence": frame_confidences})
    st.plotly_chart(px.line(conf_df, x="Time (s)", y="Confidence", title="Model Confidence per Frame"))

    # Traffic vs Free Flow pie
    st.write("**Traffic vs Free Flow Ratio**")
    closeness_df = pd.DataFrame(
        {
            "Condition": ["Traffic Closeness", "Free Flow Closeness"],
            "Percentage": [traffic_closeness, freeflow_closeness],
        }
    )
    st.plotly_chart(px.pie(closeness_df, names="Condition", values="Percentage", title="Traffic State Ratio"))

else:
    st.info("👆 Upload a video file to start analyzing traffic flow.")
