import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time

st.set_page_config(page_title="YOLOv8 Live Traffic Detection", layout="wide")
st.title("🚦 YOLOv8 Real-Time Traffic Detection")
st.write("This demo uses your local webcam for live object detection (run locally, not online).")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar settings
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 0.9, 0.5)
st.sidebar.write("Press **Stop** or close tab to end stream.")

start_button = st.button("▶ Start Live Detection")

FRAME_WINDOW = st.image([])

if start_button:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("🚫 No webcam found. Please connect a camera and restart.")
    else:
        prev_time = 0
        fps_display_interval = 1  # seconds
        frame_rate = 0
        frame_count = 0
        start_time = time.time()

        st.info("🎥 Streaming started. Press **Stop** or close window to end.")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Camera disconnected.")
                break

            frame = cv2.flip(frame, 1)  # mirror view
            results = model(frame, conf=conf_threshold)
            annotated = results[0].plot()

            # Simple Traffic Logic
            vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
            count = 0
            for box in results[0].boxes:
                if int(box.cls[0]) in vehicle_classes and float(box.conf[0]) > conf_threshold:
                    count += 1

            status = "🟢 Free Flow" if count < 5 else "🔴 Traffic"
            color = (0, 255, 0) if count < 5 else (0, 0, 255)

            cv2.putText(
                annotated,
                f"{status} | Vehicles: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )

            # Stream to Streamlit
            FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            # Optional delay (30 FPS target)
            time.sleep(0.03)

            # Stop condition if Streamlit reruns
            if not st.session_state.get("running", True):
                break

        cap.release()
        st.success("✅ Stream ended.")
