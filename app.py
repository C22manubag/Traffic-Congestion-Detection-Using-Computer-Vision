import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
from collections import deque
import os, requests

# ---------------------------------------
# Streamlit App Config
# ---------------------------------------
st.set_page_config(page_title="Smart Traffic Flow Analyzer", layout="centered")

st.title("🚦 Traffic Congestion Detection Using Computer Vision")
st.write("Analyzes road congestion using object detection — classifies traffic as Free Flow, Moderate, or Heavy based on vehicle count and road coverage.")

# ---------------------------------------
# Auto-download YOLO model if missing
# ---------------------------------------
MODEL_PATH = "yolov8s.pt"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading YOLOv8s model... please wait ⏳")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        with open(MODEL_PATH, "wb") as f:
            f.write(requests.get(url).content)
        st.success("✅ Model downloaded successfully!")

ensure_model()

# ---------------------------------------
# Load YOLO Model
# ---------------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)  # small model = better accuracy/speed balance

model = load_model()

# ---------------------------------------
# Sidebar Configuration
# ---------------------------------------
st.sidebar.header("⚙️ Configuration")
option = st.sidebar.radio("Select Input Type", ("📷 Live Camera", "🖼️ Upload Image", "🎞️ Upload Video"))
conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.4)
area_threshold = st.sidebar.slider("Traffic Area Threshold (%)", 1, 20, 10)

# Vehicle classes from COCO dataset
vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Keep recent ratios for smoothing
ratios = deque(maxlen=10)
vehicle_counts = deque(maxlen=10)

# ---------------------------------------
# Core Traffic Analysis Function
# ---------------------------------------
def analyze_flow(frame):
    """Estimate traffic congestion using both vehicle count and area ratio."""
    results = model(frame, conf=conf_threshold, verbose=False)
    height, width, _ = frame.shape
    boxes = results[0].boxes

    vehicle_count = 0
    vehicle_area = 0

    # Loop through detected objects
    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        cls_id = int(cls)
        if cls_id in vehicle_classes and conf > conf_threshold:
            x1, y1, x2, y2 = box
            vehicle_count += 1
            vehicle_area += (x2 - x1) * (y2 - y1)

    # Calculate area ratio
    area_ratio = vehicle_area / (height * width)
    ratios.append(area_ratio)
    vehicle_counts.append(vehicle_count)

    avg_ratio = np.mean(ratios)
    avg_count = np.mean(vehicle_counts)

    # Smarter human-like classification
    if avg_count <= 5 and avg_ratio < 0.05:
        status = "🟢 FREE FLOW"
    elif avg_count <= 15 or avg_ratio < 0.15:
        status = "🟡 MODERATE TRAFFIC"
    else:
        status = "🔴 HEAVY TRAFFIC"

    annotated = results[0].plot()
    annotated = cv2.putText(
        annotated,
        f"{status} | Vehicles: {int(avg_count)} | {avg_ratio*100:.1f}% area",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if "FREE" in status else (0, 255, 255) if "MODERATE" in status else (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    return annotated, status, round(avg_ratio * 100, 2), int(avg_count)


# ---------------------------------------
# 📷 LIVE CAMERA
# ---------------------------------------
if option == "📷 Live Camera":
    st.write("### Live Camera Mode")
    camera_input = st.camera_input("Capture a frame")

    if camera_input:
        img = Image.open(camera_input)
        frame = np.array(img)
        annotated, status, ratio, count = analyze_flow(frame)
        st.image(annotated, caption=f"{status} | {count} vehicles | {ratio}% coverage", use_column_width=True)
        st.success(status)

# ---------------------------------------
# 🖼️ IMAGE UPLOAD
# ---------------------------------------
elif option == "🖼️ Upload Image":
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        frame = np.array(img)
        annotated, status, ratio, count = analyze_flow(frame)
        st.image(annotated, caption=f"{status} | {count} vehicles | {ratio}% coverage", use_column_width=True)
        st.success(status)

# ---------------------------------------
# 🎞️ VIDEO UPLOAD
# ---------------------------------------
elif option == "🎞️ Upload Video":
    video = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov"])
    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        traffic_status = "Analyzing..."

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated, status, ratio, count = analyze_flow(frame)
            stframe.image(annotated, channels="RGB", use_column_width=True)
            traffic_status = status

        cap.release()
        st.success(f"Final Estimated Condition: {traffic_status}")
