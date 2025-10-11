import os, requests

# Automatically download YOLOv8s if missing
MODEL_PATH = "yolov8s.pt"
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found — downloading YOLOv8s.pt, please wait...")
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(url).content)


import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
from collections import deque

# Streamlit configuration
st.set_page_config(page_title="Smart Traffic Flow Analyzer", layout="centered")

st.title("🚦 Traffic Congestion Detection Using Computer Vision  ")
st.write("Analyzes road congestion using computer vision — detects if the road has traffic or free flow conditions.")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")  # small model, more accurate than yolov8n

model = load_model()

# Sidebar options
option = st.sidebar.radio("Select Input Type", ("📷 Live Camera", "🖼️ Upload Image", "🎞️ Upload Video"))
conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.3)

# Keep recent area ratios for smoothing
ratios = deque(maxlen=5)

# Vehicle classes (COCO IDs)
vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


def analyze_flow(frame):
    """Estimate traffic congestion based on road occupancy ratio."""
    results = model(frame, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
    height, width, _ = frame.shape

    vehicle_area = 0
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        vehicle_area += (x2 - x1) * (y2 - y1)

    ratio = vehicle_area / (height * width)
    ratios.append(ratio)
    avg_ratio = sum(ratios) / len(ratios)

    # Traffic condition classification
    if avg_ratio < 0.04:
        status = "🟢 FREE FLOW"
    else:
        status = "🔴 TRAFFIC"

    annotated = results[0].plot()
    return annotated, status, round(avg_ratio * 100, 2)


# -------------------------------
# 📷 CAMERA INPUT
# -------------------------------
if option == "📷 Live Camera":
    st.write("### Live Camera Mode")
    camera_input = st.camera_input("Capture a frame")

    if camera_input:
        img = Image.open(camera_input)
        frame = np.array(img)
        annotated, status, ratio = analyze_flow(frame)
        st.image(annotated, caption=f"Detected Condition: {status} ({ratio}% road coverage)", use_column_width=True)
        st.success(status)


# -------------------------------
# 🖼️ IMAGE UPLOAD
# -------------------------------
elif option == "🖼️ Upload Image":
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        frame = np.array(img)
        annotated, status, ratio = analyze_flow(frame)
        st.image(annotated, caption=f"Detected Condition: {status} ({ratio}% road coverage)", use_column_width=True)
        st.success(status)


# -------------------------------
# 🎞️ VIDEO UPLOAD
# -------------------------------
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
            annotated, status, ratio = analyze_flow(frame)
            stframe.image(annotated, channels="RGB", use_column_width=True)
            traffic_status = status

        cap.release()
        st.success(f"Final Estimated Condition: {traffic_status}")
