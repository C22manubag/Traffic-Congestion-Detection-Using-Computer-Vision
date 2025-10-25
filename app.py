import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
from collections import deque
import os, requests, time
import plotly.graph_objects as go

# ----------------------------------------------------
# 🌐 STREAMLIT PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Smart Traffic Flow Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Smart Traffic Flow Analyzer")
st.markdown(
    "Analyze live or recorded traffic footage using **computer vision** to detect congestion levels — "
    "**Free Flow**, **Moderate**, or **Stuck**."
)

# ----------------------------------------------------
# 📥 MODEL SETUP
# ----------------------------------------------------
MODEL_PATH = "yolov8s.pt"

def ensure_model():
    """Ensure YOLO model exists locally before loading."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 5_000_000:
        st.warning("Downloading YOLOv8s model... Please wait ⏳")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        r = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.success("✅ Model downloaded successfully!")

ensure_model()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ----------------------------------------------------
# 🧭 SIDEBAR CONFIGURATION
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Properly aligned and smaller layout
    st.markdown(
        """
        <style>
        [data-testid="stRadio"] label {display: block; padding-bottom: 6px;}
        .stSlider {margin-bottom: 15px;}
        </style>
        """,
        unsafe_allow_html=True
    )

    input_type = st.radio(
        "Select Input Source:",
        ["📷 Live Camera", "🎞️ Upload Video", "🖼️ Upload Image"],
        horizontal=False
    )

    conf_threshold = st.slider("Detection Confidence", 0.2, 0.9, 0.5, 0.05)
    area_threshold = st.slider("Area Threshold (%)", 1, 15, 8, 1)

    st.divider()
    st.subheader("🧠 Model Info")
    st.caption("YOLOv8s - pre-trained on COCO dataset")
    st.caption("Detects: car, motorcycle, bus, truck")

vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# ----------------------------------------------------
# 📊 LAYOUT: MAIN + ANALYTICS COLUMN
# ----------------------------------------------------
main_col, analytics_col = st.columns([3, 1])

ratios = deque(maxlen=10)
vehicle_counts = deque(maxlen=10)
motions = deque(maxlen=10)
prev_positions = {}

# ----------------------------------------------------
# 🧠 CORE ANALYSIS FUNCTION
# ----------------------------------------------------
def analyze_flow(frame):
    global prev_positions
    results = model.track(frame, conf=conf_threshold, persist=True, verbose=False, classes=list(vehicle_classes.keys()))
    height, width, _ = frame.shape

    boxes = results[0].boxes
    vehicle_count = 0
    vehicle_area = 0
    movements = []
    current_positions = {}

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
            vehicle_area += (x2 - x1) * (y2 - y1)

            if vid in prev_positions:
                px, py = prev_positions[vid]
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                movements.append(dist)

    prev_positions = current_positions
    avg_move = np.mean(movements) if movements else 0
    area_ratio = vehicle_area / (height * width)

    ratios.append(area_ratio)
    vehicle_counts.append(vehicle_count)
    motions.append(avg_move)

    avg_ratio = np.mean(ratios)
    avg_count = np.mean(vehicle_counts)
    avg_motion = np.mean(motions)

    # Logic
    if avg_motion > 8 and avg_ratio < 0.08:
        status = "🟢 FREE FLOW"
        color = (0, 255, 0)
    elif avg_motion > 3 or avg_ratio < 0.15:
        status = "🟡 MODERATE"
        color = (0, 255, 255)
    else:
        status = "🔴 STUCK / HEAVY"
        color = (0, 0, 255)

    annotated = results[0].plot()
    cv2.putText(
        annotated,
        f"{status} | Move:{avg_motion:.1f} | Veh:{int(avg_count)} | {avg_ratio*100:.1f}%",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA
    )
    return annotated, status, avg_ratio * 100, avg_count, avg_motion


# ----------------------------------------------------
# 🖼️ INPUT HANDLING
# ----------------------------------------------------
if input_type == "🖼️ Upload Image":
    uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        frame = np.array(img)
        annotated, status, ratio, count, motion = analyze_flow(frame)

        with main_col:
            st.image(annotated, use_column_width=True)

        with analytics_col:
            st.metric("Traffic", status)
            st.metric("Vehicles", int(count))
            st.metric("Coverage (%)", f"{ratio:.1f}")
            st.metric("Motion", f"{motion:.2f}")
            st.info("📷 Static image analysis only")

elif input_type == "🎞️ Upload Video":
    video = st.file_uploader("Upload road video", type=["mp4", "avi", "mov"])
    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = main_col.empty()

        timeline, motion_data, density_data, count_data = [], [], [], []
        start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated, status, ratio, count, motion = analyze_flow(frame)
            frame_placeholder.image(annotated, channels="RGB", use_column_width=True)

            timeline.append(time.time() - start)
            motion_data.append(motion)
            density_data.append(ratio)
            count_data.append(count)

            with analytics_col:
                st.metric("Status", status)
                st.metric("Vehicles", int(count))
                st.metric("Area (%)", f"{ratio:.2f}")
                st.metric("Motion", f"{motion:.2f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=timeline, y=density_data, mode="lines", name="Area %"))
                fig.add_trace(go.Scatter(x=timeline, y=motion_data, mode="lines", name="Motion"))
                fig.add_trace(go.Scatter(x=timeline, y=count_data, mode="lines", name="Vehicles"))
                fig.update_layout(
                    title="📈 Live Traffic Metrics",
                    xaxis_title="Time (s)",
                    yaxis_title="Value",
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        cap.release()
        with analytics_col:
            st.success(f"Final Condition: {status}")

elif input_type == "📷 Live Camera":
    st.write("### Live Camera Mode")
    camera_input = st.camera_input("Capture a frame")
    if camera_input:
        img = Image.open(camera_input)
        frame = np.array(img)
        annotated, status, ratio, count, motion = analyze_flow(frame)
        with main_col:
            st.image(annotated, use_column_width=True)
        with analytics_col:
            st.metric("Traffic", status)
            st.metric("Vehicles", int(count))
            st.metric("Coverage (%)", f"{ratio:.2f}")
            st.metric("Motion", f"{motion:.2f}")
