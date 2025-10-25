import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="🚦 Traffic Motion Analyzer", layout="wide")
st.title("🚗 Traffic Motion Analyzer using Computer Vision")
st.caption("Upload a traffic video and the system will analyze flow based on motion and object density.")

# ------------------------------------------------------------
# VIDEO UPLOAD
# ------------------------------------------------------------
uploaded_video = st.file_uploader("📹 Upload a Traffic Video", type=["mp4", "mov", "avi", "mkv"], key="video_upload")

if uploaded_video is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    st.video(tfile.name)

    st.info("⏳ Processing video... please wait.")
    time.sleep(1)

    cap = cv2.VideoCapture(tfile.name)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    if not ret:
        st.error("❌ Could not read the video. Try another file.")
        st.stop()

    FRAME_WINDOW = st.empty()
    progress = st.progress(0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1 / fps

    # Stats containers
    motion_levels = []
    traffic_states = []
    timestamps = []

    frame_idx = 0
    start_time = time.time()
    total_objects_detected = 0

    # ------------------------------------------------------------
    # VIDEO PROCESSING LOOP
    # ------------------------------------------------------------
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = frame1.shape[:2]
        line_position = int(height / 2)
        cv2.line(frame1, (0, line_position), (width, line_position), (0, 255, 255), 2)

        moving_objects = 0
        for contour in contours:
            if cv2.contourArea(contour) < 800:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            moving_objects += 1

        # --- Traffic Intensity Logic ---
        motion_intensity = len(contours)
        motion_levels.append(motion_intensity)
        total_objects_detected += moving_objects

        traffic_status = "🟢 Free Flow"
        if motion_intensity > 50:
            traffic_status = "🔴 Heavy Traffic"
        elif motion_intensity > 20:
            traffic_status = "🟡 Moderate Flow"

        traffic_states.append(traffic_status)
        timestamps.append(time.time() - start_time)

        # Annotate frame
        cv2.putText(
            frame1,
            f"{traffic_status} | Moving Objects: {moving_objects}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if "Free" in traffic_status else (0, 0, 255),
            2,
        )

        # Display
        FRAME_WINDOW.image(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        progress.progress(frame_idx / frame_count)

        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break

        frame_idx += 1
        time.sleep(frame_interval / 3)

    cap.release()
    progress.empty()
    st.success("✅ Analysis complete!")

    # ------------------------------------------------------------
    # SUMMARY SECTION
    # ------------------------------------------------------------
    st.subheader("📊 Analysis Summary")

    avg_motion = np.mean(motion_levels)
    peak_motion = np.max(motion_levels)
    avg_objects = total_objects_detected / max(1, len(motion_levels))

    if avg_motion < 20:
        overall_status = "🟢 Mostly Free Flow"
    elif avg_motion < 50:
        overall_status = "🟡 Moderate Traffic"
    else:
        overall_status = "🔴 Heavy Traffic"

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Motion Intensity", f"{avg_motion:.1f}")
    col2.metric("Peak Motion", f"{peak_motion:.1f}")
    col3.metric("Average Moving Objects", f"{avg_objects:.1f}")
    st.markdown(f"### 🧠 Overall Traffic Status: {overall_status}")

    # ------------------------------------------------------------
    # MOTION CHART
    # ------------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=motion_levels,
        mode="lines+markers",
        name="Motion Intensity",
        line=dict(width=2)
    ))
    fig.update_layout(
        title="🚦 Motion Intensity Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Motion Level",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📂 Please upload a video file to start traffic flow analysis.")
