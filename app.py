import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
from ultralytics import YOLO
import plotly.express as px
import joblib
import torch
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
st.sidebar.success(f"🚀 Device: {DEVICE.upper()}")
st.set_page_config(page_title="Traffic Monitor AI", layout="wide")
st.title("🚦 Traffic Monitoring + ML Prediction System")

# CUSTOM CLASSES
VEHICLE_CLASSES = ["bike", "bus", "car", "truck"]

# YOLO MODEL
@st.cache_resource
def load_model():
    model = YOLO("model/best.pt")
    model.fuse()
    return model

model = load_model()

# RANDOM FOREST MODEL
@st.cache_resource
def load_rf_model():
    return joblib.load("model/traffic_rf_model.pkl")

rf_model = load_rf_model()

# DETECTION 
def detect_vehicles(frame, model, conf_threshold=0.25):

    #results = model(frame, conf=conf_threshold, device="cpu", verbose=False)
    results = model(
    frame,
    conf=conf_threshold,
    device=DEVICE,
    verbose=False
)

    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:

            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name in VEHICLE_CLASSES:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": cls_name,
                    "confidence": conf
                })

    return detections


# DRAW BOXES
def draw_annotations(frame, detections):

    colors = {
        "car": (0, 255, 0),
        "bike": (255, 255, 0),
        "bus": (0, 0, 255),
        "truck": (255, 0, 0),
    }

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']} {det['confidence']:.2f}"
        color = colors.get(det["class"], (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


# COUNT VEHICLES 
def count_vehicles(detections):

    counts = {cls: 0 for cls in VEHICLE_CLASSES}

    for det in detections:
        if det["class"] in counts:
            counts[det["class"]] += 1

    return counts


# RANDOM FOREST PREDICTION
def predict_traffic(counts, model):

    features = np.array([[
        counts["bike"],
        counts["bus"],
        counts["car"],
        counts["truck"]
    ]])

    prediction = model.predict(features)[0]
    return prediction


#  VIDEO PROCESSING 
def process_video(video_path, yolo_model, conf_thresh, frame_skip, output_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    # store unique vehicle IDs
    tracked_ids = {
        cls: set() for cls in VEHICLE_CLASSES
    }

    log_data = []
    progress = st.progress(0)


    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break


        frame_count += 1


        if frame_count % frame_skip != 0:
            out.write(frame)
            continue


        small = cv2.resize(frame, (640, 384))


        # YOLO TRACKING 

        # results = yolo_model.track(
        #     small,
        #     conf=conf_thresh,
        #     persist=True,
        #     device="cpu",
        #     verbose=False
        # )
        results = yolo_model.track(
            small,
            conf=conf_thresh,
            persist=True,
            device=DEVICE,
            verbose=False
)

        detections = []


        if results[0].boxes is not None:

            for box in results[0].boxes:

                cls_id = int(box.cls[0])
                cls_name = yolo_model.names[cls_id]


                if cls_name in VEHICLE_CLASSES:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    conf = float(box.conf[0])


                    track_id = int(box.id[0]) if box.id is not None else None


                    detections.append({

                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        "class": cls_name,

                        "confidence": conf
                    })


                    # count only unique vehicle

                    if track_id is not None:
                        tracked_ids[cls_name].add(track_id)



        # scale boxes back

        sx = width / 640
        sy = height / 384


        scaled = []


        for det in detections:

            x1,y1,x2,y2 = det["bbox"]


            scaled.append({

                "bbox": (
                    int(x1*sx),
                    int(y1*sy),
                    int(x2*sx),
                    int(y2*sy)
                ),

                "class": det["class"],

                "confidence": det["confidence"]

            })


        annotated = draw_annotations(
            frame.copy(),
            scaled
        )


        out.write(annotated)


        # frame based count for graph

        frame_counts = count_vehicles(scaled)


        log_data.append({

            "frame": frame_count,

            "timestamp": frame_count/fps,

            **frame_counts

        })


        progress.progress(
            min(frame_count/total_frames,1.0)
        )


    cap.release()
    out.release()


    # final unique count

    cumulative_counts = {

        cls: len(tracked_ids[cls])

        for cls in VEHICLE_CLASSES

    }


    return log_data, cumulative_counts

# UI 
st.sidebar.header("Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.8, 0.25)
frame_skip = st.sidebar.slider("Frame Skip", 1, 6, 1)
    
tab1, tab2 = st.tabs(["📸 Image Detection", "🎥 Video + ML Prediction"])


# IMAGE
with tab1:

    img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if img:

        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detections = detect_vehicles(frame, model, conf_thresh)
        annotated = draw_annotations(frame.copy(), detections)

        counts = count_vehicles(detections)

        st.image(annotated, channels="BGR")
        st.dataframe(pd.DataFrame([counts]).T.rename(columns={0: "Count"}))


# VIDEO 
with tab2:
    video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if video:
        tfile = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )
        tfile.write(video.read())
        video_path = tfile.name

        if st.button("🚀 Process Video"):
            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            with st.spinner("Processing video..."):
                log_data, counts = process_video(
                    video_path,
                    model,
                    conf_thresh,
                    frame_skip,
                    output_path
                )

            prediction = predict_traffic(counts, rf_model)

            # Store results in session state
            st.session_state["video_processed"] = True
            st.session_state["output_path"] = output_path
            st.session_state["log_data"] = log_data
            st.session_state["counts"] = counts
            st.session_state["prediction"] = prediction

            st.success("Processing complete!")

    # Display results if video has already been processed
    if st.session_state.get("video_processed", False):

        prediction = st.session_state["prediction"]
        counts = st.session_state["counts"]
        log_data = st.session_state["log_data"]
        output_path = st.session_state["output_path"]

        st.subheader("🚦 Traffic Prediction (ML Model)")
        st.metric("Prediction", prediction)

        st.subheader("🚗 Vehicle Distribution")
        st.dataframe(
            pd.DataFrame([counts]).T.rename(
                columns={0: "Total"}
            )
        )

        # Download processed video
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download Video",
                    f,
                    "traffic_output.mp4",
                    mime="video/mp4"
                )

        # Analytics
        df = pd.DataFrame(log_data)

        if not df.empty:
            st.subheader("📊 Traffic Analytics")

            st.plotly_chart(
                px.line(
                    df,
                    x="frame",
                    y="car",
                    title="Traffic Flow (Car)"
                ),
                use_container_width=True
            )