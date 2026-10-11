import os
import tempfile
from pathlib import Path

import av
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
)

# Page configuration
st.set_page_config(
    page_title="AI Vehicle Traffic Analyzer",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Vehicle Traffic Analyzer")
st.write(
    "Upload a traffic video or start your camera "
    "to detect, track and count vehicles by direction."
)

# Vehicle classes to track from COCO dataset
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


class VehicleDetector:

    def __init__(self, confidence=0.35, up_line_ratio=0.45, down_line_ratio=0.60):
        self.model = YOLO("yolo11n.pt")
        self.confidence = confidence
        self.up_line_ratio = up_line_ratio
        self.down_line_ratio = down_line_ratio
        self.reset_counts()

    def reset_counts(self):
        self.previous_positions = {}
        self.crossed_lower = set()
        self.crossed_upper = set()
        self.counted_up = set()
        self.counted_down = set()

        self.up_count = 0
        self.down_count = 0

        self.up_vehicle = {
            name: 0 for name in VEHICLE_CLASSES.values()
        }
        self.down_vehicle = {
            name: 0 for name in VEHICLE_CLASSES.values()
        }

    def process_frame(self, frame):

        height, width = frame.shape[:2]

        up_line = int(height * self.up_line_ratio)
        down_line = int(height * self.down_line_ratio)

        results = self.model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASSES.keys()),
            conf=self.confidence,
            tracker="bytetrack.yaml",
            verbose=False
        )

        # Draw counting reference lines
        cv2.line(
            frame, (0, up_line), (width, up_line),
            (0, 0, 255), 2
        )
        cv2.line(
            frame, (0, down_line), (width, down_line),
            (255, 0, 0), 2
        )

        cv2.putText(
            frame, "UP LINE", (10, max(20, up_line - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        cv2.putText(
            frame, "DOWN LINE", (10, max(20, down_line - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:

            xyxy = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)

            for box, class_id, track_id in zip(
                xyxy, class_ids, track_ids
            ):

                vehicle_name = VEHICLE_CLASSES.get(class_id)

                if vehicle_name is None:
                    continue

                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                previous_y = self.previous_positions.get(track_id)

                if previous_y is not None:

                    # Vehicle moving upward across UP line
                    if (
                        (previous_y >= up_line and center_y < up_line)
                        or (track_id in self.crossed_lower and center_y < up_line)
                    ) and track_id not in self.counted_up:
                        self.up_count += 1
                        self.up_vehicle[vehicle_name] += 1
                        self.counted_up.add(track_id)

                    # Vehicle moving downward across DOWN line
                    if (
                        (previous_y <= down_line and center_y > down_line)
                        or (track_id in self.crossed_upper and center_y > down_line)
                    ) and track_id not in self.counted_down:
                        self.down_count += 1
                        self.down_vehicle[vehicle_name] += 1
                        self.counted_down.add(track_id)

                    # Track intermediate line crossings
                    if previous_y > down_line and center_y <= down_line:
                        self.crossed_lower.add(track_id)

                    if previous_y < up_line and center_y >= up_line:
                        self.crossed_upper.add(track_id)

                self.previous_positions[track_id] = center_y

                # Bounding box and tracking ID label
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                cv2.circle(
                    frame, (center_x, center_y),
                    4, (0, 255, 255), -1
                )

                label = f"{vehicle_name} ID:{track_id}"

                cv2.putText(
                    frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2
                )

        # On-screen counters
        cv2.putText(
            frame, f"UPWARD: {self.up_count}",
            (10, height - 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        cv2.putText(
            frame, f"DOWNWARD: {self.down_count}",
            (10, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )

        return frame

    def get_results(self):
        return {
            "total_up": self.up_count,
            "total_down": self.down_count,
            "up": self.up_vehicle.copy(),
            "down": self.down_vehicle.copy()
        }


# Frame processor for live webcam feed
class LiveVehicleProcessor(VideoProcessorBase):

    def __init__(self, confidence):
        self.detector = VehicleDetector(confidence)

    def recv(self, video_frame):
        frame = video_frame.to_ndarray(format="bgr24")
        processed = self.detector.process_frame(frame)

        return av.VideoFrame.from_ndarray(
            processed, format="bgr24"
        )


# Sidebar controls
mode = st.sidebar.radio(
    "Choose Mode",
    ["Upload Video", "Live Camera"]
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.35,
    step=0.05
)


# Statistics summary
def show_results(results):

    col1, col2 = st.columns(2)

    col1.metric("⬆️ Upward Vehicles", results["total_up"])
    col2.metric("⬇️ Downward Vehicles", results["total_down"])

    st.subheader("Vehicle-wise Counts")

    st.table({
        "Vehicle": list(VEHICLE_CLASSES.values()),
        "Upward": [
            results["up"][name]
            for name in VEHICLE_CLASSES.values()
        ],
        "Downward": [
            results["down"][name]
            for name in VEHICLE_CLASSES.values()
        ]
    })


# Mode 1: Video file processing
if mode == "Upload Video":

    uploaded_video = st.file_uploader(
        "Upload a traffic video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("🚀 Start Detection", type="primary"):

            input_path = None
            cap = None
            writer = None

            try:
                os.makedirs("outputs", exist_ok=True)

                suffix = Path(uploaded_video.name).suffix or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_file:
                    temp_file.write(uploaded_video.getvalue())
                    input_path = temp_file.name

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    st.error("Video could not be opened.")
                    st.stop()

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                if fps <= 0:
                    fps = 25

                output_path = os.path.abspath(
                    "outputs/processed_video.mp4"
                )

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    st.error("Could not create output video.")
                    st.stop()

                detector = VehicleDetector(confidence)

                preview = st.empty()
                progress = st.progress(0)
                frame_number = 0

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    processed = detector.process_frame(frame)
                    writer.write(processed)

                    frame_number += 1

                    if frame_number % 5 == 0:
                        preview.image(
                            cv2.cvtColor(
                                processed, cv2.COLOR_BGR2RGB
                            ),
                            channels="RGB",
                            use_container_width=True
                        )

                    if total_frames > 0:
                        progress.progress(
                            min(frame_number / total_frames, 1.0)
                        )

                cap.release()
                cap = None
                writer.release()
                writer = None

                st.success("Detection completed!")

                show_results(detector.get_results())

                st.subheader("Processed Video")
                st.video(output_path)

                with open(output_path, "rb") as video_file:
                    st.download_button(
                        "⬇️ Download Processed Video",
                        data=video_file.read(),
                        file_name="vehicle_analysis.mp4",
                        mime="video/mp4"
                    )

            except Exception as error:
                st.error(f"Processing failed: {error}")

            finally:
                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                if input_path and os.path.exists(input_path):
                    os.remove(input_path)


# Mode 2: Live camera stream
elif mode == "Live Camera":

    st.write(
        "Allow camera access in your browser, then start "
        "the video stream. Counts are drawn on the live feed."
    )

    ctx = webrtc_streamer(
        key="vehicle-live-camera",
        video_processor_factory=lambda: LiveVehicleProcessor(
            confidence
        ),
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    if ctx.video_processor:
        st.subheader("Live Traffic Statistics")
        show_results(ctx.video_processor.detector.get_results())
