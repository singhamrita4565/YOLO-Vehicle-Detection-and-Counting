# AI Vehicle Traffic Analyzer & Counter

A computer vision web application built with Streamlit and YOLO11 that detects, tracks, and counts vehicles moving in both upward and downward directions. It supports video file uploads as well as live camera feeds via WebRTC.

## Features

- **Object Detection & Classification**: Detects cars, motorcycles, buses, and trucks using YOLO11.
- **Directional Counting**: Counts vehicles moving upward and downward using virtual reference lines and multi-object tracking (ByteTrack).
- **Two Operating Modes**:
  - **Upload Video**: Process pre-recorded traffic videos, preview progress in real-time, inspect statistics, and download annotated output.
  - **Live Camera**: Stream video directly from your webcam using WebRTC with real-time on-screen vehicle tracking.
- **Customizable Confidence**: Adjust detection confidence threshold directly from the sidebar.
- **Detailed Summary**: Generates both total directional counts and category-wise breakdowns.

## Tech Stack

- **Python 3.10+**
- **Ultralytics YOLO11** (Object Detection & ByteTrack)
- **OpenCV** (Video I/O and frame processing)
- **Streamlit** (Web dashboard and UI)
- **Streamlit-WebRTC / PyAV** (Live webcam streaming)

## Project Structure

```text
vehicletraking/
│
├── app.py              # Main Streamlit web application
├── requirements.txt    # Project dependencies
├── yolo11n.pt          # YOLO11 nano model weights (downloaded automatically)
├── .gitignore          # Ignored files for GitHub
└── README.md           # Documentation
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/vehicle-traffic-analyzer.git
cd vehicle-traffic-analyzer
```

### 2. Set up a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: YOLO11 model weights (`yolo11n.pt`) will automatically download on the first run if not present in the directory.

### 4. Run the application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## How the Counting Works

1. **Object Detection**: Each frame is processed by YOLO11 to locate vehicles (COCO classes 2, 3, 5, 7).
2. **ByteTrack Multi-Object Tracking**: Assigns persistent tracking IDs across consecutive frames.
3. **Reference Lines**: Two virtual lines are drawn across the frame:
   - **UP LINE**: Tracks vehicles moving upwards (towards the top of the frame).
   - **DOWN LINE**: Tracks vehicles moving downwards (towards the bottom of the frame).
4. **Direction Determination**: By comparing previous and current vertical coordinates (`center_y`), the algorithm detects when a vehicle crosses a line in a given direction and updates the count without duplicate triggers.
