# AI Vehicle Traffic Analyzer & Counter

A computer vision web application built with Streamlit and YOLO11 that detects, tracks, and counts vehicles moving in both upward and downward directions. It supports video file uploads as well as live camera feeds via WebRTC.

## Features

- Object Detection & Classification**: Detects cars, motorcycles, buses, and trucks using YOLO11.
- Directional Counting: Counts vehicles moving upward and downward using virtual reference lines and multi-object tracking (ByteTrack).
- Two Operating Modes:
- Upload Video: Process pre-recorded traffic videos, preview progress in real-time, inspect statistics, and download annotated output.
- Live Camera: Stream video directly from your webcam using WebRTC with real-time on-screen vehicle tracking.
- Customizable Confidence: Adjust detection confidence threshold directly from the sidebar.
-Detailed Summary: Generates both total directional counts and category-wise breakdowns.




