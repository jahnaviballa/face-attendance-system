# Face Recognition Attendance System

A Python-based attendance system using face recognition with OpenCV and `face_recognition`.

## Features
- Detects faces via webcam.
- Matches faces against images in the `known_faces` directory.
- Logs attendance with timestamps in `attendance.csv`.

## Requirements
- Python 3.x
- Libraries: `opencv-python`, `numpy`, `face_recognition`, `pandas`
- Install via: `pip install opencv-python numpy face_recognition pandas`

## Setup
1. Place known face images in the `known_faces` directory (e.g., `known_faces/john.jpg`).
2. Run `attendance.py` to start the webcam and mark attendance.

## Usage
```bash
python attendance.py