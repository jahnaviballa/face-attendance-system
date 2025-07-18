import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
import time

# Streamlit page configuration
st.set_page_config(page_title="Face Attendance System", layout="wide")

# Path to known faces
path = 'known_faces'
images = []
classNames = []

# Load known face images and names
for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    classNames.append(os.path.splitext(file)[0])

# Encode known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodeList.append(enc[0])
    return encodeList

# Initialize encoding
encodeListKnown = findEncodings(images)
st.success("✅ Encodings Complete")

# Initialize attendance set
attendance_marked = set()

# Mark attendance
def markAttendance(name):
    if name not in attendance_marked:
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')

        # Create CSV if not exists
        if not os.path.exists('attendance.csv'):
            df = pd.DataFrame(columns=['Name', 'DateTime'])
            df.to_csv('attendance.csv', index=False)

        # Append new entry
        df = pd.read_csv('attendance.csv')
        new_row = {'Name': name, 'DateTime': dtString}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv('attendance.csv', index=False)

        attendance_marked.add(name)
        return f"🎯 Marked: {name} at {dtString}"
    return None

# Streamlit UI
st.title("Face Attendance System")
st.markdown("This app detects faces using a webcam and marks attendance.")

# Placeholder for video feed
frame_placeholder = st.empty()
# Placeholder for attendance log
log_placeholder = st.empty()
# Placeholder for attendance table
table_placeholder = st.empty()

# Start/Stop button
if 'running' not in st.session_state:
    st.session_state.running = False

if st.button("Start/Stop Webcam"):
    st.session_state.running = not st.session_state.running

# Initialize webcam
cap = cv2.VideoCapture(0)

while st.session_state.running:
    success, frame = cap.read()
    if not success:
        st.error("Failed to capture video. Check webcam connection.")
        break

    # Process frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Faster processing
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect and encode faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].capitalize()

            # Scale back up face coordinates
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance and update log
            log_message = markAttendance(name)
            if log_message:
                log_placeholder.write(log_message)

    # Display frame in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
    frame_placeholder.image(frame, channels="RGB", use_column_width=True)

    # Display attendance table
    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        table_placeholder.dataframe(df)

    # Small delay to prevent overwhelming the app
    time.sleep(0.1)

# Release webcam when stopped
if not st.session_state.running:
    cap.release()

st.write("Press the button to start/stop the webcam.")