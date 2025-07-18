import os
import cv2
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_known_faces(folder='known_faces'):
    """
    Load known face encodings and names from the specified folder.
    """
    known_encodings = []
    known_names = []
    os.makedirs(folder, exist_ok=True)
    
    try:
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                enc = face_recognition.face_encodings(img_rgb)
                if enc:
                    known_encodings.append(enc[0])
                    known_names.append(os.path.splitext(filename)[0])
                else:
                    logger.warning(f"No face detected in {img_path}")
        logger.info(f"Loaded {len(known_encodings)} known faces")
        return known_encodings, known_names
    except Exception as e:
        logger.error(f"Error loading known faces: {str(e)}")
        return [], []

def mark_attendance(name, mark_once_per_day=True):
    """
    Mark attendance for a recognized person with date and time.
    Returns (success, message) to indicate if attendance was marked and why.
    """
    path = 'attendance.csv'
    try:
        # Initialize CSV if it doesn't exist
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w') as f:
                f.write('Name,Date,Time\n')

        df = pd.read_csv(path)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Check if the person was already marked today (if filter is enabled)
        if mark_once_per_day:
            today_records = df[df['Date'] == date_str]
            if name in today_records['Name'].values:
                logger.info(f"Attendance already marked for {name} today")
                return False, f"Attendance already marked for {name} today"

        # Mark attendance
        with open(path, 'a') as f:
            f.write(f"{name},{date_str},{time_str}\n")
        logger.info(f"Attendance marked for {name} at {time_str} on {date_str}")
        return True, f"Attendance marked for {name}"

    except Exception as e:
        logger.error(f"Error marking attendance for {name}: {str(e)}")
        return False, f"Error marking attendance: {str(e)}"