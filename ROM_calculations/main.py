import mediapipe as mp
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import csv

# function to get the centeral torso point
def average_points(*points): 
        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)
        z = sum(p.z for p in points) / len(points)
        return type(landmarks[0])(x=x, y=y, z=z)

# Angle function
def angle_calculate(pointA, pointB, pointC):
    # Convert into 3D vectors
    BA = np.array([pointA.x - pointB.x, pointA.y - pointB.y, pointA.z - pointB.z])
    BC = np.array([pointC.x - pointB.x, pointC.y - pointB.y, pointC.z - pointB.z])

    # Normalise vectors
    BA_norm = BA / np.linalg.norm(BA)
    BC_norm = BC / np.linalg.norm(BC)

    # Dot product
    dot_product = np.dot(BA_norm, BC_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

# Initialise pose landmarker
model_path = 'pose_landmarker_heavy.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(base_options=BaseOptions(
                                model_asset_path=model_path),
                                running_mode = VisionRunningMode.VIDEO)

all_videos = [video for video in listdir('test') if isfile(join('test',video))]

# label novice videos
novice_videos = {48, 47, 49, 50, 46, 45, 44, 43, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13}

# create a new csv file
csv_file = open('angle_results.csv', mode='w', newline='') # Open New CSV File
csv_writer = csv.writer(csv_file)
csv_writer.writerow([ 
    "Video", "Category",
    "Shoulder Flexion Change", "Wrist Flexion Change", "Wrist Pronation Change",
    "Elbow Flexion Change", "Shoulder Abduction Change", "Wrist Palmar Flexion Change",
    "Torso Rotation Change", "Trunk Tilt Backward Change"
])

for video_file in all_videos:
    video_name = "test/"+video_file
    video = cv2.VideoCapture(video_name)
    print(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    first_right_shoulder = None      
    last_right_shoulder = None
    first_right_elbow = None    
    last_right_elbow = None
    first_right_wrist = None 
    last_right_wrist = None
    first_right_thumb = None        
    last_right_thumb = None
    first_right_index = None        
    last_right_index = None
    first_right_pinky = None         
    last_right_pinky = None
    first_torso_center = None
    last_torso_center = None
    first_left_hip = None 
    last_left_hip = None
    first_right_hip = None
    last_right_hip = None
    first_left_shoulder = None
    first_left_hip = None
    last_left_shoulder = None
    last_left_hip = None

    with PoseLandmarker.create_from_options(options) as landmarker:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((frame_index / fps) * 1000)

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            landmarks = detection_result.pose_landmarks[0]  # First detected person

            if (
                first_right_wrist is None and first_right_elbow is None and first_right_shoulder is None and
                first_right_thumb is None and first_right_index is None and first_right_pinky is None and
                first_right_hip is None and first_left_hip is None and first_left_shoulder is None
            ):
                first_right_shoulder = landmarks[12]
                first_right_elbow = landmarks[14]
                first_right_wrist = landmarks[16]
                first_right_thumb = landmarks[22]
                first_right_index = landmarks[20]
                first_right_pinky = landmarks[18]
                first_left_hip = landmarks[23]
                first_right_hip = landmarks[24]
                first_torso_center = average_points(landmarks[11], landmarks[12], landmarks[23], landmarks[24])

            last_right_shoulder = landmarks[12]
            last_right_elbow = landmarks[14]
            last_right_wrist = landmarks[16]
            last_right_thumb = landmarks[22]
            last_right_index = landmarks[20]
            last_right_pinky = landmarks[18]
            last_left_hip = landmarks[23]
            last_right_hip = landmarks[24]
            last_torso_center = average_points(landmarks[11], landmarks[12], landmarks[23], landmarks[24])

            frame_index+= 1

        # Shoulder Flexion
        shoulder_flexion_start = abs(angle_calculate(first_right_wrist,first_right_shoulder, first_right_elbow))
        shoulder_flexion_end = abs(angle_calculate(last_right_wrist, last_right_shoulder, last_right_elbow))
        shoulder_flexion_change = abs(shoulder_flexion_end - shoulder_flexion_start)

        # Wrist Flexion
        wrist_flexion_start = abs(angle_calculate(first_right_pinky, first_right_wrist, first_right_elbow))
        wrist_flexion_end = abs(angle_calculate(last_right_pinky, last_right_wrist, last_right_elbow))
        wrist_flexion_change = abs(wrist_flexion_end - wrist_flexion_start)

        # Wrist Pronation
        wrist_pronation_start = abs(angle_calculate(first_right_pinky,first_right_wrist,first_right_thumb))
        wrist_pronation_end = abs(angle_calculate(last_right_pinky, last_right_wrist, last_right_thumb))
        wrist_pronation_change = abs(wrist_pronation_end - wrist_pronation_start)

        # Elbow Flexion
        elbow_flexion_start = abs(angle_calculate(first_right_shoulder, first_right_elbow, first_right_wrist))
        elbow_flexion_end = abs(angle_calculate(last_right_shoulder, last_right_elbow, last_right_wrist))
        elbow_flexion_change = abs(elbow_flexion_end - elbow_flexion_start)

        # Shoulder Abduction
        shoulder_abduction_start = abs(angle_calculate(first_right_shoulder, first_right_elbow, first_torso_center))
        shoulder_abduction_end = abs(angle_calculate(last_right_shoulder, last_right_elbow, last_torso_center))
        shoulder_abduction_change = abs(shoulder_abduction_end - shoulder_abduction_start)

        # Wrist Palmar Flexion
        wrist_palmar_start = abs(angle_calculate(first_right_elbow, first_right_wrist, first_right_thumb))
        wrist_palmar_end = abs(angle_calculate(last_right_elbow, last_right_wrist, last_right_thumb))
        wrist_palmar_change = abs(wrist_palmar_end - wrist_palmar_start)

        # Torso Rotation
        torso_rotation_start = abs(angle_calculate(first_right_shoulder, first_right_hip, first_torso_center))
        torso_rotation_end = abs(angle_calculate(last_right_shoulder, last_right_hip, last_torso_center))
        torso_rotation_change = abs(torso_rotation_end - torso_rotation_start)

        # Trunk Tilt Backward
        trunk_tilt_back_start = abs(angle_calculate(first_right_hip, first_torso_center, first_left_hip))
        trunk_tilt_back_end = abs(angle_calculate(last_right_hip, last_torso_center, last_left_hip))
        trunk_tilt_back_change = abs(trunk_tilt_back_end - trunk_tilt_back_start)

        # Dataset Preparation
        ID = int(video_file.split('.')[0]) # Split Video Name
        category = "Novice" if ID in novice_videos else "Skilled"
        csv_writer.writerow([
            video_file, category,
            shoulder_flexion_change, wrist_flexion_change, wrist_pronation_change,
            elbow_flexion_change, shoulder_abduction_change, wrist_palmar_change,
            torso_rotation_change, trunk_tilt_back_change
        ])



csv_file.close()
