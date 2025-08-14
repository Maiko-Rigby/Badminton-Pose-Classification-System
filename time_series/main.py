import mediapipe as mp
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import csv

def average_points(*points): # function to get the centeral torso point
        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)
        z = sum(p.z for p in points) / len(points)
        return type(landmarks[0])(x=x, y=y, z=z)

def angle_calculate(pointA, pointB, pointC):
    # Convert into 3D vectors
    BA = np.array([pointA.x - pointB.x, pointA.y - pointB.y, pointA.z - pointB.z])
    BC = np.array([pointC.x - pointB.x, pointC.y - pointB.y, pointC.z - pointB.z])

    # normalise the vectors
    BA_norm = BA / np.linalg.norm(BA)
    BC_norm = BC / np.linalg.norm(BC)

    dot_product = np.dot(BA_norm, BC_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def calculate_velocity(angle_list, time):
    if len(angle_list) < 2:
        return 0  # Not enough data
    return (angle_list[-1] - angle_list[-2]) / time


def calculate_acceleration(velocity_list, time):
    if len(velocity_list) < 2:
        return 0  # Not enough data
    return (velocity_list[-1] - velocity_list[-2]) / time

# Creating a pose landmarker instance with the video mode

model_path = 'pose_landmarker_heavy.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(base_options=BaseOptions(
                                model_asset_path=model_path),
                                running_mode = VisionRunningMode.VIDEO)

all_videos = [video for video in listdir('test') if isfile(join('test',video))]

novice_videos = {48, 47, 49, 50, 46, 45, 44, 43, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13}

csv_file = open('time_series.csv', mode='w', newline='') # Open New CSV File
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'Video File', 'Category', 'Time (seconds)', 'Shoulder Flexion Angle', 'Shoulder Flexion Velocity', 'Shoulder Flexion Acceleration',
    'Wrist Flexion Angle', 'Wrist Flexion Velocity', 'Wrist Flexion Acceleration',
    'Wrist Pronation Angle', 'Wrist Pronation Velocity', 'Wrist Pronation Acceleration',
    'Elbow Flexion Angle', 'Elbow Flexion Velocity', 'Elbow Flexion Acceleration',
    'Shoulder Abduction Angle', 'Shoulder Abduction Velocity', 'Shoulder Abduction Acceleration',
    'Wrist Palmar Angle', 'Wrist Palmar Velocity', 'Wrist Palmar Acceleration',
    'Torso Rotation Angle', 'Torso Rotation Velocity', 'Torso Rotation Acceleration',
    'Trunk Tilt Back Angle', 'Trunk Tilt Back Velocity', 'Trunk Tilt Back Acceleration'
])

time = 1 / 240  # Time 


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
    shoulder_flexion_angles = []
    wrist_flexion_angles = []
    wrist_pronation_angles = []
    elbow_flexion_angles = []
    shoulder_abduction_angles = []
    wrist_palmar_angles = []
    torso_rotation_angles = []
    trunk_tilt_back_angles = []
    shoulder_flexion_velocities = []
    wrist_flexion_velocities = []
    wrist_pronation_velocities = []
    elbow_flexion_velocities = []
    shoulder_abduction_velocities = []
    wrist_palmar_velocities = []
    torso_rotation_velocities = []
    trunk_tilt_back_velocities = []
    shoulder_flexion_acceleration = []
    wrist_flexion_acceleration = []
    wrist_pronation_acceleration = []
    elbow_flexion_acceleration = []
    shoulder_abduction_acceleration = []
    wrist_palmar_acceleration = []
    torso_rotation_acceleration = []
    trunk_tilt_back_acceleration = []
    fs = 6
    cutoff = 2

    with PoseLandmarker.create_from_options(options) as landmarker:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((frame_index / fps) * 1000)

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            landmarks = detection_result.pose_world_landmarks[0] # First detected person

            if (
                first_right_wrist is None and first_right_elbow is None and first_right_shoulder is None and
                first_right_thumb is None and first_right_index is None and first_right_pinky is None and
                first_right_hip is None and first_left_hip is None
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

                first_left_shoulder = landmarks[11]
                first_left_elbow = landmarks[13]
                first_left_wrist = landmarks[15]
                first_left_thumb = landmarks[21]
                first_left_index = landmarks[19]
                first_left_pinky = landmarks[17]

            if frame_index % 5 == 0:
                time_stamp = frame_index / 240
                current_right_shoulder = landmarks[12]
                current_right_elbow = landmarks[14]
                current_right_wrist = landmarks[16]
                current_right_thumb = landmarks[22]
                current_right_index = landmarks[20]
                current_right_pinky = landmarks[18]
                current_left_hip = landmarks[23]
                current_right_hip = landmarks[24]
                current_torso_center = average_points(landmarks[11], landmarks[12], landmarks[23], landmarks[24])

                # Shoulder Flexion
                shoulder_flexion_angle = angle_calculate(current_right_wrist, current_right_shoulder, current_right_elbow)
                shoulder_flexion_angles.append(shoulder_flexion_angle)

                # Wrist Flexion
                wrist_flexion_angle = angle_calculate(current_right_pinky, current_right_wrist, current_right_elbow)
                wrist_flexion_angles.append(wrist_flexion_angle)

                # Wrist Pronation
                wrist_pronation_angle = angle_calculate(current_right_pinky, current_right_wrist, current_right_thumb)
                wrist_pronation_angles.append(wrist_pronation_angle)

                # Elbow Flexion
                elbow_flexion_angle = angle_calculate(current_right_shoulder, current_right_elbow, current_right_wrist)
                elbow_flexion_angles.append(elbow_flexion_angle)

                # Shoulder Abduction
                shoulder_abduction_angle = angle_calculate(current_right_shoulder, current_right_elbow, current_torso_center)
                shoulder_abduction_angles.append(shoulder_abduction_angle)

                # Wrist Palmar Flexion
                wrist_palmar_angle = angle_calculate(current_right_elbow, current_right_wrist, current_right_thumb)
                wrist_palmar_angles.append(wrist_palmar_angle)

                # Torso Rotation
                torso_rotation_angle = angle_calculate(current_right_shoulder, current_right_hip, current_torso_center)
                torso_rotation_angles.append(torso_rotation_angle)

                # Trunk Tilt Backward
                trunk_tilt_back_angle = angle_calculate(current_right_hip, current_torso_center, current_left_hip)
                trunk_tilt_back_angles.append(trunk_tilt_back_angle)

                if len(shoulder_flexion_angles) > 1:
                    shoulder_flexion_velocities.append(calculate_velocity(shoulder_flexion_angles, time_stamp))
                    wrist_flexion_velocities.append(calculate_velocity(wrist_flexion_angles, time_stamp))
                    wrist_pronation_velocities.append(calculate_velocity(wrist_pronation_angles, time_stamp))
                    elbow_flexion_velocities.append(calculate_velocity(elbow_flexion_angles, time_stamp))
                    shoulder_abduction_velocities.append(calculate_velocity(shoulder_abduction_angles, time_stamp))
                    wrist_palmar_velocities.append(calculate_velocity(wrist_palmar_angles, time_stamp))
                    torso_rotation_velocities.append(calculate_velocity(torso_rotation_angles, time_stamp))
                    trunk_tilt_back_velocities.append(calculate_velocity(trunk_tilt_back_angles, time_stamp))
                    if len(shoulder_flexion_velocities) > 2:
                        shoulder_flexion_acceleration.append(calculate_acceleration(shoulder_flexion_velocities, time_stamp))
                        wrist_flexion_acceleration.append(calculate_acceleration(wrist_flexion_velocities, time_stamp))
                        wrist_pronation_acceleration.append(calculate_acceleration(wrist_pronation_velocities, time_stamp))
                        elbow_flexion_acceleration.append(calculate_acceleration(elbow_flexion_velocities, time_stamp))
                        shoulder_abduction_acceleration.append(calculate_acceleration(shoulder_abduction_velocities, time_stamp))
                        wrist_palmar_acceleration.append(calculate_acceleration(wrist_palmar_velocities, time_stamp))
                        torso_rotation_acceleration.append(calculate_acceleration(torso_rotation_velocities, time_stamp))
                        trunk_tilt_back_acceleration.append(calculate_acceleration(trunk_tilt_back_velocities, time_stamp))

                        # CSV File Creation - Dataset Preparation
                        ID = int(video_file.split('.')[0]) # Split Video Name
                        category = "Novice" if ID in novice_videos else "Skilled"
                        csv_writer.writerow([
                        video_file, category, time_stamp,
                        shoulder_flexion_angles[-3], shoulder_flexion_velocities[-2], shoulder_flexion_acceleration[-1],
                        wrist_flexion_angles[-3], wrist_flexion_velocities[-2], wrist_flexion_acceleration[-1],
                        wrist_pronation_angles[-3], wrist_pronation_velocities[-2], wrist_pronation_acceleration[-1],
                        elbow_flexion_angles[-3], elbow_flexion_velocities[-2], elbow_flexion_acceleration[-1],
                        shoulder_abduction_angles[-3], shoulder_abduction_velocities[-2], shoulder_abduction_acceleration[-1],
                        wrist_palmar_angles[-3], wrist_palmar_velocities[-2], wrist_palmar_acceleration[-1],
                        torso_rotation_angles[-3], torso_rotation_velocities[-2], torso_rotation_acceleration[-1],
                        trunk_tilt_back_angles[-3], trunk_tilt_back_velocities[-2], trunk_tilt_back_acceleration[-1]
                    ])

            frame_index += 1



csv_file.close()

