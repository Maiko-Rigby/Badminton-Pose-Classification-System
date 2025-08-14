import mediapipe as mp
import numpy as np
import cv2
import csv


# Function to draw the landmarks onto the video

def draw_landmarks(rgb_image, landmarks):
  height, width, _ = frame.shape
  for landmark in landmarks:
      x, y = int(landmark.x * width), int(landmark.y * height)
      cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # red dots

# Creating a pose landmarker instance with the video mode

model_path = 'pose_landmarker_heavy.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(base_options=BaseOptions(
                                model_asset_path=model_path),
                                running_mode = VisionRunningMode.VIDEO)

video = cv2.VideoCapture("Loh Kean Yew extraordinarily Jump Smash in Slow Motion - Robin hood (720p, h264).mp4") # Load video
print(video)
fps = video.get(cv2.CAP_PROP_FPS)
frame_index = 0

with PoseLandmarker.create_from_options(options) as landmarker: 
    while video.isOpened(): # While video is running
            ret, frame = video.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((frame_index / fps) * 1000)

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms) # Landmarker results

            if detection_result.pose_landmarks:
                draw_landmarks(frame, detection_result.pose_landmarks[0]) # If there are results, draw them 

            landmarks = detection_result.pose_landmarks[0]  # First detected person

            cv2.imshow("Pose Detection", frame)

            if cv2.waitKey(1) &0xFF == ord('q'):
                break

            frame_index+= 1

    video.release()
    cv2.destroyAllWindows()
