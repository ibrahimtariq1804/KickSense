import cv2
import mediapipe as mp
import os
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frames_dir = "data/frames"
keypoints_dir = "data/keypoints"

penalty_folders = os.listdir(frames_dir)

for penalty in penalty_folders:
    # Skip if already processed
    kp_path = os.path.join("data/keypoints", penalty, "keypoints.npy")
    if os.path.exists(kp_path):
        print(f"Skipping {penalty} - already processed")
        continue
    penalty_frames_dir = os.path.join(frames_dir, penalty)
    penalty_keypoints_dir = os.path.join(keypoints_dir, penalty)
    os.makedirs(penalty_keypoints_dir, exist_ok=True)
    
    # Create folder to save pose images
    pose_images_dir = os.path.join("outputs", penalty)
    os.makedirs(pose_images_dir, exist_ok=True)
    
    frames = sorted(os.listdir(penalty_frames_dir))
    keypoints_data = []
    
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=2,
                      min_detection_confidence=0.5) as pose:
        
        for frame_file in frames:
            frame_path = os.path.join(penalty_frames_dir, frame_file)
            image = cv2.imread(frame_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Draw skeleton on image
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Extract keypoints
                keypoints = []
                for lm in results.pose_landmarks.landmark:
                    keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
                
                keypoints_data.append(keypoints)
                
                # Save pose image
                pose_image_path = os.path.join(pose_images_dir, frame_file)
                cv2.imwrite(pose_image_path, image)
    
    # Save keypoints as numpy array
    if keypoints_data:
        np.save(os.path.join(penalty_keypoints_dir, "keypoints.npy"),
                np.array(keypoints_data))
        print(f"{penalty}: {len(keypoints_data)} frames processed")
    else:
        print(f"{penalty}: No pose detected")

print("Pose estimation complete!")