import cv2
import os
import glob
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def find_strike_in_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    
    if len(frames) < 10:
        return None
        
    pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    poses = {}
    for idx, f in enumerate(frames):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            # Check for face closeups:
            # If nose is near top (0-0.3) and ankle is below 0.6, it's a standing/running person
            # If nose is at 0.5 and ankles are not visible (v < 0.1), it's a close up face!
            ank_vis = max(lms[27].visibility, lms[28].visibility)
            if ank_vis > 0.25:
                poses[idx] = lms
    pose.close()
    
    if len(poses) < 5:
        return None
        
    indices = sorted(poses.keys())
    speeds = []
    for i in range(1, len(indices)):
        f1, f2 = indices[i-1], indices[i]
        if f2 - f1 <= 2:
            l1, l2 = poses[f1], poses[f2]
            r_spd = np.hypot((l2[28].x - l1[28].x) * w, (l2[28].y - l1[28].y) * h)
            l_spd = np.hypot((l2[27].x - l1[27].x) * w, (l2[27].y - l1[27].y) * h)
            spd = max(r_spd, l_spd)
            if spd < 180: # ignore cuts
                speeds.append((f2, spd, r_spd, l_spd))
                
    if not speeds:
        return None
        
    peak = max(speeds, key=lambda s: s[1])
    return {
        'total': len(frames),
        'fps': fps,
        'detected_count': len(poses),
        'strike_frame': peak[0],
        'strike_time': peak[0] / fps,
        'peak_speed': peak[1],
        'foot': 'Right' if peak[2] > peak[3] else 'Left'
    }

for vid in ['penalty_06', 'penalty_07', 'penalty_11', 'penalty_291', 'penalty_292', 'penalty_300', 'penalty_350', 'penalty_400']:
    p = f"data/videos/{vid}.mp4"
    if os.path.exists(p):
        info = find_strike_in_clip(p)
        print(f"{vid}: {info}")
