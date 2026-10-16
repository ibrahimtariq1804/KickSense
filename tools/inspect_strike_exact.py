import cv2
import os
import sys
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def inspect_strike(vid):
    p = f"data/videos/{vid}.mp4"
    if not os.path.exists(p):
        print(f"{vid}: Not found", flush=True)
        return
    cap = cv2.VideoCapture(p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    poses = {}
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            ank_vis = max(lms[27].visibility, lms[28].visibility)
            # Full body check: nose above 0.5, ankles below 0.6
            if ank_vis > 0.35 and lms[0].y < 0.55 and max(lms[27].y, lms[28].y) > 0.60:
                poses[f_idx] = lms
        f_idx += 1
    cap.release()
    pose.close()
    
    indices = sorted(poses.keys())
    speeds = []
    for i in range(1, len(indices)):
        f1, f2 = indices[i-1], indices[i]
        if f2 - f1 <= 2:
            l1, l2 = poses[f1], poses[f2]
            r_spd = np.hypot((l2[28].x - l1[28].x) * w, (l2[28].y - l1[28].y) * h)
            l_spd = np.hypot((l2[27].x - l1[27].x) * w, (l2[27].y - l1[27].y) * h)
            spd = max(r_spd, l_spd)
            if spd < 150:
                speeds.append((f2, spd, r_spd, l_spd))
                
    if speeds:
        peak = max(speeds, key=lambda s: s[1])
        print(f"{vid}: total={total}, fps={fps}, full_body_poses={len(poses)}, peak_f={peak[0]} ({peak[0]/fps:.2f}s), spd={peak[1]:.1f}, R={peak[2]:.1f}, L={peak[3]:.1f}", flush=True)
    else:
        print(f"{vid}: no strike peak found in {len(poses)} poses", flush=True)

for vid in ['penalty_06', 'penalty_07', 'penalty_11', 'penalty_291', 'penalty_292']:
    inspect_strike(vid)
