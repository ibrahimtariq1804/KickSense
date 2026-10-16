import cv2
import os
import sys
import numpy as np
import mediapipe as mp
import pandas as pd

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection

df_labels = pd.read_excel("data/labels/penalty_labels.xlsx")
label_map = {}
for _, row in df_labels.iterrows():
    label_map[row['video_id']] = {
        'match': str(row['match']),
        'player': str(row['player']),
        'dir': str(row['shot_direction']),
        'pwr': float(row['shot_power']) if pd.notnull(row['shot_power']) else 5.0
    }

def is_full_body_striker(lms, w, h):
    """
    Returns True ONLY if the pose represents a full-body striker standing/running on the pitch.
    Returns False for face close-ups, half-body shots, and off-screen hallucinations.
    """
    nose = lms[0]
    l_sh = lms[11]
    r_sh = lms[12]
    l_hip = lms[23]
    r_hip = lms[24]
    l_knee = lms[25]
    r_knee = lms[26]
    l_ank = lms[27]
    r_ank = lms[28]
    
    # 1. Vertical hierarchy: nose < shoulders < hips < knees < ankles
    sh_y = (l_sh.y + r_sh.y) / 2
    hip_y = (l_hip.y + r_hip.y) / 2
    knee_y = max(l_knee.y, r_knee.y)
    ank_y = max(l_ank.y, r_ank.y)
    
    if not (nose.y < hip_y < ank_y):
        return False
        
    # 2. Torso and leg lengths must be substantial
    torso_h = hip_y - sh_y
    leg_h = ank_y - hip_y
    
    if torso_h < 0.04 or leg_h < 0.05:
        return False
        
    # 3. Ratio check: in a face close-up, MediaPipe puts hips near bottom edge, leg_h is tiny
    if leg_h < 0.5 * torso_h:
        return False
        
    # 4. Total body height must be reasonable for a penalty view (between 12% and 80% of screen)
    total_h = ank_y - nose.y
    if total_h < 0.12 or total_h > 0.85:
        return False
        
    # 5. Ankles must be on screen and have reasonable visibility
    max_ank_vis = max(l_ank.visibility, r_ank.visibility)
    if max_ank_vis < 0.35:
        return False
        
    # 6. Striker feet must be in lower half of screen in normal penalty camera
    if ank_y < 0.45:
        return False
        
    return True

def analyze_penalty_clip(vid):
    p = f"data/videos/{vid}.mp4"
    if not os.path.exists(p):
        return None
    cap = cv2.VideoCapture(p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) < 15:
        return None
        
    pose_tracker = mp_pose.Pose(min_detection_confidence=0.35, min_tracking_confidence=0.35)
    valid_poses = {} # frame_idx -> lms
    
    for idx, f in enumerate(frames):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = pose_tracker.process(rgb)
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            if is_full_body_striker(lms, w, h):
                valid_poses[idx] = lms
    pose_tracker.close()
    
    if len(valid_poses) < 8:
        return {
            'vid': vid,
            'status': 'FAIL_FEW_POSES',
            'valid_count': len(valid_poses),
            'total': total
        }
        
    indices = sorted(valid_poses.keys())
    
    # Calculate foot speed
    speeds = []
    for i in range(1, len(indices)):
        f1, f2 = indices[i-1], indices[i]
        if f2 - f1 <= 3:
            l1, l2 = valid_poses[f1], valid_poses[f2]
            r_spd = np.hypot((l2[28].x - l1[28].x) * w, (l2[28].y - l1[28].y) * h) / (f2 - f1)
            l_spd = np.hypot((l2[27].x - l1[27].x) * w, (l2[27].y - l1[27].y) * h) / (f2 - f1)
            spd = max(r_spd, l_spd)
            if spd < 120: # ignore cuts
                speeds.append((f2, spd, r_spd, l_spd))
                
    if not speeds:
        return {'vid': vid, 'status': 'NO_SPEEDS', 'total': total}
        
    # Peak strike frame
    peak = max(speeds, key=lambda s: s[1])
    strike_f = peak[0]
    
    # Save strike preview image with skeleton
    strike_frame_img = frames[strike_f].copy()
    lms = valid_poses[strike_f]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for c1, c2 in mp_pose.POSE_CONNECTIONS:
        if c1 < len(pts) and c2 < len(pts):
            cv2.line(strike_frame_img, pts[c1], pts[c2], (255, 200, 0), 3, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(strike_frame_img, pt, 5, (0, 100, 255), -1, cv2.LINE_AA)
        
    out_dir = "outputs/verified_strikes"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/{vid}_strike_f{strike_f}_{strike_f/fps:.2f}s.jpg", strike_frame_img)
    
    lbl = label_map.get(vid, {'player': 'Unknown', 'dir': 'Unknown', 'pwr': 5.0})
    
    return {
        'vid': vid,
        'status': 'PASS',
        'player': lbl['player'],
        'true_dir': lbl['dir'],
        'true_pwr': lbl['pwr'],
        'strike_f': strike_f,
        'strike_time': round(strike_f / fps, 2),
        'total_frames': total,
        'valid_poses': len(valid_poses),
        'peak_spd': round(peak[1], 1),
        'foot': 'Right' if peak[2] > peak[3] else 'Left'
    }

test_clips = [
    'penalty_06', 'penalty_07', 'penalty_08', 'penalty_10', 'penalty_11',
    'penalty_12', 'penalty_13', 'penalty_14', 'penalty_15',
    'penalty_291', 'penalty_292', 'penalty_300', 'penalty_350', 'penalty_400'
]

print("Testing clips with Full-Body Striker Detection Filter...", flush=True)
for c in test_clips:
    res = analyze_penalty_clip(c)
    print(res, flush=True)
