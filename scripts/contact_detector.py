import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def is_full_body_striker(lms, w=1, h=1):
    """
    Returns True ONLY if the pose represents a full-body striker standing/running on the pitch.
    Filters out face close-ups, half-body broadcast cuts, and false hallucinations.
    """
    nose = lms[0]
    l_sh, r_sh = lms[11], lms[12]
    l_hip, r_hip = lms[23], lms[24]
    l_knee, r_knee = lms[25], lms[26]
    l_ank, r_ank = lms[27], lms[28]
    
    sh_y = (l_sh.y + r_sh.y) / 2
    hip_y = (l_hip.y + r_hip.y) / 2
    knee_y = max(l_knee.y, r_knee.y)
    ank_y = max(l_ank.y, r_ank.y)
    
    # 1. Vertical anatomical hierarchy: nose < shoulders < hips < ankles
    if not (nose.y < sh_y < hip_y < ank_y):
        return False
        
    torso_h = hip_y - sh_y
    leg_h = ank_y - hip_y
    
    # 2. Torso and leg lengths must be substantial
    if torso_h < 0.03 or leg_h < 0.04:
        return False
        
    # 3. Leg to torso proportion (in face closeups, legs are non-existent or truncated)
    if leg_h < 0.45 * torso_h:
        return False
        
    # 4. Total height must be within full-body penalty view bounds
    total_h = ank_y - nose.y
    if total_h < 0.10 or total_h > 0.88:
        return False
        
    # 5. Head should not fill the frame (reject face close-ups where head alone is > 25% of frame)
    head_h = sh_y - nose.y
    if head_h > 0.25:
        return False
        
    # 6. Ankles must be on screen and visible
    ank_vis = max(l_ank.visibility, r_ank.visibility)
    if ank_vis < 0.30:
        return False
        
    # 7. Striker feet must be on lower half of the pitch
    if ank_y < 0.42:
        return False
        
    return True

def detect_contact_and_extract_sequence(video_path, target_frames=26):
    """
    Extracts 26 standardized pose frames up to the exact moment of foot-ball contact.
    Works robustly on both:
      1. TV Broadcast penalty shootout clips (filters out camera cuts and face close-ups).
      2. Field recorded clips.
    
    Returns:
      dict with:
        'sequence': np.ndarray of shape (26, 132) normalized
        'strike_frame_idx': int (frame index of foot-ball contact)
        'start_frame_idx': int
        'total_frames': int
        'fps': float
        'strike_image': BGR image of the strike moment with drawn skeleton
        'strike_landmarks': raw landmarks list
        'biomechanics': dict of approach angle, torso lean, ankle speed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0:
        cap.release()
        return None

    # Read video frames
    frames_bgr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()
    
    actual_frames = len(frames_bgr)
    if actual_frames < 5:
        return None
        
    # Run MediaPipe Pose tracker across frames with strict striker filter
    detected_poses = {} # frame_idx -> landmarks
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35
    ) as pose:
        for idx, f in enumerate(frames_bgr):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                if is_full_body_striker(lms, w, h):
                    detected_poses[idx] = lms

    # If strict filter yielded too few poses (e.g. low-res clip), try static mode with slightly relaxed thresholds
    if len(detected_poses) < 8:
        with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.25) as pose:
            for idx in range(0, actual_frames, 2):
                if idx in detected_poses:
                    continue
                rgb = cv2.cvtColor(frames_bgr[idx], cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    if is_full_body_striker(lms, w, h):
                        detected_poses[idx] = lms

    if len(detected_poses) < 5:
        return None
        
    valid_indices = sorted(detected_poses.keys())
    
    # Calculate kicking foot speed between consecutive valid striker frames
    ankle_speeds = []
    for i in range(1, len(valid_indices)):
        f1, f2 = valid_indices[i-1], valid_indices[i]
        # Only compare adjacent or near-adjacent frames
        if f2 - f1 <= 3:
            l1 = detected_poses[f1]
            l2 = detected_poses[f2]
            
            # Check both right ankle (28) and left ankle (27)
            r_spd = np.hypot((l2[28].x - l1[28].x) * w, (l2[28].y - l1[28].y) * h) / (f2 - f1)
            l_spd = np.hypot((l2[27].x - l1[27].x) * w, (l2[27].y - l1[27].y) * h) / (f2 - f1)
            kick_spd = max(r_spd, l_spd)
            
            # Camera transition filter: jumps > 150 px/frame are camera cuts, not player motion
            if kick_spd < 150:
                ankle_speeds.append((f2, kick_spd, r_spd, l_spd))

    # Strike frame determination:
    # Foot-ball contact occurs at the peak forward kicking acceleration/speed
    if ankle_speeds:
        peak = max(ankle_speeds, key=lambda s: s[1])
        strike_frame = peak[0]
    else:
        # Fallback to last valid striker frame
        strike_frame = valid_indices[-1]
        
    # Ensure strike_frame has a valid striker pose
    if strike_frame not in detected_poses:
        closest_idx = min(detected_poses.keys(), key=lambda k: abs(k - strike_frame))
        strike_frame = closest_idx
        
    # Run-up window: exactly up to the contact moment
    runup_duration_frames = int(1.8 * fps)
    start_frame = max(valid_indices[0], strike_frame - runup_duration_frames)
    
    # Sample exactly 26 frames evenly between start_frame and strike_frame (UP TO CONTACT)
    sample_indices = np.round(np.linspace(start_frame, strike_frame, target_frames)).astype(int)
    
    raw_keypoints = []
    for s_idx in sample_indices:
        if s_idx in detected_poses:
            lms = detected_poses[s_idx]
        else:
            closest_idx = min(detected_poses.keys(), key=lambda k: abs(k - s_idx))
            lms = detected_poses[closest_idx]
            
        kp_frame = [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms]
        raw_keypoints.append(kp_frame)
        
    raw_keypoints = np.array(raw_keypoints, dtype=np.float32) # (26, 33, 4)
    
    # Coordinate Normalization (Hip center + Torso height scale)
    normalized_seq = raw_keypoints.copy()
    for f_i in range(len(normalized_seq)):
        frame_lms = normalized_seq[f_i]
        left_hip = frame_lms[23, :2]
        right_hip = frame_lms[24, :2]
        hip_center = (left_hip + right_hip) / 2
        
        left_shoulder = frame_lms[11, :2]
        right_shoulder = frame_lms[12, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        torso_height = np.linalg.norm(shoulder_center - hip_center)
        if torso_height < 1e-6:
            torso_height = 1.0
            
        normalized_seq[f_i, :, 0] = (frame_lms[:, 0] - hip_center[0]) / torso_height
        normalized_seq[f_i, :, 1] = (frame_lms[:, 1] - hip_center[1]) / torso_height
        
    flattened_seq = normalized_seq.reshape(target_frames, 132)
    
    # Biomechanics Telemetry at strike
    strike_lms = detected_poses[strike_frame]
    
    # 1. Torso lean angle (degrees from vertical)
    s_mid = ((strike_lms[11].x + strike_lms[12].x)/2, (strike_lms[11].y + strike_lms[12].y)/2)
    h_mid = ((strike_lms[23].x + strike_lms[24].x)/2, (strike_lms[23].y + strike_lms[24].y)/2)
    dx = s_mid[0] - h_mid[0]
    dy = h_mid[1] - s_mid[1] # positive upward
    torso_lean_deg = float(np.degrees(np.arctan2(dx, max(1e-5, dy))))
    
    # 2. Pelvis rotation angle
    p_dx = strike_lms[24].x - strike_lms[23].x
    p_dy = strike_lms[24].y - strike_lms[23].y
    pelvis_rot_deg = float(np.degrees(np.arctan2(p_dy, p_dx)))
    
    # 3. Peak kicking ankle speed
    peak_speed_px = max([s[1] for s in ankle_speeds]) if ankle_speeds else 22.0
    
    # 4. Plant foot distance from hip center
    plant_foot_x = strike_lms[27].x # left ankle
    plant_offset = float(plant_foot_x - h_mid[0])
    
    # Render strike frame with high-contrast skeleton overlay on the striker
    strike_img = frames_bgr[strike_frame].copy()
    annotated = strike_img.copy()
    
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in strike_lms]
    connections = mp_pose.POSE_CONNECTIONS
    for c1, c2 in connections:
        if c1 < len(pts) and c2 < len(pts):
            # Cyan bone connections
            cv2.line(annotated, pts[c1], pts[c2], (255, 200, 0), 3, cv2.LINE_AA)
    for x_p, y_p in pts:
        # Amber/Coral landmark joints
        cv2.circle(annotated, (x_p, y_p), 5, (0, 100, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, (x_p, y_p), 2, (255, 255, 255), -1, cv2.LINE_AA)
        
    return {
        'sequence': flattened_seq,
        'strike_frame_idx': int(strike_frame),
        'start_frame_idx': int(start_frame),
        'total_frames': actual_frames,
        'fps': float(fps),
        'strike_image': annotated,
        'biomechanics': {
            'torso_lean_deg': round(torso_lean_deg, 1),
            'pelvis_rot_deg': round(pelvis_rot_deg, 1),
            'peak_ankle_speed_px': round(float(peak_speed_px), 1),
            'plant_offset': round(plant_offset, 3)
        }
    }
