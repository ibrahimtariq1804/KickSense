import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def detect_contact_and_extract_sequence(video_path, target_frames=26):
    """
    Extracts 26 standardized pose frames up to the moment of foot-ball contact.
    Works on both:
      1. Videos that already cut off at foot-ball contact.
      2. Full penalty kick videos showing run-up, kick, and follow-through.
    
    Returns:
      dict with:
        'sequence': np.ndarray of shape (26, 132) normalized
        'strike_frame_idx': int (frame index in original video)
        'total_frames': int
        'strike_image': BGR image of the strike moment with drawn skeleton
        'strike_landmarks': raw landmarks list
        'biomechanics': dict of approach angle, torso lean, ankle speed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0:
        cap.release()
        return None
        
    # Read video frames (downscaled for speed during pose extraction)
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
        
    # Run MediaPipe Pose tracker across frames
    detected_poses = {} # frame_idx -> landmarks
    ankle_speeds = []
    hip_displacements = []
    prev_ankle = None
    prev_hip = None
    
    # Process frames with Pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as pose:
        for idx, f in enumerate(frames_bgr):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                detected_poses[idx] = lms
                
                # Right ankle (28) and hips (23, 24) in pixel coords
                ax = lms[28].x * w
                ay = lms[28].y * h
                hx = (lms[23].x + lms[24].x) / 2 * w
                hy = (lms[23].y + lms[24].y) / 2 * h
                
                if prev_ankle is not None:
                    ankle_speeds.append((idx, np.hypot(ax - prev_ankle[0], ay - prev_ankle[1])))
                    hip_displacements.append((idx, np.hypot(hx - prev_hip[0], hy - prev_hip[1])))
                prev_ankle = (ax, ay)
                prev_hip = (hx, hy)
                
    if len(detected_poses) < 5:
        return None
        
    # Determine the strike frame (foot-ball contact):
    # If the video ends near the kick, strike_frame is near the end or peak ankle speed
    strike_frame = actual_frames - 1
    if ankle_speeds:
        # Find peak ankle speed in the video
        peak_idx = int(np.argmax([s[1] for s in ankle_speeds]))
        peak_frame = ankle_speeds[peak_idx][0]
        # Strike frame is at peak acceleration or impact
        strike_frame = min(actual_frames - 1, peak_frame + 1)
    else:
        # If speeds couldn't be calculated, strike frame is the last frame with a valid pose
        valid_indices = sorted(detected_poses.keys())
        strike_frame = valid_indices[-1]
        
    # Run-up start window: 1.5 to 2.0 seconds before strike
    runup_duration_frames = int(1.8 * fps)
    start_frame = max(0, strike_frame - runup_duration_frames)
    
    # If hip displacement indicates run-up started later, refine start_frame
    if hip_displacements:
        for frame_idx, disp in hip_displacements:
            if start_frame <= frame_idx < strike_frame and disp > 2.0:
                start_frame = max(0, frame_idx - int(0.2 * fps))
                break
                
    # Sample exactly 26 frames evenly between start_frame and strike_frame (UP TO CONTACT)
    sample_indices = np.round(np.linspace(start_frame, strike_frame, target_frames)).astype(int)
    
    raw_keypoints = []
    for s_idx in sample_indices:
        # Get closest detected pose
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
    strike_lms = detected_poses.get(strike_frame, detected_poses[min(detected_poses.keys(), key=lambda k: abs(k - strike_frame))])
    
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
    peak_speed_px = max([s[1] for s in ankle_speeds]) if ankle_speeds else 15.0
    
    # 4. Plant foot distance from hip center
    plant_foot_x = strike_lms[27].x # left ankle
    plant_offset = float(plant_foot_x - h_mid[0])
    
    # Render strike frame with skeleton overlay
    strike_img = frames_bgr[strike_frame].copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Convert strike_lms to NormalizedLandmarkList for drawing
    landmark_proto = mp_pose.PoseLandmark
    annotated = strike_img.copy()
    
    # Custom high-contrast drawing for presentation
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
