import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

sys.stdout.reconfigure(line_buffering=True)

VIDEOS_DIR = os.path.join("data", "videos", "New Recorded Videos")
OUTPUT_DIR = os.path.join("outputs", "new_videos_inspection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join("outputs", "model", "best_model_improved.keras")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded KickSense model successfully.", flush=True)
    except Exception as e:
        print("Failed to load model:", e, flush=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def normalize_keypoints_seq(keypoints):
    """Normalize (frames, 33, 4) keypoints as in preprocessing.py"""
    normalized = keypoints.copy()
    for frame_idx in range(len(keypoints)):
        frame = keypoints[frame_idx]
        left_hip = frame[23, :2]
        right_hip = frame[24, :2]
        hip_center = (left_hip + right_hip) / 2
        left_shoulder = frame[11, :2]
        right_shoulder = frame[12, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        torso_height = np.linalg.norm(shoulder_center - hip_center)
        if torso_height < 1e-6:
            torso_height = 1.0
        normalized[frame_idx, :, 0] = (frame[:, 0] - hip_center[0]) / torso_height
        normalized[frame_idx, :, 1] = (frame[:, 1] - hip_center[1]) / torso_height
    return normalized

def structure_sequence(keypoints, target_length=26):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return keypoints[indices]

def run_model_inference(keypoints_list):
    if model is None or len(keypoints_list) < 5:
        return None
    kp_arr = np.array(keypoints_list)
    norm_kp = normalize_keypoints_seq(kp_arr)
    structured = structure_sequence(norm_kp, 26)
    flattened = structured.reshape(1, 26, 132)
    preds = model.predict(flattened, verbose=0)
    if isinstance(preds, list):
        dir_pred, pwr_pred = preds[0][0], preds[1][0][0]
    else:
        dir_pred, pwr_pred = preds[0], None
    labels = ["Left", "Center", "Right"]
    idx = int(np.argmax(dir_pred))
    return {
        "direction": labels[idx],
        "confidence": float(dir_pred[idx]),
        "probs": {labels[i]: float(dir_pred[i]) for i in range(3)},
        "power": float(pwr_pred) if pwr_pred is not None else None
    }

def inspect_video_detailed(video_path):
    filename = os.path.basename(video_path)
    base_name = os.path.splitext(filename)[0]
    print(f"\n=======================================================", flush=True)
    print(f"ANALYZING: {filename}", flush=True)
    print(f"=======================================================", flush=True)
    
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {filename} with OpenCV", flush=True)
        return
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0
    
    print(f"  File Size:   {size_mb:.2f} MB", flush=True)
    print(f"  Resolution:  {w} x {h} ({'Portrait' if h > w else 'Landscape'})", flush=True)
    print(f"  FPS:         {fps:.2f}", flush=True)
    print(f"  Frames:      {total_frames}", flush=True)
    print(f"  Duration:    {duration_s:.2f} seconds", flush=True)
    
    # Process at ~15 fps (stride = round(fps / 15)) for fast and accurate timeline tracking
    sample_stride = max(1, int(round(fps / 15.0)))
    
    # MediaPipe pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frame_indices = []
    poses = []
    hip_x_coords = []
    hip_y_coords = []
    
    # We will also save some milestone frames
    sampled_count = 0
    detected_count = 0
    
    # Target resize width 1280 for fast processing
    scale = 1280.0 / w if w > 1280 else 1.0
    target_w = int(w * scale)
    target_h = int(h * scale)
    
    for f_idx in range(0, total_frames, sample_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        sampled_count += 1
        frame_indices.append(f_idx)
        
        # Resize for speed
        if scale != 1.0:
            small_frame = cv2.resize(frame, (target_w, target_h))
        else:
            small_frame = frame
            
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        if res.pose_landmarks:
            detected_count += 1
            lms = res.pose_landmarks.landmark
            kp = [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms]
            poses.append(kp)
            
            hx = (lms[23].x + lms[24].x) / 2
            hy = (lms[23].y + lms[24].y) / 2
            hip_x_coords.append((f_idx, hx, hy))
        else:
            # If no pose, record None
            poses.append(None)
            
    cap.release()
    pose.close()
    
    det_ratio = (detected_count / sampled_count) if sampled_count > 0 else 0
    print(f"  Pose Detection: {detected_count} / {sampled_count} frames ({det_ratio*100:.1f}%)", flush=True)
    
    # Analyze kicker movement over time
    if hip_x_coords:
        first_frame, first_hx, first_hy = hip_x_coords[0]
        last_frame, last_hx, last_hy = hip_x_coords[-1]
        
        # Calculate speeds/displacements to locate run-up and kick
        displacements = []
        for i in range(1, len(hip_x_coords)):
            df = hip_x_coords[i][0] - hip_x_coords[i-1][0]
            dt = df / fps if fps > 0 else 1.0
            dx = hip_x_coords[i][1] - hip_x_coords[i-1][1]
            dy = hip_x_coords[i][2] - hip_x_coords[i-1][2]
            dist = np.hypot(dx, dy)
            speed = dist / dt if dt > 0 else 0
            displacements.append((hip_x_coords[i][0], speed))
            
        # Peak movement speed is likely the run-up / kick!
        if displacements:
            speeds = [s[1] for s in displacements]
            max_speed_idx = int(np.argmax(speeds))
            peak_frame = displacements[max_speed_idx][0]
            print(f"  Peak Kicker Movement: around frame {peak_frame} ({peak_frame/fps:.2f}s)", flush=True)
            
    # Extract 26 evenly-spaced frames from the ENTIRE video (Current app.py behavior)
    cap = cv2.VideoCapture(video_path)
    app_indices = np.linspace(0, total_frames - 1, 26, dtype=int)
    app_keypoints = []
    
    pose_static = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.4
    )
    
    saved_milestones = 0
    for idx in app_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if scale != 1.0:
            frame_resized = cv2.resize(frame, (target_w, target_h))
        else:
            frame_resized = frame
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        res = pose_static.process(rgb)
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            app_keypoints.append([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms])
            
            # Save 3 sample visualizations
            if saved_milestones < 3 and (saved_milestones == 0 or idx >= total_frames // 2):
                annotated = frame_resized.copy()
                mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                out_path = os.path.join(OUTPUT_DIR, f"{base_name}_frame_{idx}.jpg")
                cv2.imwrite(out_path, annotated)
                saved_milestones += 1
                
    cap.release()
    pose_static.close()
    
    print(f"  App.py Extraction (26 frames sampled across whole clip):", flush=True)
    print(f"    - Poses detected: {len(app_keypoints)} / 26 ({len(app_keypoints)/26*100:.1f}%)", flush=True)
    
    pred_full = run_model_inference(app_keypoints)
    if pred_full:
        print(f"    - Model Prediction (Full clip): Direction={pred_full['direction']} ({pred_full['confidence']*100:.1f}%), Power={pred_full['power']:.1f}", flush=True)
        print(f"      Probabilities: {pred_full['probs']}", flush=True)
    else:
        print(f"    - Model Prediction: FAILED (less than 5 poses)", flush=True)
        
    # Check valid pose sequence from filtered detected frames
    valid_poses = [p for p in poses if p is not None]
    if len(valid_poses) >= 26:
        pred_valid = run_model_inference(valid_poses)
        if pred_valid:
            print(f"  Model Prediction (Dense detected sequence):", flush=True)
            print(f"    - Direction={pred_valid['direction']} ({pred_valid['confidence']*100:.1f}%), Power={pred_valid['power']:.1f}", flush=True)
            print(f"      Probabilities: {pred_valid['probs']}", flush=True)

# Run on all 3 videos
for v in ["IMG_1012.MOV", "IMG_1048.MOV", "IMG_1049.MOV"]:
    v_path = os.path.join(VIDEOS_DIR, v)
    if os.path.exists(v_path):
        inspect_video_detailed(v_path)
    else:
        print(f"Video {v} not found!", flush=True)

# Also run on penalty_01 for direct comparison
ref = os.path.join("data", "videos", "penalty_01.mp4")
if os.path.exists(ref):
    inspect_video_detailed(ref)

print("\n--- FAST INSPECTION COMPLETED ---", flush=True)
