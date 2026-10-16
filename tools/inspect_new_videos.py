import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

print("Starting Video Inspection...")
VIDEOS_DIR = os.path.join("data", "videos", "New Recorded Videos")
OUTPUT_DIR = os.path.join("outputs", "new_videos_inspection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join("outputs", "model", "best_model_improved.keras")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded model successfully from", MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)
else:
    print("Model not found at", MODEL_PATH)

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

def detect_ball_simple(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=50, param2=25, minRadius=4, maxRadius=45
    )
    return circles is not None

def inspect_video(video_path):
    filename = os.path.basename(video_path)
    print(f"\n==========================================")
    print(f"Inspecting: {filename}")
    print(f"==========================================")
    
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {filename} with OpenCV!")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Resolution: {width} x {height} ({'Vertical/Portrait' if height > width else 'Horizontal/Landscape'})")
    print(f"FPS: {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"FourCC Codec: {fourcc_str}")
    
    # Process frames with MediaPipe
    # We will sample every 2 frames for speed, or every frame if total_frames < 200
    stride = 1 if total_frames <= 150 else 2
    
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )
    
    detected_frames = 0
    sampled_frames = 0
    keypoints_list = []
    
    saved_samples = 0
    ball_detected_count = 0
    
    # Track joint visibility
    hip_vis = []
    knee_vis = []
    ankle_vis = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % stride == 0:
            sampled_frames += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            has_ball = detect_ball_simple(frame)
            if has_ball:
                ball_detected_count += 1
                
            if results.pose_landmarks:
                detected_frames += 1
                lms = results.pose_landmarks.landmark
                
                # Check key lower-body joints for penalty kick
                left_hip_v = lms[23].visibility
                right_hip_v = lms[24].visibility
                left_knee_v = lms[25].visibility
                right_knee_v = lms[26].visibility
                left_ankle_v = lms[27].visibility
                right_ankle_v = lms[28].visibility
                
                hip_vis.append((left_hip_v + right_hip_v) / 2)
                knee_vis.append((left_knee_v + right_knee_v) / 2)
                ankle_vis.append((left_ankle_v + right_ankle_v) / 2)
                
                kp = [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms]
                keypoints_list.append(kp)
                
                # Save first, middle, and late annotated frames
                if saved_samples < 5 and (detected_frames == 1 or detected_frames == 10 or detected_frames == 25 or detected_frames == 50):
                    annotated = frame.copy()
                    mp_drawing.draw_landmarks(
                        annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    out_name = f"{os.path.splitext(filename)[0]}_frame_{frame_idx}_pose.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), annotated)
                    saved_samples += 1
                    
        frame_idx += 1
        
    cap.release()
    pose.close()
    
    pose_ratio = (detected_frames / sampled_frames) if sampled_frames > 0 else 0
    print(f"\n--- Detection Metrics ---")
    print(f"Sampled Frames: {sampled_frames}")
    print(f"Frames with Pose Detected: {detected_frames} / {sampled_frames} ({pose_ratio*100:.1f}%)")
    print(f"Ball detections: {ball_detected_count} frames")
    
    if hip_vis:
        print(f"Average Hip Visibility: {np.mean(hip_vis):.2f}")
        print(f"Average Knee Visibility: {np.mean(knee_vis):.2f}")
        print(f"Average Ankle Visibility: {np.mean(ankle_vis):.2f}")
        
    # Check Pipeline criteria
    print(f"\n--- Model & Training Pipeline Compatibility Check ---")
    passed_min_frames = detected_frames >= 10
    passed_ratio = pose_ratio >= 0.20
    print(f"[1] Meets MIN_DETECTED_FRAMES (>= 10): {'PASS' if passed_min_frames else 'FAIL'} ({detected_frames})")
    print(f"[2] Meets MIN_POSE_RATIO (>= 20%): {'PASS' if passed_ratio else 'FAIL'} ({pose_ratio*100:.1f}%)")
    
    # Test model prediction if we have keypoints
    if model is not None and len(keypoints_list) >= 10:
        kp_arr = np.array(keypoints_list)
        norm_kp = normalize_keypoints_seq(kp_arr)
        structured = structure_sequence(norm_kp, 26)
        flattened = structured.reshape(1, 26, 132)
        
        preds = model.predict(flattened, verbose=0)
        # Check output structure
        if isinstance(preds, list):
            dir_pred, pwr_pred = preds[0], preds[1]
        else:
            dir_pred, pwr_pred = preds, None
            
        dir_labels = ["Left", "Center", "Right"]
        dir_idx = np.argmax(dir_pred[0])
        dir_conf = dir_pred[0][dir_idx]
        
        print(f"[3] Model Inference: SUCCESS")
        print(f"    - Direction: {dir_labels[dir_idx]} (confidence: {dir_conf*100:.1f}%)")
        print(f"    - Direction Probabilities: Left={dir_pred[0][0]:.3f}, Center={dir_pred[0][1]:.3f}, Right={dir_pred[0][2]:.3f}")
        if pwr_pred is not None:
            print(f"    - Power: {pwr_pred[0][0]:.1f} / 10")
    else:
        print(f"[3] Model Inference: SKIPPED (not enough detected frames)")

# Inspect the 3 new videos
for f in sorted(os.listdir(VIDEOS_DIR)):
    if f.lower().endswith((".mov", ".mp4", ".avi")):
        inspect_video(os.path.join(VIDEOS_DIR, f))

# Also inspect one reference training video for comparison
ref_path = os.path.join("data", "videos", "penalty_01.mp4")
if os.path.exists(ref_path):
    print("\n\n==========================================")
    print("BASELINE REFERENCE CLIP (Original Training Data)")
    print("==========================================")
    inspect_video(ref_path)

print("\nInspection script completed!")
