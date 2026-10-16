"""
Optimized High-Speed Batch Processing for 201 New Recorded Clips with Resume Support:
1. Fast resume (skips already processed clips)
2. 4K decode acceleration (decodes every 2nd frame for 4K 60fps)
3. Action localization & temporal dead-time trimming
4. Foreground kicker isolation & MediaPipe pose extraction
5. Ball trajectory tracking across goal frame (direction & power labeling)
6. Visual audit card generation
7. Save to data/videos/, data/keypoints/, and update penalty_labels.xlsx
"""
import os
import sys
import shutil
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

sys.stdout.reconfigure(line_buffering=True)

INPUT_DIR = os.path.join("data", "videos", "New Recorded Videos")
VIDEOS_DIR = os.path.join("data", "videos")
KEYPOINTS_DIR = os.path.join("data", "keypoints")
AUDIT_DIR = os.path.join("outputs", "incoming_audit")
LABELS_PATH = os.path.join("data", "labels", "penalty_labels.xlsx")
CSV_EXPORT_PATH = os.path.join("outputs", "incoming_audit", "new_201_labels.csv")

os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(KEYPOINTS_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

START_BASE_NUM = 291

def analyze_and_extract_clip(video_path, penalty_id, pose_tracker, pose_extractor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process at 640x360
    tw, th = 640, 360
    
    # Decode acceleration: 4K clips are 60fps, skip decode on every 2nd frame
    decode_step = 2 if (w >= 3000 and fps >= 50.0) else 1
    
    frames_buffer = []
    kicker_poses = {} # buffer_idx -> lms
    ankle_speeds = []
    ankle_speed_frames = []
    prev_ankle = None
    prev_hip = None
    hip_displacements = []
    
    raw_idx = 0
    buf_idx = 0
    
    while True:
        if decode_step > 1:
            ret = cap.grab()
            if not ret: break
            if raw_idx % decode_step != 0:
                raw_idx += 1
                continue
            ret, frame = cap.retrieve()
            if not ret: break
        else:
            ret, frame = cap.read()
            if not ret: break
            
        resized = cv2.resize(frame, (tw, th))
        frames_buffer.append(resized)
        
        # Track pose on kicker every 2 buffered frames
        if buf_idx % 2 == 0:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            res = pose_tracker.process(rgb)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                torso_h = abs(lms[24].y - lms[12].y)
                hip_y = (lms[23].y + lms[24].y) / 2
                if torso_h > 0.065 and hip_y > 0.45:
                    kicker_poses[buf_idx] = lms
                    ax, ay = lms[28].x * tw, lms[28].y * th
                    hx, hy = (lms[23].x + lms[24].x)/2 * tw, (lms[23].y + lms[24].y)/2 * th
                    if prev_ankle is not None:
                        ankle_speeds.append(np.hypot(ax - prev_ankle[0], ay - prev_ankle[1]))
                        ankle_speed_frames.append(buf_idx)
                        hip_displacements.append(np.hypot(hx - prev_hip[0], hy - prev_hip[1]))
                    prev_ankle = (ax, ay)
                    prev_hip = (hx, hy)
        buf_idx += 1
        raw_idx += 1
        
    cap.release()
    
    if not frames_buffer:
        return None
        
    actual_frames = len(frames_buffer)
    effective_fps = fps / decode_step
    
    # Action window
    strike_frame = int(actual_frames * 0.75)
    if ankle_speeds:
        peak_idx = int(np.argmax(ankle_speeds))
        strike_frame = ankle_speed_frames[peak_idx]
        
    start_frame = max(0, strike_frame - int(1.8 * effective_fps))
    if hip_displacements:
        for idx_i in range(len(hip_displacements)):
            if ankle_speed_frames[idx_i] < strike_frame and hip_displacements[idx_i] > 1.2:
                start_frame = max(0, ankle_speed_frames[idx_i] - int(0.3 * effective_fps))
                break
                
    end_frame = min(actual_frames - 1, strike_frame + int(0.8 * effective_fps))
    if end_frame - start_frame < int(1.2 * effective_fps):
        start_frame = max(0, strike_frame - int(1.2 * effective_fps))
        end_frame = min(actual_frames - 1, strike_frame + int(0.8 * effective_fps))
        
    # Ball detection in goal area
    gx1, gx2 = int(0.34 * tw), int(0.72 * tw)
    gy1, gy2 = int(0.35 * th), int(0.64 * th)
    
    ball_x_coords = []
    bg_sample = frames_buffer[min(10, actual_frames - 1)]
    bg_goal = cv2.cvtColor(bg_sample[gy1:gy2, gx1:gx2], cv2.COLOR_BGR2GRAY)
    
    search_start = max(0, strike_frame - 2)
    search_end = min(actual_frames, strike_frame + int(1.4 * effective_fps))
    
    for f in range(search_start, search_end):
        cur_frame = frames_buffer[f]
        roi = cur_frame[gy1:gy2, gx1:gx2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, bg_goal)
        _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=12,
            param1=50, param2=18, minRadius=3, maxRadius=16
        )
        if circles is not None:
            for c in circles[0]:
                bx = float(c[0] + gx1)
                ball_x_coords.append(bx / tw)
                
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 15 < area < 400:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                bx = float(cx + gx1)
                ball_x_coords.append(bx / tw)
                
    if ball_x_coords:
        median_x = float(np.median(ball_x_coords))
        if median_x < 0.475:
            direction = "left"
            placement = "bottom_left"
        elif median_x > 0.605:
            direction = "right"
            placement = "bottom_right"
        else:
            direction = "center"
            placement = "center"
    else:
        if strike_frame in kicker_poses:
            lms = kicker_poses[strike_frame]
            hip_dx = lms[24].x - lms[23].x
            direction = "right" if hip_dx > 0.05 else "left"
        else:
            direction = "center"
        placement = "center"
        median_x = 0.537
        
    if ankle_speeds:
        peak_spd = max(ankle_speeds) * decode_step
        power = float(np.clip(round(peak_spd / 1.5), 3.0, 10.0))
    else:
        power = 6.0
        
    sample_indices = np.round(np.linspace(start_frame, end_frame, 26)).astype(int)
    keypoints_seq = []
    annotated_audit = None
    
    for idx in sample_indices:
        idx = min(idx, actual_frames - 1)
        f_bgr = frames_buffer[idx]
        rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
        res = pose_extractor.process(rgb)
        
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            kp = [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms]
            keypoints_seq.append(kp)
            if idx == strike_frame or (annotated_audit is None and idx >= strike_frame):
                annotated_audit = f_bgr.copy()
                mp_drawing.draw_landmarks(annotated_audit, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            if keypoints_seq:
                keypoints_seq.append(keypoints_seq[-1])
            else:
                keypoints_seq.append([[0, 0, 0, 0]] * 33)
                
    while len(keypoints_seq) < 26:
        keypoints_seq.append(keypoints_seq[-1])
    keypoints_seq = np.array(keypoints_seq[:26], dtype=np.float32)
    
    if annotated_audit is None:
        annotated_audit = frames_buffer[strike_frame].copy()
        
    audit_card = cv2.resize(annotated_audit, (960, 540))
    scale_w, scale_h = 960 / tw, 540 / th
    
    agx1, agx2 = int(gx1 * scale_w), int(gx2 * scale_w)
    agy1, agy2 = int(gy1 * scale_h), int(gy2 * scale_h)
    
    cv2.rectangle(audit_card, (agx1, agy1), (agx2, agy2), (255, 255, 0), 2)
    cv2.line(audit_card, (int(0.475 * 960), agy1), (int(0.475 * 960), agy2), (0, 255, 255), 1)
    cv2.line(audit_card, (int(0.605 * 960), agy1), (int(0.605 * 960), agy2), (0, 255, 255), 1)
    
    color_map = {"left": (255, 100, 0), "center": (0, 200, 255), "right": (0, 255, 100)}
    banner_color = color_map.get(direction, (0, 255, 0))
    cv2.rectangle(audit_card, (15, 15), (460, 95), (20, 20, 20), -1)
    cv2.rectangle(audit_card, (15, 15), (460, 95), banner_color, 2)
    
    cv2.putText(
        audit_card,
        f"{penalty_id}: {direction.upper()}",
        (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, banner_color, 2
    )
    cv2.putText(
        audit_card,
        f"Power: {power:.1f}/10 | Dets: {len(ball_x_coords)} | Strike: f{strike_frame}",
        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2
    )
    
    audit_card_path = os.path.join(AUDIT_DIR, f"{penalty_id}_audit.jpg")
    cv2.imwrite(audit_card_path, audit_card)
    
    return {
        "video_id": penalty_id,
        "original_file": os.path.basename(video_path),
        "shot_direction": direction,
        "ball_placement": placement,
        "shot_power": float(power),
        "ball_detections": float(len(ball_x_coords)),
        "keypoints": keypoints_seq
    }

def main():
    print("=== STARTING HIGH-SPEED RESUMABLE PROCESSING OF 201 NEW CLIPS ===", flush=True)
    t_start_all = time.time()
    
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".mov", ".mp4"))])
    print(f"Total videos in source folder: {len(files)}", flush=True)
    
    pose_tracker = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    
    pose_extractor = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )
    
    # Load existing labels if CSV export already exists
    processed_records = []
    if os.path.exists(CSV_EXPORT_PATH):
        try:
            prev_df = pd.read_csv(CSV_EXPORT_PATH)
            processed_records = prev_df.to_dict("records")
            print(f"Loaded {len(processed_records)} previously processed records from CSV", flush=True)
        except Exception:
            pass
            
    processed_vids = {r["video_id"]: r for r in processed_records}
    
    for i, fname in enumerate(files):
        v_path = os.path.join(INPUT_DIR, fname)
        pen_num = START_BASE_NUM + i
        penalty_id = f"penalty_{pen_num:03d}"
        
        pen_kp_dir = os.path.join(KEYPOINTS_DIR, penalty_id)
        kp_file = os.path.join(pen_kp_dir, "keypoints.npy")
        dest_video = os.path.join(VIDEOS_DIR, f"{penalty_id}.mp4")
        audit_file = os.path.join(AUDIT_DIR, f"{penalty_id}_audit.jpg")
        
        # Check if already processed
        if os.path.exists(kp_file) and os.path.exists(dest_video) and os.path.exists(audit_file) and penalty_id in processed_vids:
            print(f"[{i+1}/{len(files)}] {penalty_id} already processed -> SKIP", flush=True)
            continue
            
        t0 = time.time()
        print(f"[{i+1}/{len(files)}] Processing {fname} -> {penalty_id}...", end="", flush=True)
        
        try:
            res = analyze_and_extract_clip(v_path, penalty_id, pose_tracker, pose_extractor)
            if res is None:
                print(" FAILED to open", flush=True)
                continue
                
            os.makedirs(pen_kp_dir, exist_ok=True)
            np.save(kp_file, res["keypoints"])
            shutil.copy2(v_path, dest_video)
            
            record = {
                "video_id": penalty_id,
                "match": "Self-recorded session 2026",
                "player": "RecordedPlayer",
                "shot_direction": res["shot_direction"],
                "outcome": "goal",
                "dominant_foot": "right",
                "tournament": "Field Recording 2026",
                "ball_placement": res["ball_placement"],
                "shot_power": res["shot_power"],
                "ball_detections": res["ball_detections"]
            }
            
            processed_vids[penalty_id] = record
            
            # Save incremental CSV
            pd.DataFrame(list(processed_vids.values())).to_csv(CSV_EXPORT_PATH, index=False)
            
            elapsed = time.time() - t0
            print(f" OK ({res['shot_direction'].upper()}, Pwr={res['shot_power']:.1f}, {elapsed:.1f}s)", flush=True)
            
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            
    pose_tracker.close()
    pose_extractor.close()
    
    all_new_records = list(processed_vids.values())
    print(f"\nAll {len(all_new_records)} clips processed! Total elapsed: {time.time()-t_start_all:.1f}s", flush=True)
    
    new_df = pd.DataFrame(all_new_records)
    new_df.to_csv(CSV_EXPORT_PATH, index=False)
    
    # Merge into penalty_labels.xlsx
    print("Updating master penalty_labels.xlsx...", flush=True)
    existing_df = pd.read_excel(LABELS_PATH)
    
    backup_path = os.path.join("data", "labels", "penalty_labels_backup_before_201.xlsx")
    if not os.path.exists(backup_path):
        existing_df.to_excel(backup_path, index=False)
        print(f"Created master backup at {backup_path}", flush=True)
        
    # Remove any existing new penalties before concatenating to avoid duplicates
    existing_df = existing_df[~existing_df["video_id"].isin(new_df["video_id"])]
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_excel(LABELS_PATH, index=False)
    print(f"Master Excel updated! Total labeled clips: {len(combined_df)} (previously {len(existing_df)})", flush=True)
    
    print("\nCombined Direction Distribution:")
    print(combined_df["shot_direction"].value_counts())
    
    print("\n=== BATCH PROCESSING FINISHED SUCCESSFULLY ===", flush=True)

if __name__ == "__main__":
    main()
