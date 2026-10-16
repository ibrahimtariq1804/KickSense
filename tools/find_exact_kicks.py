import cv2
import numpy as np
import os

def find_cuts_and_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prev_gray = None
    cuts = []
    diffs = []
    
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            # Mean absolute difference
            diff = np.mean(cv2.absdiff(gray, prev_gray))
            diffs.append((i, diff))
            if diff > 35.0: # clear camera cut
                cuts.append((i, round(i/fps, 2), round(diff, 1)))
        prev_gray = gray
        
    cap.release()
    return fps, total, cuts

for vid in ['penalty_01', 'penalty_02', 'penalty_04', 'penalty_05', 'penalty_06', 'penalty_07', 'penalty_11', 'penalty_291', 'penalty_292']:
    p = os.path.join("data", "videos", f"{vid}.mp4")
    if os.path.exists(p):
        fps, total, cuts = find_cuts_and_motion(p)
        print(f"{vid} (total={total}, fps={fps}): cuts={cuts}")
