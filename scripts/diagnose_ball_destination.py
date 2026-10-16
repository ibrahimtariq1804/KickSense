import cv2
import numpy as np
import os
import sys

def check_one(vpath):
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Reading {vpath}: {total} frames at {fps:.1f} fps", flush=True)
    
    # Read with grab stride if high fps
    step = 2 if fps > 40 else 1
    frames = []
    idx = 0
    while True:
        if step > 1 and idx % step != 0:
            if not cap.grab():
                break
            idx += 1
            continue
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(f, (640, 360)))
        idx += 1
    cap.release()
    print(f"Loaded {len(frames)} resized frames", flush=True)
    
    w, h = 640, 360
    # Goal net bounds (where the net hangs):
    # Left post: 0.355 * w, Right post: 0.725 * w
    # Crossbar: 0.320 * h, Goal line: 0.520 * h
    gx1, gx2 = int(0.355 * w), int(0.725 * w)
    gy1, gy2 = int(0.320 * h), int(0.520 * h)
    
    # Baseline net image from early frames (frames 2-8)
    bg_roi = np.median([frames[i][gy1:gy2, gx1:gx2] for i in range(2, min(9, len(frames)))], axis=0).astype(np.uint8)
    bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
    
    # Scan frames
    detections = []
    for f_idx in range(10, len(frames)):
        cur_roi = cv2.cvtColor(frames[f_idx][gy1:gy2, gx1:gx2], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(cur_roi, bg_gray)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Check connected components in net
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if 25 < area < 800:
                cx = (centroids[i][0] + gx1) / w
                cy = (centroids[i][1] + gy1) / h
                detections.append((f_idx, cx, cy, area))
                
    print(f"Total net detections: {len(detections)}", flush=True)
    if detections:
        # Group by frame
        by_f = {}
        for f_idx, cx, cy, a in detections:
            by_f.setdefault(f_idx, []).append((cx, cy, a))
        # Find frames with top motion
        top_frames = sorted(by_f.keys(), key=lambda f: sum(m[2] for m in by_f[f]), reverse=True)[:5]
        for tf in top_frames:
            xs = [m[0] for m in by_f[tf]]
            areas = [m[2] for m in by_f[tf]]
            print(f"  Frame {tf}: mean_X = {np.mean(xs):.3f}, xs = {[round(x, 3) for x in xs]}, total_area = {sum(areas)}", flush=True)

check_one("data/videos/penalty_291.mp4")
check_one("data/videos/penalty_292.mp4")
check_one("data/videos/penalty_293.mp4")
