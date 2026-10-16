import cv2
import numpy as np
import os
import pandas as pd

df = pd.read_csv('outputs/incoming_audit/new_201_labels.csv')
sample_vids = df.head(8)['video_id'].tolist()

tw, th = 640, 360
gx1, gx2 = int(0.34 * tw), int(0.72 * tw)
gy1, gy2 = int(0.35 * th), int(0.64 * th)

for vid in sample_vids:
    vpath = os.path.join('data', 'videos', f"{vid}.mp4")
    if not os.path.exists(vpath):
        continue
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    for _ in range(min(total, 90)):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(f, (tw, th)))
    cap.release()
    
    if len(frames) < 20:
        continue
        
    bg = cv2.cvtColor(frames[5][gy1:gy2, gx1:gx2], cv2.COLOR_BGR2GRAY)
    
    diff_records = []
    for idx, f in enumerate(frames[10:], start=10):
        roi = cv2.cvtColor(f[gy1:gy2, gx1:gx2], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(roi, bg)
        _, th_img = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 15 < area < 400:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                abs_x = (cx + gx1) / tw
                abs_y = (cy + gy1) / th
                diff_records.append((idx, abs_x, abs_y, area))
                
    curr_lbl = df[df['video_id'] == vid]['shot_direction'].values[0]
    if diff_records:
        diff_records.sort(key=lambda x: x[0])
        first_few = [round(r[1], 3) for r in diff_records[:4]]
        last_few = [round(r[1], 3) for r in diff_records[-4:]]
        min_x = min(r[1] for r in diff_records)
        max_x = max(r[1] for r in diff_records)
        print(f"{vid} (Curr: {curr_lbl}): Count={len(diff_records)}, MinX={min_x:.3f}, MaxX={max_x:.3f}, FirstX={first_few}, LastX={last_few}")
    else:
        print(f"{vid} (Curr: {curr_lbl}): No detections")
