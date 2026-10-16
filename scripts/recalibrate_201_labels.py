import cv2
import numpy as np
import os
import pandas as pd

df_curr = pd.read_csv('outputs/incoming_audit/new_201_labels.csv')

recalibrated = []

for idx, row in df_curr.iterrows():
    vid_id = row['video_id']
    vpath = f"data/videos/{vid_id}.mp4"
    if not os.path.exists(vpath):
        recalibrated.append((vid_id, row['shot_direction'], 'missing'))
        continue
        
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 2 if fps > 40 else 1
    
    frames = []
    f_idx = 0
    while True:
        if step > 1 and f_idx % step != 0:
            if not cap.grab():
                break
            f_idx += 1
            continue
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(f, (640, 360)))
        f_idx += 1
    cap.release()
    
    if len(frames) < 15:
        recalibrated.append((vid_id, row['shot_direction'], 'too_short'))
        continue
        
    w, h = 640, 360
    # Strike frame detection
    diff_sum = [np.sum(cv2.absdiff(frames[t+1], frames[t])[int(0.5*h):int(0.85*h), int(0.2*w):int(0.8*w)]) for t in range(len(frames)-1)]
    strike_t = int(np.argmax(diff_sum))
    
    # Net window: strike + 4 to strike + 18
    arrival_start = min(len(frames)-1, strike_t + 3)
    arrival_end = min(len(frames)-1, strike_t + 18)
    
    # Check motion in net (y: 115 to 195 px in 360h, x: 227 to 464 px in 640w)
    # Goal boundaries: Left=227, Center-div1=306, Center-div2=385, Right=464
    xs = []
    weights = []
    
    for t in range(arrival_start, arrival_end):
        d = cv2.absdiff(frames[t], frames[max(0, strike_t - 4)])
        d_gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        roi = d_gray[115:195, 227:464]
        _, th = cv2.threshold(roi, 25, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if 15 < a < 600:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                xs.append(cx + 227)
                weights.append(a)
                
    if xs:
        avg_x = np.average(xs, weights=weights)
        if avg_x < 306:
            direct = "left"
        elif avg_x > 385:
            direct = "right"
        else:
            direct = "center"
    else:
        direct = row['shot_direction']
        
    recalibrated.append((vid_id, direct, 'ok'))

df_recal = pd.DataFrame(recalibrated, columns=['video_id', 'new_direction', 'status'])
print("Original distribution in new_201_labels.csv:")
print(df_curr['shot_direction'].value_counts())
print("\nRecalibrated distribution:")
print(df_recal['new_direction'].value_counts())
