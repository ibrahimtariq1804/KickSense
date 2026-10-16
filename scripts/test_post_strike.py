import cv2
import numpy as np
import os

def detect_ball_post_strike(vid_id):
    vpath = f"data/videos/{vid_id}.mp4"
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
        frames.append(cv2.resize(f, (960, 540)))
        idx += 1
    cap.release()
    
    if len(frames) < 15:
        return None
        
    w, h = 960, 540
    # Pitch motion to find strike frame
    diff_sum = []
    for t in range(len(frames)-1):
        d = cv2.absdiff(frames[t+1], frames[t])
        # Only in lower half (kicker leg swing)
        diff_sum.append(np.sum(d[int(0.45*h):int(0.85*h), int(0.2*w):int(0.8*w)]))
        
    strike_t = int(np.argmax(diff_sum))
    
    # Net arrival window: 3 to 18 frames after strike
    arrival_start = min(len(frames)-1, strike_t + 3)
    arrival_end = min(len(frames)-1, strike_t + 20)
    
    # Baseline net image from before strike (e.g. max(0, strike_t - 8))
    bg_t = max(0, strike_t - 6)
    bg_net = cv2.cvtColor(frames[bg_t][175:295, 340:705], cv2.COLOR_BGR2GRAY)
    
    post_strike_xs = []
    post_strike_weights = []
    
    for t in range(arrival_start, arrival_end):
        cur_net = cv2.cvtColor(frames[t][175:295, 340:705], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(cur_net, bg_net)
        _, th = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 15 < area < 1000:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                abs_x = cx + 340
                post_strike_xs.append(abs_x)
                post_strike_weights.append(area)
                
    if not post_strike_xs:
        # Fallback to consecutive diff
        for t in range(arrival_start, arrival_end-1):
            d = cv2.absdiff(frames[t+1][175:295, 340:705], frames[t][175:295, 340:705])
            d_gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(d_gray, 25, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if 15 < area < 1000:
                    (cx, cy), r = cv2.minEnclosingCircle(c)
                    abs_x = cx + 340
                    post_strike_xs.append(abs_x)
                    post_strike_weights.append(area)
                    
    if post_strike_xs:
        weighted_x = np.average(post_strike_xs, weights=post_strike_weights)
        # Goal boundaries: 340 to 695
        # Left: 340 to 458
        # Center: 458 to 577
        # Right: 577 to 695
        if weighted_x < 458:
            dir_pred = "left"
        elif weighted_x > 577:
            dir_pred = "right"
        else:
            dir_pred = "center"
        return weighted_x, dir_pred, len(post_strike_xs)
    return None, "center", 0

for v in ["penalty_291", "penalty_292", "penalty_293", "penalty_300", "penalty_320"]:
    x_val, d_val, count = detect_ball_post_strike(v)
    print(f"{v}: Goal X={x_val:.1f} (counts={count}) -> {d_val.upper()}", flush=True)
