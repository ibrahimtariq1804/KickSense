import cv2
import numpy as np
import os

os.makedirs("outputs/trajectory_viz", exist_ok=True)

def visualize_trajectory(vid_id):
    vpath = f"data/videos/{vid_id}.mp4"
    if not os.path.exists(vpath):
        return
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
        return
        
    w, h = 960, 540
    # Goal net bounds on 960x540:
    # Left post: ~340 (0.354), Right post: ~695 (0.724)
    # Crossbar: ~175 (0.324), Ground line: ~295 (0.546)
    
    # Let's find the strike frame by looking at kicker or motion
    # Between y = 0.30*h and 0.70*h, compute consecutive frame diffs
    diff_sum = []
    for t in range(len(frames)-1):
        d = cv2.absdiff(frames[t+1], frames[t])
        diff_sum.append(np.sum(d[int(0.3*h):int(0.7*h), int(0.3*w):int(0.75*w)]))
    
    peak_t = int(np.argmax(diff_sum))
    # Ball flight is peak_t to peak_t + 18
    start_t = max(0, peak_t - 2)
    end_t = min(len(frames)-1, peak_t + 20)
    
    # Create composite motion overlay
    base = frames[start_t].copy()
    max_diff = np.zeros((h, w), dtype=np.uint8)
    
    trajectory_points = []
    
    for t in range(start_t, end_t):
        f_gray = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        f_next = cv2.cvtColor(frames[t+1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(f_next, f_gray)
        
        # Only pitch & goal region
        roi_mask = np.zeros_like(diff)
        roi_mask[int(0.32*h):int(0.72*h), int(0.32*w):int(0.75*w)] = 255
        diff = cv2.bitwise_and(diff, diff, mask=roi_mask)
        
        _, th = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        # Find contours that are small and fast moving
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 15 < area < 400:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                # Ignore kicker's body area if it's too far left/bottom
                trajectory_points.append((t, cx, cy, area))
                # draw red circle on base
                cv2.circle(base, (int(cx), int(cy)), int(r), (0, 0, 255), 2)
                
    # Draw goal frame and 3 zones
    cv2.rectangle(base, (340, 175), (695, 295), (255, 255, 0), 2)
    # Zone 1 (Left): 340 to 458
    # Zone 2 (Center): 458 to 577
    # Zone 3 (Right): 577 to 695
    cv2.line(base, (458, 175), (458, 295), (0, 255, 255), 2)
    cv2.line(base, (577, 175), (577, 295), (0, 255, 255), 2)
    
    cv2.putText(base, "LEFT (340-458)", (340, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(base, "CENTER (458-577)", (460, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(base, "RIGHT (577-695)", (580, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    out_path = f"outputs/trajectory_viz/{vid_id}_trajectory.jpg"
    cv2.imwrite(out_path, base)
    print(f"Saved {out_path}", flush=True)

for vid in ["penalty_291", "penalty_292", "penalty_293", "penalty_300", "penalty_320"]:
    visualize_trajectory(vid)
