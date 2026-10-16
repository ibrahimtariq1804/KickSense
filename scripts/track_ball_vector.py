import cv2
import numpy as np
import os
import pandas as pd

def compute_ball_flight_vector(vid_id):
    vpath = f"data/videos/{vid_id}.mp4"
    if not os.path.exists(vpath):
        return None
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
        frames.append(cv2.resize(f, (640, 360)))
        idx += 1
    cap.release()
    
    if len(frames) < 15:
        return None
        
    w, h = 640, 360
    
    # Find strike frame by looking at peak motion in kicking zone (y: 0.55 to 0.85)
    diff_sum = []
    for t in range(len(frames)-1):
        d = cv2.absdiff(frames[t+1], frames[t])
        diff_sum.append(np.sum(d[int(0.55*h):int(0.85*h), int(0.3*w):int(0.7*w)]))
        
    strike_t = int(np.argmax(diff_sum))
    
    # In frames [strike_t - 2, strike_t + 10], track the fastest moving compact object moving upwards
    ball_positions = []
    
    # Grass color mask (green grass)
    # The ball is white / high contrast against green grass
    for t in range(strike_t, min(len(frames)-1, strike_t + 12)):
        f_cur = frames[t]
        f_next = frames[t+1]
        diff = cv2.absdiff(f_next, f_cur)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Only in grass region between penalty spot and goal line (y: 0.40 to 0.75)
        grass_mask = np.zeros_like(diff_gray)
        grass_mask[int(0.40*h):int(0.75*h), int(0.25*w):int(0.75*w)] = 255
        roi = cv2.bitwise_and(diff_gray, diff_gray, mask=grass_mask)
        
        _, th = cv2.threshold(roi, 35, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Candidate ball contours
        best_c = None
        best_dist = 999
        for c in cnts:
            area = cv2.contourArea(c)
            if 15 < area < 300:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                # If we have previous ball position, pick the one closest to expected trajectory
                if ball_positions:
                    prev_x, prev_y = ball_positions[-1][1], ball_positions[-1][2]
                    # Ball must move upward (cy <= prev_y + 2)
                    if cy <= prev_y + 3:
                        dist = np.hypot(cx - prev_x, cy - prev_y)
                        if 3 < dist < 45 and dist < best_dist:
                            best_dist = dist
                            best_c = (t, cx, cy, area)
                else:
                    # Initial ball detection near penalty spot
                    best_c = (t, cx, cy, area)
                    
        if best_c:
            ball_positions.append(best_c)
            
    if len(ball_positions) >= 2:
        xs = [p[1] for p in ball_positions]
        ys = [p[2] for p in ball_positions]
        # Linear regression to get trajectory line x(y)
        # y decreases as ball approaches goal (goal line is at y ≈ 190 px in 360h)
        slope, intercept = np.polyfit(ys, xs, 1)
        # Goal line y ≈ 0.53 * 360 = 191 px
        projected_x_at_goal = slope * 191 + intercept
        
        # Goal boundaries on 640x360:
        # Left post: ~227 px (0.355 * 640)
        # Right post: ~464 px (0.725 * 640)
        # Goal width: ~237 px
        # Left third: 227 to 306 px
        # Center third: 306 to 385 px
        # Right third: 385 to 464 px
        
        if projected_x_at_goal < 306:
            direction = "left"
        elif projected_x_at_goal > 385:
            direction = "right"
        else:
            direction = "center"
            
        return {
            'vid_id': vid_id,
            'pts': len(ball_positions),
            'proj_x': projected_x_at_goal,
            'direction': direction,
            'positions': [(round(p[1], 1), round(p[2], 1)) for p in ball_positions]
        }
    return None

sample_ids = [f"penalty_{i}" for i in range(291, 305)]
for sid in sample_ids:
    res = compute_ball_flight_vector(sid)
    if res:
        print(f"{res['vid_id']}: ProjX={res['proj_x']:.1f} ({res['direction'].upper()}), Pts={res['pts']}, Path={res['positions'][:4]}", flush=True)
    else:
        print(f"{sid}: Not enough points", flush=True)
