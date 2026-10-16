import cv2
import numpy as np

def track_kick_flight(vpath):
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
    
    # In each clip, the kicker runs up and strikes the ball.
    # Between y = 0.52 (goal line) and y = 0.75 (penalty area),
    # let's look at frame differencing between consecutive frames t and t+1!
    # Consecutive frame differencing (f[t+1] - f[t]) ELIMINATES static background,
    # eliminates stationary people, and only shows fast moving objects:
    # 1. The kicker's swinging leg
    # 2. The flying ball!
    w, h = 640, 360
    
    # Scan from frame 10 to len(frames)-5
    ball_traj = []
    for t in range(5, len(frames)-2):
        f1 = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(frames[t+1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(f2, f1)
        
        # Region of pitch between penalty spot and goal
        # x between 0.30 and 0.75, y between 0.40 and 0.70
        ry1, ry2 = int(0.38 * h), int(0.70 * h)
        rx1, rx2 = int(0.30 * w), int(0.75 * w)
        
        roi_diff = diff[ry1:ry2, rx1:rx2]
        _, th = cv2.threshold(roi_diff, 35, 255, cv2.THRESH_BINARY)
        
        # Look for small circular / compact components (ball)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = float(cw) / max(1, ch)
            # A flying ball in consecutive frame diff has area between 15 and 250, aspect ratio ~ 0.5 to 2.0
            if 15 < area < 250 and 0.4 < aspect < 2.5:
                cx = (centroids[i][0] + rx1) / w
                cy = (centroids[i][1] + ry1) / h
                ball_traj.append((t, cx, cy, area))
                
    return ball_traj

for p in [291, 292, 293, 294, 295, 296, 297]:
    traj = track_kick_flight(f"data/videos/penalty_{p}.mp4")
    print(f"penalty_{p}: {len(traj)} candidate ball points", flush=True)
    # Filter for upward motion towards goal (decreasing y)
    moving_up = []
    for i in range(len(traj)-1):
        t1, x1, y1, a1 = traj[i]
        t2, x2, y2, a2 = traj[i+1]
        if 1 <= t2 - t1 <= 3 and y2 < y1: # moving towards goal
            moving_up.append((t1, x1, y1, t2, x2, y2, (x2 - x1)/(y1 - y2 + 1e-5)))
    print(f"  upward vectors: {len(moving_up)}", flush=True)
    for v in moving_up[:5]:
        print(f"    f{v[0]}->f{v[3]}: ({v[1]:.3f}, {v[2]:.3f}) -> ({v[4]:.3f}, {v[5]:.3f}), slope dx/dy = {v[6]:.2f}", flush=True)
