import cv2
import numpy as np
import pandas as pd
import os

def detect_ball(frame, prev_ball_pos=None):
    """Detect football using Hough Circle Transform"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=25,
        minRadius=5,
        maxRadius=40
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        if prev_ball_pos is not None:
            # Pick circle closest to previous position
            min_dist = float('inf')
            best = None
            for c in circles[0]:
                dist = np.sqrt((c[0] - prev_ball_pos[0])**2 + (c[1] - prev_ball_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best = c
            return best
        else:
            return circles[0][0]
    
    return None

def get_ball_placement(x, y, frame_width, frame_height):
    """Determine ball placement in goal based on final position"""
    # Horizontal zones
    if x < frame_width * 0.33:
        h_zone = "left"
    elif x > frame_width * 0.66:
        h_zone = "right"
    else:
        h_zone = "center"
    
    # Vertical zones
    if y < frame_height * 0.5:
        v_zone = "top"
    else:
        v_zone = "bottom"
    
    if h_zone == "center":
        return "center"
    return f"{v_zone}_{h_zone}"

def calculate_shot_power(positions, fps=25):
    """Estimate shot power from ball speed between frames"""
    if len(positions) < 2:
        return 5  # default medium power
    
    speeds = []
    for i in range(1, min(5, len(positions))):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        speed = np.sqrt(dx**2 + dy**2) * fps
        speeds.append(speed)
    
    avg_speed = np.mean(speeds) if speeds else 0
    
    # Normalize to 1-10 scale
    power = min(10, max(1, int(avg_speed / 50)))
    return power

def track_ball_in_clip(video_path):
    """Track ball through entire penalty clip"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ball_positions = []
    prev_pos = None
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        ball = detect_ball(frame, prev_pos)
        
        if ball is not None:
            x, y, r = ball
            ball_positions.append((int(x), int(y)))
            prev_pos = (int(x), int(y))
        
        frame_count += 1
    
    cap.release()
    
    # Get final ball position (last detected position)
    if ball_positions:
        final_x, final_y = ball_positions[-1]
        placement = get_ball_placement(final_x, final_y, frame_width, frame_height)
        power = calculate_shot_power(ball_positions, fps)
        return placement, power, len(ball_positions)
    else:
        return "unknown", 5, 0

# Process all penalty clips
videos_dir = "data/videos"
penalty_clips = sorted([f for f in os.listdir(videos_dir) if f.startswith("penalty_")])

results = []
for clip in penalty_clips:
    clip_path = os.path.join(videos_dir, clip)
    clip_name = clip.replace(".mp4", "")
    
    print(f"Tracking ball in {clip_name}...")
    placement, power, detections = track_ball_in_clip(clip_path)
    
    results.append({
        "video_id": clip_name,
        "ball_placement": placement,
        "shot_power": power,
        "ball_detections": detections
    })
    print(f"  Placement: {placement} | Power: {power} | Detections: {detections}")

# Save results
results_df = pd.DataFrame(results)
print("\n=== Ball Tracking Results ===")
print(results_df)

# Merge with existing labels
labels_df = pd.read_excel("data/labels/penalty_labels.xlsx")

# Drop old ball tracking columns if they exist
labels_df = labels_df.drop(columns=['ball_placement', 'shot_power', 'ball_detections'], errors='ignore')

# Merge fresh
merged_df = labels_df.merge(results_df, on="video_id", how="left")
merged_df.to_excel("data/labels/penalty_labels.xlsx", index=False)
print("\nLabels updated with ball tracking data!")
print(merged_df[["video_id", "ball_placement", "shot_power", "ball_detections"]])