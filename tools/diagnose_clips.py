import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def diagnose(video_id):
    p = f"data/videos/{video_id}.mp4"
    cap = cv2.VideoCapture(p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n=================== {video_id} (total={total}, fps={fps}) ===================")
    pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    
    records = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            l_ank = lms[27]
            r_ank = lms[28]
            l_hip = lms[23]
            r_hip = lms[24]
            nose = lms[0]
            
            # calculate bounding height
            body_top = min(lms[11].y, lms[12].y, nose.y)
            body_bottom = max(l_ank.y, r_ank.y)
            body_h = body_bottom - body_top
            
            records.append({
                'f': idx,
                't': idx / fps,
                'l_ank_v': l_ank.visibility,
                'r_ank_v': r_ank.visibility,
                'body_h': body_h,
                'l_ank_y': l_ank.y,
                'r_ank_y': r_ank.y,
                'l_ank_x': l_ank.x,
                'r_ank_x': r_ank.x,
                'nose_y': nose.y
            })
        idx += 1
    cap.release()
    pose.close()
    
    print(f"Total detected frames: {len(records)} / {total}")
    if records:
        print(f"Frames range: {records[0]['f']} to {records[-1]['f']}")
        # sample some records
        for r in records[::max(1, len(records)//15)]:
            print(f"  frame {r['f']:3d} ({r['t']:.2f}s): body_h={r['body_h']:.2f}, nose_y={r['nose_y']:.2f}, L_ank_v={r['l_ank_v']:.2f}, R_ank_v={r['r_ank_v']:.2f}")

for vid in ['penalty_01', 'penalty_02', 'penalty_04', 'penalty_05']:
    diagnose(vid)
