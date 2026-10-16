import cv2
import mediapipe as mp
import numpy as np
import glob
import os

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection

def check_clip_frames(vid):
    files = sorted(glob.glob(f"outputs/inspect_timeline/{vid}/*.jpg"))
    if not files:
        return
    print(f"\n====================== {vid} ======================")
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.4) as pose, \
         mp_face.FaceDetection(min_detection_confidence=0.4) as face_det:
        for f in files:
            img = cv2.imread(f)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            f_res = face_det.process(rgb)
            p_res = pose.process(rgb)
            
            face_desc = "None"
            if f_res.detections:
                h_rel = f_res.detections[0].location_data.relative_bounding_box.height
                face_desc = f"Face(h={h_rel:.2f})"
                
            pose_desc = "None"
            if p_res.pose_landmarks:
                lms = p_res.pose_landmarks.landmark
                l_ank_v = lms[27].visibility
                r_ank_v = lms[28].visibility
                b_top = min(lms[11].y, lms[12].y)
                b_bot = max(lms[27].y, lms[28].y)
                b_h = b_bot - b_top
                pose_desc = f"Pose(h={b_h:.2f}, L_ank_v={l_ank_v:.2f}, R_ank_v={r_ank_v:.2f})"
                
            print(f"  {os.path.basename(f)}: Face={face_desc} | Pose={pose_desc}")

for v in ['penalty_01', 'penalty_02', 'penalty_04', 'penalty_05', 'penalty_06', 'penalty_07', 'penalty_11']:
    check_clip_frames(v)
