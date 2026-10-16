import cv2
import mediapipe as mp
import os

mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("data/videos/penalty_01.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

with mp_face.FaceDetection(min_detection_confidence=0.5) as face_det, \
     mp_pose.Pose(min_detection_confidence=0.3) as pose_det:
    
    for f_idx in [0, 30, 60, 85, 87, 95, 107, 110, 130, 160]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = face_det.process(rgb)
        pose_res = pose_det.process(rgb)
        
        face_count = len(face_res.detections) if face_res.detections else 0
        face_box_h = 0
        if face_res.detections:
            face_box_h = face_res.detections[0].location_data.relative_bounding_box.height
            
        has_pose = pose_res.pose_landmarks is not None
        pose_l_ank_v = 0
        pose_r_ank_v = 0
        body_h = 0
        if has_pose:
            lms = pose_res.pose_landmarks.landmark
            pose_l_ank_v = lms[27].visibility
            pose_r_ank_v = lms[28].visibility
            body_h = max(lms[27].y, lms[28].y) - min(lms[11].y, lms[12].y)
            
        print(f"Frame {f_idx:3d} ({f_idx/fps:.2f}s): faces={face_count} (first face h={face_box_h:.2f}), pose={has_pose}, body_h={body_h:.2f}, L_ank_v={pose_l_ank_v:.2f}, R_ank_v={pose_r_ank_v:.2f}")

cap.release()
