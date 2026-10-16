import cv2
import os

def analyze_timeline(video_id):
    path = os.path.join("data", "videos", f"{video_id}.mp4")
    if not os.path.exists(path):
        print(f"{video_id}: Not found")
        return
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n--- {video_id} (Total: {total} frames, FPS: {fps:.1f}, Duration: {total/fps:.2f}s) ---")
    
    # Save a frame strip every 0.5 seconds or 10-15 frames
    step = int(fps * 0.5)
    os.makedirs(f"outputs/inspect_timeline/{video_id}", exist_ok=True)
    f_idx = 0
    saved = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx % step == 0:
            h, w = frame.shape[:2]
            # write frame
            cv2.imwrite(f"outputs/inspect_timeline/{video_id}/f_{f_idx:03d}_{f_idx/fps:.2f}s.jpg", frame)
            saved.append((f_idx, f_idx/fps))
        f_idx += 1
    cap.release()
    print(f"Saved {len(saved)} frames to outputs/inspect_timeline/{video_id}/")

for vid in ['penalty_01', 'penalty_02', 'penalty_04', 'penalty_05', 'penalty_06', 'penalty_07', 'penalty_11']:
    analyze_timeline(vid)
