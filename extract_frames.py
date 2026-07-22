import cv2
import os

videos_dir = "data/videos"
frames_dir = "data/frames"

# Only process penalty clips not the full video
penalty_clips = [f for f in os.listdir(videos_dir) if f.startswith("penalty_")]

for clip in penalty_clips:
    clip_path = os.path.join(videos_dir, clip)
    clip_name = clip.replace(".mp4", "")
    
    # Create a folder for each penalty clip
    output_dir = os.path.join(frames_dir, clip_name)
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(clip_path)
    frame_count = 0
    saved = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every frame
        cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
        saved += 1
        frame_count += 1
    
    cap.release()
    print(f"{clip_name}: {saved} frames extracted")

print("All frames extracted successfully!")