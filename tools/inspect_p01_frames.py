import cv2
import os

cap = cv2.VideoCapture("data/videos/penalty_01.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs("outputs/inspect_p01", exist_ok=True)
for i in range(total):
    ret, f = cap.read()
    if not ret:
        break
    if i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 86, 87, 90, 100, 107, 110, 120, 150, 180]:
        cv2.imwrite(f"outputs/inspect_p01/frame_{i:03d}_{i/fps:.2f}s.jpg", f)
cap.release()
print("Saved p01 frames")
