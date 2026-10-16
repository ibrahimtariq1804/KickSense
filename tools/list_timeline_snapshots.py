import cv2
import glob
import os

for vid in ['penalty_01', 'penalty_02', 'penalty_04', 'penalty_05', 'penalty_06', 'penalty_07', 'penalty_11', 'penalty_291', 'penalty_292']:
    folder = f"outputs/inspect_timeline/{vid}"
    if not os.path.exists(folder):
        continue
    files = sorted(glob.glob(f"{folder}/*.jpg"))
    print(f"\n================ {vid} ({len(files)} timeline snapshots) ================")
    for f in files:
        base = os.path.basename(f)
        print(f"  {base}")
