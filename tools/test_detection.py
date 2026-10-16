import os
import sys
sys.path.insert(0, ".")
import cv2
from scripts.contact_detector import detect_contact_and_extract_sequence

clips = ['penalty_01', 'penalty_02', 'penalty_04', 'penalty_05', 'penalty_06', 'penalty_07', 'penalty_11', 'penalty_291', 'penalty_292']
for c in clips:
    p = os.path.join('data', 'videos', f'{c}.mp4')
    if not os.path.exists(p):
        print(f"{c}: NOT FOUND")
        continue
    res = detect_contact_and_extract_sequence(p)
    if res is None:
        print(f"{c}: FAILED (None returned)")
    else:
        sf = res['strike_frame_idx']
        fps = res['fps']
        tf = res['total_frames']
        st = sf / fps if fps > 0 else 0
        print(f"{c}: total={tf}, fps={fps:.1f}, strike_frame={sf}, strike_time={st:.2f}s")
