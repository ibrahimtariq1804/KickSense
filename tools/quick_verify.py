import sys
sys.path.insert(0, ".")
from scripts.contact_detector import detect_contact_and_extract_sequence

for vid in ['penalty_06', 'penalty_07', 'penalty_11', 'penalty_291', 'penalty_292']:
    res = detect_contact_and_extract_sequence(f"data/videos/{vid}.mp4")
    if res:
        sf = res['strike_frame_idx']
        fps = res['fps']
        print(f"{vid}: strike_frame={sf}, strike_time={sf/fps:.2f}s, seq_shape={res['sequence'].shape}", flush=True)
    else:
        print(f"{vid}: FAILED", flush=True)
