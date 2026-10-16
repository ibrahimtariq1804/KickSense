import urllib.request
import json

samples = ['penalty_01', 'penalty_02', 'penalty_04', 'penalty_05', 'penalty_291', 'penalty_292']
for s in samples:
    url = f"http://127.0.0.1:8000/api/predict-sample?sample_id={s}"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"Sample {s}: OK -> {data['direction']}, strike_frame={data['strike_frame_idx']}, fps={data['fps']}, total={data['total_frames']}")
    except Exception as e:
        print(f"Sample {s}: ERROR -> {e}")
