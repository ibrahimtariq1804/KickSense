import urllib.request
import json

req = urllib.request.urlopen("http://127.0.0.1:8000/api/sample-clips")
data = json.loads(req.read().decode())
print("Available sample clips on server:")
for s in data['samples']:
    print(f"  {s['id']}: {s['name']} (true_dir={s['true_direction']}, pwr={s['true_power']})")

print("\nTesting /api/predict-sample?sample_id=penalty_06 ...")
post_req = urllib.request.Request("http://127.0.0.1:8000/api/predict-sample?sample_id=penalty_06", data=b"", method="POST")
res = urllib.request.urlopen(post_req)
pred = json.loads(res.read().decode())
print("Prediction response for penalty_06:")
print("  Status:", pred.get("status"))
print("  Direction:", pred.get("direction"))
print("  Confidence:", pred.get("confidence"))
print("  Power:", pred.get("power"))
print("  Strike frame:", pred.get("strike_frame_idx"))
print("  Total frames:", pred.get("total_frames"))
print("  Strike time:", pred.get("strike_frame_idx") / pred.get("fps"), "s")
print("  Has skeleton base64:", bool(pred.get("strike_skeleton_base64")))
print("  Biomechanics:", pred.get("biomechanics"))
