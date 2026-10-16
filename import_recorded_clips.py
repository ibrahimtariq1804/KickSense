"""
Import self-recorded clips from data/videos/incoming/Recorded Clips/,
auto-label shot direction / placement / power via ball tracking,
append rows to penalty_labels.xlsx.
"""
import os
import shutil
import csv

import cv2
import numpy as np
import pandas as pd

INCOMING_DIR = os.path.join("data", "videos", "incoming", "Recorded Clips")
VIDEOS_DIR = os.path.join("data", "videos")
LABELS_PATH = os.path.join("data", "labels", "penalty_labels.xlsx")
MAP_PATH = os.path.join("data", "videos", "incoming", "import_map.csv")
AUDIT_DIR = os.path.join("outputs", "incoming_audit", "labeled")

MIN_DURATION_S = 1.8
MAX_DURATION_S = 14.0
MIN_BALL_DETECTIONS = 5


def detect_ball(frame, prev_ball_pos=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=50, param2=25, minRadius=4, maxRadius=45,
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    if prev_ball_pos is not None:
        best, min_dist = None, float("inf")
        for c in circles[0]:
            dist = np.hypot(c[0] - prev_ball_pos[0], c[1] - prev_ball_pos[1])
            if dist < min_dist:
                min_dist, best = dist, c
        return best
    return circles[0][0]


def horizontal_zone(x, width):
    if x < width * 0.38:
        return "left"
    if x > width * 0.62:
        return "right"
    return "center"


def placement_from_xy(x, y, width, height):
    h = horizontal_zone(x, width)
    v = "top" if y < height * 0.55 else "bottom"
    if h == "center":
        return "center"
    return f"{v}_{h}"


def power_from_positions(positions, fps):
    if len(positions) < 2:
        return 5
    speeds = []
    for i in range(1, min(6, len(positions))):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        speeds.append(np.hypot(dx, dy) * fps)
    avg = float(np.mean(speeds)) if speeds else 0.0
    return int(min(10, max(1, round(avg / 50))))


def analyze_clip(video_path, frame_stride=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = n / fps if fps > 0 else 0.0

    positions = []
    prev = None
    mid_frame = None
    late_frame = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx == n // 2:
            mid_frame = frame.copy()
        if n > 0 and idx == max(n - 5, 0):
            late_frame = frame.copy()
        # Skip frames for speed; still enough for trajectory
        if idx % frame_stride == 0:
            ball = detect_ball(frame, prev)
            if ball is not None:
                x, y, r = int(ball[0]), int(ball[1]), int(ball[2])
                positions.append((x, y, idx))
                prev = (x, y)
        idx += 1
    cap.release()

    if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
        return {
            "ok": False,
            "reason": f"duration {duration:.1f}s out of range",
            "duration": duration,
            "detections": len(positions),
        }

    if len(positions) < MIN_BALL_DETECTIONS:
        return {
            "ok": False,
            "reason": f"only {len(positions)} ball detections",
            "duration": duration,
            "detections": len(positions),
        }

    # Direction from late-trajectory median x (more stable than single last hit)
    late = positions[max(0, len(positions) - 8):]
    xs = [p[0] for p in late]
    ys = [p[1] for p in late]
    # Also use delta: early vs late x movement toward goal
    early = positions[: min(5, len(positions))]
    early_x = float(np.median([p[0] for p in early]))
    late_x = float(np.median(xs))
    late_y = float(np.median(ys))

    direction = horizontal_zone(late_x, w)
    # If ball barely moved horizontally, prefer zone of late position
    placement = placement_from_xy(late_x, late_y, w, h)
    power = power_from_positions([(p[0], p[1]) for p in positions], fps)

    # Confidence heuristic
    spread = max(xs) - min(xs)
    conf = "high"
    if abs(late_x - w * 0.5) < w * 0.08 or spread < w * 0.05:
        conf = "low"
    elif direction == "center":
        conf = "medium"

    return {
        "ok": True,
        "reason": "ok",
        "duration": duration,
        "detections": len(positions),
        "shot_direction": direction,
        "ball_placement": placement,
        "shot_power": power,
        "confidence": conf,
        "late_x": late_x,
        "early_x": early_x,
        "mid_frame": mid_frame,
        "late_frame": late_frame,
        "width": w,
        "height": h,
    }


def next_penalty_number():
    nums = []
    for name in os.listdir(VIDEOS_DIR):
        if name.startswith("penalty_") and name.endswith(".mp4"):
            try:
                nums.append(int(name.replace("penalty_", "").replace(".mp4", "")))
            except ValueError:
                pass
    return (max(nums) + 1) if nums else 1


def main():
    print("Starting import_recorded_clips...", flush=True)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    clips = sorted(
        f for f in os.listdir(INCOMING_DIR)
        if f.lower().endswith(".mp4")
    )
    print(f"Found {len(clips)} incoming clips", flush=True)

    start_num = next_penalty_number()
    print(f"Starting import at penalty_{start_num:03d}")

    df = pd.read_excel(LABELS_PATH)
    # Ensure tracking columns exist
    for col in ("ball_placement", "shot_power", "ball_detections"):
        if col not in df.columns:
            df[col] = np.nan

    existing_ids = set(df["video_id"].astype(str))
    new_rows = []
    map_rows = []
    skipped = []
    num = start_num

    for filename in clips:
        src = os.path.join(INCOMING_DIR, filename)
        print(f"\nAnalyzing {filename}...")
        result = analyze_clip(src)
        if not result["ok"]:
            print(f"  SKIP: {result['reason']}")
            skipped.append({"original": filename, "reason": result["reason"]})
            continue

        video_id = f"penalty_{num:03d}"
        # Avoid collisions with odd ids like penalty_200 already in Excel
        while video_id in existing_ids or os.path.exists(
            os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        ):
            num += 1
            video_id = f"penalty_{num:03d}"

        dest = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        shutil.copy2(src, dest)

        # Audit overlays
        for tag, frame in (("mid", result.get("mid_frame")), ("late", result.get("late_frame"))):
            if frame is None:
                continue
            vis = frame.copy()
            cv2.putText(
                vis,
                f"{video_id} dir={result['shot_direction']} pwr={result['shot_power']} ({result['confidence']})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.imwrite(os.path.join(AUDIT_DIR, f"{video_id}_{tag}.jpg"), vis)

        row = {
            "video_id": video_id,
            "match": "Self-recorded session",
            "player": "LocalPlayer",
            "shot_direction": result["shot_direction"],
            "outcome": "goal",
            "dominant_foot": "right",
            "tournament": "Local recording 2024",
            "ball_placement": result["ball_placement"],
            "shot_power": float(result["shot_power"]),
            "ball_detections": float(result["detections"]),
        }
        new_rows.append(row)
        existing_ids.add(video_id)
        map_rows.append({
            "video_id": video_id,
            "original": filename,
            "shot_direction": result["shot_direction"],
            "ball_placement": result["ball_placement"],
            "shot_power": result["shot_power"],
            "ball_detections": result["detections"],
            "confidence": result["confidence"],
            "duration": round(result["duration"], 2),
        })
        print(
            f"  -> {video_id}: {result['shot_direction']} / "
            f"{result['ball_placement']} / power={result['shot_power']} "
            f"({result['confidence']}, dets={result['detections']})"
        )
        num += 1

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_excel(LABELS_PATH, index=False)

    with open(MAP_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id", "original", "shot_direction", "ball_placement",
                "shot_power", "ball_detections", "confidence", "duration",
            ],
        )
        writer.writeheader()
        writer.writerows(map_rows)

    print("\n=== IMPORT COMPLETE ===")
    print(f"Imported: {len(new_rows)}")
    print(f"Skipped:  {len(skipped)}")
    for s in skipped:
        print(f"  - {s['original']}: {s['reason']}")
    if map_rows:
        from collections import Counter
        print("Direction counts:", Counter(r["shot_direction"] for r in map_rows))
        print("Confidence:", Counter(r["confidence"] for r in map_rows))
    print(f"Total labels now: {len(df)}")
    print(f"Map saved: {MAP_PATH}")


if __name__ == "__main__":
    main()
