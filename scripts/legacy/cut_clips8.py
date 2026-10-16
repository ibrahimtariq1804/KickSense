import subprocess

video_file = "data/videos/ITALY 1-1 ENGLAND PENALTY SHOOT-OUT, EURO 2020 FINAL.mp4"

penalties = [
    ("0:01", "0:05", "penalty_64"),
    ("0:10", "0:18", "penalty_65"),
    ("0:18", "0:27", "penalty_66"),
    ("0:28", "0:36", "penalty_67"),
    ("0:35", "0:43", "penalty_68"),
    ("0:41", "0:52", "penalty_69"),
    ("0:50", "0:57", "penalty_70"),
    ("0:59", "1:05", "penalty_71"),
    ("1:05", "1:12", "penalty_72"),
    ("1:14", "1:21", "penalty_73"),
]

for start, end, name in penalties:
    output = f"data/videos/{name}.mp4"
    command = [
        "ffmpeg",
        "-ss", start,
        "-to", end,
        "-i", video_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        output
    ]
    print(f"Cutting {name}...")
    subprocess.run(command)
    print(f"Saved: {output}")

print("All penalties cut successfully!")