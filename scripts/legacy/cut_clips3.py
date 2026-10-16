import subprocess

video_file = "data/videos/Full Penalty Shoot-out ｜ Croatia vs Brazil ｜ 2022 FIFA World Cup Quarter-Final.mp4"

penalties = [
    ("5:49", "5:58", "penalty_17"),
    ("6:43", "6:49", "penalty_18"),
    ("7:21", "7:29", "penalty_19"),
    ("8:09", "8:17", "penalty_20"),
    ("8:51", "8:59", "penalty_21"),
    ("9:35", "9:44", "penalty_22"),
    ("10:22", "10:32", "penalty_23"),
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