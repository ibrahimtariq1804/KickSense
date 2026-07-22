import subprocess
import os

video_file = "data/videos/Full Penalty Shoot-out ｜ Messi's Argentina vs Mbappe's France ｜ FIFA World Cup 2022 Final.mp4"

penalties = [
    ("1:28", "1:36", "penalty_01"),
    ("2:18", "2:26", "penalty_02"),
    ("2:58", "3:06", "penalty_03"),
    ("3:43", "3:51", "penalty_04"),
    ("4:45", "4:51", "penalty_05"),
    ("5:33", "5:40", "penalty_06"),
    ("6:27", "6:32", "penalty_07"),
    ("7:14", "7:21", "penalty_08"),
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