import subprocess

video_file = "data/videos/Manchester_United_vs_Chelsea_UCL2008.mp4"

penalties = [
    ("0:16", "0:22", "penalty_113"),
    ("0:27", "0:34", "penalty_114"),
    ("0:39", "0:47", "penalty_115"),
    ("1:04", "1:12", "penalty_116"),
    ("1:30", "1:35", "penalty_117"),
    ("1:44", "1:50", "penalty_118"),
    ("1:54", "2:00", "penalty_119"),
    ("2:06", "2:12", "penalty_120"),
    ("2:29", "2:36", "penalty_121"),x
    ("2:57", "3:04", "penalty_122"),
    ("3:12", "3:18", "penalty_123"),
    ("3:27", "3:33", "penalty_124"),
    ("3:52", "3:58", "penalty_125"),
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