import subprocess

video_file = "data/videos/Nigeria_vs_Morocco_AFCON2025.mp4"

penalties = [
    ("1:09", "1:16", "penalty_74"),
    ("1:58", "2:05", "penalty_75"),
    ("2:57", "3:04", "penalty_76"),
    ("3:56", "4:03", "penalty_77"),
    ("4:56", "5:03", "penalty_78"),
    ("5:59", "6:08", "penalty_79"),
    ("6:52", "6:59", "penalty_80"),
    ("7:46", "7:52", "penalty_81"),
    ("8:31", "8:38", "penalty_82"),
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