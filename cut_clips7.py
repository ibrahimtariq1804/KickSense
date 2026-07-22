import subprocess

video_file = "data/videos/Inter Miami vs Nashville 1-1  ｜ PENALTY (10-9) ｜ Leagues Cup FINAL 2023.mp4"

penalties = [
    ("0:01", "0:08", "penalty_48"),
    ("0:16", "0:23", "penalty_49"),
    ("0:31", "0:39", "penalty_50"),
    ("0:48", "0:56", "penalty_51"),
    ("1:06", "1:13", "penalty_52"),
    ("1:30", "1:37", "penalty_53"),
    ("1:39", "1:46", "penalty_54"),
    ("1:50", "1:57", "penalty_55"),
    ("2:06", "2:12", "penalty_56"),
    ("2:27", "2:33", "penalty_57"),
    ("2:37", "2:45", "penalty_58"),
    ("2:53", "2:59", "penalty_59"),
    ("3:09", "3:16", "penalty_60"),
    ("3:29", "3:37", "penalty_61"),
    ("4:05", "4:12", "penalty_62"),
    ("4:12", "4:18", "penalty_63"),
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