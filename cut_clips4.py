import subprocess

video_file = "data/videos/FULL PENALTY SHOOT-OUT ｜ England v Switzerland ｜ UEFA Euros 2024 ｜ Highlights.mp4"

penalties = [
    ("0:13", "0:21", "penalty_24"),
    ("0:42", "0:51", "penalty_25"),
    ("1:12", "1:21", "penalty_26"),
    ("1:46", "1:55", "penalty_27"),
    ("2:25", "2:36", "penalty_28"),
    ("2:57", "3:05", "penalty_29"),
    ("3:33", "3:43", "penalty_30"),
    ("4:16", "4:24", "penalty_31"),
    ("4:55", "5:05", "penalty_32"),
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