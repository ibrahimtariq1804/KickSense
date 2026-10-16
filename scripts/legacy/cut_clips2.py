import subprocess

video_file = "data/videos/Netherlands vs Argentina： Full Penalty Shootout ｜ FIFA World Cup 2022.mp4"

penalties = [
    ("0:33", "0:42", "penalty_09"),
    ("1:30", "1:38", "penalty_10"),
    ("3:05", "3:15", "penalty_11"),
    ("4:43", "4:50", "penalty_12"),
    ("5:27", "5:36", "penalty_13"),
    ("6:26", "6:35", "penalty_14"),
    ("7:09", "7:16", "penalty_15"),
    ("8:08", "8:17", "penalty_16"),
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