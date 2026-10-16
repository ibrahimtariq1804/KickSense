import subprocess

video_file = "data/videos/Japan vs Croatia： Full Penalty Shootout ｜ FIFA World Cup 2022.mp4"

penalties = [
    ("0:09", "0:17", "penalty_33"),
    ("0:50", "0:56", "penalty_34"),
    ("1:17", "1:25", "penalty_35"),
    ("1:58", "2:05", "penalty_36"),
    ("2:38", "2:45", "penalty_37"),
    ("3:18", "3:26", "penalty_38"),
    ("3:55", "4:03", "penalty_39"),
    ("4:33", "4:42", "penalty_40"),
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