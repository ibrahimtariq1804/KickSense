import subprocess

video_file = "data/videos/Morocco vs Spain： Full Penalty Shootout ｜ FIFA World Cup 2022.mp4"

penalties = [
    ("0:05", "0:14", "penalty_41"),
    ("0:56", "1:06", "penalty_42"),
    ("1:49", "1:57", "penalty_43"),
    ("2:42", "2:50", "penalty_44"),
    ("3:34", "3:42", "penalty_45"),
    ("4:29", "4:37", "penalty_46"),
    ("5:35", "5:43", "penalty_47"),
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