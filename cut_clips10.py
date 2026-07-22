import subprocess

video_file = "data/videos/Egypt_vs_Nigeria_AFCON2025.mp4"

penalties = [
    ("0:03", "0:11", "penalty_83"),
    ("1:03", "1:11", "penalty_84"),
    ("2:06", "2:13", "penalty_85"),
    ("2:56", "3:02", "penalty_86"),
    ("4:09", "4:16", "penalty_87"),
    ("5:08", "5:16", "penalty_88"),
    ("6:07", "6:17", "penalty_89"),
    ("7:01", "7:08", "penalty_90"),
    ("8:11", "8:18", "penalty_91"),
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