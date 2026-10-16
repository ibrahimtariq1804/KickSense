import subprocess

video_file = "data/videos/Australia_vs_France_WWC2023.mp4"

penalties = [
    ("0:01", "0:07", "penalty_92"),
    ("1:08", "1:14", "penalty_93"),
    ("1:51", "1:58", "penalty_94"),
    ("2:30", "2:36", "penalty_95"),
    ("3:11", "3:17", "penalty_96"),
    ("3:52", "3:58", "penalty_97"),
    ("4:25", "4:32", "penalty_98"),
    ("5:02", "5:08", "penalty_99"),
    ("5:31", "5:38", "penalty_100"),
    ("6:05", "6:12", "penalty_101"),
    ("7:00", "7:06", "penalty_102"),
    ("8:03", "8:10", "penalty_103"),
    ("8:33", "8:39", "penalty_104"),
    ("9:12", "9:19", "penalty_105"),
    ("9:47", "9:55", "penalty_106"),
    ("10:31", "10:38", "penalty_107"),
    ("11:03", "11:09", "penalty_108"),
    ("12:18", "12:25", "penalty_109"),
    ("13:05", "13:11", "penalty_110"),
    ("13:42", "13:48", "penalty_111"),
    ("14:16", "14:21", "penalty_112"),
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