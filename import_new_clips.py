import os
import shutil

# Source folders
source_folders = [
    "C:/Users/Ibrahim/Downloads/Clips 1",
    "C:/Users/Ibrahim/Downloads/Clips 2"
]

# Destination
dest_folder = "data/videos"

# Find the next penalty number
existing = [f for f in os.listdir(dest_folder) if f.startswith("penalty_")]
if existing:
    last_num = max([int(f.split("_")[1].split(".")[0]) for f in existing])
else:
    last_num = 0

next_num = last_num + 1
print(f"Starting from penalty_{next_num:02d}")

# Copy all valid clips
copied = []
skipped = []

for folder in source_folders:
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".mp4"):
            continue
        
        filepath = os.path.join(folder, filename)
        size = os.path.getsize(filepath)
        
        # Skip empty/corrupted files (less than 10KB)
        if size < 10000:
            skipped.append(filename)
            continue
        
        # Skip non-penalty content
        skip_keywords = ["RESUMEN", "LALIGA", "highlights", 
                        "HIGHLIGHTS", "goals", "GOALS"]
        if any(kw in filename for kw in skip_keywords):
            skipped.append(filename)
            print(f"Skipping non-penalty: {filename}")
            continue
        
        # Copy with new name
        new_name = f"penalty_{next_num:02d}.mp4"
        dest_path = os.path.join(dest_folder, new_name)
        shutil.copy2(filepath, dest_path)
        copied.append({
            "new_name": new_name,
            "original": filename,
            "size_kb": size // 1024
        })
        print(f"Copied: {new_name} ← {filename[:60]}...")
        next_num += 1

print(f"\n=== IMPORT COMPLETE ===")
print(f"Copied: {len(copied)} clips")
print(f"Skipped: {len(skipped)} clips")
print(f"\nNew clips added:")
for c in copied:
    print(f"  {c['new_name']} — {c['original'][:70]}...")