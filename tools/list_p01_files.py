import cv2
import glob

files = sorted(glob.glob("outputs/inspect_p01/*.jpg"))
print("Files in outputs/inspect_p01:")
for f in files:
    img = cv2.imread(f)
    print(f"{f}: shape={img.shape}")
