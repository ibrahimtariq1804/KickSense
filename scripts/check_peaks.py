import numpy as np

X_train = np.load('data/processed/X_train.npy')
peak_frames = []
for seq in X_train:
    r_ankle = seq[:, 112:114]
    diffs = np.linalg.norm(r_ankle[1:] - r_ankle[:-1], axis=1)
    peak_frames.append(int(np.argmax(diffs)))

counts = np.bincount(peak_frames, minlength=25)
print('Peak right ankle velocity distribution across 26 frames:')
for f, c in enumerate(counts):
    print(f'Frame {f:2d}: {"#" * (c // 2)} ({c})')
