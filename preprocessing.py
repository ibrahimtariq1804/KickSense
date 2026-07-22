import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

print("=== KICKSENSE PREPROCESSING PIPELINE ===\n")

keypoints_dir = "data/keypoints"
os.makedirs("data/processed", exist_ok=True)

# ============================================================
# STEP 1: COORDINATE NORMALIZATION
# ============================================================
print("STEP 1: Normalizing keypoint coordinates...")

def normalize_keypoints(keypoints):
    """
    Normalize keypoints relative to hip center
    keypoints shape: (frames, 33, 4) -> x, y, z, visibility
    """
    normalized = keypoints.copy()

    for frame_idx in range(len(keypoints)):
        frame = keypoints[frame_idx]

        # Hip center = average of left hip (23) and right hip (24)
        left_hip = frame[23, :2]
        right_hip = frame[24, :2]
        hip_center = (left_hip + right_hip) / 2

        # Shoulder center = average of left (11) and right (12)
        left_shoulder = frame[11, :2]
        right_shoulder = frame[12, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # Torso height for scaling
        torso_height = np.linalg.norm(shoulder_center - hip_center)
        if torso_height < 1e-6:
            torso_height = 1.0

        # Normalize X and Y
        normalized[frame_idx, :, 0] = (frame[:, 0] - hip_center[0]) / torso_height
        normalized[frame_idx, :, 1] = (frame[:, 1] - hip_center[1]) / torso_height
        # Z stays as is, visibility stays as is

    return normalized

# Process all penalty keypoints
all_sequences = []
all_video_ids = []
failed = []

penalty_folders = sorted(os.listdir(keypoints_dir))

for folder in penalty_folders:
    kp_path = os.path.join(keypoints_dir, folder, "keypoints.npy")

    if not os.path.exists(kp_path):
        print(f"  Skipping {folder} - no keypoints file")
        failed.append(folder)
        continue

    keypoints = np.load(kp_path)

    if len(keypoints) < 10:
        print(f"  Skipping {folder} - too few frames ({len(keypoints)})")
        failed.append(folder)
        continue

    # Normalize
    normalized = normalize_keypoints(keypoints)
    all_sequences.append(normalized)
    all_video_ids.append(folder)

print(f"\nSuccessfully normalized: {len(all_sequences)} penalties")
print(f"Skipped: {len(failed)} penalties")
if failed:
    print(f"Skipped list: {failed}")

# ============================================================
# STEP 2: SEQUENCE STRUCTURING (26 frames per clip)
# ============================================================
print("\nSTEP 2: Structuring sequences to 26 frames...")

SEQUENCE_LENGTH = 26

def structure_sequence(keypoints, target_length=26):
    """
    Resize any sequence to exactly target_length frames
    by sampling evenly across the clip
    """
    current_length = len(keypoints)

    if current_length == target_length:
        return keypoints

    # Sample evenly spaced frames
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return keypoints[indices]

structured_sequences = []

for i, (seq, vid_id) in enumerate(zip(all_sequences, all_video_ids)):
    structured = structure_sequence(seq, SEQUENCE_LENGTH)
    structured_sequences.append(structured)

structured_sequences = np.array(structured_sequences)
print(f"Structured sequences shape: {structured_sequences.shape}")
print(f"Expected: ({len(all_video_ids)}, {SEQUENCE_LENGTH}, 33, 4)")

# ============================================================
# STEP 3: FLATTEN FEATURES
# ============================================================
print("\nSTEP 3: Flattening features for model input...")

# Shape: (samples, 26, 33*4) = (samples, 26, 132)
n_samples = structured_sequences.shape[0]
n_frames = structured_sequences.shape[1]
n_features = structured_sequences.shape[2] * structured_sequences.shape[3]

X = structured_sequences.reshape(n_samples, n_frames, n_features)
print(f"Feature matrix shape: {X.shape}")
print(f"Each sample: {n_frames} frames x {n_features} features")

# ============================================================
# STEP 4: LOAD AND ENCODE LABELS
# ============================================================
print("\nSTEP 4: Loading and encoding labels...")

df = pd.read_excel("data/labels/penalty_labels.xlsx")

# Match labels to sequences
direction_map = {'left': 0, 'center': 1, 'right': 2}
placement_map = {
    'bottom_left': 0, 'bottom_right': 1,
    'center': 2, 'top_left': 3, 'top_right': 4
}

y_direction = []
y_power = []
y_placement = []
valid_ids = []

for vid_id in all_video_ids:
    row = df[df['video_id'] == vid_id]
    if len(row) == 0:
        print(f"  Warning: {vid_id} not found in labels")
        continue
    row = row.iloc[0]
    y_direction.append(direction_map.get(row['shot_direction'], -1))
    y_power.append(row['shot_power'])
    y_placement.append(placement_map.get(row['ball_placement'], -1))
    valid_ids.append(vid_id)

y_direction = np.array(y_direction)
y_power = np.array(y_power)
y_placement = np.array(y_placement)

print(f"Labels encoded for {len(valid_ids)} penalties")
print(f"Shot direction classes: {np.unique(y_direction)}")
print(f"Ball placement classes: {np.unique(y_placement)}")
print(f"Shot power range: {y_power.min()} - {y_power.max()}")

# ============================================================
# STEP 5: OUTLIER REMOVAL
# ============================================================
print("\nSTEP 5: Removing outliers...")

# Remove sequences where labels are invalid
valid_mask = (y_direction != -1) & (y_placement != -1)
X = X[valid_mask]
y_direction = y_direction[valid_mask]
y_power = y_power[valid_mask]
y_placement = y_placement[valid_mask]

print(f"After outlier removal: {len(X)} valid sequences")

# ============================================================
# STEP 6: TRAIN / VALIDATION / TEST SPLIT
# ============================================================
print("\nSTEP 6: Splitting into train/validation/test...")

# First split: 85% train+val, 15% test
X_trainval, X_test, yd_trainval, yd_test, yp_trainval, yp_test, ypl_trainval, ypl_test = \
    train_test_split(X, y_direction, y_power, y_placement,
                     test_size=0.15, random_state=42, stratify=y_direction)

# Second split: 70% train, 15% val from trainval
X_train, X_val, yd_train, yd_val, yp_train, yp_val, ypl_train, ypl_val = \
    train_test_split(X_trainval, yd_trainval, yp_trainval, ypl_trainval,
                     test_size=0.176, random_state=42, stratify=yd_trainval)

print(f"Training set:   {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set:       {len(X_test)} samples")

# ============================================================
# STEP 7: SAVE PROCESSED DATA
# ============================================================
print("\nSTEP 7: Saving processed data...")

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_direction_train.npy", yd_train)
np.save("data/processed/y_direction_val.npy", yd_val)
np.save("data/processed/y_direction_test.npy", yd_test)
np.save("data/processed/y_power_train.npy", yp_train)
np.save("data/processed/y_power_val.npy", yp_val)
np.save("data/processed/y_power_test.npy", yp_test)
np.save("data/processed/y_placement_train.npy", ypl_train)
np.save("data/processed/y_placement_val.npy", ypl_val)
np.save("data/processed/y_placement_test.npy", ypl_test)

print("\n=== PREPROCESSING COMPLETE ===")
print(f"All files saved to data/processed/")
print(f"\nSummary:")
print(f"  Total samples:     {len(X)}")
print(f"  Training:          {len(X_train)}")
print(f"  Validation:        {len(X_val)}")
print(f"  Test:              {len(X_test)}")
print(f"  Input shape:       {X_train.shape}")
print(f"  Direction classes: 3 (left, center, right)")
print(f"  Placement classes: 5 (top_left, top_right, bottom_left, bottom_right, center)")