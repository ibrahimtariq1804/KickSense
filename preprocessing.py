import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

print("=== KICKSENSE PREPROCESSING PIPELINE ===\n")

keypoints_dir = "data/keypoints"
frames_dir = "data/frames"
os.makedirs("data/processed", exist_ok=True)

# Drop clips where MediaPipe found a pose on too few frames.
# Low-ratio clips mostly pad/repeat junk into the 26-frame sequence.
MIN_DETECTED_FRAMES = 10
MIN_POSE_RATIO = 0.20

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

def count_frame_images(folder):
    path = os.path.join(frames_dir, folder)
    if not os.path.isdir(path):
        return 0
    return sum(
        1 for name in os.listdir(path)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )

# Process all penalty keypoints
all_sequences = []
all_video_ids = []
failed = []

penalty_folders = sorted(
    f for f in os.listdir(keypoints_dir)
    if os.path.isdir(os.path.join(keypoints_dir, f))
)

for folder in penalty_folders:
    kp_path = os.path.join(keypoints_dir, folder, "keypoints.npy")

    if not os.path.exists(kp_path):
        print(f"  Skipping {folder} - no keypoints file")
        failed.append(folder)
        continue

    keypoints = np.load(kp_path)
    n_detected = len(keypoints)
    n_total = count_frame_images(folder)
    pose_ratio = (n_detected / n_total) if n_total > 0 else 0.0

    if n_detected < MIN_DETECTED_FRAMES:
        print(f"  Skipping {folder} - too few detected frames ({n_detected})")
        failed.append(folder)
        continue

    if n_total > 0 and pose_ratio < MIN_POSE_RATIO:
        print(
            f"  Skipping {folder} - low pose ratio "
            f"({n_detected}/{n_total} = {pose_ratio*100:.1f}%)"
        )
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

# Keep X rows aligned with label rows (drop unlabeled ids)
labeled_sequences = []
for i, vid_id in enumerate(all_video_ids):
    row = df[df['video_id'] == vid_id]
    if len(row) == 0:
        print(f"  Warning: {vid_id} not found in labels")
        continue
    row = row.iloc[0]
    direction_key = str(row['shot_direction']).strip().lower()
    placement_key = (
        str(row['ball_placement']).strip().lower()
        if pd.notna(row['ball_placement']) else ''
    )
    y_direction.append(direction_map.get(direction_key, -1))
    y_power.append(row['shot_power'] if pd.notna(row['shot_power']) else np.nan)
    y_placement.append(placement_map.get(placement_key, -1))
    valid_ids.append(vid_id)
    labeled_sequences.append(X[i])

X = np.array(labeled_sequences)
y_direction = np.array(y_direction)
y_power = np.array(y_power, dtype=float)
y_placement = np.array(y_placement)

print(f"Labels encoded for {len(valid_ids)} penalties")
print(f"Shot direction classes: {np.unique(y_direction)}")
print(f"Ball placement classes: {np.unique(y_placement)}")
print(f"Shot power range: {np.nanmin(y_power)} - {np.nanmax(y_power)}")

# ============================================================
# STEP 5: OUTLIER REMOVAL
# ============================================================
print("\nSTEP 5: Removing outliers...")

# Direction + power are required for the dual-head model.
# Placement is optional (encoded for future use; not used by current model).
valid_mask = (y_direction != -1) & (~np.isnan(y_power))
X = X[valid_mask]
y_direction = y_direction[valid_mask]
y_power = y_power[valid_mask]
y_placement = y_placement[valid_mask]
valid_ids = [vid for vid, keep in zip(valid_ids, valid_mask) if keep]

print(f"After outlier removal: {len(X)} valid sequences")

# ============================================================
# STEP 6: TRAIN / VALIDATION / TEST SPLIT
# ============================================================
print("\nSTEP 6: Splitting into train/validation/test...")

ids = np.array(valid_ids)

# First split: 85% train+val, 15% test
(X_trainval, X_test, yd_trainval, yd_test, yp_trainval, yp_test,
 ypl_trainval, ypl_test, id_trainval, id_test) = train_test_split(
    X, y_direction, y_power, y_placement, ids,
    test_size=0.15, random_state=42, stratify=y_direction
)

# Second split: 70% train, 15% val from trainval
(X_train, X_val, yd_train, yd_val, yp_train, yp_val,
 ypl_train, ypl_val, id_train, id_val) = train_test_split(
    X_trainval, yd_trainval, yp_trainval, ypl_trainval, id_trainval,
    test_size=0.176, random_state=42, stratify=yd_trainval
)

print(f"Training set:   {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set:       {len(X_test)} samples")
print(f"Direction counts (all): {dict(zip(*np.unique(y_direction, return_counts=True)))}")

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
np.save("data/processed/ids_train.npy", id_train)
np.save("data/processed/ids_val.npy", id_val)
np.save("data/processed/ids_test.npy", id_test)

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
print(f"  Pose filter:       min {MIN_DETECTED_FRAMES} frames, ratio >= {MIN_POSE_RATIO:.0%}")
