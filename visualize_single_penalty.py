import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ============================================================
# CHANGE THIS TO ANY PENALTY YOU WANT TO VISUALIZE
# ============================================================
PENALTY_ID = "penalty_05"
# ============================================================

frames_dir = f"data/frames/{PENALTY_ID}"
keypoints_path = f"data/keypoints/{PENALTY_ID}/keypoints.npy"
os.makedirs(f"outputs/single_penalty/{PENALTY_ID}", exist_ok=True)

print(f"Visualizing: {PENALTY_ID}")

# ============================================================
# PART 1: Show sample frames with skeleton overlay
# ============================================================
frames = sorted(os.listdir(frames_dir))
total_frames = len(frames)
print(f"Total frames: {total_frames}")

# Pick 8 evenly spaced frames to show
indices = np.linspace(0, total_frames - 1, 8, dtype=int)
selected_frames = [frames[i] for i in indices]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(f'Skeleton Pose Estimation — {PENALTY_ID}', 
             fontsize=16, fontweight='bold', y=1.01)

with mp_pose.Pose(static_image_mode=True,
                  model_complexity=2,
                  min_detection_confidence=0.3) as pose:

    for idx, (ax, frame_file) in enumerate(zip(axes.flatten(), selected_frames)):
        frame_path = os.path.join(frames_dir, frame_file)
        image = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            ax.set_title(f'Frame {indices[idx]}  ✓ Pose detected',
                        fontsize=9, color='green')
        else:
            ax.set_title(f'Frame {indices[idx]}  ✗ No pose',
                        fontsize=9, color='red')

        ax.imshow(image_rgb)
        ax.axis('off')

plt.tight_layout()
plt.savefig(f'outputs/single_penalty/{PENALTY_ID}/01_skeleton_frames.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_skeleton_frames.png")

# ============================================================
# PART 2: Keypoint trajectory over time
# ============================================================
if os.path.exists(keypoints_path):
    keypoints = np.load(keypoints_path)
    print(f"Keypoints shape: {keypoints.shape}")
    # keypoints shape: (frames, 33, 4) → x, y, z, visibility

    # MediaPipe landmark indices
    landmarks = {
        'Left Hip': 23, 'Right Hip': 24,
        'Left Knee': 25, 'Right Knee': 26,
        'Left Ankle': 27, 'Right Ankle': 28,
        'Left Shoulder': 11, 'Right Shoulder': 12
    }

    colors_list = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD',
                   '#BA7517', '#D4537E', '#E24B4A', '#888780']

    # X coordinate trajectory
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Joint Position Over Time — {PENALTY_ID}',
                 fontsize=14, fontweight='bold')

    for (name, idx), color in zip(landmarks.items(), colors_list):
        if idx < keypoints.shape[1]:
            x_vals = keypoints[:, idx, 0]
            y_vals = keypoints[:, idx, 1]
            axes[0].plot(x_vals, label=name, color=color, linewidth=2)
            axes[1].plot(y_vals, label=name, color=color, linewidth=2)

    axes[0].set_title('X-Coordinate (Horizontal Position)', fontweight='bold')
    axes[0].set_xlabel('Frame Number')
    axes[0].set_ylabel('Normalized X Position')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].set_title('Y-Coordinate (Vertical Position)', fontweight='bold')
    axes[1].set_xlabel('Frame Number')
    axes[1].set_ylabel('Normalized Y Position')
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'outputs/single_penalty/{PENALTY_ID}/02_joint_trajectory.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: 02_joint_trajectory.png")

    # ============================================================
    # PART 3: Keypoint visibility heatmap
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    joint_names = [
        'Nose', 'L.Eye(in)', 'L.Eye', 'L.Eye(out)', 'R.Eye(in)',
        'R.Eye', 'R.Eye(out)', 'L.Ear', 'R.Ear', 'Mouth(L)',
        'Mouth(R)', 'L.Shoulder', 'R.Shoulder', 'L.Elbow', 'R.Elbow',
        'L.Wrist', 'R.Wrist', 'L.Pinky', 'R.Pinky', 'L.Index',
        'R.Index', 'L.Thumb', 'R.Thumb', 'L.Hip', 'R.Hip',
        'L.Knee', 'R.Knee', 'L.Ankle', 'R.Ankle', 'L.Heel',
        'R.Heel', 'L.Foot', 'R.Foot'
    ]

    visibility = keypoints[:, :, 3]

    im = ax.imshow(visibility.T, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Visibility Score')
    ax.set_yticks(range(len(joint_names)))
    ax.set_yticklabels(joint_names, fontsize=8)
    ax.set_xlabel('Frame Number', fontsize=11)
    ax.set_title(f'Keypoint Visibility Heatmap — {PENALTY_ID}\n'
                 f'(Green = Visible, Red = Not Visible)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'outputs/single_penalty/{PENALTY_ID}/03_visibility_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: 03_visibility_heatmap.png")

    # ============================================================
    # PART 4: Stick figure at key moment (middle frame)
    # ============================================================
    mid_frame = len(keypoints) // 2
    kp = keypoints[mid_frame]

    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
        (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
    ]

    fig, ax = plt.subplots(figsize=(8, 10))

    for start, end in connections:
        if start < len(kp) and end < len(kp):
            if kp[start, 3] > 0.3 and kp[end, 3] > 0.3:
                ax.plot([kp[start, 0], kp[end, 0]],
                        [kp[start, 1], kp[end, 1]],
                        'b-', linewidth=3, alpha=0.8)

    for i, (x, y, z, vis) in enumerate(kp):
        if vis > 0.3:
            color = '#378ADD' if i in [23, 24, 25, 26, 27, 28] else '#1D9E75'
            ax.scatter(x, y, c=color, s=80, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('equal')
    ax.set_title(f'Stick Figure — {PENALTY_ID}\n'
                 f'Frame {mid_frame} (Blue = Leg joints)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('X Position (Normalized)', fontsize=11)
    ax.set_ylabel('Y Position (Normalized)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    blue_patch = mpatches.Patch(color='#378ADD', label='Leg joints')
    green_patch = mpatches.Patch(color='#1D9E75', label='Upper body joints')
    ax.legend(handles=[blue_patch, green_patch], fontsize=10)

    plt.tight_layout()
    plt.savefig(f'outputs/single_penalty/{PENALTY_ID}/04_stick_figure.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: 04_stick_figure.png")

    # ============================================================
    # PART 5: Frame-by-frame stats
    # ============================================================
    print(f"\n--- {PENALTY_ID} STATS ---")
    print(f"Total frames with keypoints: {len(keypoints)}")
    print(f"Avg visibility score: {keypoints[:, :, 3].mean():.2f}")
    print(f"Best detected joint: {joint_names[np.argmax(keypoints[:, :, 3].mean(axis=0))]}")
    print(f"Worst detected joint: {joint_names[np.argmin(keypoints[:, :, 3].mean(axis=0))]}")

else:
    print(f"No keypoints found for {PENALTY_ID}")

print(f"\nAll visualizations saved to outputs/single_penalty/{PENALTY_ID}/")