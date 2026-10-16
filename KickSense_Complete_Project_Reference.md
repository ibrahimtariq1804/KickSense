# KickSense — Complete Project Reference Document

**Bahria University Islamabad — Department of Computer Science**  
**Final Year Project**  
**Team:** Ashir Ijaz (01-135231-017) & Ibrahim Tariq (01-135231-037)  
**Supervisor:** Ms. Zainab Iftikhar  

---

> **Purpose of this document:** A complete technical reference capturing every decision, architecture detail, problem encountered, and improvement pathway for this project. Use this as your starting blueprint if rebuilding from scratch.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Full Pipeline Architecture](#3-full-pipeline-architecture)
   - 3.1 Frame Extraction
   - 3.2 Pose Estimation
   - 3.3 Preprocessing
   - 3.4 Model Architecture
   - 3.5 Training Config
   - 3.6 Inference / App
4. [Current Results & Performance](#4-current-results--performance)
5. [Problems Encountered](#5-problems-encountered)
6. [Dependency & Environment Issues](#6-dependency--environment-issues)
7. [What Is Built vs What Is Not](#7-what-is-built-vs-what-is-not)
8. [Deployment Architecture (Planned)](#8-deployment-architecture-planned)
9. [Accuracy Improvement Options](#9-accuracy-improvement-options)
10. [Alternative Architectures](#10-alternative-architectures)
11. [If Starting From Scratch — Recommendations](#11-if-starting-from-scratch--recommendations)
12. [File Map](#12-file-map)

---

## 1. Project Overview

KickSense analyzes football penalty kicks using computer vision and deep learning. Given a short video clip of a penalty kick, it predicts:

- **Shot direction** — Left / Center / Right (3-class classification)
- **Shot power** — Score 1–10 (regression)

The core idea: a goalkeeper reading body language before the kick can gain an advantage. KickSense automates this by tracking the kicker's skeletal pose through the motion and inferring where the ball will go.

**Task type:** Multi-output supervised learning (classification + regression)  
**Input:** Short MP4/AVI/MOV video clip (penalty kick, ~2–10 seconds)  
**Output:** `{ direction, direction_confidence, power }`

---

## 2. Dataset

### Source
- 259 labeled penalty clips scraped from publicly available match footage
- Matches include World Cup 2022 and other tournaments
- Players: Mbappe, Messi, Coman, Dybala, Tchouameni, and others

### Label Schema (`data/labels/penalty_labels.xlsx`)

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Folder name, e.g. `penalty_01` |
| `match` | str | Match name, e.g. `Argentina vs France` |
| `player` | str | Player name |
| `shot_direction` | str | `left`, `center`, `right` |
| `outcome` | str | `goal`, `saved`, `missed` |
| `dominant_foot` | str | `left`, `right` |
| `tournament` | str | Competition name |
| `ball_placement` | str | `top_left`, `top_right`, `bottom_left`, `bottom_right`, `center` |
| `shot_power` | float | 1–10 scale |
| `ball_detections` | float | Number of ball positions tracked by Hough circle detector |

### Class Distribution (Direction)

| Class | Count | Percentage |
|-------|-------|------------|
| Right | 88 | ~34% |
| Left | 71 | ~27% |
| Center | 39 | ~15% |
| Unlabeled/dropped | ~61 | ~24% |

**Class imbalance** is a significant problem. Center class is severely underrepresented (39 vs 88 for Right). Random baseline for 3-class is 33%.

### Shot Power Labeling Method
Power was not directly measured. It was approximated using **Hough Circle Transform** (OpenCV) to detect the ball in each frame and calculate ball velocity between frames. This method is:
- Noisy (Hough circles frequently misfire on crowd backgrounds)
- Unreliable for clips where the ball is off-screen or occluded
- Not validated against ground-truth speed measurement

**This is a known weakness.** Shot power labels should be treated as approximate.

### Ball Placement
Ball placement labels (goal zone: top-left, top-right, etc.) were prepared in preprocessing but **never connected to the model output**. The model only outputs direction and power. Ball placement is a dead column in the current model.

---

## 3. Full Pipeline Architecture

```
Raw Video
    │
    ▼
extract_frames.py ──── Extracts all frames as JPEGs
    │
    ▼
pose_estimation.py ──── MediaPipe Pose on each frame
    │                    Saves keypoints.npy (frames × 33 × 4)
    ▼
preprocessing.py ──── Hip-normalize → 26-frame resample
    │                  → flatten to 132 features → label encode
    │                  → stratified 70/15/15 split → save .npy
    ▼
train_model_improved.py ──── CNN-LSTM, dual heads
    │                         Saves best_model_improved.keras
    ▼
evaluate_model.py ──── Confusion matrix, F1, MAE
    │
    ▼
app.py (Streamlit) ──── Upload video → live inference → display results
```

---

### 3.1 Frame Extraction (`extract_frames.py`)

- Reads video files from `data/videos/`
- Saves all frames as JPEGs to `data/frames/<penalty_id>/`
- No frame-rate normalization at this stage — all frames extracted

**Issue:** Frame count varies wildly between clips (some clips have 50 frames, others 200+). This is handled later in preprocessing by resampling to 26 frames, but it means the model never sees "real" temporal resolution — it always sees a fixed 26-frame snapshot regardless of clip length.

---

### 3.2 Pose Estimation (`pose_estimation.py`)

**Library:** MediaPipe Pose (Google)  
**Model:** `model_complexity=2` (most accurate, slowest)  
**Mode:** `static_image_mode=True` — each frame processed independently, no tracking across frames  
**Detection threshold:** `min_detection_confidence=0.5`

**Per-frame output:** 33 landmarks × 4 values (x, y, z, visibility)  
**Saved as:** `data/keypoints/<penalty_id>/keypoints.npy` — shape `(detected_frames, 33, 4)`

**Critical detail:** Frames where MediaPipe detects no pose are **silently skipped**. They are not saved to the keypoints file. This means a clip with 200 frames but only 6 detected poses produces a 6-frame keypoints file. After preprocessing resamples to 26 frames, those 6 frames get stretched/repeated across 26 slots — producing a sequence full of duplicated/interpolated data.

**Detection ratio distribution (from audit of `data/keypoints/`):**
- Some clips: ~100% detection (close-up, clear body visible)
- Some clips: as low as 3% (e.g. `penalty_10`: 6/200 frames detected)
- Average across dataset: unknown, not computed

**App uses different settings:**
- `model_complexity=1` (faster)
- `min_detection_confidence=0.3` (more permissive)
- This inconsistency between training extraction and inference is a source of distribution shift

**MediaPipe 33 Landmarks Used:**

| Index | Landmark | Notes |
|-------|----------|-------|
| 11 | Left shoulder | Used for torso height |
| 12 | Right shoulder | Used for torso height |
| 23 | Left hip | Used as normalization origin |
| 24 | Right hip | Used as normalization origin |
| All 33 | Full body | All included in feature vector |

---

### 3.3 Preprocessing (`preprocessing.py`)

**7 steps, applied in this order:**

#### Step 1: Quality Filtering
- Drop clips with fewer than **10 detected frames** (`MIN_DETECTED_FRAMES = 10`)
- Drop clips with pose detection ratio below **20%** (`MIN_POSE_RATIO = 0.20`)
- This filters out the worst clips, but many marginal clips (e.g. 25–40% detection ratio) still pass through

#### Step 2: Hip-Centered Coordinate Normalization
For each detected frame:
```
hip_center = (landmark[23][:2] + landmark[24][:2]) / 2
shoulder_center = (landmark[11][:2] + landmark[12][:2]) / 2
torso_height = ||shoulder_center - hip_center||  (Euclidean distance)

x_normalized = (x - hip_center_x) / torso_height
y_normalized = (y - hip_center_y) / torso_height
z, visibility = kept as-is
```

**Purpose:** Makes the representation scale-invariant and position-invariant. A player standing close to the camera produces the same normalized representation as a distant player.

**Weakness:** If torso_height ≈ 0 (e.g. player is bending forward, or a bad detection), division-by-zero or near-zero is possible. Not explicitly handled.

#### Step 3: Resample to 26 Frames
```python
indices = np.round(np.linspace(0, len(detected_frames) - 1, 26)).astype(int)
sequence = detected_frames[indices]
```
This gives exactly 26 frames per clip regardless of original clip length. If a clip only had 8 good frames, the 26-frame sequence will heavily repeat frames.

**Why 26?** Chosen empirically. No ablation study was done to find the optimal sequence length.

#### Step 4: Feature Flattening
```
(26, 33, 4) → (26, 132)
```
Each frame's 33 landmarks × 4 values are flattened into a 132-element vector.

#### Step 5: Label Encoding
```
shot_direction: left=0, center=1, right=2
shot_power: float (1–10), normalized to 0–1 by dividing by 10 during training
ball_placement: bottom_left=0, bottom_right=1, center=2, top_left=3, top_right=4
```

#### Step 6: Outlier Removal
- Drop samples where `shot_direction == -1` (unmatched video_id)
- Drop samples where `shot_power is NaN`

#### Step 7: Stratified Train/Val/Test Split
```
85% train+val, 15% test  (stratified by direction, random_state=42)
Of 85%: ~82% train, ~18% val  →  effective split ≈ 70/15/15
```

**Saves 15 `.npy` files to `data/processed/`:**
- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_direction_train.npy`, `y_direction_val.npy`, `y_direction_test.npy`
- `y_power_train.npy`, `y_power_val.npy`, `y_power_test.npy`
- `y_placement_train.npy`, `y_placement_val.npy`, `y_placement_test.npy`
- `ids_train.npy`, `ids_val.npy`, `ids_test.npy`

---

### 3.4 Model Architecture (`train_model_improved.py`)

**Model type:** CNN-LSTM hybrid with dual output heads  
**Input shape:** `(26, 132)` — 26 time steps, 132 features per step  
**Framework:** TensorFlow / Keras

```
Input (26, 132)
    │
    ├─ Conv1D(64, kernel=3, relu, same)
    ├─ BatchNorm
    ├─ Dropout(0.2)
    ├─ Conv1D(128, kernel=3, relu, same)
    ├─ BatchNorm
    ├─ MaxPooling1D(pool=2)          ← sequence: 26 → 13
    ├─ Dropout(0.2)
    │
    ├─ LSTM(128, return_sequences=True)
    ├─ Dropout(0.3)
    ├─ LSTM(64, return_sequences=False)
    ├─ Dropout(0.3)
    │
    ├─ Dense(128, relu)
    ├─ BatchNorm
    ├─ Dropout(0.3)
    ├─ Dense(64, relu)
    ├─ Dropout(0.2)
    │
    ├─── Direction head: Dense(3, softmax)    → Left / Center / Right
    └─── Power head:     Dense(1, sigmoid)    → Power 0–1 (×10 for display)
```

**Parameter count:** ~500K–700K (estimated, not measured)

**Why CNN before LSTM?**  
The Conv1D layers act as a local temporal feature extractor — learning short-window patterns (e.g. "leg lift" over 3 consecutive frames) before passing abstract features to the LSTM which captures long-range temporal dependencies (e.g. full kick motion trajectory).

---

### 3.5 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr=0.0005 |
| Direction loss | Categorical cross-entropy |
| Power loss | MSE |
| Direction loss weight | 1.0 |
| Power loss weight | 0.3 |
| Direction metric | Accuracy |
| Power metric | MAE |
| Batch size | 16 |
| Max epochs | 150 |
| Early stopping | patience=20, monitor=val_direction_accuracy, restore_best_weights=True |
| LR scheduler | ReduceLROnPlateau, factor=0.5, patience=8, min_lr=0.00001 |
| Best model save | ModelCheckpoint → `outputs/model/best_model_improved.keras` |

**Data Augmentation (5× multiplier applied to training set only):**

| Version | Transformation |
|---------|----------------|
| Original | Unmodified |
| Gaussian noise | `+N(0, 0.01)` per coordinate |
| Scaling | Random scale factor in [0.95, 1.05] |
| Horizontal mirror | Negate X, swap 16 left/right landmark pairs, flip direction label (left↔right) |
| Time shift | `np.roll(sequence, shift=1 or 2, axis=0)` |

**Class imbalance handling:** `class_weight` computed via `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` and passed to `model.fit()`.

---

### 3.6 Inference / App (`app.py`)

**Stack:** Streamlit  
**Flow:**
1. User uploads video
2. App loads model once via `@st.cache_resource`
3. Extract 26 evenly-spaced frames from video
4. Run MediaPipe Pose on each frame (`model_complexity=1`, `min_detection_confidence=0.3`)
5. Apply same hip-center normalization as preprocessing
6. If fewer than 5 poses detected → show error
7. Reshape to `(1, 26, 132)` → `model.predict()` → direction softmax + power sigmoid
8. Display: direction + confidence + power level + per-class probability bars + skeleton image

---

## 4. Current Results & Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Direction accuracy | ~50% | Vs 33% random baseline |
| Direction precision (weighted) | ~50% | Reflects class imbalance |
| Direction recall (weighted) | ~50% | |
| Direction F1 (weighted) | ~50% | |
| Shot power MAE | Unknown | Not reported in accessible files |
| Training clips | 117–198 (varies by run) | Only clips passing quality filter |
| Test set size | ~15% of passing clips | Stratified |

**Interpretation:**
- 50% accuracy is about 1.5× better than random chance — the model is learning something, but not reliably
- The model likely performs well on easy cases (clear body movement, high detection ratio) but poorly on ambiguous or occluded clips
- Class imbalance means Center is almost certainly predicted poorly; Left and Right get most of the model's attention

**Confusion matrix observations (from visual inspection of `confusion_matrix.png`):**
- Model tends to predict Right most often (dominant class)
- Center is the most misclassified class
- Left and Right are sometimes confused with each other

---

## 5. Problems Encountered

### 5.1 Pose Detection Failure Rate
**Problem:** Many clips have very low pose detection ratios. Some clips as low as 3–6% of frames have a detected pose. After resampling to 26 frames, the model receives a sequence that is 94%+ repeated frames.

**Why it happens:** 
- Wide-angle broadcast footage: player is small in frame
- Crowd/advertisement boards in background confuse the model
- Partial occlusion (goal posts, other players)
- `static_image_mode=True` means no tracking continuity — each frame is independently assessed

**Impact:** Low-quality clips add noise to training and likely degrade accuracy significantly.

**What was done:** Clips below 20% detection ratio are filtered out in preprocessing.

**What was NOT done:** No investigation into which specific clips are low-quality, no manual review, no interpolation between detected frames.

---

### 5.2 Class Imbalance
**Problem:** Right (88) >> Left (71) >> Center (39). Center class is severely underrepresented.

**What was done:** `class_weight='balanced'` passed to training.

**What was NOT done:** Oversampling (SMOTE), undersampling, or specifically collecting more center kicks.

---

### 5.3 Shot Power Label Noise
**Problem:** Power labels were generated by tracking ball movement using Hough Circle Transform. This detector frequently returns false positives on crowd/jersey patterns, and misses the ball when it is moving fast (motion blur).

**Impact:** Power regression is trained on noisy ground truth, making the power output unreliable.

**Alternative approach (not implemented):** Manual power labeling, or using ball velocity from professional tracking data.

---

### 5.4 Ball Placement Not Wired to Model
**Problem:** `ball_placement` column was encoded in preprocessing and saved in `.npy` files, but never used as an output head in the model.

**Status:** The model has 2 output heads (direction + power). Ball placement would require a 3rd head. Never implemented.

---

### 5.5 Inconsistency Between Training Extraction and Inference
**Problem:** During training, pose extraction used:
- `model_complexity=2`
- `min_detection_confidence=0.5`

During inference (`app.py`), it uses:
- `model_complexity=1`
- `min_detection_confidence=0.3`

The model was trained on high-complexity, high-confidence pose data, but receives lower-quality pose estimates at inference time. This is a distribution shift that degrades predictions.

---

### 5.6 `static_image_mode=True` Throughout
**Problem:** MediaPipe Pose runs independently on each frame with no temporal tracking. The video-tracking mode (`static_image_mode=False`) would use Kalman filter-like tracking to maintain landmark continuity across frames, significantly improving detection in fast-motion video. It was never used.

---

### 5.7 Versioned Script Proliferation
**Problem:** The data labeling and clip cutting process was iterated many times without a clean pipeline. The project has:
- `cut_clips.py` through `cut_clips12.py`
- `update_labels.py` through `update_labels4.py`

These are historical iteration files. They are not a unified pipeline. The logic is spread across many files and it is unclear which one represents the final state.

---

### 5.8 No Reproducibility Infrastructure
**Problem:** No automated tests, no CI/CD, no Docker container, no locked `requirements.txt` with exact versions.

**Impact:** Environment setup is fragile. The protobuf conflict between mediapipe and TensorFlow is a direct consequence of loose version pins.

**Protobuf conflict specifically:**
- `mediapipe==0.10.21` requires `protobuf<5`
- `tensorflow>=2.16` requires `protobuf>=6.31.1`
- These are **mutually exclusive** — it is impossible to satisfy both at the same time with a standard pip install

**Resolution:** Use `mediapipe==0.10.35` (drops protobuf requirement from its dependency list) or run mediapipe and TensorFlow in separate processes.

---

### 5.9 Small Dataset
**Problem:** 198 passing clips is very small for a 3-class temporal classification problem. Deep learning models typically require thousands of examples to generalize well. With 5× augmentation the effective size is ~990 training samples, which is still small.

---

### 5.10 No Ball Tracking Integration in Model
**Problem:** `ball_tracker.py` exists but its output (ball position trajectory) was never fed into the model as an additional feature. The model only uses human pose keypoints, missing potentially discriminative ball flight information.

---

## 6. Dependency & Environment Issues

### Working Environment Setup (correct way)

```bash
python -m venv kicksense_env
kicksense_env\Scripts\activate   # Windows

# Install in this exact order to avoid protobuf conflicts:
pip install "mediapipe==0.10.35"
pip install "tensorflow==2.16.2"
pip install fastapi uvicorn python-multipart
pip install streamlit opencv-python numpy pandas scikit-learn matplotlib seaborn openpyxl python-docx
```

### Known Conflicts

| Package A | Version | Package B | Version | Conflict |
|-----------|---------|-----------|---------|----------|
| mediapipe 0.10.21 | requires protobuf<5 | tensorflow ≥2.16 | requires protobuf≥6.31 | Cannot coexist |
| mediapipe 0.10.35 | no protobuf pin | tensorflow 2.16–2.21 | requires protobuf≥6.31 | Compatible |

### Correct Pinned Requirements (for future use)

```
tensorflow==2.16.2
mediapipe==0.10.35
fastapi==0.110.0
uvicorn==0.29.0
python-multipart==0.0.9
opencv-python==4.9.0.80
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
matplotlib==3.8.0
seaborn==0.13.2
openpyxl==3.1.2
streamlit==1.32.0
```

---

## 7. What Is Built vs What Is Not

### Built and Working
- [x] Full data extraction pipeline (frames → keypoints → processed .npy)
- [x] CNN-LSTM model with dual output heads (direction + power)
- [x] Data augmentation (5 strategies)
- [x] Class imbalance weighting
- [x] Stratified 70/15/15 train/val/test split
- [x] Evaluation scripts (confusion matrix, F1, per-class metrics)
- [x] Streamlit demo app
- [x] FastAPI backend (`backend.py`) — coded but broken due to dependency issues
- [x] 259-row labeled dataset

### Partially Built
- [~] Ball placement output (preprocessed, not modeled)
- [~] Ball tracker (written, not integrated into model features)

### Not Built
- [ ] Flutter mobile/web app
- [ ] On-device inference (TFLite / TensorFlow.js)
- [ ] Automated tests
- [ ] Docker container / reproducible environment
- [ ] CI/CD pipeline
- [ ] Video-mode pose tracking (non-static)
- [ ] Shot outcome prediction (goal/saved/missed)
- [ ] Player identification
- [ ] REST API deployment (cloud)

---

## 8. Deployment Architecture (Planned)

The decided deployment architecture for the final product:

```
┌─────────────────────────────────────────────────┐
│              Flutter App                        │
│   (one codebase → Web + Android)                │
│                                                 │
│  ┌──────────────┐     ┌──────────────────────┐  │
│  │ Upload/Record│     │   Results Screen     │  │
│  │ Video Screen │────▶│  Direction + Power   │  │
│  └──────────────┘     │  + Skeleton Overlay  │  │
│                       └──────────────────────┘  │
└─────────────────────────────────────────────────┘
                         │  POST /predict  (multipart video)
                         ▼
┌─────────────────────────────────────────────────┐
│         FastAPI Backend (Python)                │
│                                                 │
│  1. Receive video → temp file                   │
│  2. MediaPipe Pose extraction (26 frames)       │
│  3. Hip-center normalization                    │
│  4. best_model_improved.keras inference         │
│  5. Return JSON:                                │
│     { direction, direction_confidence, power }  │
└─────────────────────────────────────────────────┘
```

**Why Flutter?** Single codebase compiles to both web (`flutter build web`) and Android (`flutter build apk`). No duplicate code for two platforms.

**Build Order:**
1. Fix FastAPI backend locally → verify JSON output matches Streamlit output for same clip
2. Basic error handling (bad file type, no pose detected, file too large)
3. Flutter scaffold: upload screen → POST to backend → display results
4. Test on `flutter run -d chrome` and Android emulator
5. Polish (loading states, skeleton overlay, UI)
6. (Later phase) On-device offline mode: TFLite on Android, TF.js on web

---

## 9. Accuracy Improvement Options

Listed in order of expected impact:

### 9.1 Fix Pose Extraction Quality (Highest Impact)
**Current problem:** Low detection ratio clips poison training data.

**Fix options:**
- Switch to `static_image_mode=False` (video tracking mode) — maintains landmark tracking across frames, greatly reduces missed detections
- Use `model_complexity=2` consistently in both training AND inference (currently mismatched)
- Filter out clips with detection ratio below 50% (current threshold is 20%)
- Interpolate between detected keyframes instead of repeating frames

**Expected gain:** +5–15% direction accuracy

---

### 9.2 Collect More Data
**Current problem:** 198 clips is tiny for deep learning.

**Fix:** Collect 500–1000+ penalty clips. Focus specifically on Center kicks (currently only 39) to balance classes.

**Sources:** YouTube penalty compilations, football data providers (StatsBomb, Wyscout, Opta).

**Expected gain:** +10–20% accuracy with 500 clips, +15–25% with 1000

---

### 9.3 Use Consistent Pose Settings Across Pipeline
**Fix:** Use identical MediaPipe parameters in `pose_estimation.py` and `app.py`.

**Recommended:** `model_complexity=2`, `min_detection_confidence=0.5` for both.

**Expected gain:** +2–5% (eliminates distribution shift)

---

### 9.4 Add More Discriminative Features
**Current features:** Full 33 landmarks × 4 values = 132 features/frame

**Additional features to add:**
- Joint angles (knee angle, hip angle, ankle angle) — directly encode kicking motion
- Velocity features: `keypoints[t] - keypoints[t-1]` per landmark per frame
- Ball position trajectory (from `ball_tracker.py` output)
- Dominant foot label as a static feature appended to each frame
- Distance from ball to key joints

**Expected gain:** +5–10% for joint angles, +5–8% for velocity features

---

### 9.5 Increase Sequence Length
**Current:** 26 frames (fixed)

**Try:** 32, 48, 64 frames. A longer sequence captures more of the approach run and follow-through, which contain pre-kick cues.

**Expected gain:** Unknown — needs ablation study. Could be +5% or could overfit on small data.

---

### 9.6 Better Augmentation
**Current augmentation:** Noise, scale, mirror, time shift

**Additional augmentations to try:**
- Random landmark dropout (simulate partial occlusion)
- Speed variation (stretch/compress time axis non-uniformly)
- Random crop of sequence (use 20 of 26 frames at random offset)
- Mixup augmentation on sequences
- Random rotation of torso angle

---

### 9.7 Fix Shot Power Labels
**Current:** Noisy Hough-circle-based estimates

**Fix:** Manual labeling or use a sports analytics speed measurement tool.

**Expected gain:** Improved power MAE; also potentially improves direction accuracy (if power and direction are correlated in the data, noisy power labels add training noise)

---

## 10. Alternative Architectures

### Current: CNN-LSTM
```
Conv1D → Conv1D → MaxPool → LSTM → LSTM → Dense → [Direction, Power]
```
Pros: Simple, fast to train, well-understood  
Cons: Conv1D kernels are small (3 frames), may miss global patterns; LSTM is sequential, not easily parallelizable

---

### Option A: Transformer / Attention-based
```
Positional Encoding → Multi-Head Self-Attention × N → MLP → [Direction, Power]
```
**Why better:**
- Self-attention directly models relationships between any two frames (frame 1 and frame 26 can directly attend to each other)
- Captures global temporal context without the sequential bottleneck of LSTM
- Transformers for action recognition (e.g. Video Swin Transformer, TimeSformer) have shown SOTA results on human pose classification

**Caveat:** Transformers need more data. With 198 clips even after augmentation, a large Transformer may overfit. Use a small variant (2–4 attention heads, 2–3 layers) or add strong regularization.

**Implementation:** PoseFormer (existing model designed for pose-based action recognition, readily available on GitHub) or a custom small Transformer in Keras.

---

### Option B: Graph Neural Network (GNN)
```
Skeleton as graph (nodes=joints, edges=bones) → GCN → [Direction, Power]
```
**Why relevant:**
- Human body has natural graph structure (33 joints, anatomical bone connections)
- Graph Convolutional Networks (GCN, ST-GCN) are designed specifically for skeleton-based action recognition
- ST-GCN (Spatial-Temporal Graph Convolutional Network) is the most successful approach for skeleton action recognition and would be a strong architecture for KickSense

**Key papers:**
- Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" (AAAI 2018)
- PoseFormer (ICCV 2021)
- CTRGCN (CVPR 2021)

**Implementation complexity:** High. Requires building a graph adjacency matrix for MediaPipe's 33 landmarks and using a GCN library (PyTorch Geometric, DGL).

**Recommended for next project:** ST-GCN or its variants would likely outperform the CNN-LSTM significantly.

---

### Option C: 3D CNN on Video Frames
```
Raw video frames → 3D Conv → Global Average Pool → [Direction, Power]
```
**Why it could work:**
- Uses raw pixels instead of extracted keypoints — less sensitive to pose estimation quality
- Can learn its own spatial-temporal features

**Why it's risky:**
- Much larger model (millions of parameters)
- Needs far more training data (thousands of clips)
- Computationally heavy
- Not suitable with 198 clips

---

### Option D: Two-Stream Network
```
Stream 1: Pose keypoints → CNN-LSTM
Stream 2: Optical flow → CNN
Fusion → Joint prediction
```
**Why better:**
- Optical flow captures actual pixel-level motion (ball movement, leg speed)
- Complementary to keypoint stream
- Standard approach in video action recognition (Two-Stream Inflated 3D CNN)

**Implementation:** Extract optical flow using OpenCV `cv2.calcOpticalFlowFarneback`, train as second input stream, fuse with late fusion (concatenate features before heads).

---

### Option E: Pre-trained Pose Action Recognition
Use a pre-trained model fine-tuned on penalty data:
- **PoseC3D** — trained on NTU RGB+D, can be fine-tuned on sports
- **MMAction2** (OpenMMLab) — framework with many pre-trained action recognition models

**Why:** Transfer learning from large action recognition datasets could dramatically improve performance with small data.

---

## 11. If Starting From Scratch — Recommendations

### Architecture
- **Use ST-GCN** (Spatial-Temporal GCN) instead of CNN-LSTM. It is purpose-built for skeleton action recognition.
- If you want simplicity, keep CNN-LSTM but add joint angle features and velocity features from day one.

### Data
- Start with at least 300 clips before training any model
- Balance classes intentionally — collect center kicks specifically
- Use at least 3 tournaments to avoid player bias
- Have a human manually review and correct all power labels (don't rely on Hough circles)
- Record detection ratio for every clip before including it in training; discard clips below 60%

### Pose Extraction
- Always use `static_image_mode=False` with MediaPipe for video input
- Use the same `model_complexity` and `min_detection_confidence` in both training extraction and inference
- Store the raw detection confidence per frame in the keypoints file — you'll want it later

### Preprocessing
- Add velocity features (frame-to-frame difference) as standard
- Compute and store joint angles (especially: left/right knee, hip, ankle)
- Normalize sequence length using proper temporal interpolation, not just nearest-neighbor index selection

### Training
- Use stratified k-fold cross-validation (k=5) instead of a single split — with small data, a single split gives high variance results
- Log all metrics to a file (not just to console) — you lose your eval numbers if the terminal closes
- Save a `training_config.json` alongside every model checkpoint

### Deployment
- Use Docker from day one — eliminates the protobuf/dependency hell
- Pin all dependency versions in `requirements.txt` at the start
- Use a virtual environment consistently and never install to system Python

### Project Structure
- Have one canonical labeling script, not 12 versioned iterations
- Store the final pipeline as a single `pipeline.py` or a `Makefile`

---

## 12. File Map

```
Kicksense/
├── Core ML Pipeline
│   ├── extract_frames.py          — Raw video → JPEG frames
│   ├── pose_estimation.py         — Frames → MediaPipe keypoints.npy
│   ├── preprocessing.py           — Keypoints → processed .npy splits
│   ├── train_model.py             — Original CNN-LSTM (baseline)
│   ├── train_model_improved.py    — Improved CNN-LSTM (current best)
│   └── evaluate_model.py          — Test-set evaluation + charts
│
├── Apps
│   ├── app.py                     — Streamlit demo
│   └── backend.py                 — FastAPI backend (broken, fix env first)
│
├── Utilities
│   ├── ball_tracker.py            — Hough circle ball detection
│   ├── eda_analysis.py            — Exploratory data analysis
│   ├── data_visualization.py      — Dataset charts
│   ├── create_labels.py           — Label creation helper
│   └── import_recorded_clips.py   — Import new clips workflow
│
├── Historical / Do Not Touch
│   ├── cut_clips.py → cut_clips12.py    — 12 iterations of clip cutting
│   └── update_labels.py → update_labels4.py  — 4 iterations of label updates
│
├── Data
│   ├── data/videos/               — Raw penalty clips (not in git)
│   ├── data/frames/               — Extracted JPEG frames (not in git)
│   ├── data/keypoints/            — keypoints.npy per clip (not in git)
│   ├── data/processed/            — X_train/val/test + label .npy files
│   └── data/labels/penalty_labels.xlsx  — 259-row master label file
│
├── Outputs
│   ├── outputs/model/best_model_improved.keras  — Current best model
│   ├── outputs/evaluation/                      — PNG evaluation charts
│   └── outputs/incoming_audit/                  — New clips pending review
│
├── Config
│   ├── requirements.txt           — Python dependencies (loose pins)
│   ├── CLAUDE.md                  — AI assistant context file
│   └── README.md                  — Project readme
│
└── Notebooks
    └── EDA.ipynb                  — Exploratory data analysis notebook
```

---

*Document compiled: August 2026*  
*KickSense — Bahria University Islamabad Final Year Project*  
*Ashir Ijaz & Ibrahim Tariq | Supervisor: Ms. Zainab Iftikhar*
