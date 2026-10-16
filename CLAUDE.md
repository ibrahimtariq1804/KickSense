# KickSense — project brief for Claude Code

## What this project is

KickSense analyzes football penalty kicks using computer vision and deep
learning. It extracts body pose keypoints from penalty clips with MediaPipe,
processes them into temporal sequences, and uses a CNN-LSTM model to predict
shot direction (Left/Center/Right) and shot power (1–10). Final year project,
Bahria University Islamabad. Team: Ashir Ijaz & Ibrahim Tariq.

## Current state — already built, working, do not casually rewrite

- 198 labeled penalty clips, pose keypoints extracted, full train/val/test split
- `pose_estimation.py` — MediaPipe Pose (model_complexity=2), extracts 33
  landmarks/frame, saves `keypoints.npy` (frames × 33 × 4)
- `preprocessing.py` — 7-step pipeline: hip-centered normalization, resample
  to exactly 26 frames, flatten to 132 features/frame (33×4), label encoding,
  stratified 70/15/15 split
- `train_model.py` / `train_model_improved.py` — CNN-LSTM, dual output heads
  (direction softmax + power regression). Improved model adds
  BatchNorm/Dropout, augmentation, class weighting. Saved as
  `outputs/model/best_model_improved.keras` — **this is the model to use**,
  input shape (26, 132)
- `evaluate_model.py`, `ball_tracker.py`, `eda_analysis.py`,
  `data_visualization.py` — evaluation and analysis scripts
- `app.py` — existing Streamlit demo: upload video → 26 evenly-spaced frames
  → MediaPipe → CNN-LSTM inference → direction + power with confidence
- `requirements.txt` and `README.md` already exist in the repo

## Known limitations (from the progress report — don't be surprised by these)

- Direction accuracy is ~50% (above the 33% random baseline, not great)
- Shot power labels (from Hough-circle ball tracking) are noisy/approximate
- Ball placement head was prepped in preprocessing but never wired into the model
- Pose estimation uses `static_image_mode` per-frame, not video-optimized tracking
- Pose detection is unreliable on many clips: checking detection ratio (frames
  with a detected pose / total frames) across `data/keypoints/` shows some
  clips near 100%, but others as low as 3% (e.g. penalty_10: 6/200 frames).
  Low-ratio clips contribute mostly padded/repeated frames to the training
  sequence, which likely explains part of the ~50% direction accuracy —
  worth investigating as a separate fix (e.g. video-mode tracking, frame
  interpolation, or filtering out low-ratio clips from training)
- Class imbalance: Center (39 clips) vs Right (88) / Left (71)
- Many `cut_clips*.py` / `update_labels*.py` are versioned iterations, not a
  unified pipeline — leave them alone unless asked to consolidate them
- No automated tests, no CI/CD

## Where we're headed: web + Android app

Decided architecture (don't re-litigate this without discussion):

1. **A small Python backend (FastAPI)** wraps the existing pipeline behind
   one endpoint, e.g. `POST /predict` — accepts a video file, internally runs
   `pose_estimation.py`'s extraction logic + `preprocessing.py`'s exact
   normalization + `best_model_improved.keras` inference, returns JSON:
   `{direction, direction_confidence, power}`. **The ML code itself should
   not change** — the backend is a thin wrapper, not a rewrite.
2. **One Flutter app**, built once, deployed to both **web** and **Android**
   — a video capture/upload screen and a results screen, calling the backend
   above. This is why Flutter: one codebase, two targets.
3. **Fully offline/on-device inference is a later, separate phase** — it is
   NOT shared code between web and mobile (mobile needs TFLite + native
   MediaPipe; web needs TensorFlow.js + MediaPipe's JS pose landmarker).
   Don't attempt this until phases 1–2 are solid.

## Suggested build order

1. FastAPI backend with `/predict` — get it working locally against a sample
   clip from `data/videos/` or a clip already in `outputs/`, verify the JSON
   output matches what `app.py` currently shows in Streamlit for that clip
2. Basic error handling (bad file type, corrupt video, no pose detected)
3. Flutter app scaffold: capture/upload screen → call backend → results screen
4. Get it running on `flutter run -d chrome` (web) and an Android
   emulator/device
5. Polish UI, pose overlay visualization, loading states
6. (Stretch, later) on-device offline mode

## Working conventions

- Python env is `kicksense_env` (venv). Activate it before running anything.
- Preprocessing must match training exactly — same hip-centering, same
  26-frame resampling, same 132-feature flattening — or the model's
  predictions will silently degrade. If touching preprocessing, cross-check
  against `preprocessing.py` line by line.
- Ask before modifying or deleting anything under `data/` or `outputs/`.
- Don't touch `train_model.py`, `train_model_improved.py`, or the versioned
  `cut_clips*.py` / `update_labels*.py` scripts unless specifically asked.
- Use plan mode for anything touching more than one file.
