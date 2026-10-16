# KickSense

Intelligent Penalty Kick Analysis Using Pose Estimation

**Bahria University Islamabad** — Department of Computer Science  
Final Year Project (Session 2026–2027)

**Team:** Ashir Ijaz (01-135231-017) & Ibrahim Tariq (01-135231-037)  
**Supervisor:** Ms. Zainab Iftikhar

## Overview

KickSense analyzes football penalty kicks with computer vision and deep learning. It extracts body pose keypoints with MediaPipe, processes them into temporal sequences, and uses a CNN-LSTM model to predict:

- **Shot direction** — Left / Center / Right
- **Shot power** — Scale 1–10

A Streamlit web app lets you upload a penalty clip and get predictions with pose visualization.

## Dataset (current)

| Item | Value |
|------|--------|
| Labeled clips | **259** (`data/labels/penalty_labels.xlsx`) |
| Usable after quality filter | **225** |
| Train / val / test | **157 / 34 / 34** (stratified ~70/15/15) |
| Model input | `(26, 132)` — 26 frames × 33 landmarks × 4 |
| Typical video resolution | 640×360 or 1280×720 |

## Project Structure

```
Kicksense/
├── app.py                      # Streamlit demo (upload → predict)
├── backend.py                  # Optional FastAPI /predict wrapper
├── extract_frames.py           # Videos → JPEG frames
├── pose_estimation.py          # MediaPipe pose → keypoints.npy
├── preprocessing.py            # Filter, hip-norm, 26-frame, split
├── train_model.py              # Baseline CNN-LSTM
├── train_model_improved.py     # Production CNN-LSTM (+ aug)
├── evaluate_model.py           # Test metrics & plots
├── ball_tracker.py             # Hough ball track → power labels
├── eda_analysis.py             # Exploratory charts
├── data_visualization.py       # Dataset dashboards
├── docs/
│   ├── SUPERVISOR_VIVA_CHEATSHEET.md
│   └── DEMO_CLIPS.txt
├── scripts/legacy/             # Archived cut_clips* / update_labels*
├── data/
│   ├── videos/                 # Raw penalty MP4s
│   ├── frames/                 # Extracted JPEGs
│   ├── keypoints/              # MediaPipe dumps
│   ├── labels/                 # penalty_labels.xlsx
│   └── processed/              # Train/val/test .npy arrays
└── outputs/
    ├── model/                  # best_model_improved.keras
    └── evaluation/             # Confusion matrix & metrics
```

## Setup

```bash
python -m venv kicksense_env
# Windows:
kicksense_env\Scripts\activate
pip install -r requirements.txt
```

## Run the web app

```bash
streamlit run app.py
```

Use a clear, close-up penalty clip (see `docs/DEMO_CLIPS.txt`).

## Pipeline

```bash
python extract_frames.py
python pose_estimation.py
python preprocessing.py
python train_model_improved.py
python evaluate_model.py
```

**Canonical model:** `outputs/model/best_model_improved.keras`

## Note on data

Raw videos, extracted frames, and keypoint dumps may be omitted from git (size limits). Labels and trained model artifacts are included where available. Place clips under `data/videos/` to run the full pipeline.

## License

Academic / educational use — Bahria University FYP.
