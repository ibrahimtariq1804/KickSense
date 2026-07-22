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

## Project Structure

```
Kicksense/
├── app.py                      # Streamlit web application
├── pose_estimation.py          # MediaPipe pose extraction
├── preprocessing.py            # Normalize & split sequences
├── train_model.py              # Baseline CNN-LSTM
├── train_model_improved.py     # Improved model + augmentation
├── evaluate_model.py           # Test-set evaluation
├── ball_tracker.py             # Ball placement & power estimation
├── extract_frames.py           # Frame extraction from clips
├── eda_analysis.py             # Exploratory charts
├── data_visualization.py       # Dataset dashboards
├── data/labels/                # Penalty labels (Excel)
├── outputs/model/              # Trained models
└── outputs/evaluation/         # Metrics & plots
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

## Pipeline (optional)

```bash
python extract_frames.py
python pose_estimation.py
python preprocessing.py
python train_model_improved.py
python evaluate_model.py
```

## Note on data

Raw videos, extracted frames, and keypoint dumps are **not** included in this repository (size limits). Labels and trained model artifacts are included where available. Place your own clips under `data/videos/` to run the full pipeline.

## License

Academic / educational use — Bahria University FYP.
