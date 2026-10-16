"""Generates every figure used in the KickSense FYP report.

All numbers come from the real project artefacts:
  data/labels/penalty_labels.xlsx
  data/keypoints/*/keypoints.npy
  data/frames/*/
  data/videos/*.mp4
  data/processed/*.npy
  outputs/model/best_model_improved.keras
  outputs/model/retrain_visual_labels_log.txt
  report_assets/ablation_results.json

Output: report/figures/*.png
"""
import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = Path("report_fyp/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2E6FB7"
GREEN = "#1D9E75"
ORANGE = "#D9772B"
PURPLE = "#7A5AA8"
RED = "#C0392B"
GREY = "#8A8A8A"
LIGHT = "#EAF1F8"

DIR_ORDER = ["left", "center", "right"]
DIR_LABELS = ["Left", "Center", "Right"]


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


# ----------------------------------------------------------------------------
# Shared helpers for block diagrams
# ----------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=9, bold=False, tc="#12293F"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            linespacing=1.45)


def arrow(ax, start, end, color="#33556F", style="-|>", rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13,
        color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def blank_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_labels():
    return pd.read_excel("data/labels/penalty_labels.xlsx")


def load_detection_stats():
    """Pose detection ratio and sequence length for each clip."""
    rows = []
    kp_root = Path("data/keypoints")
    fr_root = Path("data/frames")
    for kp in sorted(kp_root.glob("penalty_*/keypoints.npy")):
        clip = kp.parent.name
        try:
            detected = len(np.load(kp))
        except Exception:
            continue
        frame_dir = fr_root / clip
        total = 0
        if frame_dir.is_dir():
            total = sum(1 for f in frame_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rows.append({
            "clip": clip,
            "detected": detected,
            "total_frames": total,
            "ratio": detected / total if total else np.nan,
        })
    return pd.DataFrame(rows)


def load_video_stats():
    import cv2
    rows = []
    for v in sorted(Path("data/videos").glob("*.mp4")):
        cap = cv2.VideoCapture(str(v))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        rows.append({"file": v.name, "w": w, "h": h, "frames": n, "fps": fps})
    return pd.DataFrame(rows)


def parse_training_log(path):
    """Extract per-epoch train/val metrics from a Keras training log."""
    raw = Path(path).read_bytes()
    encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    text = raw.decode(encoding, errors="ignore")
    rows = []
    for line in text.splitlines():
        if "val_direction_accuracy:" not in line:
            continue

        def grab(key):
            m = re.search(rf"(?<!val_){key}:\s*([0-9.eE+-]+)", line)
            return float(m.group(1)) if m else None

        def grab_val(key):
            m = re.search(rf"val_{key}:\s*([0-9.eE+-]+)", line)
            return float(m.group(1)) if m else None

        acc = grab("direction_accuracy")
        vacc = grab_val("direction_accuracy")
        dloss = grab("direction_loss")
        vdloss = grab_val("direction_loss")
        if acc is None or vacc is None:
            continue
        rows.append({
            "train_acc": acc, "val_acc": vacc,
            "train_loss": dloss, "val_loss": vdloss,
        })
    df = pd.DataFrame(rows)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def evaluate_model():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_recall_fscore_support, precision_score, recall_score,
    )

    model = tf.keras.models.load_model("report_assets/reference_model.keras")
    X = np.load("data/processed/X_test.npy")
    yd = np.load("data/processed/y_direction_test.npy")
    yp = np.load("data/processed/y_power_test.npy")
    preds = model.predict(X, verbose=0)
    raw_dir = preds["direction"] if isinstance(preds, dict) else preds[0]
    raw_pow = preds["power"] if isinstance(preds, dict) else preds[1]
    pred_dir = np.argmax(raw_dir, axis=1)
    pred_pow = np.asarray(raw_pow).flatten() * 10.0

    p, r, f, _ = precision_recall_fscore_support(
        yd, pred_dir, labels=[0, 1, 2], zero_division=0)

    return {
        "y_true": yd, "y_pred": pred_dir,
        "power_true": yp, "power_pred": pred_pow,
        "accuracy": accuracy_score(yd, pred_dir),
        "precision": precision_score(yd, pred_dir, average="weighted", zero_division=0),
        "recall": recall_score(yd, pred_dir, average="weighted", zero_division=0),
        "f1": f1_score(yd, pred_dir, average="weighted", zero_division=0),
        "cm": confusion_matrix(yd, pred_dir, labels=[0, 1, 2]),
        "per_class": (p * 100, r * 100, f * 100),
        "mae": float(np.abs(pred_pow - yp).mean()),
        "rmse": float(np.sqrt(((pred_pow - yp) ** 2).mean())),
    }


# ============================================================================
# FIGURE 1 — Concept overview
# ============================================================================
def fig_concept():
    fig, ax = blank_axes((11, 2.9))
    stages = [
        ("Penalty kick\nvideo clip", LIGHT, BLUE),
        ("Frame\nextraction", LIGHT, BLUE),
        ("MediaPipe pose\n33 landmarks", "#E6F4EE", GREEN),
        ("Normalised\n26x132 sequence", "#E6F4EE", GREEN),
        ("CNN-LSTM\nmulti-task model", "#F3EDF9", PURPLE),
        ("Direction +\nPower output", "#FBEDE2", ORANGE),
    ]
    w, h, gap = 0.145, 0.42, 0.026
    x = 0.01
    for text, fc, ec in stages:
        box(ax, x, 0.29, w, h, text, fc=fc, ec=ec, fs=9, bold=True)
        if x + w + gap < 1.0:
            arrow(ax, (x + w, 0.5), (x + w + gap, 0.5))
        x += w + gap
    ax.text(0.5, 0.10, "Skeleton motion is the only signal used for prediction; "
                       "raw pixels are never fed to the classifier.",
            ha="center", fontsize=9, style="italic", color="#44586B")
    save(fig, "fig01_concept_overview.png")


# ============================================================================
# FIGURE 2 — Research taxonomy
# ============================================================================
def fig_taxonomy():
    fig, ax = blank_axes((10.5, 5.6))
    box(ax, 0.33, 0.86, 0.34, 0.11,
        "Penalty / kick outcome prediction", fc="#DCE9F6", ec=BLUE, fs=10.5, bold=True)

    branches = [
        (0.02, "Perceptual /\nnotational studies",
         "Eye tracking and manual\npose annotation of\npenalty strategy\n(Savelsbergh et al., 2002;\nPinheiro et al., 2022)",
         "#FBEDE2", ORANGE),
        (0.26, "Pixel-based\nvideo models",
         "3D CNN over RGB volumes\n(Tran et al., 2015;\nCarreira and\nZisserman, 2017)", "#F3EDF9", PURPLE),
        (0.50, "Segmentation plus\n3D pose pipelines",
         "TAM segmentation with\nMMPose 3D keypoints\nand 3D CNN\n(Salazar and\nAlatrista-Salas, 2024)", "#EDE7F6", "#5E4B9B"),
        (0.74, "Skeleton sequence\nmodels",
         "Graph and recurrent\nmodels on keypoints\n(Yan et al., 2018;\nDuan et al., 2022).\nKickSense sits here.", "#E6F4EE", GREEN),
    ]
    for x, title, detail, fc, ec in branches:
        box(ax, x, 0.55, 0.24, 0.16, title, fc=fc, ec=ec, fs=9.5, bold=True)
        box(ax, x, 0.18, 0.24, 0.29, detail, fc="white", ec=GREY, fs=8.2)
        arrow(ax, (0.5, 0.86), (x + 0.12, 0.71), rad=0.0)
        arrow(ax, (x + 0.12, 0.55), (x + 0.12, 0.47))

    ax.text(0.5, 0.06, "KickSense follows the skeleton-sequence branch, uses a lightweight "
                       "2D pose extractor, and adds a\nsecond regression head for shot power.",
            ha="center", fontsize=9, style="italic", color="#44586B")
    save(fig, "fig02_research_taxonomy.png")


# ============================================================================
# FIGURE 3 — Use case diagram
# ============================================================================
def fig_usecase():
    fig, ax = blank_axes((10.5, 6.4))

    def actor(cx, cy, name):
        ax.plot([cx], [cy + 0.045], marker="o", ms=11, mfc="white",
                mec="#12293F", mew=1.6)
        ax.plot([cx, cx], [cy + 0.032, cy - 0.035], color="#12293F", lw=1.6)
        ax.plot([cx - 0.032, cx + 0.032], [cy + 0.012, cy + 0.012],
                color="#12293F", lw=1.6)
        ax.plot([cx, cx - 0.026], [cy - 0.035, cy - 0.085], color="#12293F", lw=1.6)
        ax.plot([cx, cx + 0.026], [cy - 0.035, cy - 0.085], color="#12293F", lw=1.6)
        ax.text(cx, cy - 0.12, name, ha="center", fontsize=9.5, fontweight="bold")

    ax.add_patch(FancyBboxPatch(
        (0.27, 0.06), 0.46, 0.88, boxstyle="round,pad=0.005,rounding_size=0.01",
        facecolor="#FAFCFE", edgecolor=BLUE, linewidth=1.5))
    ax.text(0.50, 0.955, "KickSense System", ha="center",
            fontsize=10.5, fontweight="bold", color=BLUE)

    actor(0.10, 0.66, "Analyst /\nCoach")
    actor(0.90, 0.66, "Researcher /\nDeveloper")

    use_cases = [
        (0.86, "Upload penalty clip"),
        (0.755, "View pose overlay"),
        (0.65, "Predict shot direction"),
        (0.545, "Predict shot power"),
        (0.44, "Inspect class confidence"),
        (0.335, "Label a new clip"),
        (0.23, "Retrain the model"),
        (0.125, "Review evaluation report"),
    ]
    for y, text in use_cases:
        e = plt.matplotlib.patches.Ellipse(
            (0.50, y), 0.36, 0.082, facecolor=LIGHT, edgecolor=BLUE, lw=1.2)
        ax.add_patch(e)
        ax.text(0.50, y, text, ha="center", va="center", fontsize=9)

    for y in [0.86, 0.755, 0.65, 0.545, 0.44]:
        ax.plot([0.145, 0.32], [0.62, y], color=GREY, lw=1.0)
    for y in [0.335, 0.23, 0.125, 0.44]:
        ax.plot([0.855, 0.68], [0.62, y], color=GREY, lw=1.0)

    save(fig, "fig03_usecase_diagram.png")


# ============================================================================
# FIGURE 4 — System architecture
# ============================================================================
def fig_architecture():
    fig, ax = blank_axes((10.5, 6.8))

    layers = [
        (0.80, "Presentation Layer", "#DCE9F6", BLUE,
         ["Streamlit web demo\n(app.py)", "Video upload and\npose overlay view",
          "Prediction panel\n(direction + power)"]),
        (0.575, "Service Layer", "#E6F4EE", GREEN,
         ["FastAPI wrapper\n(backend.py)", "POST /predict\nJSON response",
          "Input validation and\nerror handling"]),
        (0.35, "Processing Layer", "#F3EDF9", PURPLE,
         ["MediaPipe Pose\nkeypoint extraction", "Hip-centred normalisation\nand 26-frame resample",
          "CNN-LSTM inference\n(Keras)"]),
        (0.125, "Data Layer", "#FBEDE2", ORANGE,
         ["data/videos, frames,\nkeypoints", "penalty_labels.xlsx\n(annotations)",
          "outputs/model\nbest_model_improved.keras"]),
    ]

    for y, title, fc, ec, blocks in layers:
        ax.add_patch(FancyBboxPatch(
            (0.03, y - 0.025), 0.94, 0.185,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=fc, edgecolor=ec, linewidth=1.6, alpha=0.55))
        ax.text(0.055, y + 0.132, title, fontsize=10.5, fontweight="bold", color=ec)
        bw = 0.275
        for i, b in enumerate(blocks):
            box(ax, 0.065 + i * (bw + 0.035), y + 0.002, bw, 0.105,
                b, fc="white", ec=ec, fs=8.5)

    for y0, y1 in [(0.80, 0.735), (0.575, 0.51), (0.35, 0.285)]:
        arrow(ax, (0.5, y0 - 0.025), (0.5, y1 - 0.03), lw=1.8)
        arrow(ax, (0.44, y1 - 0.03), (0.44, y0 - 0.025), lw=1.8, color=GREY)

    save(fig, "fig04_system_architecture.png")


# ============================================================================
# FIGURE 5 — Pipeline / high level design
# ============================================================================
def fig_pipeline():
    fig, ax = blank_axes((10.5, 7.2))

    steps = [
        (0.885, "data/videos/*.mp4",
         "301 collected clips, 259 fully labelled", "#FBEDE2", ORANGE),
        (0.755, "extract_frames.py",
         "Every frame decoded to JPEG\n41,476 frames, median 161 per clip", "#DCE9F6", BLUE),
        (0.625, "pose_estimation.py",
         "MediaPipe Pose at complexity 2\nkeypoints.npy of shape (T, 33, 4)", "#E6F4EE", GREEN),
        (0.495, "preprocessing.py",
         "Quality gate, hip normalisation,\n26-frame resample, flatten to 132", "#E6F4EE", GREEN),
        (0.365, "Stratified split",
         "225 usable sequences split\n157 train / 34 val / 34 test", "#F3EDF9", PURPLE),
        (0.235, "train_model_improved.py",
         "CNN-LSTM, 5x augmentation to 785\nsequences, balanced class weights", "#F3EDF9", PURPLE),
        (0.105, "evaluate_model.py / app.py",
         "Metrics, confusion matrix,\nStreamlit and FastAPI inference", "#DCE9F6", BLUE),
    ]
    for y, title, detail, fc, ec in steps:
        box(ax, 0.06, y, 0.32, 0.095, title, fc=fc, ec=ec, fs=9.5, bold=True)
        box(ax, 0.41, y, 0.53, 0.095, detail, fc="white", ec=GREY, fs=8.6)
        if y > 0.11:
            arrow(ax, (0.22, y), (0.22, y - 0.035), lw=1.7)

    ax.text(0.5, 0.02, "Every stage writes to disk, so any step can be re-run "
                       "independently and reproduced.",
            ha="center", fontsize=9, style="italic", color="#44586B")
    save(fig, "fig05_pipeline_flow.png")


# ============================================================================
# FIGURE 6 — Model architecture
# ============================================================================
def fig_model_architecture():
    fig, ax = blank_axes((8.6, 9.4))

    layers = [
        ("Input: pose sequence", "(None, 26, 132)", "0", "#DCE9F6", BLUE),
        ("Conv1D(64, k=3, ReLU)", "(None, 26, 64)", "25,408", "#E6F4EE", GREEN),
        ("BatchNorm + Dropout(0.2)", "(None, 26, 64)", "256", "#E6F4EE", GREEN),
        ("Conv1D(128, k=3, ReLU)", "(None, 26, 128)", "24,704", "#E6F4EE", GREEN),
        ("BatchNorm + MaxPool1D(2)", "(None, 13, 128)", "512", "#E6F4EE", GREEN),
        ("LSTM(128, sequences)", "(None, 13, 128)", "131,584", "#F3EDF9", PURPLE),
        ("Dropout(0.3)", "(None, 13, 128)", "0", "#F3EDF9", PURPLE),
        ("LSTM(64)", "(None, 64)", "49,408", "#F3EDF9", PURPLE),
        ("Dense(128) + BN + Dropout", "(None, 128)", "8,832", "#FFF6DC", "#B8952E"),
        ("Dense(64) + Dropout(0.2)", "(None, 64)", "8,256", "#FFF6DC", "#B8952E"),
    ]

    y = 0.955
    h = 0.068
    gap = 0.0165
    for name, shape, params, fc, ec in layers:
        box(ax, 0.14, y - h, 0.50, h, name, fc=fc, ec=ec, fs=9)
        ax.text(0.665, y - h / 2, shape, fontsize=8.2, va="center",
                color="#33556F", family="monospace")
        ax.text(0.985, y - h / 2, params, fontsize=8.2, va="center",
                ha="right", color=GREY, family="monospace")
        arrow(ax, (0.39, y - h), (0.39, y - h - gap), lw=1.4)
        y -= h + gap

    ax.text(0.665, 0.985, "output shape", fontsize=8.4, fontweight="bold", color="#12293F")
    ax.text(0.985, 0.985, "params", fontsize=8.4, fontweight="bold",
            color="#12293F", ha="right")

    head_y = y - 0.02
    box(ax, 0.04, head_y - 0.075, 0.29, 0.075,
        "Direction head\nDense(3, softmax)", fc="#FBEDE2", ec=ORANGE, fs=9, bold=True)
    box(ax, 0.40, head_y - 0.075, 0.29, 0.075,
        "Power head\nDense(1, sigmoid)", fc="#FBEDE2", ec=ORANGE, fs=9, bold=True)
    arrow(ax, (0.39, y), (0.185, head_y), rad=0.15, lw=1.4)
    arrow(ax, (0.39, y), (0.545, head_y), rad=-0.15, lw=1.4)

    ax.text(0.185, head_y - 0.115, "Left / Center / Right", ha="center",
            fontsize=8.4, color="#44586B")
    ax.text(0.545, head_y - 0.115, "Power score 1-10", ha="center",
            fontsize=8.4, color="#44586B")
    ax.text(0.5, head_y - 0.175,
            "Total trainable parameters: 248,580   (249,220 including BatchNorm statistics)",
            ha="center", fontsize=9, fontweight="bold", color="#12293F")
    save(fig, "fig06_model_architecture.png")


# ============================================================================
# FIGURE 7 — MediaPipe landmark topology
# ============================================================================
def fig_landmarks():
    coords = {
        0: (0.00, 1.00), 1: (-0.04, 1.04), 2: (-0.07, 1.04), 3: (-0.10, 1.03),
        4: (0.04, 1.04), 5: (0.07, 1.04), 6: (0.10, 1.03),
        7: (-0.12, 0.99), 8: (0.12, 0.99), 9: (-0.04, 0.94), 10: (0.04, 0.94),
        11: (-0.20, 0.80), 12: (0.20, 0.80),
        13: (-0.28, 0.58), 14: (0.28, 0.58),
        15: (-0.32, 0.38), 16: (0.32, 0.38),
        17: (-0.36, 0.32), 18: (0.36, 0.32),
        19: (-0.30, 0.30), 20: (0.30, 0.30),
        21: (-0.33, 0.34), 22: (0.33, 0.34),
        23: (-0.14, 0.50), 24: (0.14, 0.50),
        25: (-0.16, 0.27), 26: (0.16, 0.27),
        27: (-0.17, 0.05), 28: (0.17, 0.05),
        29: (-0.20, 0.01), 30: (0.20, 0.01),
        31: (-0.10, 0.00), 32: (0.10, 0.00),
    }
    edges = [(11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
             (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
             (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
             (24, 26), (26, 28), (28, 30), (28, 32), (0, 9), (0, 10), (9, 10)]

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    for a, b in edges:
        xa, ya = coords[a]
        xb, yb = coords[b]
        ax.plot([xa, xb], [ya, yb], color=GREY, lw=1.5, zorder=1)

    key = {11, 12, 23, 24}
    legs = {25, 26, 27, 28, 29, 30, 31, 32}
    for idx, (x, y) in coords.items():
        if idx in key:
            c, s = RED, 90
        elif idx in legs:
            c, s = GREEN, 62
        else:
            c, s = BLUE, 46
        ax.scatter([x], [y], s=s, color=c, zorder=3, edgecolors="white", linewidths=0.9)
        ax.annotate(str(idx), (x, y), textcoords="offset points",
                    xytext=(6, 5), fontsize=7.2, color="#12293F")

    ax.scatter([], [], s=90, color=RED, label="Normalisation anchors (11, 12, 23, 24)")
    ax.scatter([], [], s=62, color=GREEN, label="Lower body / kicking chain (25-32)")
    ax.scatter([], [], s=46, color=BLUE, label="Head, arms and remaining joints")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8.6,
              frameon=False, ncol=1)
    ax.set_title("MediaPipe Pose: 33 landmarks used as the feature source")
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.10, 1.15)
    ax.axis("off")
    save(fig, "fig07_mediapipe_landmarks.png")


# ============================================================================
# FIGURE 8 — Preprocessing activity diagram
# ============================================================================
def fig_preprocessing():
    fig, ax = blank_axes((10.6, 5.0))
    steps = [
        ("1. Load keypoints\n(T x 33 x 4)", "#DCE9F6", BLUE),
        ("2. Quality filter\n>=10 frames, >=20% ratio", "#DCE9F6", BLUE),
        ("3. Hip-centred\nnormalisation", "#E6F4EE", GREEN),
        ("4. Resample to\n26 frames", "#E6F4EE", GREEN),
        ("5. Flatten to\n26 x 132", "#F3EDF9", PURPLE),
        ("6. Encode labels\ndirection + power", "#F3EDF9", PURPLE),
        ("7. Stratified split\n70/15/15", "#FBEDE2", ORANGE),
    ]
    w, h = 0.125, 0.30
    gap = 0.0185
    x = 0.007
    for text, fc, ec in steps:
        box(ax, x, 0.45, w, h, text, fc=fc, ec=ec, fs=8.3, bold=True)
        if x + w + gap < 1.0:
            arrow(ax, (x + w, 0.60), (x + w + gap, 0.60))
        x += w + gap

    box(ax, 0.16, 0.10, 0.30, 0.20,
        "Rejected: 34 clips\n(low pose ratio or invalid label)",
        fc="#FBE9E7", ec=RED, fs=8.6)
    arrow(ax, (0.21, 0.45), (0.28, 0.30), color=RED)
    box(ax, 0.56, 0.10, 0.34, 0.20,
        "Accepted: 225 sequences\nsaved as .npy under data/processed/",
        fc="#E8F6EE", ec=GREEN, fs=8.6)
    arrow(ax, (0.80, 0.45), (0.74, 0.30), color=GREEN)
    save(fig, "fig08_preprocessing_flow.png")


# ============================================================================
# FIGURE 9 — Inference sequence diagram
# ============================================================================
def fig_sequence():
    fig, ax = blank_axes((10.4, 6.2))
    actors = ["User", "Streamlit /\nFastAPI", "Frame\nsampler",
              "MediaPipe\nPose", "Normaliser", "CNN-LSTM\nmodel"]
    xs = np.linspace(0.09, 0.91, len(actors))
    for x, name in zip(xs, actors):
        box(ax, x - 0.068, 0.885, 0.136, 0.085, name, fc=LIGHT, ec=BLUE, fs=8.6, bold=True)
        ax.plot([x, x], [0.885, 0.10], color=GREY, ls="--", lw=1.0, zorder=0)

    msgs = [
        (0, 1, 0.815, "upload clip (.mp4)"),
        (1, 2, 0.735, "request 26 frames"),
        (2, 3, 0.655, "frame batch"),
        (3, 2, 0.585, "33 landmarks / frame"),
        (2, 4, 0.505, "detected keypoints"),
        (4, 5, 0.425, "tensor (1, 26, 132)"),
        (5, 1, 0.345, "softmax + power"),
        (1, 0, 0.265, "direction, confidence, power"),
    ]
    for a, b, y, text in msgs:
        x0, x1 = xs[a], xs[b]
        colour = "#33556F" if x1 > x0 else ORANGE
        arrow(ax, (x0, y), (x1, y), color=colour, lw=1.4)
        ax.text((x0 + x1) / 2, y + 0.022, text, ha="center", fontsize=8.2,
                color=colour)

    box(ax, 0.30, 0.10, 0.40, 0.10,
        "If fewer than 5 poses are detected the request is rejected\n"
        "with a clear error instead of an unreliable prediction",
        fc="#FBE9E7", ec=RED, fs=8.4)
    save(fig, "fig09_inference_sequence.png")


# ============================================================================
# FIGURE 10 — Direction class distribution
# ============================================================================
def fig_class_distribution(df):
    counts = df["shot_direction"].value_counts().reindex(DIR_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    bars = axes[0].bar(DIR_LABELS, counts.values,
                       color=[BLUE, ORANGE, GREEN], width=0.58, edgecolor="white")
    for b, v in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}",
                     ha="center", fontweight="bold", fontsize=10)
    axes[0].set_title("Labelled clips per shot direction")
    axes[0].set_ylabel("Number of clips")
    axes[0].set_ylim(0, max(counts.values) * 1.18)
    axes[0].axhline(len(df) / 3, color=RED, ls="--", lw=1.3,
                    label=f"Perfectly balanced ({len(df)/3:.0f})")
    axes[0].legend(fontsize=8.5, frameon=False)

    axes[1].pie(counts.values, labels=DIR_LABELS, autopct="%1.1f%%",
                colors=[BLUE, ORANGE, GREEN],
                wedgeprops={"edgecolor": "white", "linewidth": 1.6},
                textprops={"fontsize": 9.5})
    axes[1].set_title("Class share of the labelled corpus")

    fig.suptitle(f"Shot direction distribution across {len(df)} labelled penalty clips",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig10_class_distribution.png")


# ============================================================================
# FIGURE 11 — Shot power distribution
# ============================================================================
def fig_power_distribution(df):
    power = df["shot_power"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    vals = sorted(power.unique())
    counts = [(power == v).sum() for v in vals]
    bars = axes[0].bar(vals, counts, color=PURPLE, width=0.62, edgecolor="white")
    for b, c in zip(bars, counts):
        axes[0].text(b.get_x() + b.get_width() / 2, c + 1.2, str(c),
                     ha="center", fontsize=8.5)
    axes[0].set_title("Shot power label frequency")
    axes[0].set_xlabel("Shot power score (1-10)")
    axes[0].set_ylabel("Number of clips")
    axes[0].set_xticks(range(1, 11))
    axes[0].set_ylim(0, max(counts) * 1.18)

    data = [df.loc[df["shot_direction"] == d, "shot_power"].dropna().values
            for d in DIR_ORDER]
    bp = axes[1].boxplot(data, labels=DIR_LABELS, patch_artist=True, widths=0.5)
    for patch, colour in zip(bp["boxes"], [BLUE, ORANGE, GREEN]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set_color("#12293F")
        median.set_linewidth(1.6)
    axes[1].set_title("Power spread per direction class")
    axes[1].set_ylabel("Shot power score")

    fig.suptitle(f"Shot power labels (mean {power.mean():.2f}, "
                 f"std {power.std():.2f}, n={len(power)})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig11_power_distribution.png")


# ============================================================================
# FIGURE 12 — Resolution and frame rate
# ============================================================================
def fig_resolution(vdf):
    res = Counter(zip(vdf["w"], vdf["h"]))
    labels = [f"{w}x{h}" for (w, h) in res.keys()]
    values = list(res.values())

    fps_round = vdf["fps"].round(0)
    fps_counts = fps_round.value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    bars = axes[0].bar(labels, values, color=[BLUE, GREEN][:len(labels)],
                       width=0.5, edgecolor="white")
    for b, v in zip(bars, values):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 2, str(v),
                     ha="center", fontweight="bold")
    axes[0].set_title("Video resolution distribution")
    axes[0].set_ylabel("Number of video files")
    axes[0].set_ylim(0, max(values) * 1.18)

    bars = axes[1].bar([f"{int(f)}" for f in fps_counts.index], fps_counts.values,
                       color=ORANGE, width=0.5, edgecolor="white")
    for b, v in zip(bars, fps_counts.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 2, str(v),
                     ha="center", fontweight="bold")
    axes[1].set_title("Frame rate distribution")
    axes[1].set_xlabel("Frames per second")
    axes[1].set_ylabel("Number of video files")
    axes[1].set_ylim(0, max(fps_counts.values) * 1.18)

    fig.suptitle(f"Acquisition properties of the {len(vdf)} collected video files",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig12_resolution_framerate.png")


# ============================================================================
# FIGURE 13 — Frames per clip
# ============================================================================
def fig_frame_counts(det):
    total = det.loc[det["total_frames"] > 0, "total_frames"]
    shown = total[total <= 400]
    outliers = int((total > 400).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].hist(shown, bins=28, color=BLUE, edgecolor="white", alpha=0.9)
    axes[0].axvline(total.median(), color=GREEN, ls="-.", lw=1.6,
                    label=f"Median {total.median():.0f} frames")
    axes[0].axvline(26, color=PURPLE, ls=":", lw=2.0,
                    label="26 frames consumed by the model")
    axes[0].set_title(f"Extracted frames per clip (n={len(shown)} shown, "
                      f"{outliers} above 400 excluded)")
    axes[0].set_xlabel("Number of extracted frames")
    axes[0].set_ylabel("Number of clips")
    axes[0].legend(fontsize=8.6, frameon=False)

    used = np.minimum(total, 26)
    discarded = np.clip(total - 26, 0, None)
    axes[1].bar(["Frames extracted", "Frames used by\nthe model",
                 "Frames discarded by\nresampling"],
                [total.sum(), used.sum(), discarded.sum()],
                color=[BLUE, GREEN, GREY], width=0.5, edgecolor="white")
    for i, v in enumerate([total.sum(), used.sum(), discarded.sum()]):
        axes[1].text(i, v + total.sum() * 0.02, f"{int(v):,}", ha="center",
                     fontweight="bold", fontsize=9.5)
    axes[1].set_ylabel("Total frames across the corpus")
    axes[1].set_ylim(0, total.sum() * 1.15)
    axes[1].set_title("Frame budget: extracted versus actually consumed")

    fig.suptitle(f"Temporal footprint of the corpus (mean {total.mean():.0f}, "
                 f"median {total.median():.0f}, range {total.min()} to {total.max()})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig13_frame_count_distribution.png")


# ============================================================================
# FIGURE 14 — Pose detection ratio
# ============================================================================
def fig_detection_ratio(det):
    sub = det.dropna(subset=["ratio"])
    ratio = sub["ratio"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    axes[0].hist(ratio, bins=25, color=GREEN, edgecolor="white", alpha=0.9)
    axes[0].axvline(20, color=RED, ls="--", lw=1.8,
                    label="20% quality threshold")
    axes[0].set_title("Pose detection ratio per clip")
    axes[0].set_xlabel("Frames with a detected pose (%)")
    axes[0].set_ylabel("Number of clips")
    axes[0].legend(fontsize=9, frameon=False)

    kept = int((ratio >= 20).sum())
    dropped = int((ratio < 20).sum())
    bars = axes[1].bar(["Above threshold", "Below threshold"], [kept, dropped],
                       color=[GREEN, RED], width=0.5, edgecolor="white")
    for b, v in zip(bars, [kept, dropped]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 2, str(v),
                     ha="center", fontweight="bold")
    axes[1].set_title("Effect of the detection-ratio filter")
    axes[1].set_ylabel("Number of clips")
    axes[1].set_ylim(0, max(kept, dropped) * 1.2)

    fig.suptitle(f"MediaPipe detection reliability (mean ratio {ratio.mean():.1f}%)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig14_detection_ratio.png")


# ============================================================================
# FIGURE 15 — Split composition
# ============================================================================
def fig_split():
    splits = {}
    for name in ["train", "val", "test"]:
        y = np.load(f"data/processed/y_direction_{name}.npy")
        splits[name] = [int((y == k).sum()) for k in range(3)]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    x = np.arange(3)
    width = 0.26
    for i, (name, colour) in enumerate(zip(["train", "val", "test"],
                                           [BLUE, ORANGE, GREEN])):
        vals = splits[name]
        bars = axes[0].bar(x + (i - 1) * width, vals, width,
                           label=f"{name} (n={sum(vals)})",
                           color=colour, edgecolor="white")
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.8, str(v),
                         ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(DIR_LABELS)
    axes[0].set_title("Stratified class counts per split")
    axes[0].set_ylabel("Number of sequences")
    axes[0].legend(fontsize=8.5, frameon=False)

    totals = [sum(splits[s]) for s in ["train", "val", "test"]]
    axes[1].pie(totals, labels=[f"Train\n{totals[0]}", f"Validation\n{totals[1]}",
                                f"Test\n{totals[2]}"],
                autopct="%1.0f%%", colors=[BLUE, ORANGE, GREEN],
                wedgeprops={"edgecolor": "white", "linewidth": 1.6},
                textprops={"fontsize": 9.5})
    axes[1].set_title("Overall split proportions")

    fig.suptitle(f"Dataset partitioning of the {sum(totals)} usable sequences",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig15_split_distribution.png")


# ============================================================================
# FIGURE 16 — Outcome and ball placement
# ============================================================================
def fig_outcome_placement(df):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    oc = df["outcome"].value_counts()
    bars = axes[0].bar([s.capitalize() for s in oc.index], oc.values,
                       color=[GREEN, ORANGE, RED][:len(oc)], width=0.5,
                       edgecolor="white")
    for b, v in zip(bars, oc.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 3, str(v),
                     ha="center", fontweight="bold")
    axes[0].set_title("Kick outcome distribution")
    axes[0].set_ylabel("Number of clips")
    axes[0].set_ylim(0, max(oc.values) * 1.18)

    order = ["top_left", "top_right", "center", "bottom_left", "bottom_right"]
    pc = df["ball_placement"].value_counts().reindex(order).fillna(0)
    pretty = [s.replace("_", " ").title() for s in order]
    bars = axes[1].barh(pretty, pc.values, color=PURPLE, height=0.55,
                        edgecolor="white")
    for b, v in zip(bars, pc.values):
        axes[1].text(v + 1.5, b.get_y() + b.get_height() / 2, f"{int(v)}",
                     va="center", fontsize=9)
    axes[1].set_title("Ball placement label distribution")
    axes[1].set_xlabel("Number of clips")
    axes[1].set_xlim(0, max(pc.values) * 1.2)
    axes[1].invert_yaxis()

    fig.suptitle("Auxiliary annotations available in penalty_labels.xlsx",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig16_outcome_placement.png")


# ============================================================================
# FIGURE 17 — Source composition
# ============================================================================
def fig_sources(df):
    tc = df["tournament"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    bars = ax.barh(tc.index[::-1], tc.values[::-1], color=BLUE, height=0.6,
                   edgecolor="white")
    for b, v in zip(bars, tc.values[::-1]):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, str(v),
                va="center", fontsize=9)
    ax.set_title("Top ten footage sources in the KickSense corpus")
    ax.set_xlabel("Number of labelled clips")
    ax.set_xlim(0, max(tc.values) * 1.15)
    fig.tight_layout()
    save(fig, "fig17_source_distribution.png")


# ============================================================================
# FIGURE 18 — Training curves
# ============================================================================
def fig_training_curves():
    """Training history of the seeded reference model used for all test metrics."""
    hist = json.loads(Path("report_assets/reference_history.json").read_text())["history"]
    df = pd.DataFrame({
        "epoch": range(1, len(hist["loss"]) + 1),
        "train_acc": hist["direction_accuracy"],
        "val_acc": hist["val_direction_accuracy"],
        "train_loss": hist["direction_loss"],
        "val_loss": hist["val_direction_loss"],
    })

    best_idx = int(df["val_acc"].idxmax())
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val = float(df.loc[best_idx, "val_acc"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    axes[0].plot(df["epoch"], df["train_acc"] * 100, color=BLUE, lw=1.9,
                 label="Training accuracy")
    axes[0].plot(df["epoch"], df["val_acc"] * 100, color=ORANGE, lw=1.9,
                 label="Validation accuracy")
    axes[0].axhline(33.3, color=RED, ls="--", lw=1.3, label="Random baseline (33.3%)")
    axes[0].scatter([best_epoch], [best_val * 100], s=80, color=GREEN, zorder=5,
                    label=f"Best epoch {best_epoch} ({best_val*100:.1f}%)")
    axes[0].set_title("Direction accuracy per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=8.4, frameon=False, loc="lower right")

    axes[1].plot(df["epoch"], df["train_loss"], color=BLUE, lw=1.9,
                 label="Training direction loss")
    axes[1].plot(df["epoch"], df["val_loss"], color=ORANGE, lw=1.9,
                 label="Validation direction loss")
    axes[1].axvline(best_epoch, color=GREEN, ls=":", lw=1.6,
                    label=f"Checkpoint restored (epoch {best_epoch})")
    axes[1].set_title("Categorical cross-entropy per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8.4, frameon=False)

    fig.suptitle("Training history of the seeded reference CNN-LSTM used for "
                 "all reported test metrics",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig18_training_curves.png")
    return {"best_epoch": best_epoch, "best_val": best_val, "epochs": len(df)}


# ============================================================================
# FIGURE 28 — Historical best run curves
# ============================================================================
def fig_historical_curves():
    df = parse_training_log("outputs/model/retrain_visual_labels_log.txt")
    if df.empty:
        print("WARNING: no historical curve rows parsed")
        return None

    best_idx = int(df["val_acc"].idxmax())
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val = float(df.loc[best_idx, "val_acc"])

    fig, ax = plt.subplots(figsize=(9.8, 4.3))
    ax.plot(df["epoch"], df["train_acc"] * 100, color=BLUE, lw=1.9,
            label="Training accuracy")
    ax.plot(df["epoch"], df["val_acc"] * 100, color=ORANGE, lw=1.9,
            label="Validation accuracy")
    ax.axhline(33.3, color=RED, ls="--", lw=1.3, label="Random baseline (33.3%)")
    ax.scatter([best_epoch], [best_val * 100], s=90, color=GREEN, zorder=5,
               label=f"Best recorded validation accuracy: "
                     f"{best_val*100:.1f}% at epoch {best_epoch}")
    ax.fill_between(df["epoch"], df["train_acc"] * 100, df["val_acc"] * 100,
                    where=(df["train_acc"] > df["val_acc"]),
                    color=GREY, alpha=0.18, label="Generalisation gap")
    ax.set_title("Highest validation accuracy recorded across all training runs "
                 "(retrain_visual_labels_log.txt)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.4, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig28_historical_best_run.png")
    return {"best_epoch": best_epoch, "best_val": best_val, "epochs": len(df)}


# ============================================================================
# FIGURE 19 — Confusion matrix
# ============================================================================
def fig_confusion(ev):
    cm = ev["cm"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    im = axes[0].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                         fontweight="bold",
                         color="white" if cm[i, j] > cm.max() * 0.55 else "#12293F")
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(DIR_LABELS)
    axes[0].set_yticks(range(3)); axes[0].set_yticklabels(DIR_LABELS)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion matrix (absolute counts)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100
    im2 = axes[1].imshow(norm, cmap="Oranges", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{norm[i, j]:.0f}%", ha="center", va="center",
                         fontsize=13, fontweight="bold",
                         color="white" if norm[i, j] > 55 else "#12293F")
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(DIR_LABELS)
    axes[1].set_yticks(range(3)); axes[1].set_yticklabels(DIR_LABELS)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_title("Row-normalised confusion matrix (recall view)")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    fig.suptitle(f"Shot direction confusion on the held-out test set "
                 f"(n={cm.sum()}, accuracy {ev['accuracy']*100:.1f}%)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig19_confusion_matrix.png")


# ============================================================================
# FIGURE 20 — Per class metrics
# ============================================================================
def fig_per_class(ev):
    p, r, f = ev["per_class"]
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    x = np.arange(3)
    width = 0.26
    for i, (vals, name, colour) in enumerate([
            (p, "Precision", BLUE), (r, "Recall", GREEN), (f, "F1-score", ORANGE)]):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name,
                      color=colour, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.0f}",
                    ha="center", fontsize=8.4)
    ax.set_xticks(x); ax.set_xticklabels(DIR_LABELS)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Per-class precision, recall and F1 on the test set")
    ax.legend(fontsize=9, frameon=False, ncol=3)
    fig.tight_layout()
    save(fig, "fig20_per_class_metrics.png")


# ============================================================================
# FIGURE 21 — Aggregate metrics
# ============================================================================
def fig_metrics_overview(ev):
    names = ["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)", "F1-score\n(weighted)"]
    vals = [ev["accuracy"] * 100, ev["precision"] * 100,
            ev["recall"] * 100, ev["f1"] * 100]
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    bars = ax.bar(names, vals, color=[BLUE, GREEN, ORANGE, PURPLE],
                  width=0.52, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.1f}%",
                ha="center", fontweight="bold", fontsize=10)
    ax.axhline(33.3, color=RED, ls="--", lw=1.5, label="Random baseline (33.3%)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Aggregate direction-classification metrics on the test set")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "fig21_metrics_overview.png")


# ============================================================================
# FIGURE 22 — Power regression behaviour
# ============================================================================
def fig_power_regression(ev):
    yt, yp = ev["power_true"], ev["power_pred"]
    err = np.abs(yp - yt)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    axes[0].scatter(yt, yp, s=62, color=BLUE, alpha=0.65, edgecolors="white")
    lo = min(yt.min(), yp.min()) - 0.5
    hi = max(yt.max(), yp.max()) + 0.5
    axes[0].plot([lo, hi], [lo, hi], color=RED, ls="--", lw=1.5,
                 label="Perfect prediction")
    axes[0].axhline(yp.mean(), color=GREEN, ls="-.", lw=1.4,
                    label=f"Mean prediction ({yp.mean():.2f})")
    axes[0].set_xlabel("Actual power label")
    axes[0].set_ylabel("Predicted power")
    axes[0].set_title("Actual versus predicted shot power")
    axes[0].legend(fontsize=8.6, frameon=False)

    axes[1].hist(err, bins=12, color=ORANGE, edgecolor="white", alpha=0.9)
    axes[1].axvline(ev["mae"], color=RED, ls="--", lw=1.7,
                    label=f"MAE {ev['mae']:.2f}")
    axes[1].axvline(ev["rmse"], color=PURPLE, ls=":", lw=1.7,
                    label=f"RMSE {ev['rmse']:.2f}")
    axes[1].set_xlabel("Absolute error (power points)")
    axes[1].set_ylabel("Number of test clips")
    axes[1].set_title("Distribution of absolute power error")
    axes[1].legend(fontsize=8.6, frameon=False)

    fig.suptitle("Shot power regression head on the held-out test set",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig22_power_regression.png")


# ============================================================================
# FIGURE 23 — Ablation study
# ============================================================================
def fig_ablation():
    path = Path("report_assets/ablation_results.json")
    if not path.exists():
        print("WARNING: ablation_results.json missing, skipping fig23")
        return None
    results = json.loads(path.read_text())
    if not results:
        return None

    names = [r["name"].replace(" (", "\n(") for r in results]
    val = [r["best_val_direction_accuracy"] * 100 for r in results]
    test = [r["test_direction_accuracy"] * 100 for r in results]
    mae = [r["test_power_mae"] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0),
                             gridspec_kw={"height_ratios": [1.25, 1]})
    x = np.arange(len(results))
    width = 0.36
    b1 = axes[0].bar(x - width / 2, val, width, label="Best validation accuracy",
                     color=BLUE, edgecolor="white")
    b2 = axes[0].bar(x + width / 2, test, width, label="Test accuracy",
                     color=ORANGE, edgecolor="white")
    for bars, vals in [(b1, val), (b2, test)]:
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                         ha="center", fontsize=8.2)
    axes[0].axhline(33.3, color=RED, ls="--", lw=1.4, label="Random baseline (33.3%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8.2)
    axes[0].set_ylabel("Direction accuracy (%)")
    axes[0].set_ylim(0, max(max(val), max(test)) * 1.28)
    axes[0].set_title("Ablation study: direction accuracy per configuration")
    axes[0].legend(fontsize=8.6, frameon=False, ncol=3)

    bars = axes[1].bar(x, mae, width=0.5, color=PURPLE, edgecolor="white")
    for b, v in zip(bars, mae):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}",
                     ha="center", fontsize=8.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=8.2)
    axes[1].set_ylabel("Power MAE (1-10 scale)")
    axes[1].set_ylim(0, max(mae) * 1.25)
    axes[1].set_title("Ablation study: shot power mean absolute error")

    fig.tight_layout()
    save(fig, "fig23_ablation_study.png")
    return results


# ============================================================================
# FIGURE 24 — Logged run comparison
# ============================================================================
def fig_run_comparison():
    runs = [
        ("retrain_log", 44.0, 48.0, 3.73),
        ("retrain_mirrorfix", 47.1, 44.1, 3.06),
        ("retrain_with_recorded", 44.1, 44.1, 2.99),
        ("retrain_user_labels", 47.1, 32.4, 2.83),
        ("retrain_visual_labels", 55.9, 32.4, 3.25),
    ]
    names = [r[0].replace("retrain_", "").replace("_", "\n") for r in runs]
    val = [r[1] for r in runs]
    test = [r[2] for r in runs]
    mae = [r[3] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(runs))
    width = 0.36
    b1 = axes[0].bar(x - width / 2, val, width, label="Best validation accuracy",
                     color=BLUE, edgecolor="white")
    b2 = axes[0].bar(x + width / 2, test, width, label="Test accuracy",
                     color=ORANGE, edgecolor="white")
    for bars, vals in [(b1, val), (b2, test)]:
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                         ha="center", fontsize=8.2)
    axes[0].axhline(33.3, color=RED, ls="--", lw=1.4, label="Random baseline")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8.2)
    axes[0].set_ylabel("Direction accuracy (%)")
    axes[0].set_ylim(0, 72)
    axes[0].set_title("Accuracy recorded in each saved training log")
    axes[0].legend(fontsize=8.4, frameon=False)

    bars = axes[1].bar(names, mae, color=PURPLE, width=0.5, edgecolor="white")
    for b, v in zip(bars, mae):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}",
                     ha="center", fontsize=8.4)
    axes[1].set_ylabel("Power MAE")
    axes[1].set_ylim(0, max(mae) * 1.25)
    axes[1].tick_params(axis="x", labelsize=8.2)
    axes[1].set_title("Power MAE per logged run")

    fig.suptitle("Experiment history across the five recorded training runs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig24_run_comparison.png")


# ============================================================================
# FIGURE 25 — Qualitative sample grid
# ============================================================================
def fig_qualitative(det, labels):
    """Montage of real frames from clips with high and low pose detection."""
    import cv2

    merged = det.merge(labels[["video_id", "shot_direction"]],
                       left_on="clip", right_on="video_id", how="inner")
    merged = merged.dropna(subset=["ratio"])
    good = merged.sort_values("ratio", ascending=False).head(3)
    poor = merged[merged["ratio"] > 0].sort_values("ratio").head(3)
    picks = list(good.itertuples()) + list(poor.itertuples())

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.4))
    for ax, row in zip(axes.flat, picks):
        frame_dir = Path("data/frames") / row.clip
        frames = sorted(f for f in frame_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not frames:
            ax.axis("off")
            continue
        img = cv2.imread(str(frames[len(frames) // 2]))
        if img is None:
            ax.axis("off")
            continue
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        quality = "usable" if row.ratio >= 0.20 else "rejected"
        colour = GREEN if row.ratio >= 0.20 else RED
        ax.set_title(f"{row.clip}  ({row.shot_direction})\n"
                     f"detection {row.ratio*100:.0f}%  [{quality}]",
                     fontsize=8.8, color=colour)
        ax.axis("off")
    for ax in axes.flat[len(picks):]:
        ax.axis("off")

    fig.suptitle("Qualitative dataset samples: high-detection clips (top row) "
                 "versus low-detection clips (bottom row)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig25_qualitative_samples.png")


# ============================================================================
# FIGURE 26 — Pose trajectory of a single clip
# ============================================================================
def fig_pose_trajectory():
    X = np.load("data/processed/X_train.npy")
    y = np.load("data/processed/y_direction_train.npy")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), sharey=True)
    joints = {"Left ankle (27)": 27, "Right ankle (28)": 28, "Hip centre proxy (24)": 24}

    for ax, cls, colour in zip(axes, range(3), [BLUE, ORANGE, GREEN]):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            ax.axis("off")
            continue
        seq = X[idx[0]].reshape(26, 33, 4)
        for (name, j), ls in zip(joints.items(), ["-", "--", ":"]):
            ax.plot(range(26), seq[:, j, 0], ls, lw=1.8, color=colour, label=name)
        ax.set_title(f"{DIR_LABELS[cls]} example")
        ax.set_xlabel("Resampled frame index (1-26)")
        ax.axhline(0, color=GREY, lw=0.9)
        ax.legend(fontsize=7.6, frameon=False)
    axes[0].set_ylabel("Normalised horizontal position")

    fig.suptitle("Normalised horizontal joint trajectories fed to the CNN-LSTM",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig26_pose_trajectories.png")


# ============================================================================
# FIGURE 27 — Feature tensor heatmap
# ============================================================================
def fig_feature_heatmap():
    X = np.load("data/processed/X_train.npy")
    sample = X[0]
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    im = ax.imshow(sample.T, aspect="auto", cmap="coolwarm",
                   vmin=np.percentile(sample, 2), vmax=np.percentile(sample, 98))
    ax.set_xlabel("Resampled frame index (26 time steps)")
    ax.set_ylabel("Feature index (132 = 33 landmarks x 4 values)")
    ax.set_title("Model input tensor for one clip: shape (26, 132)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="Normalised value")
    fig.tight_layout()
    save(fig, "fig27_feature_tensor.png")


# ============================================================================
def main():
    labels = load_labels()
    print("labels loaded", labels.shape)

    det = load_detection_stats()
    det.to_csv("report_assets/detection_stats.csv", index=False)
    print("detection stats", det.shape)

    vdf = load_video_stats()
    vdf.to_csv("report_assets/video_stats.csv", index=False)
    print("video stats", vdf.shape)

    ev = evaluate_model()
    summary = {
        "accuracy": ev["accuracy"], "precision": ev["precision"],
        "recall": ev["recall"], "f1": ev["f1"],
        "mae": ev["mae"], "rmse": ev["rmse"],
        "confusion_matrix": ev["cm"].tolist(),
        "per_class_precision": ev["per_class"][0].tolist(),
        "per_class_recall": ev["per_class"][1].tolist(),
        "per_class_f1": ev["per_class"][2].tolist(),
        "detection_ratio_mean": float(det["ratio"].dropna().mean()),
        "detection_below_20pct": int((det["ratio"].dropna() < 0.20).sum()),
        "frames_mean": float(det.loc[det.total_frames > 0, "total_frames"].mean()),
        "frames_min": int(det.loc[det.total_frames > 0, "total_frames"].min()),
        "frames_max": int(det.loc[det.total_frames > 0, "total_frames"].max()),
    }

    fig_concept()
    fig_taxonomy()
    fig_usecase()
    fig_architecture()
    fig_pipeline()
    fig_model_architecture()
    fig_landmarks()
    fig_preprocessing()
    fig_sequence()
    fig_class_distribution(labels)
    fig_power_distribution(labels)
    fig_resolution(vdf)
    fig_frame_counts(det)
    fig_detection_ratio(det)
    fig_split()
    fig_outcome_placement(labels)
    fig_sources(labels)
    curve = fig_training_curves()
    if curve:
        summary["reference_training"] = curve
    hist_curve = fig_historical_curves()
    if hist_curve:
        summary["historical_best_run"] = hist_curve
    fig_confusion(ev)
    fig_per_class(ev)
    fig_metrics_overview(ev)
    fig_power_regression(ev)
    abl = fig_ablation()
    if abl:
        summary["ablation"] = abl
    fig_run_comparison()
    fig_qualitative(det, labels)
    fig_pose_trajectory()
    fig_feature_heatmap()

    with open("report_assets/report_numbers.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nSUMMARY")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("ablation",)}, indent=2))
    print(f"\nfigures in {OUT}: {len(list(OUT.glob('*.png')))}")


if __name__ == "__main__":
    main()
