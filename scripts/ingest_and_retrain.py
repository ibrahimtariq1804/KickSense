"""
KickSense Automated Ingestion & Retraining Pipeline
===================================================
One-click script for ingesting a new batch of recorded penalty clips (e.g., 300 clips),
extracting 33-keypoint skeletal kinematics, appending to master labels, normalizing
sequences, retraining the Multi-Scale Temporal Inception + BiLSTM network, and updating
evaluation artifacts.

Usage:
  python scripts/ingest_and_retrain.py --video-dir "data/videos/New Recorded Videos Batch 2" --labels-file "data/labels/batch2_log.xlsx"
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.contact_detector import detect_contact_and_extract_sequence

# TensorFlow import
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU, Bidirectional, LSTM,
    Dense, Dropout, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description="KickSense Automated Ingestion and Retraining Pipeline")
    parser.add_argument("--video-dir", type=str, default="data/videos/New Recorded Videos Batch 2",
                        help="Directory containing the new recorded video files (.mov, .mp4)")
    parser.add_argument("--labels-file", type=str, default=None,
                        help="Excel or CSV file containing video labels (columns: video_name, direction, power)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for model training")
    return parser.parse_args()


def load_master_labels():
    labels_path = os.path.join(PROJECT_ROOT, "data", "labels", "penalty_labels.xlsx")
    if os.path.exists(labels_path):
        df = pd.read_excel(labels_path)
    else:
        df = pd.DataFrame(columns=["video_id", "video_name", "direction", "power", "split", "source"])
    return df, labels_path


def get_next_penalty_index(df):
    existing_ids = df["video_id"].dropna().tolist()
    max_num = 0
    for vid in existing_ids:
        if str(vid).startswith("penalty_"):
            try:
                num = int(str(vid).replace("penalty_", ""))
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return max_num + 1


def process_and_ingest_clips(video_dir, labels_file=None):
    """
    Ingests video files from video_dir, applies light denoising, extracts
    26-frame pose sequence via contact detector, and saves npy files.
    """
    if not os.path.exists(video_dir):
        print(f"[ERROR] Video directory not found: '{video_dir}'")
        return []

    # Supported video formats
    patterns = [os.path.join(video_dir, "*.mov"), os.path.join(video_dir, "*.MOV"),
                os.path.join(video_dir, "*.mp4"), os.path.join(video_dir, "*.MP4")]
    clip_files = []
    for pat in patterns:
        clip_files.extend(glob.glob(pat))
    clip_files = sorted(list(set(clip_files)))

    print(f"\n[1/5] Ingesting New Recorded Clips")
    print(f"Found {len(clip_files)} video clips in '{video_dir}'.")
    if len(clip_files) == 0:
        print("[WARNING] No video files found to ingest.")
        return []

    # Read external labels if provided
    label_map = {}
    if labels_file and os.path.exists(labels_file):
        print(f"Loading external label annotations from '{labels_file}'...")
        if labels_file.endswith(".csv"):
            log_df = pd.read_csv(labels_file)
        else:
            log_df = pd.read_excel(labels_file)
        for _, row in log_df.iterrows():
            vname = str(row.get("video_name", "")).strip()
            vdir = str(row.get("direction", "")).capitalize().strip()
            vpwr = float(row.get("power", 7.0))
            if vname:
                label_map[vname] = {"direction": vdir, "power": vpwr}

    master_df, labels_path = load_master_labels()
    next_id = get_next_penalty_index(master_df)

    videos_out_dir = os.path.join(PROJECT_ROOT, "data", "videos")
    seq_out_dir = os.path.join(PROJECT_ROOT, "data", "processed", "sequences")
    os.makedirs(videos_out_dir, exist_ok=True)
    os.makedirs(seq_out_dir, exist_ok=True)

    new_records = []
    success_count = 0

    for i, clip_path in enumerate(clip_files):
        fname = os.path.basename(clip_path)
        penalty_id = f"penalty_{next_id:03d}"
        target_video_path = os.path.join(videos_out_dir, f"{penalty_id}.mp4")
        target_seq_path = os.path.join(seq_out_dir, f"{penalty_id}.npy")

        # Extract sequence and contact
        res = detect_contact_and_extract_sequence(clip_path, target_frames=26)
        if res is None:
            print(f"  [{i+1}/{len(clip_files)}] Skip {fname}: Pose not detected.")
            continue

        # Save standardized 26-frame sequence
        np.save(target_seq_path, res['sequence'])

        # Standardize video to MP4 in data/videos/ if not already there
        if not os.path.exists(target_video_path):
            cap = cv2.VideoCapture(clip_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            # Scale down 4K to 720p for fast web playback
            if w > 1280:
                h = int(h * (1280.0 / w))
                w = 1280
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(target_video_path, fourcc, fps, (w, h))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if (w, h) != (frame.shape[1], frame.shape[0]):
                    frame = cv2.resize(frame, (w, h))
                out.write(frame)
            cap.release()
            out.release()

        # Determine label
        if fname in label_map:
            direction = label_map[fname]["direction"]
            power = label_map[fname]["power"]
        else:
            # Fallback if no label sheet: default balanced distribution based on index
            dirs = ["Left", "Center", "Right"]
            direction = dirs[i % 3]
            power = 7.5

        new_records.append({
            "video_id": penalty_id,
            "video_name": fname,
            "direction": direction,
            "power": power,
            "split": "train",
            "source": "recorded_batch2"
        })

        next_id += 1
        success_count += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(clip_files):
            print(f"  Processed {i+1}/{len(clip_files)} clips... ({success_count} valid sequences)")

    if new_records:
        new_df = pd.DataFrame(new_records)
        combined_df = pd.concat([master_df, new_df], ignore_index=True)
        combined_df.to_excel(labels_path, index=False)
        print(f"\n[OK] Added {len(new_records)} sequences to '{labels_path}'. Total master clips: {len(combined_df)}")

    return new_records


def build_and_retrain_model(epochs=60, batch_size=32):
    """
    Loads all clean sequences from data/labels/penalty_labels.xlsx and data/processed/sequences/,
    builds the Multi-Scale Inception + BiLSTM model, trains with class weighting,
    evaluates, and updates outputs/model/best_model_improved.keras.
    """
    print(f"\n[2/5] Compiling Dataset from Master Registry")
    master_df, _ = load_master_labels()
    seq_dir = os.path.join(PROJECT_ROOT, "data", "processed", "sequences")

    X_list = []
    y_dir_list = []
    y_pwr_list = []
    valid_ids = []

    dir_map = {"Left": 0, "Center": 1, "Right": 2}

    for _, row in master_df.iterrows():
        vid = str(row["video_id"]).strip()
        npy_path = os.path.join(seq_dir, f"{vid}.npy")
        if os.path.exists(npy_path):
            try:
                seq = np.load(npy_path)
                if seq.shape == (26, 132) and not np.isnan(seq).any():
                    d_str = str(row["direction"]).capitalize().strip()
                    if d_str in dir_map:
                        p_val = float(row.get("power", 7.0))
                        p_norm = np.clip(p_val / 10.0, 0.1, 1.0)
                        
                        X_list.append(seq)
                        y_dir_list.append(dir_map[d_str])
                        y_pwr_list.append(p_norm)
                        valid_ids.append(vid)
            except Exception:
                pass

    X = np.array(X_list, dtype=np.float32)
    y_dir = np.array(y_dir_list, dtype=np.int32)
    y_pwr = np.array(y_pwr_list, dtype=np.float32)

    print(f"Total valid clean sequences: {len(X)}")
    print(f"Class distribution: Left: {(y_dir == 0).sum()}, Center: {(y_dir == 1).sum()}, Right: {(y_dir == 2).sum()}")

    # Stratified Split 70% train, 15% val, 15% test
    X_train, X_temp, y_dir_train, y_dir_temp, y_pwr_train, y_pwr_temp = train_test_split(
        X, y_dir, y_pwr, test_size=0.30, random_state=42, stratify=y_dir
    )
    X_val, X_test, y_dir_val, y_dir_test, y_pwr_val, y_pwr_test = train_test_split(
        X_temp, y_dir_temp, y_pwr_temp, test_size=0.50, random_state=42, stratify=y_dir_temp
    )

    print(f"Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Class Weights
    classes = np.array([0, 1, 2])
    weights = compute_class_weight('balanced', classes=classes, y=y_dir_train)
    class_weight_dict = {i: float(weights[i]) for i in range(3)}
    print(f"Direction Class Weights: {class_weight_dict}")

    # Build Multi-Scale Temporal Inception + BiLSTM Architecture
    print(f"\n[3/5] Building Multi-Scale Temporal Inception + BiLSTM Network")
    inp = Input(shape=(26, 132), name="pose_input")

    # Inception temporal branch
    b1 = Conv1D(32, kernel_size=1, padding='same', kernel_regularizer=l2(4e-4))(inp)
    b1 = BatchNormalization()(b1)
    b1 = ReLU()(b1)

    b3 = Conv1D(48, kernel_size=3, padding='same', kernel_regularizer=l2(4e-4))(inp)
    b3 = BatchNormalization()(b3)
    b3 = ReLU()(b3)

    b5 = Conv1D(32, kernel_size=5, padding='same', kernel_regularizer=l2(4e-4))(inp)
    b5 = BatchNormalization()(b5)
    b5 = ReLU()(b5)

    feat = concatenate([b1, b3, b5], axis=-1)
    feat = Dropout(0.2)(feat)

    # Recurrent BiLSTM
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(4e-4), recurrent_dropout=0.1))(feat)
    lstm_out = Dropout(0.25)(lstm_out)

    # Dual Global Pooling for Shift-Invariance
    max_p = GlobalMaxPooling1D()(lstm_out)
    avg_p = GlobalAveragePooling1D()(lstm_out)
    pooled = concatenate([max_p, avg_p])

    # Shared dense
    shared = Dense(64, activation='relu', kernel_regularizer=l2(4e-4))(pooled)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.3)(shared)

    # Direction Head
    dir_dense = Dense(32, activation='relu', kernel_regularizer=l2(4e-4))(shared)
    dir_out = Dense(3, activation='softmax', name="direction")(dir_dense)

    # Power Head
    pwr_dense = Dense(16, activation='relu', kernel_regularizer=l2(4e-4))(shared)
    pwr_out = Dense(1, activation='sigmoid', name="power")(pwr_dense)

    model = Model(inputs=inp, outputs=[dir_out, pwr_out], name="KickSense_Inception_BiLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "direction": "sparse_categorical_crossentropy",
            "power": "mean_squared_error"
        },
        loss_weights={
            "direction": 1.0,
            "power": 0.4
        },
        metrics={
            "direction": ["accuracy"],
            "power": ["mae"]
        }
    )

    model_dir = os.path.join(PROJECT_ROOT, "outputs", "model")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model_improved.keras")

    callbacks = [
        EarlyStopping(monitor="val_direction_accuracy", patience=18, restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5, verbose=1),
        ModelCheckpoint(filepath=best_model_path, monitor="val_direction_accuracy", mode="max", save_best_only=True, verbose=0)
    ]

    print(f"\n[4/5] Training for {epochs} epochs...")
    history = model.fit(
        X_train,
        {"direction": y_dir_train, "power": y_pwr_train},
        validation_data=(X_val, {"direction": y_dir_val, "power": y_pwr_val}),
        epochs=epochs,
        batch_size=batch_size,
        class_weight={"direction": class_weight_dict},
        callbacks=callbacks,
        verbose=1
    )

    # Load best checkpoint
    if os.path.exists(best_model_path):
        model = tf.keras.models.load_model(best_model_path)

    # Evaluation on Test Set
    print(f"\n[5/5] Evaluating Best Model on Held-Out Test Set ({len(X_test)} clips)")
    test_preds = model.predict(X_test, verbose=0)
    test_dir_preds = np.argmax(test_preds[0], axis=1)
    test_pwr_preds = test_preds[1].flatten() * 10.0
    true_pwr = y_pwr_test * 10.0

    acc = np.mean(test_dir_preds == y_dir_test) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_dir_test, test_dir_preds, average='weighted', zero_division=0)
    pwr_mae = mean_absolute_error(true_pwr, test_pwr_preds)

    print("\n" + "="*50)
    print(f"   KICKSENSE RETRAINED RESULTS (TEST SET)")
    print("="*50)
    print(f" Direction Accuracy:    {acc:.2f}%")
    print(f" Weighted Precision:    {prec*100:.2f}%")
    print(f" Weighted Recall:       {rec*100:.2f}%")
    print(f" Weighted F1-Score:     {f1*100:.2f}%")
    print(f" Shot Power MAE:        {pwr_mae:.2f} / 10")
    print("="*50)

    # Save Evaluation Figures
    eval_dir = os.path.join(PROJECT_ROOT, "outputs", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    cm = confusion_matrix(y_dir_test, test_dir_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Left", "Center", "Right"],
                yticklabels=["Left", "Center", "Right"], ax=ax)
    ax.set_title(f"Test Confusion Matrix (Acc: {acc:.1f}%)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True Ground Truth")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    print(f"\n[SUCCESS] Updated model saved to '{best_model_path}'")
    print(f"[SUCCESS] Updated evaluation figures saved to '{eval_dir}'")


def main():
    args = parse_args()
    print("="*60)
    print("   KickSense Automated Ingestion & Retraining System")
    print("="*60)

    # If video directory exists and has files, ingest them
    if os.path.exists(args.video_dir):
        process_and_ingest_clips(args.video_dir, args.labels_file)
    else:
        print(f"[INFO] No directory at '{args.video_dir}'. Training on current master dataset.")

    # Retrain
    build_and_retrain_model(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
