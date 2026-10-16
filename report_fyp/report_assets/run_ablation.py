"""Ablation study for the KickSense report.

Trains architecture/training variants on the existing processed split and records
validation/test direction accuracy plus power MAE for each variant.
Results are written to report_assets/ablation_results.json
"""
import json
import os
import random

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv1D, Dense, Dropout, GlobalAveragePooling1D,
    Input, LSTM, MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
DATA = "data/processed"
OUT = "report_assets/ablation_results.json"

POSE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def mirror_pose_sequence(sequence):
    n = sequence.shape[0]
    frames = sequence.reshape(n, 33, 4).copy()
    frames[:, :, 0] *= -1
    for left, right in POSE_LR_PAIRS:
        swapped = frames[:, left, :].copy()
        frames[:, left, :] = frames[:, right, :]
        frames[:, right, :] = swapped
    return frames.reshape(n, 132)


def augment(X, yd, yp):
    Xa, ya, pa = [], [], []
    for seq, lbl, pwr in zip(X, yd, yp):
        Xa.append(seq); ya.append(lbl); pa.append(pwr)

        Xa.append(seq + np.random.normal(0, 0.01, seq.shape)); ya.append(lbl); pa.append(pwr)
        Xa.append(seq * np.random.uniform(0.95, 1.05)); ya.append(lbl); pa.append(pwr)

        mirrored = mirror_pose_sequence(seq)
        mlbl = 2 if lbl == 0 else (0 if lbl == 2 else lbl)
        Xa.append(mirrored); ya.append(mlbl); pa.append(pwr)

        Xa.append(np.roll(seq, np.random.randint(1, 3), axis=0)); ya.append(lbl); pa.append(pwr)
    return np.array(Xa), np.array(ya), np.array(pa)


def build(kind, regularise=True):
    inp = Input(shape=(26, 132), name="pose_input")
    x = inp

    if kind in ("cnn_lstm", "cnn_only"):
        x = Conv1D(64, 3, activation="relu", padding="same")(x)
        if regularise:
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
        x = Conv1D(128, 3, activation="relu", padding="same")(x)
        if regularise:
            x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        if regularise:
            x = Dropout(0.2)(x)

    if kind in ("cnn_lstm", "lstm_only"):
        x = LSTM(128, return_sequences=True)(x)
        if regularise:
            x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        if regularise:
            x = Dropout(0.3)(x)
    else:
        x = GlobalAveragePooling1D()(x)

    x = Dense(128, activation="relu")(x)
    if regularise:
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    if regularise:
        x = Dropout(0.2)(x)

    direction = Dense(3, activation="softmax", name="direction")(x)
    power = Dense(1, activation="sigmoid", name="power")(x)

    model = Model(inp, {"direction": direction, "power": power})
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss={"direction": "categorical_crossentropy", "power": "mse"},
        loss_weights={"direction": 1.0, "power": 0.3},
        metrics={"direction": "accuracy", "power": "mae"},
    )
    return model


def run(name, kind="cnn_lstm", use_aug=True, use_weights=True, regularise=True):
    set_seeds()

    X_tr = np.load(f"{DATA}/X_train.npy")
    X_va = np.load(f"{DATA}/X_val.npy")
    X_te = np.load(f"{DATA}/X_test.npy")
    yd_tr = np.load(f"{DATA}/y_direction_train.npy")
    yd_va = np.load(f"{DATA}/y_direction_val.npy")
    yd_te = np.load(f"{DATA}/y_direction_test.npy")
    yp_tr = np.load(f"{DATA}/y_power_train.npy")
    yp_va = np.load(f"{DATA}/y_power_val.npy")
    yp_te = np.load(f"{DATA}/y_power_test.npy")

    if use_aug:
        X_fit, yd_fit, yp_fit = augment(X_tr, yd_tr, yp_tr)
    else:
        X_fit, yd_fit, yp_fit = X_tr, yd_tr, yp_tr

    # Keras rejects class_weight for multi-output models, so balanced weights are
    # applied as per-sample weights on the direction head instead.
    sample_weight = None
    if use_weights:
        classes = np.unique(yd_fit)
        weights = compute_class_weight("balanced", classes=classes, y=yd_fit)
        lookup = {int(c): float(w) for c, w in zip(classes, weights)}
        dir_w = np.array([lookup[int(l)] for l in yd_fit], dtype="float32")
        sample_weight = {
            "direction": dir_w,
            "power": np.ones_like(dir_w),
        }

    model = build(kind, regularise)

    hist = model.fit(
        X_fit,
        {"direction": to_categorical(yd_fit, 3), "power": yp_fit / 10.0},
        validation_data=(X_va, {"direction": to_categorical(yd_va, 3), "power": yp_va / 10.0}),
        epochs=150,
        batch_size=16,
        sample_weight=sample_weight,
        callbacks=[
            EarlyStopping(monitor="val_direction_accuracy", patience=20,
                          restore_best_weights=True, mode="max"),
            ReduceLROnPlateau(monitor="val_direction_loss", factor=0.5,
                              patience=8, min_lr=1e-5, mode="min"),
        ],
        verbose=0,
    )

    val_acc = float(max(hist.history["val_direction_accuracy"]))
    preds = model.predict(X_te, verbose=0)
    pred_dir = preds["direction"] if isinstance(preds, dict) else preds[0]
    pred_pow = preds["power"] if isinstance(preds, dict) else preds[1]
    test_acc = float((np.argmax(pred_dir, axis=1) == yd_te).mean())
    test_mae = float(np.abs(np.asarray(pred_pow).flatten() * 10 - yp_te).mean())

    result = {
        "name": name,
        "variant": kind,
        "augmentation": use_aug,
        "class_weights": use_weights,
        "regularisation": regularise,
        "params": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "best_val_direction_accuracy": val_acc,
        "test_direction_accuracy": test_acc,
        "test_power_mae": test_mae,
        "train_samples": int(len(X_fit)),
    }
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    os.makedirs("report_assets", exist_ok=True)
    configs = [
        ("Full model (CNN-LSTM + Aug + CW)", dict()),
        ("Without augmentation", dict(use_aug=False)),
        ("Without class weights", dict(use_weights=False)),
        ("Without BatchNorm/Dropout", dict(regularise=False)),
        ("CNN only (no LSTM)", dict(kind="cnn_only")),
        ("LSTM only (no CNN)", dict(kind="lstm_only")),
    ]

    results = []
    for name, kwargs in configs:
        print(f"=== {name} ===", flush=True)
        results.append(run(name, **kwargs))
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=2)

    print("ABLATION COMPLETE", flush=True)
