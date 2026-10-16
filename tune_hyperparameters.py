"""
KickSense wise hyperparameter tuning.

1. Stratified 5-fold CV on combined train+val (test held out)
2. Seeded random search (~10 configs), baseline included as trial 0
3. Retrain best config on full train+val; evaluate test once

Does NOT overwrite outputs/model/best_model_improved.keras.
Writes artefacts under outputs/tuning/.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

SEED = 42
DATA = Path("data/processed")
OUT = Path("outputs/tuning")
N_FOLDS = 5
N_TRIALS = 10  # trial 0 = baseline; 9 sampled configs
SEARCH_EPOCHS = 80
SEARCH_PATIENCE = 12
FINAL_EPOCHS = 150
FINAL_PATIENCE = 20

POSE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

WIDTHS = {
    "small": {"conv1": 32, "conv2": 64, "lstm1": 64, "lstm2": 32, "dense1": 64, "dense2": 32},
    "base": {"conv1": 64, "conv2": 128, "lstm1": 128, "lstm2": 64, "dense1": 128, "dense2": 64},
    "wide": {"conv1": 96, "conv2": 192, "lstm1": 192, "lstm2": 96, "dense1": 192, "dense2": 96},
}

BASELINE = {
    "learning_rate": 5e-4,
    "batch_size": 16,
    "dropout_cnn": 0.2,
    "dropout_lstm": 0.3,
    "power_loss_weight": 0.3,
    "width_scale": "base",
}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def mirror_pose_sequence(sequence: np.ndarray) -> np.ndarray:
    n = sequence.shape[0]
    frames = sequence.reshape(n, 33, 4).copy()
    frames[:, :, 0] *= -1
    for left, right in POSE_LR_PAIRS:
        swapped = frames[:, left, :].copy()
        frames[:, left, :] = frames[:, right, :]
        frames[:, right, :] = swapped
    return frames.reshape(n, 132)


def augment(X: np.ndarray, yd: np.ndarray, yp: np.ndarray):
    Xa, ya, pa = [], [], []
    for seq, lbl, pwr in zip(X, yd, yp):
        Xa.append(seq)
        ya.append(lbl)
        pa.append(pwr)

        Xa.append(seq + np.random.normal(0, 0.01, seq.shape))
        ya.append(lbl)
        pa.append(pwr)

        Xa.append(seq * np.random.uniform(0.95, 1.05))
        ya.append(lbl)
        pa.append(pwr)

        mirrored = mirror_pose_sequence(seq)
        mlbl = 2 if lbl == 0 else (0 if lbl == 2 else lbl)
        Xa.append(mirrored)
        ya.append(mlbl)
        pa.append(pwr)

        Xa.append(np.roll(seq, np.random.randint(1, 3), axis=0))
        ya.append(lbl)
        pa.append(pwr)

    X_out = np.array(Xa, dtype=np.float32)
    y_out = np.array(ya)
    p_out = np.array(pa, dtype=np.float32)
    idx = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx], p_out[idx]


def direction_sample_weights(yd: np.ndarray) -> dict:
    classes = np.unique(yd)
    weights = compute_class_weight("balanced", classes=classes, y=yd)
    lookup = {int(c): float(w) for c, w in zip(classes, weights)}
    dir_w = np.array([lookup[int(l)] for l in yd], dtype=np.float32)
    return {"direction": dir_w, "power": np.ones_like(dir_w)}


def build_model(cfg: dict) -> Model:
    w = WIDTHS[cfg["width_scale"]]
    d_cnn = float(cfg["dropout_cnn"])
    d_lstm = float(cfg["dropout_lstm"])

    inputs = Input(shape=(26, 132), name="pose_input")
    x = Conv1D(w["conv1"], 3, activation="relu", padding="same", name="conv1")(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = Dropout(d_cnn, name="dropout1")(x)

    x = Conv1D(w["conv2"], 3, activation="relu", padding="same", name="conv2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = MaxPooling1D(2, name="pool1")(x)
    x = Dropout(d_cnn, name="dropout2")(x)

    x = LSTM(w["lstm1"], return_sequences=True, name="lstm1")(x)
    x = Dropout(d_lstm, name="dropout3")(x)
    x = LSTM(w["lstm2"], return_sequences=False, name="lstm2")(x)
    x = Dropout(d_lstm, name="dropout4")(x)

    x = Dense(w["dense1"], activation="relu", name="dense1")(x)
    x = BatchNormalization(name="bn3")(x)
    x = Dropout(d_lstm, name="dropout5")(x)
    x = Dense(w["dense2"], activation="relu", name="dense2")(x)
    x = Dropout(d_cnn, name="dropout6")(x)

    direction = Dense(3, activation="softmax", name="direction")(x)
    power = Dense(1, activation="sigmoid", name="power")(x)

    model = Model(inputs=inputs, outputs={"direction": direction, "power": power})
    model.compile(
        optimizer=Adam(learning_rate=float(cfg["learning_rate"])),
        loss={"direction": "categorical_crossentropy", "power": "mse"},
        loss_weights={"direction": 1.0, "power": float(cfg["power_loss_weight"])},
        metrics={"direction": "accuracy", "power": "mae"},
    )
    return model


def train_once(
    X_tr, yd_tr, yp_tr, X_va, yd_va, yp_va, cfg: dict, epochs: int, patience: int, verbose: int = 0
):
    X_fit, yd_fit, yp_fit = augment(X_tr, yd_tr, yp_tr)
    sample_weight = direction_sample_weights(yd_fit)
    model = build_model(cfg)
    history = model.fit(
        X_fit,
        {"direction": to_categorical(yd_fit, 3), "power": yp_fit / 10.0},
        validation_data=(
            X_va,
            {"direction": to_categorical(yd_va, 3), "power": yp_va / 10.0},
        ),
        epochs=epochs,
        batch_size=int(cfg["batch_size"]),
        sample_weight=sample_weight,
        callbacks=[
            EarlyStopping(
                monitor="val_direction_accuracy",
                patience=patience,
                restore_best_weights=True,
                mode="max",
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_direction_loss",
                factor=0.5,
                patience=max(4, patience // 2),
                min_lr=1e-5,
                mode="min",
                verbose=0,
            ),
        ],
        verbose=verbose,
    )
    best_val = float(max(history.history["val_direction_accuracy"]))
    preds = model.predict(X_va, verbose=0)
    pred_dir = preds["direction"] if isinstance(preds, dict) else preds[0]
    pred_pow = preds["power"] if isinstance(preds, dict) else preds[1]
    va_acc = float(accuracy_score(yd_va, np.argmax(pred_dir, axis=1)))
    va_mae = float(mean_absolute_error(yp_va, np.asarray(pred_pow).flatten() * 10.0))
    return model, {
        "best_val_direction_accuracy": best_val,
        "fold_val_accuracy": va_acc,
        "fold_val_power_mae": va_mae,
        "epochs_run": len(history.history["loss"]),
        "params": int(model.count_params()),
    }


def sample_configs(n_trials: int) -> list[dict]:
    """Trial 0 is the current improved-model baseline; remaining are seeded draws."""
    rng = random.Random(SEED)
    configs = [dict(BASELINE)]
    lrs = [1e-4, 3e-4, 5e-4, 1e-3]
    batches = [8, 16, 32]
    dropouts = [0.1, 0.2, 0.3, 0.4]
    power_ws = [0.1, 0.3, 0.5]
    widths = ["small", "base", "wide"]

    seen = {json.dumps(BASELINE, sort_keys=True)}
    while len(configs) < n_trials:
        cfg = {
            "learning_rate": rng.choice(lrs),
            "batch_size": rng.choice(batches),
            "dropout_cnn": rng.choice(dropouts),
            "dropout_lstm": rng.choice(dropouts),
            "power_loss_weight": rng.choice(power_ws),
            "width_scale": rng.choice(widths),
        }
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs


def load_data():
    X_train = np.load(DATA / "X_train.npy")
    X_val = np.load(DATA / "X_val.npy")
    X_test = np.load(DATA / "X_test.npy")
    yd_train = np.load(DATA / "y_direction_train.npy")
    yd_val = np.load(DATA / "y_direction_val.npy")
    yd_test = np.load(DATA / "y_direction_test.npy")
    yp_train = np.load(DATA / "y_power_train.npy").astype(np.float32)
    yp_val = np.load(DATA / "y_power_val.npy").astype(np.float32)
    yp_test = np.load(DATA / "y_power_test.npy").astype(np.float32)

    X_cv = np.concatenate([X_train, X_val], axis=0)
    yd_cv = np.concatenate([yd_train, yd_val], axis=0)
    yp_cv = np.concatenate([yp_train, yp_val], axis=0)
    return X_cv, yd_cv, yp_cv, X_test, yd_test, yp_test


def run_cv_search(X_cv, yd_cv, yp_cv) -> list[dict]:
    configs = sample_configs(N_TRIALS)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = []

    for trial_idx, cfg in enumerate(configs):
        print(f"\n=== Trial {trial_idx}/{N_TRIALS - 1}: {cfg} ===", flush=True)
        fold_accs, fold_maes, fold_details = [], [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_cv, yd_cv)):
            set_seeds(SEED + trial_idx * 100 + fold_idx)
            print(f"  Fold {fold_idx + 1}/{N_FOLDS} ...", flush=True)
            _, metrics = train_once(
                X_cv[tr_idx],
                yd_cv[tr_idx],
                yp_cv[tr_idx],
                X_cv[va_idx],
                yd_cv[va_idx],
                yp_cv[va_idx],
                cfg,
                epochs=SEARCH_EPOCHS,
                patience=SEARCH_PATIENCE,
                verbose=0,
            )
            fold_accs.append(metrics["fold_val_accuracy"])
            fold_maes.append(metrics["fold_val_power_mae"])
            fold_details.append(metrics)
            tf.keras.backend.clear_session()

        trial = {
            "trial": trial_idx,
            "is_baseline": trial_idx == 0,
            "config": cfg,
            "mean_cv_direction_accuracy": float(np.mean(fold_accs)),
            "std_cv_direction_accuracy": float(np.std(fold_accs)),
            "mean_cv_power_mae": float(np.mean(fold_maes)),
            "std_cv_power_mae": float(np.std(fold_maes)),
            "fold_accuracies": [float(x) for x in fold_accs],
            "fold_power_maes": [float(x) for x in fold_maes],
            "folds": fold_details,
        }
        results.append(trial)
        print(
            f"  -> CV acc {trial['mean_cv_direction_accuracy'] * 100:.1f}% "
            f"+/- {trial['std_cv_direction_accuracy'] * 100:.1f}% | "
            f"power MAE {trial['mean_cv_power_mae']:.2f}",
            flush=True,
        )

        with open(OUT / "cv_results.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    return results


def pick_best(results: list[dict]) -> dict:
    return max(results, key=lambda r: r["mean_cv_direction_accuracy"])


def final_retrain_and_test(best: dict, X_cv, yd_cv, yp_cv, X_test, yd_test, yp_test) -> dict:
    cfg = best["config"]
    print(f"\n=== Final retrain on full train+val with best config ===\n{cfg}", flush=True)
    set_seeds(SEED)

    # 1) Stratified 85/15 split of train+val only for early-stopping epoch selection.
    X_tr, X_va, yd_tr, yd_va, yp_tr, yp_va = train_test_split(
        X_cv,
        yd_cv,
        yp_cv,
        test_size=0.15,
        stratify=yd_cv,
        random_state=SEED,
    )
    _, es_meta = train_once(
        X_tr,
        yd_tr,
        yp_tr,
        X_va,
        yd_va,
        yp_va,
        cfg,
        epochs=FINAL_EPOCHS,
        patience=FINAL_PATIENCE,
        verbose=1,
    )
    tf.keras.backend.clear_session()

    # 2) Retrain from scratch on ALL train+val for the ES-selected epoch count.
    n_epochs = max(10, int(es_meta["epochs_run"]))
    print(f"\nRetraining on all {len(X_cv)} train+val samples for {n_epochs} epochs...", flush=True)
    set_seeds(SEED)
    X_fit, yd_fit, yp_fit = augment(X_cv, yd_cv, yp_cv)
    sample_weight = direction_sample_weights(yd_fit)
    model = build_model(cfg)
    history = model.fit(
        X_fit,
        {"direction": to_categorical(yd_fit, 3), "power": yp_fit / 10.0},
        epochs=n_epochs,
        batch_size=int(cfg["batch_size"]),
        sample_weight=sample_weight,
        verbose=1,
    )
    train_meta = {
        "es_phase": es_meta,
        "full_retrain_epochs": n_epochs,
        "full_retrain_final_direction_accuracy": float(
            history.history["direction_accuracy"][-1]
        ),
        "params": int(model.count_params()),
        "train_samples_augmented": int(len(X_fit)),
    }

    model_path = OUT / "best_model_tuned.keras"
    model.save(model_path)
    print(f"Saved: {model_path}", flush=True)

    preds = model.predict(X_test, verbose=0)
    pred_dir = preds["direction"] if isinstance(preds, dict) else preds[0]
    pred_pow = preds["power"] if isinstance(preds, dict) else preds[1]
    y_hat = np.argmax(pred_dir, axis=1)
    power_hat = np.asarray(pred_pow).flatten() * 10.0

    labels = ["Left", "Center", "Right"]
    acc = float(accuracy_score(yd_test, y_hat))
    prec = float(precision_score(yd_test, y_hat, average="weighted", zero_division=0))
    rec = float(recall_score(yd_test, y_hat, average="weighted", zero_division=0))
    f1 = float(f1_score(yd_test, y_hat, average="weighted", zero_division=0))
    mae = float(mean_absolute_error(yp_test, power_hat))
    cm = confusion_matrix(yd_test, y_hat).tolist()
    report = classification_report(
        yd_test, y_hat, target_names=labels, zero_division=0, output_dict=True
    )

    metrics = {
        "best_trial": best["trial"],
        "config": cfg,
        "cv_mean_direction_accuracy": best["mean_cv_direction_accuracy"],
        "cv_std_direction_accuracy": best["std_cv_direction_accuracy"],
        "cv_mean_power_mae": best["mean_cv_power_mae"],
        "final_train_meta": train_meta,
        "test_direction_accuracy": acc,
        "test_weighted_precision": prec,
        "test_weighted_recall": rec,
        "test_weighted_f1": f1,
        "test_power_mae": mae,
        "test_confusion_matrix": cm,
        "classification_report": report,
        "random_baseline": 1.0 / 3.0,
        "model_path": str(model_path).replace("\\", "/"),
        "note": "Demo/app still use outputs/model/best_model_improved.keras until explicitly switched.",
    }

    with open(OUT / "best_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "trial": best["trial"],
                "is_baseline": best["is_baseline"],
                "config": cfg,
                "mean_cv_direction_accuracy": best["mean_cv_direction_accuracy"],
                "std_cv_direction_accuracy": best["std_cv_direction_accuracy"],
                "mean_cv_power_mae": best["mean_cv_power_mae"],
                "std_cv_power_mae": best["std_cv_power_mae"],
            },
            fh,
            indent=2,
        )

    with open(OUT / "test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        f"Tuned model — test confusion matrix ({acc * 100:.1f}% acc)",
        fontweight="bold",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== HELD-OUT TEST (once) ===", flush=True)
    print(f"  Direction accuracy: {acc * 100:.1f}%", flush=True)
    print(f"  Weighted F1:        {f1 * 100:.1f}%", flush=True)
    print(f"  Power MAE:          {mae:.2f}", flush=True)
    print(f"  Confusion matrix:   {cm}", flush=True)
    return metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    set_seeds(SEED)

    print("=== KickSense wise hyperparameter tuning ===", flush=True)
    X_cv, yd_cv, yp_cv, X_test, yd_test, yp_test = load_data()
    print(
        f"CV pool: {len(X_cv)} | Test held out: {len(X_test)} | "
        f"Folds: {N_FOLDS} | Trials: {N_TRIALS}",
        flush=True,
    )

    results = run_cv_search(X_cv, yd_cv, yp_cv)
    best = pick_best(results)
    print(
        f"\nBest trial {best['trial']}: "
        f"CV {best['mean_cv_direction_accuracy'] * 100:.1f}% "
        f"+/- {best['std_cv_direction_accuracy'] * 100:.1f}%",
        flush=True,
    )

    final_retrain_and_test(best, X_cv, yd_cv, yp_cv, X_test, yd_test, yp_test)
    print("\n=== TUNING COMPLETE ===", flush=True)
    print(f"Artefacts written under {OUT}/", flush=True)


if __name__ == "__main__":
    main()
