"""Trains the reference model for the report with a fixed seed and saves it.

Everything in Chapter 4 that reports test-set numbers is computed from this
single checkpoint, so the confusion matrix, per-class metrics and power
regression figures are all mutually consistent and reproducible.
"""
import json
import os
import sys

import numpy as np

# Run from the repository root so the relative data paths resolve, regardless
# of the directory this script is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "report_assets"))

from run_ablation import DATA, build, augment, set_seeds  # noqa: E402

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

OUT_MODEL = "report_assets/reference_model.keras"
OUT_HIST = "report_assets/reference_history.json"


def main():
    set_seeds()

    X_tr = np.load(f"{DATA}/X_train.npy")
    X_va = np.load(f"{DATA}/X_val.npy")
    yd_tr = np.load(f"{DATA}/y_direction_train.npy")
    yd_va = np.load(f"{DATA}/y_direction_val.npy")
    yp_tr = np.load(f"{DATA}/y_power_train.npy")
    yp_va = np.load(f"{DATA}/y_power_val.npy")

    X_fit, yd_fit, yp_fit = augment(X_tr, yd_tr, yp_tr)

    classes = np.unique(yd_fit)
    weights = compute_class_weight("balanced", classes=classes, y=yd_fit)
    lookup = {int(c): float(w) for c, w in zip(classes, weights)}
    dir_w = np.array([lookup[int(l)] for l in yd_fit], dtype="float32")

    model = build("cnn_lstm", regularise=True)
    model.summary()

    hist = model.fit(
        X_fit,
        {"direction": to_categorical(yd_fit, 3), "power": yp_fit / 10.0},
        validation_data=(X_va, {"direction": to_categorical(yd_va, 3),
                                "power": yp_va / 10.0}),
        epochs=150,
        batch_size=16,
        sample_weight={"direction": dir_w, "power": np.ones_like(dir_w)},
        callbacks=[
            EarlyStopping(monitor="val_direction_accuracy", patience=20,
                          restore_best_weights=True, mode="max"),
            ReduceLROnPlateau(monitor="val_direction_loss", factor=0.5,
                              patience=8, min_lr=1e-5, mode="min"),
        ],
        verbose=2,
    )

    model.save(OUT_MODEL)
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    with open(OUT_HIST, "w") as fh:
        json.dump({
            "class_weights": lookup,
            "augmented_train_samples": int(len(X_fit)),
            "params": int(model.count_params()),
            "history": history,
        }, fh, indent=2)

    print(f"saved {OUT_MODEL}")
    print(f"best val direction accuracy: "
          f"{max(history['val_direction_accuracy']):.4f}")


if __name__ == "__main__":
    main()
