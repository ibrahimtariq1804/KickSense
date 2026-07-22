import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, 
    Dense, Dropout, BatchNormalization, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/model", exist_ok=True)

print("=== KICKSENSE MODEL TRAINING ===\n")

# ============================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================
print("STEP 1: Loading preprocessed data...")

X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
X_test = np.load("data/processed/X_test.npy")

yd_train = np.load("data/processed/y_direction_train.npy")
yd_val = np.load("data/processed/y_direction_val.npy")
yd_test = np.load("data/processed/y_direction_test.npy")

yp_train = np.load("data/processed/y_power_train.npy")
yp_val = np.load("data/processed/y_power_val.npy")
yp_test = np.load("data/processed/y_power_test.npy")

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")
print(f"Input shape:        {X_train.shape}")

# ============================================================
# STEP 2: ENCODE LABELS
# ============================================================
print("\nSTEP 2: Encoding labels...")

# One-hot encode direction labels
# left=0 → [1,0,0], center=1 → [0,1,0], right=2 → [0,0,1]
yd_train_cat = to_categorical(yd_train, num_classes=3)
yd_val_cat = to_categorical(yd_val, num_classes=3)
yd_test_cat = to_categorical(yd_test, num_classes=3)

# Normalize shot power to 0-1 range for better training
yp_train_norm = yp_train / 10.0
yp_val_norm = yp_val / 10.0
yp_test_norm = yp_test / 10.0

print(f"Direction labels shape: {yd_train_cat.shape}")
print(f"Power labels range: {yp_train_norm.min():.2f} - {yp_train_norm.max():.2f}")

# ============================================================
# STEP 3: BUILD CNN-LSTM MODEL
# ============================================================
print("\nSTEP 3: Building CNN-LSTM model...")

def build_model(input_shape):
    inputs = Input(shape=input_shape, name='pose_input')

    # CNN layers - learn spatial patterns per frame
    x = Conv1D(filters=64, kernel_size=3, activation='relu', 
               padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu',
               padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling1D(pool_size=2, name='pool1')(x)
    x = Dropout(0.3, name='dropout1')(x)

    # LSTM layer - learn temporal patterns across frames
    x = LSTM(128, return_sequences=True, name='lstm1')(x)
    x = LSTM(64, return_sequences=False, name='lstm2')(x)
    x = Dropout(0.3, name='dropout2')(x)

    # Dense shared layer
    x = Dense(64, activation='relu', name='dense_shared')(x)
    x = Dropout(0.2, name='dropout3')(x)

    # Output Head 1: Shot Direction (Classification)
    direction_output = Dense(3, activation='softmax', 
                            name='direction')(x)

    # Output Head 2: Shot Power (Regression)
    power_output = Dense(1, activation='sigmoid', 
                        name='power')(x)

    model = Model(inputs=inputs, 
                  outputs=[direction_output, power_output])
    return model

input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape)

# Print model summary
model.summary()

# ============================================================
# STEP 4: COMPILE MODEL
# ============================================================
print("\nSTEP 4: Compiling model...")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'direction': 'categorical_crossentropy',
        'power': 'mse'
    },
    loss_weights={
        'direction': 1.0,
        'power': 0.5
    },
    metrics={
        'direction': ['accuracy'],
        'power': ['mae']
    }
)

print("Model compiled successfully!")

# ============================================================
# STEP 5: TRAIN MODEL
# ============================================================
print("\nSTEP 5: Training model...")

early_stop = EarlyStopping(
    monitor='val_direction_accuracy',
    patience=15,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

checkpoint = ModelCheckpoint(
    'outputs/model/best_model.keras',
    monitor='val_direction_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    X_train,
    {'direction': yd_train_cat, 'power': yp_train_norm},
    validation_data=(
        X_val,
        {'direction': yd_val_cat, 'power': yp_val_norm}
    ),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\nTraining complete!")

# ============================================================
# STEP 6: EVALUATE MODEL
# ============================================================
print("\nSTEP 6: Evaluating on test set...")

results = model.evaluate(
    X_test,
    {'direction': yd_test_cat, 'power': yp_test_norm},
    verbose=0
)

print(f"\nTest Results:")
print(f"  Direction Accuracy: {results[3]*100:.1f}%")
print(f"  Power MAE:          {results[4]*10:.2f} (on scale 1-10)")

# ============================================================
# STEP 7: PLOT TRAINING CURVES
# ============================================================
print("\nSTEP 7: Plotting training curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('KickSense Model Training', fontsize=14, fontweight='bold')

# Direction Accuracy
axes[0].plot(history.history['direction_accuracy'], 
             label='Train Accuracy', color='#378ADD', linewidth=2)
axes[0].plot(history.history['val_direction_accuracy'],
             label='Val Accuracy', color='#1D9E75', linewidth=2)
axes[0].set_title('Shot Direction Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Power Loss
axes[1].plot(history.history['power_mae'],
             label='Train MAE', color='#378ADD', linewidth=2)
axes[1].plot(history.history['val_power_mae'],
             label='Val MAE', color='#1D9E75', linewidth=2)
axes[1].set_title('Shot Power MAE', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/model/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: training_curves.png")

# ============================================================
# STEP 8: SAVE FINAL MODEL
# ============================================================
print("\nSTEP 8: Saving final model...")
model.save('outputs/model/kicksense_model.keras')
print("Model saved to outputs/model/kicksense_model.keras")

print("\n=== TRAINING COMPLETE ===")
print(f"Best model saved to outputs/model/best_model.keras")