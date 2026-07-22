import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/model", exist_ok=True)

print("=== KICKSENSE IMPROVED MODEL TRAINING ===\n")

# ============================================================
# STEP 1: LOAD DATA
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

# ============================================================
# STEP 2: DATA AUGMENTATION
# ============================================================
print("\nSTEP 2: Augmenting training data...")

def augment_sequence(sequence, label):
    """
    Create augmented versions of a sequence
    sequence shape: (26, 132)
    """
    augmented_sequences = []
    augmented_labels = []

    # Original
    augmented_sequences.append(sequence)
    augmented_labels.append(label)

    # Augmentation 1: Add small random noise
    noise = np.random.normal(0, 0.01, sequence.shape)
    augmented_sequences.append(sequence + noise)
    augmented_labels.append(label)

    # Augmentation 2: Slight scaling
    scale = np.random.uniform(0.95, 1.05)
    augmented_sequences.append(sequence * scale)
    augmented_labels.append(label)

    # Augmentation 3: Mirror horizontally (flip left/right)
    # For mirroring flip the label too: left(0) → right(2), right(2) → left(0)
    mirrored = sequence.copy()
    mirrored[:, 0::4] = -mirrored[:, 0::4]  # flip x coordinates
    mirrored_label = label
    if label == 0:    # left → right
        mirrored_label = 2
    elif label == 2:  # right → left
        mirrored_label = 0
    augmented_sequences.append(mirrored)
    augmented_labels.append(mirrored_label)

    # Augmentation 4: Time shift (slight temporal shift)
    shift = np.random.randint(1, 3)
    shifted = np.roll(sequence, shift, axis=0)
    augmented_sequences.append(shifted)
    augmented_labels.append(label)

    return augmented_sequences, augmented_labels

# Apply augmentation to training data
X_aug = []
y_aug = []

for seq, lbl in zip(X_train, yd_train):
    seqs, lbls = augment_sequence(seq, lbl)
    X_aug.extend(seqs)
    y_aug.extend(lbls)

X_train_aug = np.array(X_aug)
y_train_aug = np.array(y_aug)

# Shuffle augmented data
shuffle_idx = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[shuffle_idx]
y_train_aug = y_train_aug[shuffle_idx]

print(f"Original training samples: {len(X_train)}")
print(f"Augmented training samples: {len(X_train_aug)}")

# ============================================================
# STEP 3: ENCODE LABELS
# ============================================================
print("\nSTEP 3: Encoding labels...")

yd_train_cat = to_categorical(y_train_aug, num_classes=3)
yd_val_cat = to_categorical(yd_val, num_classes=3)
yd_test_cat = to_categorical(yd_test, num_classes=3)

yp_train_norm = np.load("data/processed/y_power_train.npy")
yp_train_norm_aug = np.tile(
    yp_train_norm,
    int(np.ceil(len(X_train_aug) / len(yp_train_norm)))
)[:len(X_train_aug)] / 10.0
yp_val_norm = yp_val / 10.0
yp_test_norm = yp_test / 10.0

print(f"Augmented labels shape: {yd_train_cat.shape}")

# ============================================================
# STEP 4: CLASS WEIGHTS
# ============================================================
print("\nSTEP 4: Computing class weights...")

classes = np.unique(y_train_aug)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_aug
)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Class weights: {class_weight_dict}")

# ============================================================
# STEP 5: BUILD IMPROVED MODEL
# ============================================================
print("\nSTEP 5: Building improved model...")

def build_improved_model(input_shape):
    inputs = Input(shape=input_shape, name='pose_input')

    # CNN block 1
    x = Conv1D(filters=64, kernel_size=3, activation='relu',
               padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.2, name='dropout1')(x)

    # CNN block 2
    x = Conv1D(filters=128, kernel_size=3, activation='relu',
               padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling1D(pool_size=2, name='pool1')(x)
    x = Dropout(0.2, name='dropout2')(x)

    # LSTM block
    x = LSTM(128, return_sequences=True, name='lstm1')(x)
    x = Dropout(0.3, name='dropout3')(x)
    x = LSTM(64, return_sequences=False, name='lstm2')(x)
    x = Dropout(0.3, name='dropout4')(x)

    # Dense block
    x = Dense(128, activation='relu', name='dense1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Dropout(0.3, name='dropout5')(x)
    x = Dense(64, activation='relu', name='dense2')(x)
    x = Dropout(0.2, name='dropout6')(x)

    # Output heads
    direction_output = Dense(3, activation='softmax',
                             name='direction')(x)
    power_output = Dense(1, activation='sigmoid',
                         name='power')(x)

    model = Model(inputs=inputs,
                  outputs=[direction_output, power_output])
    return model

input_shape = (X_train_aug.shape[1], X_train_aug.shape[2])
model = build_improved_model(input_shape)
model.summary()

# ============================================================
# STEP 6: COMPILE
# ============================================================
print("\nSTEP 6: Compiling model...")

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss={
        'direction': 'categorical_crossentropy',
        'power': 'mse'
    },
    loss_weights={
        'direction': 1.0,
        'power': 0.3
    },
    metrics={
        'direction': ['accuracy'],
        'power': ['mae']
    }
)

# ============================================================
# STEP 7: TRAIN
# ============================================================
print("\nSTEP 7: Training improved model...")

early_stop = EarlyStopping(
    monitor='val_direction_accuracy',
    patience=20,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

checkpoint = ModelCheckpoint(
    'outputs/model/best_model_improved.keras',
    monitor='val_direction_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Reduce learning rate when stuck
reduce_lr = ReduceLROnPlateau(
    monitor='val_direction_accuracy',
    factor=0.5,
    patience=8,
    mode='max',
    min_lr=0.00001,
    verbose=1
)

history = model.fit(
    X_train_aug,
    {'direction': yd_train_cat, 'power': yp_train_norm_aug},
    validation_data=(
        X_val,
        {'direction': yd_val_cat, 'power': yp_val_norm}
    ),
    epochs=150,
    batch_size=16,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

print("\nTraining complete!")

# ============================================================
# STEP 8: EVALUATE
# ============================================================
print("\nSTEP 8: Evaluating on test set...")

results = model.evaluate(
    X_test,
    {'direction': yd_test_cat, 'power': yp_test_norm},
    verbose=0
)

print(f"\nTest Results:")
print(f"  Direction Accuracy: {results[3]*100:.1f}%")
print(f"  Power MAE:          {results[4]*10:.2f} (on scale 1-10)")

# ============================================================
# STEP 9: PLOT TRAINING CURVES
# ============================================================
print("\nSTEP 9: Plotting training curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('KickSense Improved Model Training', 
             fontsize=14, fontweight='bold')

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
plt.savefig('outputs/model/training_curves_improved.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: training_curves_improved.png")

# ============================================================
# STEP 10: SAVE MODEL
# ============================================================
print("\nSTEP 10: Saving final model...")
model.save('outputs/model/kicksense_model_improved.keras')
print("Model saved!")

print("\n=== IMPROVED TRAINING COMPLETE ===")