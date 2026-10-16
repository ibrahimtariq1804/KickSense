import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Bidirectional,
    Dense, Dropout, BatchNormalization, concatenate,
    GlobalAveragePooling1D, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=== TRAINING PRODUCTION MODEL ===")

# 1. Load data
X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
X_test = np.load("data/processed/X_test.npy")

yd_train = np.load("data/processed/y_direction_train.npy")
yd_val = np.load("data/processed/y_direction_val.npy")
yd_test = np.load("data/processed/y_direction_test.npy")

yp_train = np.load("data/processed/y_power_train.npy")
yp_val = np.load("data/processed/y_power_val.npy")
yp_test = np.load("data/processed/y_power_test.npy")

print(f"Train samples: {len(X_train)} (counts: {np.bincount(yd_train)})")
print(f"Val samples:   {len(X_val)} (counts: {np.bincount(yd_val)})")
print(f"Test samples:  {len(X_test)} (counts: {np.bincount(yd_test)})")

# 2. Kinematically-Valid Augmentation with Right-Class Balancing
def augment_training_set(X, yd, yp):
    X_aug, yd_aug, yp_aug = [], [], []
    for seq, d, p in zip(X, yd, yp):
        # Original
        X_aug.append(seq)
        yd_aug.append(d)
        yp_aug.append(p)
        
        # Jitter (sensor & landmark noise)
        n = np.random.normal(0, 0.012, seq.shape)
        X_aug.append(seq + n)
        yd_aug.append(d)
        yp_aug.append(p)
        
        # Scale (player height / camera distance variation)
        s = np.random.uniform(0.93, 1.07)
        X_aug.append(seq * s)
        yd_aug.append(d)
        yp_aug.append(p)
        
        # Temporal roll (+-1 frame shift)
        shift = np.random.choice([-1, 1])
        X_aug.append(np.roll(seq, shift, axis=0))
        yd_aug.append(d)
        yp_aug.append(p)
        
        # For Right kicks (class 2, minority), add 3 additional variations
        # to achieve balanced representation across all 3 classes
        if d == 2:
            for _ in range(3):
                s_right = np.random.uniform(0.94, 1.06)
                n_right = np.random.normal(0, 0.01, seq.shape)
                X_aug.append((seq * s_right) + n_right)
                yd_aug.append(d)
                yp_aug.append(p)
                
    X_aug = np.array(X_aug)
    yd_aug = np.array(yd_aug)
    yp_aug = np.array(yp_aug, dtype=float)
    
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], yd_aug[idx], yp_aug[idx]

X_tr_aug, yd_tr_aug, yp_tr_aug = augment_training_set(X_train, yd_train, yp_train)
print(f"Augmented Train samples: {len(X_tr_aug)}, balanced counts: {np.bincount(yd_tr_aug)}")

yd_tr_cat = to_categorical(yd_tr_aug, num_classes=3)
yd_va_cat = to_categorical(yd_val, num_classes=3)
yd_te_cat = to_categorical(yd_test, num_classes=3)

yp_tr_norm = yp_tr_aug / 10.0
yp_va_norm = yp_val / 10.0
yp_te_norm = yp_test / 10.0

# 3. Class weights to ensure balanced gradient updates
weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=yd_tr_aug)
class_weights_dict = {i: float(w) for i, w in enumerate(weights)}
print("Class weights:", class_weights_dict)

# 4. Multi-Scale Temporal Inception + BiLSTM Architecture
def build_production_model(input_shape=(26, 132)):
    inp = Input(shape=input_shape, name='pose_input')
    
    # Feature Projection (Bottleneck to remove noise in 132 face/hand landmarks)
    proj = Dense(64, activation='relu', kernel_regularizer=l2(4e-4))(inp)
    proj = BatchNormalization()(proj)
    proj = Dropout(0.25)(proj)
    
    # Multi-Scale Temporal Convolution:
    # Kernel 1: Static posture features
    # Kernel 3: Velocity features (finite difference)
    # Kernel 5: Acceleration & stride dynamics
    conv1 = Conv1D(32, kernel_size=1, padding='same', activation='relu', kernel_regularizer=l2(4e-4))(proj)
    conv3 = Conv1D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(4e-4))(proj)
    conv5 = Conv1D(32, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(4e-4))(proj)
    
    m = concatenate([conv1, conv3, conv5], axis=-1)
    m = BatchNormalization()(m)
    m = Dropout(0.25)(m)
    
    # Temporal sequence modeling via Bidirectional LSTM
    lstm_out = Bidirectional(LSTM(48, return_sequences=True, kernel_regularizer=l2(4e-4)))(m)
    
    # Shift-Invariance:
    # Max-pooling captures peak strike impact wherever it occurs in the 26 frames
    # Average-pooling captures run-up angle and body posture
    max_p = GlobalMaxPooling1D()(lstm_out)
    avg_p = GlobalAveragePooling1D()(lstm_out)
    pooled = concatenate([max_p, avg_p])
    pooled = Dropout(0.35)(pooled)
    
    # Dense trunk
    fc = Dense(64, activation='relu', kernel_regularizer=l2(4e-4))(pooled)
    fc = BatchNormalization()(fc)
    fc = Dropout(0.3)(fc)
    
    # Output Heads
    out_dir = Dense(3, activation='softmax', name='direction')(fc)
    out_pwr = Dense(1, activation='sigmoid', name='power')(fc)
    
    model = Model(inputs=inp, outputs=[out_dir, out_pwr], name="KickSense_Production_Model")
    return model

model = build_production_model()
model.summary()

# 5. Compile with optimal loss balance
model.compile(
    optimizer=Adam(learning_rate=0.0006),
    loss={'direction': 'categorical_crossentropy', 'power': 'mse'},
    loss_weights={'direction': 1.0, 'power': 0.25},
    metrics={'direction': ['accuracy'], 'power': ['mae']}
)

checkpoint_path = "outputs/model/best_model_improved.keras"
callbacks = [
    EarlyStopping(monitor='val_direction_accuracy', patience=25, restore_best_weights=True, mode='max', verbose=1),
    ModelCheckpoint(checkpoint_path, monitor='val_direction_accuracy', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_direction_accuracy', factor=0.5, patience=8, mode='max', min_lr=1e-5, verbose=1)
]

# 6. Train model
history = model.fit(
    X_tr_aug, {'direction': yd_tr_cat, 'power': yp_tr_norm},
    validation_data=(X_val, {'direction': yd_va_cat, 'power': yp_va_norm}),
    epochs=120,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Also save final model
model.save("outputs/model/kicksense_model_improved.keras")
print(f"Saved best checkpoint to {checkpoint_path}")

# 7. Evaluate on Test Set
test_preds = model.predict(X_test, verbose=0)
pred_dirs = np.argmax(test_preds[0], axis=1)
pred_pwrs = test_preds[1].flatten() * 10.0

acc = accuracy_score(yd_test, pred_dirs)
pwr_mae = mean_absolute_error(yp_test, pred_pwrs)

print(f"\n==========================================")
print(f"FINAL PRODUCTION MODEL EVALUATION:")
print(f"Direction Accuracy: {acc*100:.2f}%")
print(f"Power MAE:          {pwr_mae:.2f}")
print(f"==========================================")
print(classification_report(yd_test, pred_dirs, target_names=['Left', 'Center', 'Right'], zero_division=0))

# 8. Plot and save training curves
os.makedirs("outputs/model", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('KickSense Production Model Training Curves', fontsize=14, fontweight='bold')

axes[0].plot(history.history['direction_accuracy'], label='Train Accuracy', color='#378ADD', linewidth=2)
axes[0].plot(history.history['val_direction_accuracy'], label='Val Accuracy', color='#1D9E75', linewidth=2)
axes[0].set_title('Shot Direction Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].plot(history.history['power_mae'], label='Train MAE', color='#378ADD', linewidth=2)
axes[1].plot(history.history['val_power_mae'], label='Val MAE', color='#1D9E75', linewidth=2)
axes[1].set_title('Shot Power MAE', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/model/training_curves_improved.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/model/training_curves_improved.png")
