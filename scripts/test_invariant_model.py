import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    LSTM, Bidirectional, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D,
    concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

# Load data
X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
X_test = np.load("data/processed/X_test.npy")

yd_train = np.load("data/processed/y_direction_train.npy")
yd_val = np.load("data/processed/y_direction_val.npy")
yd_test = np.load("data/processed/y_direction_test.npy")

yp_train = np.load("data/processed/y_power_train.npy")
yp_val = np.load("data/processed/y_power_val.npy")
yp_test = np.load("data/processed/y_power_test.npy")

# Data augmentation
def augment_clean(X, yd, yp):
    X_aug, yd_aug, yp_aug = [], [], []
    for x, d, p in zip(X, yd, yp):
        # 1. Original
        X_aug.append(x)
        yd_aug.append(d)
        yp_aug.append(p)
        
        # 2. Gaussian noise
        noise = np.random.normal(0, 0.012, x.shape)
        X_aug.append(x + noise)
        yd_aug.append(d)
        yp_aug.append(p)
        
        # 3. Scaling
        scale = np.random.uniform(0.94, 1.06)
        X_aug.append(x * scale)
        yd_aug.append(d)
        yp_aug.append(p)
        
        # 4. Small frame roll
        shift = np.random.choice([-1, 1])
        X_aug.append(np.roll(x, shift, axis=0))
        yd_aug.append(d)
        yp_aug.append(p)
        
        # Minority class boost (Right = 2)
        if d == 2:
            for _ in range(2):
                s = np.random.uniform(0.95, 1.05)
                n = np.random.normal(0, 0.01, x.shape)
                X_aug.append(x * s + n)
                yd_aug.append(d)
                yp_aug.append(p)
                
    X_aug = np.array(X_aug)
    yd_aug = np.array(yd_aug)
    yp_aug = np.array(yp_aug, dtype=float)
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], yd_aug[idx], yp_aug[idx]

np.random.seed(42)
tf.random.set_seed(42)

X_tr_aug, yd_tr_aug, yp_tr_aug = augment_clean(X_train, yd_train, yp_train)

yd_tr_cat = to_categorical(yd_tr_aug, num_classes=3)
yd_va_cat = to_categorical(yd_val, num_classes=3)
yd_te_cat = to_categorical(yd_test, num_classes=3)

yp_tr_norm = yp_tr_aug / 10.0
yp_va_norm = yp_val / 10.0
yp_te_norm = yp_test / 10.0

def build_temporally_invariant_model():
    inp = Input(shape=(26, 132), name='pose_input')
    
    # Feature bottleneck: project noisy 132 landmarks to 64 clean latent dimensions
    x = Dense(64, activation='relu', kernel_regularizer=l2(5e-4))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 1D Temporal Convolution (kernel size 3 captures local 3-frame velocity / acceleration)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Bidirectional temporal model
    lstm_out = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(5e-4)))(x)
    
    # TEMPORAL INVARIANCE:
    # Max-pooling captures the peak kicking moment regardless of when it occurred
    # Avg-pooling captures the average run-up stance
    max_p = GlobalMaxPooling1D()(lstm_out)
    avg_p = GlobalAveragePooling1D()(lstm_out)
    pooled = concatenate([max_p, avg_p])
    pooled = Dropout(0.4)(pooled)
    
    # Classification & Regression trunk
    fc = Dense(48, activation='relu', kernel_regularizer=l2(5e-4))(pooled)
    fc = BatchNormalization()(fc)
    fc = Dropout(0.3)(fc)
    
    out_dir = Dense(3, activation='softmax', name='direction')(fc)
    out_pwr = Dense(1, activation='sigmoid', name='power')(fc)
    
    model = Model(inputs=inp, outputs=[out_dir, out_pwr], name="TemporallyInvariant_Model")
    return model

model = build_temporally_invariant_model()
model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0008),
    loss={'direction': 'categorical_crossentropy', 'power': 'mse'},
    loss_weights={'direction': 1.0, 'power': 0.25},
    metrics={'direction': ['accuracy'], 'power': ['mae']}
)

callbacks = [
    EarlyStopping(monitor='val_direction_accuracy', patience=25, restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_direction_accuracy', factor=0.5, patience=8, mode='max', min_lr=1e-5)
]

history = model.fit(
    X_tr_aug, {'direction': yd_tr_cat, 'power': yp_tr_norm},
    validation_data=(X_val, {'direction': yd_va_cat, 'power': yp_va_norm}),
    epochs=120,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

preds = model.predict(X_test, verbose=0)
pred_dir = np.argmax(preds[0], axis=1)
acc = accuracy_score(yd_test, pred_dir)
pwr_mae = mean_absolute_error(yp_test, preds[1].flatten() * 10.0)

print(f"\n==========================================")
print(f"TEST RESULTS WITH TEMPORAL INVARIANCE:")
print(f"Direction Accuracy: {acc*100:.2f}%")
print(f"Power MAE:          {pwr_mae:.2f}")
print(f"==========================================")
print(classification_report(yd_test, pred_dir, target_names=['Left', 'Center', 'Right'], zero_division=0))
