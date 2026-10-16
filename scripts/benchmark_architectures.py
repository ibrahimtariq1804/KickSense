import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, AveragePooling1D, LSTM, GRU,
    Bidirectional, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, mean_absolute_error

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

print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
print(f"Train direction counts: {np.bincount(yd_train)}")

# Sophisticated, kinematically valid augmentation (no left-right flipping)
def augment_dataset(X, yd, yp, right_multiplier=2):
    X_aug, yd_aug, yp_aug = [], [], []
    for x_seq, y_dir, y_pwr in zip(X, yd, yp):
        # 1. Original
        X_aug.append(x_seq)
        yd_aug.append(y_dir)
        yp_aug.append(y_pwr)
        
        # 2. Gaussian jitter (slight sensor / pose noise)
        noise = np.random.normal(0, 0.015, x_seq.shape)
        X_aug.append(x_seq + noise)
        yd_aug.append(y_dir)
        yp_aug.append(y_pwr)
        
        # 3. Scale jitter (player distance / body size variation)
        scale = np.random.uniform(0.93, 1.07)
        X_aug.append(x_seq * scale)
        yd_aug.append(y_dir)
        yp_aug.append(y_pwr)
        
        # 4. Temporal shift (+1 frame roll)
        shift = np.random.choice([-1, 1])
        rolled = np.roll(x_seq, shift, axis=0)
        X_aug.append(rolled)
        yd_aug.append(y_dir)
        yp_aug.append(y_pwr)
        
        # 5. Joint dropout (simulate minor landmark occlusion)
        drop_mask = (np.random.rand(*x_seq.shape) > 0.03).astype(float)
        X_aug.append(x_seq * drop_mask)
        yd_aug.append(y_dir)
        yp_aug.append(y_pwr)
        
        # For minority class (Right, label 2), add extra variations to balance dataset
        if y_dir == 2:
            for _ in range(right_multiplier):
                s = np.random.uniform(0.95, 1.05)
                n = np.random.normal(0, 0.012, x_seq.shape)
                X_aug.append((x_seq * s) + n)
                yd_aug.append(y_dir)
                yp_aug.append(y_pwr)
                
    X_aug = np.array(X_aug)
    yd_aug = np.array(yd_aug)
    yp_aug = np.array(yp_aug, dtype=float)
    
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], yd_aug[idx], yp_aug[idx]

np.random.seed(42)
tf.random.set_seed(42)

X_tr_aug, yd_tr_aug, yp_tr_aug = augment_dataset(X_train, yd_train, yp_train)
print(f"Augmented Train samples: {len(X_tr_aug)}, direction counts: {np.bincount(yd_tr_aug)}")

yd_tr_cat = to_categorical(yd_tr_aug, num_classes=3)
yd_va_cat = to_categorical(yd_val, num_classes=3)
yd_te_cat = to_categorical(yd_test, num_classes=3)

yp_tr_norm = yp_tr_aug / 10.0
yp_va_norm = yp_val / 10.0
yp_te_norm = yp_test / 10.0

input_shape = (26, 132)

# Candidate Architecture 1: Standard CNN + BiLSTM with L2 Regularization
def build_model_1():
    inp = Input(shape=input_shape, name='pose_input')
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    out_dir = Dense(3, activation='softmax', name='direction')(x)
    out_pwr = Dense(1, activation='sigmoid', name='power')(x)
    return Model(inputs=inp, outputs=[out_dir, out_pwr], name="Arch1_CNN_BiLSTM")

# Candidate Architecture 2: Multi-Scale Temporal Inception + BiGRU
def build_model_2():
    inp = Input(shape=input_shape, name='pose_input')
    
    # Branch 1: kernel 3 (short temporal movements)
    b1 = Conv1D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inp)
    # Branch 2: kernel 5 (medium temporal movements)
    b2 = Conv1D(32, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inp)
    # Branch 3: kernel 7 (longer approach kinematics)
    b3 = Conv1D(32, kernel_size=7, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inp)
    
    m = concatenate([b1, b2, b3], axis=-1)
    m = BatchNormalization()(m)
    m = Dropout(0.2)(m)
    
    m = Conv1D(96, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(m)
    m = BatchNormalization()(m)
    m = MaxPooling1D(2)(m)
    m = Dropout(0.25)(m)
    
    m = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(1e-4)))(m)
    m = Dropout(0.3)(m)
    m = Bidirectional(GRU(32, return_sequences=False, kernel_regularizer=l2(1e-4)))(m)
    m = Dropout(0.3)(m)
    
    d = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(m)
    d = BatchNormalization()(d)
    d = Dropout(0.3)(d)
    
    out_dir = Dense(3, activation='softmax', name='direction')(d)
    out_pwr = Dense(1, activation='sigmoid', name='power')(d)
    return Model(inputs=inp, outputs=[out_dir, out_pwr], name="Arch2_MultiScale_BiGRU")

# Candidate Architecture 3: Temporal Conv + Global Pooling Ensemble (ResNet-style)
def build_model_3():
    inp = Input(shape=input_shape, name='pose_input')
    
    # Dense spatial projection
    x = Dense(96, activation='relu', kernel_regularizer=l2(1e-4))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Conv block 1
    c1 = Conv1D(96, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.2)(c1)
    c2 = Conv1D(96, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(c1)
    c2 = BatchNormalization()(c2)
    res1 = Add()([x, c2])
    
    # Pool
    p1 = MaxPooling1D(2)(res1)
    
    # BiLSTM
    lstm_out = Bidirectional(LSTM(48, return_sequences=True, kernel_regularizer=l2(1e-4)))(p1)
    
    # Global average + max pooling over time
    avg_pool = GlobalAveragePooling1D()(lstm_out)
    max_pool = GlobalMaxPooling1D()(lstm_out)
    pooled = concatenate([avg_pool, max_pool])
    pooled = Dropout(0.35)(pooled)
    
    fc = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(pooled)
    fc = BatchNormalization()(fc)
    fc = Dropout(0.3)(fc)
    
    out_dir = Dense(3, activation='softmax', name='direction')(fc)
    out_pwr = Dense(1, activation='sigmoid', name='power')(fc)
    return Model(inputs=inp, outputs=[out_dir, out_pwr], name="Arch3_ResConv_BiLSTM_Pool")

models = [
    build_model_1(),
    build_model_2(),
    build_model_3()
]

results = []
for model in models:
    name = model.name
    print(f"\n==========================================")
    print(f"Training {name}...")
    print(f"==========================================")
    
    model.compile(
        optimizer=Adam(learning_rate=0.0007),
        loss={'direction': 'categorical_crossentropy', 'power': 'mse'},
        loss_weights={'direction': 1.0, 'power': 0.25},
        metrics={'direction': ['accuracy'], 'power': ['mae']}
    )
    
    callbacks = [
        EarlyStopping(monitor='val_direction_accuracy', patience=18, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_direction_accuracy', factor=0.5, patience=7, mode='max', min_lr=1e-5)
    ]
    
    h = model.fit(
        X_tr_aug, {'direction': yd_tr_cat, 'power': yp_tr_norm},
        validation_data=(X_val, {'direction': yd_va_cat, 'power': yp_va_norm}),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=0
    )
    
    best_val_epoch = np.argmax(h.history['val_direction_accuracy'])
    best_val_acc = h.history['val_direction_accuracy'][best_val_epoch]
    
    preds = model.predict(X_test, verbose=0)
    pred_dir = np.argmax(preds[0], axis=1)
    test_acc = accuracy_score(yd_test, pred_dir)
    test_mae = mean_absolute_error(yp_test, preds[1].flatten() * 10.0)
    
    print(f"{name} -> Best Val Acc: {best_val_acc*100:.1f}%, Test Acc: {test_acc*100:.1f}%, Test Power MAE: {test_mae:.2f}")
    results.append({
        'name': name,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_mae': test_mae,
        'model': model
    })

print("\n\n=== BENCHMARK SUMMARY ===")
for r in results:
    print(f"{r['name']}: Val Acc={r['val_acc']*100:.1f}%, Test Acc={r['test_acc']*100:.1f}%, Power MAE={r['test_mae']:.2f}")
