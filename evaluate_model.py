import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score,
    recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)
import os

os.makedirs("outputs/evaluation", exist_ok=True)

print("=== KICKSENSE MODEL EVALUATION ===\n")

# ============================================================
# STEP 1: LOAD MODEL AND DATA
# ============================================================
print("STEP 1: Loading model and test data...")

model = tf.keras.models.load_model(
    'outputs/model/best_model_improved.keras'
)

X_test = np.load("data/processed/X_test.npy")
yd_test = np.load("data/processed/y_direction_test.npy")
yp_test = np.load("data/processed/y_power_test.npy")

from tensorflow.keras.utils import to_categorical
yd_test_cat = to_categorical(yd_test, num_classes=3)
yp_test_norm = yp_test / 10.0

print(f"Test samples: {len(X_test)}")

# ============================================================
# STEP 2: GET PREDICTIONS
# ============================================================
print("\nSTEP 2: Getting predictions...")

predictions = model.predict(X_test, verbose=0)
direction_probs = predictions[0]
power_preds = predictions[1]

# Convert probabilities to class labels
direction_preds = np.argmax(direction_probs, axis=1)
power_preds_scaled = power_preds.flatten() * 10.0

label_names = ['Left', 'Center', 'Right']

# ============================================================
# STEP 3: DIRECTION CLASSIFICATION METRICS
# ============================================================
print("\nSTEP 3: Classification metrics...")

accuracy = accuracy_score(yd_test, direction_preds)
precision = precision_score(yd_test, direction_preds,
                            average='weighted', zero_division=0)
recall = recall_score(yd_test, direction_preds,
                      average='weighted', zero_division=0)
f1 = f1_score(yd_test, direction_preds,
              average='weighted', zero_division=0)

print(f"\n{'='*40}")
print(f"SHOT DIRECTION CLASSIFICATION RESULTS")
print(f"{'='*40}")
print(f"Accuracy:  {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall:    {recall*100:.1f}%")
print(f"F1-Score:  {f1*100:.1f}%")

print(f"\nDetailed Classification Report:")
print(classification_report(
    yd_test, direction_preds,
    target_names=label_names,
    zero_division=0
))

# ============================================================
# STEP 4: POWER REGRESSION METRICS
# ============================================================
print("\nSTEP 4: Regression metrics...")

mae = mean_absolute_error(yp_test, power_preds_scaled)
mse = mean_squared_error(yp_test, power_preds_scaled)
rmse = np.sqrt(mse)

print(f"\n{'='*40}")
print(f"SHOT POWER REGRESSION RESULTS")
print(f"{'='*40}")
print(f"MAE:  {mae:.2f} (avg error on scale 1-10)")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# ============================================================
# STEP 5: CONFUSION MATRIX
# ============================================================
print("\nSTEP 5: Plotting confusion matrix...")

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(yd_test, direction_preds)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_names,
    yticklabels=label_names,
    ax=ax, linewidths=0.5,
    annot_kws={'size': 14, 'weight': 'bold'}
)
ax.set_title('Confusion Matrix — Shot Direction',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/evaluation/confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# ============================================================
# STEP 6: METRICS BAR CHART
# ============================================================
print("\nSTEP 6: Plotting metrics chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('KickSense Model Evaluation Results',
             fontsize=14, fontweight='bold')

# Classification metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy*100, precision*100, recall*100, f1*100]
colors = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD']

bars = axes[0].bar(metrics, values, color=colors,
                   edgecolor='white', width=0.5)
for bar, val in zip(bars, values):
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{val:.1f}%',
        ha='center', fontsize=11, fontweight='bold'
    )
axes[0].set_title('Classification Metrics', fontweight='bold')
axes[0].set_ylabel('Score (%)')
axes[0].set_ylim(0, 100)
axes[0].axhline(y=33.3, color='red', linestyle='--',
                linewidth=1.5, label='Random baseline (33%)')
axes[0].legend(fontsize=9)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Power prediction scatter
axes[1].scatter(yp_test, power_preds_scaled,
                color='#378ADD', alpha=0.7, s=80)
axes[1].plot([1, 10], [1, 10], 'r--',
             linewidth=2, label='Perfect prediction')
axes[1].set_title('Shot Power: Actual vs Predicted',
                  fontweight='bold')
axes[1].set_xlabel('Actual Power')
axes[1].set_ylabel('Predicted Power')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/evaluation/evaluation_results.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: evaluation_results.png")

# ============================================================
# STEP 7: PER CLASS ANALYSIS
# ============================================================
print("\nSTEP 7: Per class analysis...")

fig, ax = plt.subplots(figsize=(10, 5))

per_class_f1 = f1_score(yd_test, direction_preds,
                         average=None, zero_division=0)
per_class_precision = precision_score(yd_test, direction_preds,
                                       average=None, zero_division=0)
per_class_recall = recall_score(yd_test, direction_preds,
                                 average=None, zero_division=0)

x = np.arange(len(label_names))
width = 0.25

bars1 = ax.bar(x - width, per_class_precision*100,
               width, label='Precision',
               color='#378ADD', edgecolor='white')
bars2 = ax.bar(x, per_class_recall*100,
               width, label='Recall',
               color='#1D9E75', edgecolor='white')
bars3 = ax.bar(x + width, per_class_f1*100,
               width, label='F1-Score',
               color='#D85A30', edgecolor='white')

ax.set_title('Per Class Metrics', fontsize=14,
             fontweight='bold', pad=15)
ax.set_xlabel('Shot Direction')
ax.set_ylabel('Score (%)')
ax.set_xticks(x)
ax.set_xticklabels(label_names)
ax.legend()
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/evaluation/per_class_metrics.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: per_class_metrics.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*40}")
print("FINAL EVALUATION SUMMARY")
print(f"{'='*40}")
print(f"Model: KickSense CNN-LSTM")
print(f"Test samples: {len(X_test)}")
print(f"\nShot Direction Classification:")
print(f"  Accuracy:  {accuracy*100:.1f}%")
print(f"  Precision: {precision*100:.1f}%")
print(f"  Recall:    {recall*100:.1f}%")
print(f"  F1-Score:  {f1*100:.1f}%")
print(f"\nShot Power Regression:")
print(f"  MAE:  {mae:.2f}")
print(f"  MSE:  {mse:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"\nAll evaluation files saved to outputs/evaluation/")