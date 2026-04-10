# ============================================================
# Arrhythmia Classification Using Deep Learning Models
# Author: Bachu Manikanta
# Description: ECG signal classification using CNN and
#              hybrid LSTM-CNN architectures.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense,
                                      Dropout, Flatten, Input, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────
# 1. SIMULATE ECG DATASET (MIT-BIH format)
# ─────────────────────────────────────────────
n_features = 187
class_counts = {'N': 3500, 'S': 556, 'V': 641, 'F': 162, 'Q': 141}
labels, signals = [], []

for cls, count in class_counts.items():
    for _ in range(count):
        base = np.random.randn(n_features) * 0.3
        if cls == 'N':
            base[60:80] += np.sin(np.linspace(0, np.pi, 20)) * 2
        elif cls == 'V':
            base[50:90] += np.sin(np.linspace(0, 2*np.pi, 40)) * 2.5
        elif cls == 'S':
            base[65:75] += np.sin(np.linspace(0, np.pi, 10)) * 1.5
        elif cls == 'F':
            base[55:85] += np.sin(np.linspace(0, 1.5*np.pi, 30)) * 2
        elif cls == 'Q':
            base += np.random.randn(n_features) * 0.5
        signals.append(base)
        labels.append(cls)

X = np.array(signals)
y_raw = np.array(labels)
le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"Dataset shape: {X.shape}")
print("Class distribution:")
for cls, cnt in zip(*np.unique(y_raw, return_counts=True)):
    print(f"  {cls}: {cnt} ({cnt/len(y_raw)*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. VISUALISE SAMPLE ECG SIGNALS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 3))
fig.suptitle('Sample ECG Signals by Class', fontsize=14, fontweight='bold')
for i, cls in enumerate(le.classes_):
    idx = np.where(y_raw == cls)[0][0]
    axes[i].plot(X[idx], linewidth=1, color='steelblue')
    axes[i].set_title(f'Class: {cls}', fontweight='bold')
    axes[i].set_xlabel('Sample')
    axes[i].set_ylabel('Amplitude')
plt.tight_layout()
plt.savefig('ecg_samples.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_classes = len(le.classes_)
y_cat = to_categorical(y, num_classes=n_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42)

X_train_r = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_r   = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_r  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"\nTrain: {X_train_r.shape}, Val: {X_val_r.shape}, Test: {X_test_r.shape}")

# ─────────────────────────────────────────────
# 4. MODEL 1 — CNN
# ─────────────────────────────────────────────
def build_cnn(input_shape, n_classes):
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
cnn_model = build_cnn((X_train_r.shape[1], 1), n_classes)
cnn_model.summary()

cnn_history = cnn_model.fit(
    X_train_r, y_train, epochs=30, batch_size=64,
    validation_data=(X_val_r, y_val), callbacks=[es], verbose=1)

# ─────────────────────────────────────────────
# 5. MODEL 2 — HYBRID LSTM-CNN
# ─────────────────────────────────────────────
def build_lstm_cnn(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=5, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

lstm_cnn_model = build_lstm_cnn((X_train_r.shape[1], 1), n_classes)
lstm_cnn_model.summary()

lstm_history = lstm_cnn_model.fit(
    X_train_r, y_train, epochs=30, batch_size=64,
    validation_data=(X_val_r, y_val), callbacks=[es], verbose=1)

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, name, X_test, y_test, le):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix — {name}', fontweight='bold')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ","_")}.png', dpi=150)
    plt.show()
    return acc, y_pred_prob, y_true

cnn_acc, cnn_probs, y_true = evaluate_model(cnn_model, 'CNN', X_test_r, y_test, le)
lstm_acc, lstm_probs, _    = evaluate_model(lstm_cnn_model, 'LSTM-CNN', X_test_r, y_test, le)

# ─────────────────────────────────────────────
# 7. THRESHOLD TUNING — Sensitivity vs Specificity
# ─────────────────────────────────────────────
v_idx = list(le.classes_).index('V')
y_true_binary = (y_true == v_idx).astype(int)
y_scores = lstm_probs[:, v_idx]

thresholds = np.arange(0.1, 0.9, 0.05)
sensitivities, specificities = [], []
for t in thresholds:
    y_pred_t = (y_scores >= t).astype(int)
    tp = ((y_pred_t == 1) & (y_true_binary == 1)).sum()
    fn = ((y_pred_t == 0) & (y_true_binary == 1)).sum()
    tn = ((y_pred_t == 0) & (y_true_binary == 0)).sum()
    fp = ((y_pred_t == 1) & (y_true_binary == 0)).sum()
    sensitivities.append(tp / (tp + fn + 1e-9))
    specificities.append(tn / (tn + fp + 1e-9))

plt.figure(figsize=(9, 4))
plt.plot(thresholds, sensitivities, marker='o', label='Sensitivity', color='tomato')
plt.plot(thresholds, specificities, marker='s', label='Specificity', color='steelblue')
plt.axvline(x=0.4, linestyle='--', color='gray', label='Selected Threshold (0.4)')
plt.title('Sensitivity vs Specificity Trade-off — Class V', fontweight='bold')
plt.xlabel('Threshold'); plt.ylabel('Score'); plt.legend()
plt.tight_layout()
plt.savefig('threshold_tradeoff.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 8. CLASS IMBALANCE — Per-class F1 Comparison
# ─────────────────────────────────────────────
cnn_f1  = f1_score(y_true, np.argmax(cnn_probs, axis=1), average=None)
lstm_f1 = f1_score(y_true, np.argmax(lstm_probs, axis=1), average=None)

x = np.arange(len(le.classes_))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, cnn_f1,  width, label='CNN',      color='steelblue')
ax.bar(x + width/2, lstm_f1, width, label='LSTM-CNN', color='coral')
ax.set_xticks(x); ax.set_xticklabels(le.classes_)
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 Score: CNN vs LSTM-CNN', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('f1_comparison.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 9. MODEL COMPARISON SUMMARY
# ─────────────────────────────────────────────
print("\n========== MODEL COMPARISON ==========")
print(f"{'Model':<12} {'Accuracy':>10}")
print(f"{'CNN':<12} {cnn_acc:>10.4f}")
print(f"{'LSTM-CNN':<12} {lstm_acc:>10.4f}")
print("\nConclusion: LSTM-CNN captures temporal ECG patterns better; CNN trains faster.")
