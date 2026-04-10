# Arrhythmia Classification Using Deep Learning Models

## Overview
ECG signal classification using CNN and hybrid LSTM-CNN architectures. Built on the MIT-BIH Arrhythmia dataset format with structured preprocessing, threshold tuning, and robustness analysis under class imbalance.

## Project Structure
```
arrhythmia-classification/
│
├── arrhythmia_classification.ipynb   # Main notebook
├── ecg_samples.png                   # Sample ECG signals per class
├── confusion_matrix_CNN.png
├── confusion_matrix_LSTM-CNN.png
├── threshold_tradeoff.png            # Sensitivity vs Specificity plot
├── f1_comparison.png                 # Per-class F1 comparison
└── README.md
```

## What This Project Does
- Simulates a 5-class ECG dataset (N, S, V, F, Q) mirroring real MIT-BIH class imbalance
- Builds and trains two architectures: **CNN** and **hybrid LSTM-CNN**
- Evaluates using Accuracy, Sensitivity, Specificity, and F1-score per class
- Tunes classification thresholds to analyse sensitivity–specificity trade-offs (critical for Ventricular class)
- Conducts robustness analysis under class imbalance conditions
- Compares both architectures to determine optimal performance

## Model Architectures

| Model | Layers | Best For |
|-------|--------|----------|
| CNN | 3x Conv1D + BatchNorm + Dense | Speed & simplicity |
| LSTM-CNN | Conv1D + LSTM + Dense | Temporal dependencies |

## Technologies Used
- Python 3
- TensorFlow / Keras
- Scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn

## How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
   ```
2. Open and run the notebook:
   ```bash
   jupyter notebook arrhythmia_classification.ipynb
   ```

## Key Findings
- LSTM-CNN captures sequential ECG patterns better than CNN alone
- Threshold tuning for class V (Ventricular) is clinically critical — maximising sensitivity reduces missed detections
- Class imbalance affects minority class (S, F, Q) F1 scores — oversampling recommended for production

## Author
**Bachu Manikanta**  
Master of Data Science — RMIT University  
[LinkedIn](https://www.linkedin.com/in/manikanta-bachu)
