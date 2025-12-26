# 🧠 EEG Eye State Detection: A BCI Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project focuses on the development of a **Brain-Computer Interface (BCI)** system capable of classifying eye states (Open vs. Closed) using continuous Electroencephalogram (EEG) signals.

Using data from a 14-channel **Emotiv EPOC+ headset**, I implemented and compared two distinct methodologies: **Classical Machine Learning** (with manual feature engineering) and **Deep Learning** (sequence labeling). The goal was to build a robust model for applications such as drowsiness detection and neurological monitoring.

**Key Achievement:** Achieved **99.4% accuracy** using a Hybrid CNN-LSTM architecture, outperforming classical SVM and Random Forest models.

---

## 📂 Dataset
* **Source:** EEG data collected via Emotiv EPOC+.
* **Size:** 14,980 samples.
* **Sensors:** 14 continuous EEG measurements (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4).
* **Target:** Binary Classification (`0`: Eye Closed, `1`: Eye Open).
* **Sampling Rate:** 128 Hz.

---

## ⚙️ Methodology & Pipeline

### 1. Signal Preprocessing
Raw EEG data is noisy and prone to artifacts. I implemented a rigorous cleaning pipeline:
* **Artifact Removal:** Applied **Z-score thresholding** to detect outliers caused by muscle movement (EMG) or blinks (EOG).
* **Interpolation:** Reconstructed removed artifacts using **Cubic Spline Interpolation**.
* **Filtering:** Designed a **Butterworth Bandpass Filter** (Zero-phase `filtfilt`) to isolate the **Alpha Band (8–13 Hz)**, which is neurophysiologically correlated with eye closure (relaxation states).
* **Normalization:** Applied standard scaling to prepare data for identifying boundaries in SVM and Neural Networks.

### 2. Feature Extraction (For Approach A)
I engineered domain-specific features to capture the signal characteristics:
* **Temporal Domain:** Mean, Variance, Skewness, Kurtosis, Peak-to-Peak, Zero Crossing Rate, and **Shannon Entropy**.
* **Frequency Domain (FFT):** Power Spectral Density (PSD) extracted for Delta, Theta, Alpha, Beta, and Gamma bands.
* **Dimensionality Reduction:** Utilized **PCA (Principal Component Analysis)** to retain 95% of the variance and removed highly correlated features (>0.8) to reduce multicollinearity.

---

## 🧠 Modeling Approaches

### Approach A: Independent Window Classification (Classical ML)
The continuous signal was segmented into overlapping windows. Features were extracted per window, treating each as an independent sample.
* **Models:** Support Vector Machine (RBF Kernel), Random Forest, K-Nearest Neighbors (KNN).
* **Validation:** 10-Fold Stratified Cross-Validation.
* **Statistical Analysis:** Performed **Wilcoxon Signed-Rank Tests** to ensure statistically significant performance differences (p < 0.05).

### Approach B: Sequence Labeling (Deep Learning)
Modeled the temporal dependencies between consecutive samples, similar to Named Entity Recognition (NER) in NLP.
* **Bidirectional LSTM:** Captured long-term dependencies in both forward and backward directions.
* **Hybrid CNN-LSTM:** Utilized 1D Convolutional layers for automatic local feature extraction, fed into LSTM layers for temporal sequencing.
* **Architecture:** Included Dropout for regularization and TimeDistributed Dense layers for per-timestep prediction.

---

## 📊 Results & Performance

The **Hybrid CNN-LSTM** model demonstrated superior performance, proving that automatic feature extraction combined with temporal modeling outperforms manual feature engineering for this task.

| Approach | Model | Accuracy | Feature Type | Granularity |
| :--- | :--- | :--- | :--- | :--- |
| **B (Deep Learning)** | **CNN-LSTM** | **99.41%** | **Automatic (Spatial-Temporal)** | **Per Timestep** |
| A (Classical) | SVM (RBF) | 96.56% | Manual (Spectral/Temp) | Per Window |
| B (Deep Learning) | Bi-LSTM | 96.49% | Automatic (Temporal) | Per Timestep |
| A (Classical) | KNN | 96.34% | Manual (Spectral/Temp) | Per Window |
| A (Classical) | Random Forest | 93.54% | Manual (Spectral/Temp) | Per Window |

---

## 🛠 Tools & Technologies
* **Languages:** Python 3.x
* **Data Manipulation:** Pandas, NumPy, SciPy (Signal Processing)
* **Machine Learning:** Scikit-Learn (SVM, RF, KNN, PCA)
* **Deep Learning:** TensorFlow / Keras (LSTM, CNN)
* **Visualization:** Matplotlib, Seaborn

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/nadaahx/eeg-eye-state-detection.git](https://github.com/nadaahx/eeg-eye-state-detection.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the notebook:
    ```bash
    jupyter notebook eeg_analysis.ipynb
    ```
