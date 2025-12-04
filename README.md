🧠 NeuroSignal: EEG Seizure Detection System

A Machine Learning Pipeline for Automated Epilepsy Diagnosis using Spectral Analysis.

🏥 Project Overview

NeuroSignal is a Clinical Decision Support System (CDSS) designed to assist neurologists in distinguishing between Ictal (Seizure) and Inter-ictal (Normal) brain activity.

Unlike standard tabular ML projects, this system processes raw physiological time-series data. It utilizes Digital Signal Processing (DSP) techniques to filter noise and extract biological features (Alpha, Beta, Delta waves) before feeding them into a Random Forest Classifier.

🎯 Objective

To bridge the gap between raw biological signals and clinical diagnosis, proving that automated pipelines can detect seizures with >98% accuracy.

📸 The Dashboard (Doctor's View)

The system provides a real-time interface for clinicians to visualize signals, analyze frequency bands, and receive an automated diagnostic probability.

(Place your screenshot of the Streamlit Dashboard here)

⚙️ Technical Architecture

The pipeline follows a standard Neuro-Data Science workflow:

1. Data Acquisition & Physics

Dataset: UCI Epileptic Seizure Recognition Data Set (University of Bonn).

Signal: Single-channel EEG recordings sampled at 178 Hz.

Challenge: Separating chaotic seizure spikes from normal rhythmic brain activity.

2. Signal Preprocessing (DSP)

Raw EEG data is noisy (muscle movement, electrical mains hum). I applied a Butterworth Bandpass Filter (0.5 Hz – 40 Hz) using scipy.signal.

High-Pass (0.5 Hz): Removes DC drift and sweating artifacts.

Low-Pass (40 Hz): Removes electrical line noise (50/60Hz) and muscle tension.

3. Feature Engineering (The "Biomarkers")

Instead of using raw time-series data, I extracted domain-specific neuro-features:

Time-Domain: Statistical moments (Mean, Variance, Skewness) and Kurtosis (to detect the "spikiness" of seizure signals).

Frequency-Domain (Spectral Analysis): Used Welch’s Method (FFT) to decompose signals into brainwave bands:

Delta (0.5-4 Hz): Deep sleep / Pathological slowing.

Theta (4-8 Hz): Drowsiness.

Alpha (8-13 Hz): Relaxed wakefulness.

Beta (13-30 Hz): Active thinking / High anxiety.

4. Machine Learning Classification

Model: Random Forest Classifier (n_estimators=100).

Reasoning: Chosen for its robustness against overfitting on tabular data and ability to rank Feature Importance (interpretability).

📊 Clinical Results

The model achieved 98.2% Accuracy on the held-out test set.

Confusion Matrix:

Sensitivity (Recall): 95% (Crucial for minimizing False Negatives in a medical context).

Specificity: 99% (Minimizing False Alarms).

🚀 Installation & Usage

Prerequisites

Python 3.8+

Pip

1. Clone the Repository

git clone [https://github.com/your-username/NeuroSignal.git](https://github.com/your-username/NeuroSignal.git)
cd NeuroSignal


2. Install Dependencies

pip install pandas numpy scikit-learn matplotlib seaborn scipy streamlit plotly joblib


3. Run the Dashboard

To launch the Clinical Interface:

streamlit run app.py


4. Retrain the Model (Optional)

To regenerate features and retrain the Random Forest:

python preprocess.py
python feature_extraction.py
python train_model.py


📂 Project Structure

NeuroSignal/
├── data/
│   ├── data.csv            # Raw dataset
│   ├── data_clean.csv      # Filtered signals (0.5-40Hz)
│   └── features.csv        # Extracted Biomarkers
├── app.py                  # Streamlit Dashboard (Frontend)
├── preprocess.py           # Butterworth Filter logic
├── feature_extraction.py   # PSD and Statistical extraction
├── train_model.py          # ML Training & Evaluation
├── seizure_model.pkl       # Serialized Model
└── README.md               # Documentation


🔬 Future Work

Deep Learning: Implementing 1D-CNNs (Convolutional Neural Networks) to learn features directly from raw waveforms without manual extraction.

Hardware Integration: Connecting the pipeline to a live OpenBCI headset for real-time streaming analysis.

Author: [Raja Abdulrahman]
