# 🧠 NeuroSignal: EEG Seizure Detection System

A Machine Learning Pipeline for Automated Epilepsy Diagnosis using Spectral Analysis.

---

## 🏥 Project Overview

**NeuroSignal** is a Clinical Decision Support System (CDSS) designed to help neurologists distinguish between **Ictal (Seizure)** and **Inter-ictal (Normal)** EEG activity.

Unlike standard tabular ML projects, this system processes **raw physiological time-series EEG signals**.  
It applies **Digital Signal Processing (DSP)** to clean noise and extract biological features (Alpha, Beta, Delta waves), which are then used by a **Random Forest Classifier**.

---

## 🎯 Objective

To bridge the gap between **raw neurophysiology** and **clinical diagnosis**, showing that automated pipelines can detect seizures with **>98% accuracy**.

---

## 📸 The Dashboard (Doctor's View)

The system provides a real-time interface for clinicians to:

- Visualize EEG signals  
- Analyze frequency bands  
- Receive an automated seizure probability  

> *(Place your Streamlit dashboard screenshot here)*

---

## ⚙️ Technical Architecture

### **1. Data Acquisition & Physics**

- **Dataset:** UCI Epileptic Seizure Recognition Data Set (University of Bonn)  
- **Signal:** Single-channel EEG sampled at **178 Hz**  
- **Challenge:** Distinguishing chaotic seizure spikes from normal rhythmic brain waves  

---

### **2. Signal Preprocessing (DSP)**

Raw EEG data contains noise: muscle movement, sweat artifacts, and electrical hum.

A **Butterworth Bandpass Filter (0.5–40 Hz)** was applied using `scipy.signal`.

- **High-Pass (0.5 Hz):** Removes drift + sweat artifacts  
- **Low-Pass (40 Hz):** Removes mains hum (50/60Hz) + muscle noise  

---

### **3. Feature Engineering (Neuro-Biomarkers)**

Rather than feeding raw EEG into ML, domain-specific features were extracted.

#### **Time-Domain Features:**
- Mean  
- Variance  
- Skewness  
- Kurtosis (captures seizure “spikiness”)  

#### **Frequency-Domain (Spectral Analysis):**  
Using **Welch’s Method (FFT)** to derive brainwave energy in:

| Band | Range | Meaning |
|------|--------|---------|
| **Delta** | 0.5–4 Hz | Deep sleep / Pathology |
| **Theta** | 4–8 Hz | Drowsiness |
| **Alpha** | 8–13 Hz | Relaxed wakefulness |
| **Beta**  | 13–30 Hz | Active thinking / Anxiety |

---

### **4. Machine Learning Classification**

- **Model:** Random Forest (n_estimators=100)  
- **Why:** Robust, low-overfitting risk, strong interpretability (feature importance)  

---

## 📊 Clinical Results

- **Accuracy:** 98.2%  
- **Sensitivity (Recall):** 95% — critical for minimizing missed seizures  
- **Specificity:** 99% — reduces false alarms  

Confusion Matrix is available in the training output.

---

## 🚀 Installation & Usage

### **Prerequisites**
- Python **3.8+**
- pip

---

### **Clone the Repository**
```bash
git clone https://github.com/your-username/NeuroSignal.git
cd NeuroSignal
