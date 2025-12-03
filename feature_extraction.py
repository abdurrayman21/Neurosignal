import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# 1. Load the Cleaned Data
print("Loading data...")
df = pd.read_csv('data/data_clean.csv')

# 2. Separate Labels and Signals SAFELY
# First, extract the target 'y' if it exists
if 'y' in df.columns:
    labels = df['y']
    # Drop 'y' so it isn't treated as a signal
    df_signals = df.drop('y', axis=1)
else:
    # If 'y' is missing for some reason, we can't label data.
    raise ValueError("Critical: Column 'y' not found in data_clean.csv")

# CRITICAL FIX: Ensure we ONLY take numbers.
# This removes any accidental ID columns (strings) that caused your error.
data = df_signals.select_dtypes(include=[np.number]).values

fs = 178  # Sampling Frequency

print(f"Data Loaded. Shape: {data.shape} (Should be roughly 11500 x 178)")


# 3. Define Helper Function for Frequency Bands
def get_spectral_features(row_data, fs):
    # 'welch' method estimates power at different frequencies
    freqs, psd = welch(row_data, fs, nperseg=fs)

    # Calculate total power in each band
    delta = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
    theta = np.sum(psd[(freqs >= 4) & (freqs < 8)])
    alpha = np.sum(psd[(freqs >= 8) & (freqs < 13)])
    beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])

    return [delta, theta, alpha, beta]


# 4. Extract Features
print("Extracting features... this will take 10-20 seconds.")

features_list = []

for i in range(len(data)):
    row = data[i]

    # Safety Check: If row is empty or not numeric, skip
    if len(row) == 0: continue

    # A. Time Domain Features
    mean_val = np.mean(row)
    std_val = np.std(row)
    max_val = np.max(row)
    min_val = np.min(row)
    skew_val = skew(row)
    kurt_val = kurtosis(row)

    # B. Frequency Domain Features
    delta, theta, alpha, beta = get_spectral_features(row, fs)

    features_list.append([mean_val, std_val, max_val, min_val, skew_val, kurt_val,
                          delta, theta, alpha, beta])

# 5. Save Final Dataset
columns = ['mean', 'std', 'max', 'min', 'skew', 'kurtosis', 'delta', 'theta', 'alpha', 'beta']
df_features = pd.DataFrame(features_list, columns=columns)

# Convert labels: 1=Seizure, Other=Normal(0)
df_features['label'] = labels.apply(lambda x: 1 if x == 1 else 0)

df_features.to_csv('data/features.csv', index=False)

print(f"Feature Extraction Complete!")
print(f"New Shape: {df_features.shape}")
print(df_features.head())