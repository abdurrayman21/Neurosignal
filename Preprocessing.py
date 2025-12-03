import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. Load Data
df = pd.read_csv('data/data.csv')
# Columns 1 to 179 contain the signal data (X1...X178)
X = df.iloc[:, 1:179].values

# 2. Define Filter Parameters
fs = 178       # Sampling frequency (Hz)
lowcut = 0.5   # Low cut-off frequency
highcut = 40.0 # High cut-off frequency
order = 5      # Filter order

# Calculate coefficients once
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(order, [low, high], btype='band')

# 3. Apply Filter (Vectorized Fix)
# instead of looping row-by-row, we filter the whole matrix (axis=1)
print("Applying Bandpass Filter...")
X_filtered = filtfilt(b, a, X, axis=1)
print(f"Filtering Complete! New Shape: {X_filtered.shape}")

# 4. Visualize Before vs. After
row_idx = 0
plt.figure(figsize=(12, 6))

# Original Signal
plt.subplot(2, 1, 1)
plt.plot(X[row_idx, :], color='black', alpha=0.7, label='Original (Raw)')
plt.title("Original Signal (Row 0)")
plt.grid(True)
plt.legend()

# Filtered Signal
plt.subplot(2, 1, 2)
plt.plot(X_filtered[row_idx, :], color='blue', label='Filtered (0.5-40Hz)')
plt.title("Filtered Signal (Cleaned)")
plt.xlabel('Time Points')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('filtering_demo.png')
print("Comparison saved to 'filtering_demo.png'")

# 5. Save the Cleaned Data for ML
df_clean = pd.DataFrame(X_filtered)
df_clean['y'] = df['y'] # Add labels back
df_clean.to_csv('data/data_clean.csv', index=False)
print("Cleaned dataset saved as 'data/data_clean.csv'")