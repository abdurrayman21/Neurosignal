import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/data.csv')

#seprating signaling data (X1-X178)
signal_data = df.iloc[:, 1:179]
labels = df['y']

#transforming labels to binary
binary_labels = labels.apply(lambda x:1 if x==1 else 0)
# selecting one sample of each for comparison
seizure_index = binary_labels[binary_labels == 1].index[0]
normal_index = binary_labels[binary_labels == 0].index[0]

seizure_signal = signal_data.iloc[seizure_index].values
normal_signal = signal_data.iloc[normal_index].values

plt.figure(figsize=(14, 5))

# Plot Seizure
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.plot(seizure_signal, label='Seizure Activity', color='red')
plt.title('Ictal Activity (Seizure)', fontsize=14, fontweight='bold')
plt.xlabel('Time (samples)', fontsize=12)
plt.ylabel('Microvolts (uV)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Normal
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.plot(normal_signal, label='Normal Activity', color='green')
plt.title('Inter-ictal Activity (Normal)', fontsize=14, fontweight='bold')
plt.xlabel('Time (samples)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot (Great for your GitHub README later)
plt.tight_layout()
plt.savefig('eeg_comparison.png')
plt.show()

print('generated')