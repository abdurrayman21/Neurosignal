import pandas as pd
import numpy as np

csv_path = 'data/data.csv'
try:
    df =pd.read_csv(csv_path)
    print("success")
except:
    print("error")
    exit()

print(f"\n data pverview")
print(f"Shape: {df.shape} (Rows, Column)")
print(f"\n data distribution")
print(df['y'].value_counts().sort_index())
print(df.head())

