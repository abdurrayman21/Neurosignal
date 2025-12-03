import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv('data/features.csv')

#Split into X (Features) and y (Target)
X = df.drop('label', axis=1)
y = df['label']
#Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Train the Model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#evaluate
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

#Visualize Confusion Matrix (The "Clinical Proof")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal', 'Seizure'],
            yticklabels=['Normal', 'Seizure'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Seizure Detection')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion Matrix saved to 'confusion_matrix.png'")

joblib.dump(model, 'seizure_model.pkl')
print("Model saved as 'seizure_model.pkl'")