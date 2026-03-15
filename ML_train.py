import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Simulate dataset: EAR sequences (10 frames) as features; 1=focused, 0=distracted (CSE206)
np.random.seed(42)
X = np.random.rand(2000, 10) * 0.5  # Eye Aspect Ratios
y = np.array([1 if np.mean(row) > 0.22 else 0 for row in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
print(classification_report(y_test, preds))

joblib.dump(knn, 'focus_knn.pkl')
print("Model saved as focus_knn.pkl")
