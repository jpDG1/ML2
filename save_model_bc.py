"""
save_model_bc.py — URUCHOM RAZ przed app.py
Trenuje najlepszy model Breast Cancer Wisconsin i zapisuje go
jako 'najlepszy_model_breast_cancer.pkl' dla interfejsu Gradio.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

print("=" * 50)
print("  Zapisywanie modelu Breast Cancer → .pkl")
print("=" * 50)

# Dane Wisconsin (wbudowane w sklearn, nie UCI)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_raw = data.target  # 0=malignant, 1=benign

# Mapowanie na 4 (złośliwy) i 2 (łagodny) — spójne z zajęciami 4
y = pd.Series(y_raw).map({0: 4, 1: 2})

# Wybierz tylko 9 cech analogicznych do UCI Wisconsin
FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter',
    'mean area', 'mean smoothness', 'mean compactness',
    'mean concavity', 'mean concave points', 'mean symmetry'
]
# Przeskaluj do zakresu 1-10 dla interfejsu suwaków
X_sel = X[FEATURES].copy()
for col in FEATURES:
    col_min, col_max = X_sel[col].min(), X_sel[col].max()
    X_sel[col] = ((X_sel[col] - col_min) / (col_max - col_min) * 9 + 1).round().astype(int)

FEATURE_NAMES = [f.replace(' ', '_') for f in FEATURES]
X_sel.columns = FEATURE_NAMES

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline z najlepszym modelem (Random Forest — zwykle wygrywa CV)
pipelines = {
    "Random Forest": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
    "SVM": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, probability=True, random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "K-NN": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=7))
    ]),
    "MLP": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    "Naive Bayes": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', GaussianNB())
    ]),
}

# CV 5-fold — wybierz najlepszy
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_name, best_score, best_pipe = "", 0, None

print(f"\n{'Model':22} | {'CV Acc (5-fold)':>15}")
print("-" * 42)
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_sel, y, cv=kf, scoring='accuracy')
    mean_score = scores.mean()
    print(f"{name:22} | {mean_score:.4f} ± {scores.std():.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_name = name
        best_pipe = pipe

# Trenuj finalny model na całym zbiorze treningowym
best_pipe.fit(X_train, y_train)
test_acc = best_pipe.score(X_test, y_test)

print("-" * 42)
print(f"Zwycięzca:  {best_name}")
print(f"CV Acc:     {best_score:.4f}")
print(f"Test Acc:   {test_acc:.4f}")

# Zapis
payload = {
    'model': best_pipe,
    'features': FEATURE_NAMES,
    'model_name': best_name,
    'cv_accuracy': best_score,
    'test_accuracy': test_acc,
}
joblib.dump(payload, 'najlepszy_model_breast_cancer.pkl')
print("\n✅ Zapisano: najlepszy_model_breast_cancer.pkl")
print("   Możesz teraz uruchomić: python app.py")