"""
train_model.py
──────────────
Run this script ONCE before launching the Streamlit app.
It trains an ANN (sklearn MLPClassifier) on the expanded dataset,
then saves the model and scaler to the model/ directory.

Usage:
    python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "STUDENT_PERFORMANCE_EXPANDED.csv"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "student_ann_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
SEED       = 42

# ── Risk classification ───────────────────────────────────────────────────────
def classify_risk(g):
    if g < 10:  return 0   # At Risk
    elif g < 14: return 1  # Average
    else:        return 2  # High Performer

# ── Load data ─────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")

df['risk_level'] = df['G3'].apply(classify_risk)

# ── Encode ────────────────────────────────────────────────────────────────────
print("🔧 Encoding features...")
le = LabelEncoder()
for c in df.columns:
    if df[c].dtype.name in ['object', 'string'] or 'string' in str(df[c].dtype).lower():
        df[c] = le.fit_transform(df[c].astype(str))

df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

X = df.drop(columns=['G3', 'risk_level']).astype(np.float32)
y = df['risk_level'].astype(int)

print(f"   Features : {X.shape[1]}")
print(f"   Samples  : {X.shape[0]}")
print(f"   Classes  : {dict(y.value_counts().sort_index())}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.20, random_state=SEED, stratify=y)

# ── Scale ─────────────────────────────────────────────────────────────────────
print("📐 Scaling features...")
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train MLP ─────────────────────────────────────────────────────────────────
print("🚀 Training ANN (MLPClassifier) — 100 epochs...")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=100,
    batch_size=32,
    random_state=SEED,
    early_stopping=False,       # Train all 100 iterations
    verbose=True,
    n_iter_no_change=100,       # Never stop early
    tol=1e-10,
)
model.fit(X_train_s, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n📊 Evaluating...")
y_pred = model.predict(X_test_s)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred, average='weighted')
f1   = f1_score(y_test, y_pred, average='weighted')

print("=" * 50)
print("  FINAL MODEL PERFORMANCE")
print("=" * 50)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")
print("=" * 50)
print(classification_report(y_test, y_pred,
      target_names=['At Risk', 'Average', 'High Performer']))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"✅ Model saved  → {MODEL_PATH}")
print(f"✅ Scaler saved → {SCALER_PATH}")
print("\n🎉 Done! Now run:  streamlit run app.py")
