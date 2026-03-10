# ─────────────────────────────────────────
# GeneRisk AI — ML Pipeline
# Algorithm: Random Forest Classifier
# Dataset: Wisconsin Breast Cancer Dataset
# ─────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 50)
print("  GeneRisk AI — ML Training Pipeline")
print("=" * 50)

# ─────────────────────────────────────────
# STEP 1 — Load and Clean Data
# ─────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_csv('cancer_data.csv')

# Drop useless columns
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Drop missing values
df = df.dropna()

print(f"Dataset shape after cleaning: {df.shape}")
print(f"\nClass distribution:")
print(df['diagnosis'].value_counts())
print(f"\nM = Malignant (Cancer), B = Benign (No Cancer)")

# ─────────────────────────────────────────
# STEP 2 — Preprocessing
# ─────────────────────────────────────────
print("\n[2/7] Preprocessing data...")

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Encode M=1, B=0
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'models/label_encoder.pkl')

print(f"Classes: {le.classes_} → {le.transform(le.classes_)}")
print(f"Features: {list(X.columns)}")
print(f"Total features: {X.shape[1]}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.pkl')

# ─────────────────────────────────────────
# STEP 3 — Split Data
# ─────────────────────────────────────────
print("\n[3/7] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ─────────────────────────────────────────
# STEP 4 — Train Model
# ─────────────────────────────────────────
print("\n[4/7] Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model trained successfully!")

# ─────────────────────────────────────────
# STEP 5 — Evaluate
# ─────────────────────────────────────────
print("\n[5/7] Evaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
roc_auc   = roc_auc_score(y_test, y_prob)

print(f"\n{'='*40}")
print(f"  RESULTS")
print(f"{'='*40}")
print(f"  Accuracy:          {accuracy*100:.2f}%")
print(f"  ROC-AUC Score:     {roc_auc:.4f}")
print(f"  CV Score (5-fold): {cv_scores.mean()*100:.2f}%"
      f" (±{cv_scores.std()*100:.2f}%)")
print(f"{'='*40}")
print(f"\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Benign', 'Malignant']
))

# ─────────────────────────────────────────
# STEP 6 — Generate Plots
# ─────────────────────────────────────────
print("\n[6/7] Generating plots...")

# Plot 1 — Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign','Malignant'],
            yticklabels=['Benign','Malignant'])
plt.title('GeneRisk AI — Confusion Matrix',
          fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.close()
print("  Saved: confusion_matrix.png")

# Plot 2 — ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, color='#00B4D8', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GeneRisk AI — ROC Curve',
          fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=150)
plt.close()
print("  Saved: roc_curve.png")

# Plot 3 — Feature Importance (Top 15)
plt.figure(figsize=(10, 6))
feature_names = list(X.columns)
importances   = model.feature_importances_
indices       = np.argsort(importances)[::-1][:15]
plt.bar(range(len(indices)),
        importances[indices],
        color='#00B4D8')
plt.xticks(range(len(indices)),
           [feature_names[i] for i in indices],
           rotation=45, ha='right')
plt.title('GeneRisk AI — Top 15 Feature Importances',
          fontsize=14, fontweight='bold')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.close()
print("  Saved: feature_importance.png")

# Plot 4 — Cross Validation
plt.figure(figsize=(8, 5))
plt.bar(range(1, 6), cv_scores * 100, color='#00B4D8',
        edgecolor='white')
plt.axhline(y=cv_scores.mean()*100,
            color='red', linestyle='--',
            label=f'Mean: {cv_scores.mean()*100:.2f}%')
plt.xlabel('Fold')
plt.ylabel('Accuracy (%)')
plt.ylim([90, 100])
plt.title('GeneRisk AI — 5-Fold Cross Validation',
          fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('plots/cross_validation.png', dpi=150)
plt.close()
print("  Saved: cross_validation.png")

# ─────────────────────────────────────────
# STEP 7 — Save Model
# ─────────────────────────────────────────
print("\n[7/7] Saving model...")
joblib.dump(model, 'models/generisk_model.pkl')

metadata = {
    'model': 'Random Forest Classifier',
    'accuracy': round(float(accuracy), 4),
    'roc_auc': round(float(roc_auc), 4),
    'cv_score_mean': round(float(cv_scores.mean()), 4),
    'cv_score_std': round(float(cv_scores.std()), 4),
    'n_estimators': 200,
    'training_samples': len(X_train),
    'testing_samples': len(X_test),
    'features': feature_names,
    'classes': le.classes_.tolist()
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*50)
print("  TRAINING COMPLETE!")
print("="*50)
print(f"  Accuracy:    {accuracy*100:.2f}%")
print(f"  ROC-AUC:     {roc_auc:.4f}")
print(f"  CV Score:    {cv_scores.mean()*100:.2f}%")
print(f"  Model saved: models/generisk_model.pkl")
print(f"  Plots saved: plots/ folder")
print("="*50)