# ─────────────────────────────────────────
# GeneRisk AI — Advanced ML Pipeline
# Algorithms: RF, SVM, XGBoost, LightGBM, Neural Network
# Dataset: Wisconsin Breast Cancer (UCI)
# Explainability: SHAP
# ─────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
import joblib
import json

print("=" * 60)
print("   GeneRisk AI — Advanced ML Training Pipeline")
print("=" * 60)

# ─────────────────────────────────────────
# STEP 1 — Load Dataset
# ─────────────────────────────────────────
print("\n[1/8] Loading dataset...")

try:
    df = pd.read_csv('gene_expression.csv')
    print("Using gene expression dataset!")
except:
    df = pd.read_csv('cancer_data.csv')
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    print("Using Wisconsin cancer dataset!")

df = df.dropna()
print(f"Dataset shape: {df.shape}")

# ─────────────────────────────────────────
# STEP 2 — Preprocessing
# ─────────────────────────────────────────
print("\n[2/8] Preprocessing...")

# Fix — properly separate features and target
if 'diagnosis' in df.columns:
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
else:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'models/label_encoder.pkl')
print(f"Classes: {le.classes_} → {le.transform(le.classes_)}")
print(f"Class distribution: {dict(zip(le.classes_, np.bincount(y_encoded)))}")

# Feature selection — top 20
print(f"Original features: {X.shape[1]}")
k = min(20, X.shape[1])
selector = SelectKBest(f_classif, k=k)
X_selected = selector.fit_transform(X, y_encoded)
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected top {k} features: {selected_features}")
joblib.dump(selector, 'models/feature_selector.pkl')

with open('models/selected_features.json', 'w') as f:
    json.dump(selected_features, f)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
joblib.dump(scaler, 'models/scaler.pkl')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────
# STEP 3 — Train 5 Algorithms
# ─────────────────────────────────────────
print("\n[3/8] Training 5 ML algorithms...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=6,
        learning_rate=0.1, random_state=42,
        eval_metric='logloss', verbosity=0
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200, max_depth=6,
        learning_rate=0.1, random_state=42,
        verbose=-1
    ),
    'SVM': SVC(
        kernel='rbf', C=1.0,
        probability=True, random_state=42
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv)

    results[name] = {
        'model':    model,
        'accuracy': accuracy,
        'roc_auc':  roc_auc,
        'cv_mean':  cv_scores.mean(),
        'cv_std':   cv_scores.std(),
        'y_pred':   y_pred,
        'y_prob':   y_prob
    }

    print(f"  {name}: Accuracy={accuracy*100:.2f}% | "
          f"ROC-AUC={roc_auc:.4f} | "
          f"CV={cv_scores.mean()*100:.2f}%"
          f"(±{cv_scores.std()*100:.2f}%)")

# ─────────────────────────────────────────
# STEP 4 — Find Best Model
# ─────────────────────────────────────────
print("\n[4/8] Finding best model...")
best_name = max(results, key=lambda x: results[x]['roc_auc'])
best      = results[best_name]
print(f"\n  Best model: {best_name}")
print(f"  Accuracy:   {best['accuracy']*100:.2f}%")
print(f"  ROC-AUC:    {best['roc_auc']:.4f}")

joblib.dump(best['model'], 'models/generisk_model.pkl')
print(f"  Saved: models/generisk_model.pkl")

# ─────────────────────────────────────────
# STEP 5 — Algorithm Comparison Plot
# ─────────────────────────────────────────
print("\n[5/8] Generating comparison plots...")

names      = list(results.keys())
accuracies = [results[n]['accuracy']*100 for n in names]
roc_aucs   = [results[n]['roc_auc'] for n in names]
cv_means   = [results[n]['cv_mean']*100 for n in names]
colors     = ['#FF6B6B' if n == best_name else '#00B4D8' for n in names]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('GeneRisk AI — Algorithm Comparison',
             fontsize=16, fontweight='bold')

axes[0].bar(names, accuracies, color=colors)
axes[0].set_title('Test Accuracy (%)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([80, 102])
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(accuracies):
    axes[0].text(i, v+0.3, f'{v:.1f}%',
                 ha='center', fontsize=9)

axes[1].bar(names, roc_aucs, color=colors)
axes[1].set_title('ROC-AUC Score')
axes[1].set_ylabel('ROC-AUC')
axes[1].set_ylim([0.85, 1.02])
axes[1].tick_params(axis='x', rotation=30)
for i, v in enumerate(roc_aucs):
    axes[1].text(i, v+0.003, f'{v:.3f}',
                 ha='center', fontsize=9)

axes[2].bar(names, cv_means, color=colors)
axes[2].set_title('5-Fold CV Score (%)')
axes[2].set_ylabel('CV Accuracy (%)')
axes[2].set_ylim([80, 102])
axes[2].tick_params(axis='x', rotation=30)
for i, v in enumerate(cv_means):
    axes[2].text(i, v+0.3, f'{v:.1f}%',
                 ha='center', fontsize=9)

best_patch = mpatches.Patch(
    color='#FF6B6B', label=f'Best: {best_name}'
)
axes[0].legend(handles=[best_patch])
plt.tight_layout()
plt.savefig('plots/algorithm_comparison.png', dpi=150)
plt.close()
print("  Saved: algorithm_comparison.png")

# ROC Curves
plt.figure(figsize=(10, 7))
for name in names:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc = results[name]['roc_auc']
    lw  = 3 if name == best_name else 1.5
    plt.plot(fpr, tpr, lw=lw,
             label=f'{name} (AUC={auc:.3f})')
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GeneRisk AI — ROC Curves (All Models)',
          fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/roc_curves_comparison.png', dpi=150)
plt.close()
print("  Saved: roc_curves_comparison.png")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title(f'GeneRisk AI — Confusion Matrix ({best_name})',
          fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.close()
print("  Saved: confusion_matrix.png")

# Cross Validation
plt.figure(figsize=(10, 6))
cv_data = [cross_val_score(
    results[n]['model'], X_scaled, y_encoded, cv=cv
) * 100 for n in names]
plt.boxplot(cv_data, labels=names, patch_artist=True,
            boxprops=dict(facecolor='#00B4D8', alpha=0.7))
plt.title('GeneRisk AI — Cross Validation Distribution',
          fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('plots/cross_validation.png', dpi=150)
plt.close()
print("  Saved: cross_validation.png")

# ─────────────────────────────────────────
# STEP 6 — SHAP Explainability
# ─────────────────────────────────────────
print("\n[6/8] Generating SHAP explainability...")

try:
    rf_model  = results['Random Forest']['model']
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_test[:100])

    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv, X_test[:100],
        feature_names=selected_features,
        show=False
    )
    plt.title('GeneRisk AI — SHAP Feature Importance',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_summary.png")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv, X_test[:100],
        feature_names=selected_features,
        plot_type='bar', show=False
    )
    plt.title('GeneRisk AI — SHAP Feature Impact',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/shap_bar.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_bar.png")

except Exception as e:
    print(f"  SHAP warning: {e}")

# ─────────────────────────────────────────
# STEP 7 — Neural Network Diagram
# ─────────────────────────────────────────
print("\n[7/8] Generating Neural Network diagram...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#0A1628')
fig.patch.set_facecolor('#0A1628')

layers = [
    ('Input Layer\n20 features', 20, '#00B4D8'),
    ('Hidden Layer 1\n128 neurons\nReLU', 128, '#0096B7'),
    ('Hidden Layer 2\n64 neurons\nReLU', 64, '#0078A0'),
    ('Hidden Layer 3\n32 neurons\nReLU', 32, '#005A7A'),
    ('Output Layer\n2 classes\nSoftmax', 2, '#FF6B6B'),
]

x_pos = [1.5, 4.0, 6.5, 9.0, 11.5]

for i, ((label, size, color), x) in enumerate(
    zip(layers, x_pos)
):
    radius = min(size / 128 * 1.5 + 0.5, 1.8)
    circle = plt.Circle(
        (x, 5), radius, color=color, alpha=0.85
    )
    ax.add_patch(circle)
    ax.text(x, 5, label, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

    if i < len(layers) - 1:
        next_x = x_pos[i+1]
        ax.annotate(
            '', xy=(next_x - 0.3, 5),
            xytext=(x + radius, 5),
            arrowprops=dict(
                arrowstyle='->', color='white', lw=2
            )
        )

ax.text(7, 1.5,
        'GeneRisk Neural Network — MLP Classifier\n'
        'Input: Gene Features → Hidden Layers → Cancer Risk Output',
        ha='center', va='center',
        fontsize=10, color='#00B4D8')

ax.set_title(
    'GeneRisk AI — Neural Network Architecture',
    fontsize=14, fontweight='bold', color='white', pad=20
)
plt.tight_layout()
plt.savefig('plots/neural_network.png',
            dpi=150, bbox_inches='tight',
            facecolor='#0A1628')
plt.close()
print("  Saved: neural_network.png")

# ─────────────────────────────────────────
# STEP 8 — Save Metadata
# ─────────────────────────────────────────
print("\n[8/8] Saving metadata...")

metadata = {
    'best_model': best_name,
    'accuracy': round(float(best['accuracy']), 4),
    'roc_auc': round(float(best['roc_auc']), 4),
    'cv_score_mean': round(float(best['cv_mean']), 4),
    'cv_score_std': round(float(best['cv_std']), 4),
    'features': selected_features,
    'n_features': len(selected_features),
    'training_samples': len(X_train),
    'testing_samples': len(X_test),
    'all_results': {
        name: {
            'accuracy': round(float(r['accuracy']), 4),
            'roc_auc':  round(float(r['roc_auc']), 4),
            'cv_mean':  round(float(r['cv_mean']), 4),
            'cv_std':   round(float(r['cv_std']), 4),
        }
        for name, r in results.items()
    }
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*60)
print("   ADVANCED TRAINING COMPLETE!")
print("="*60)
print(f"\n  Best Model:  {best_name}")
print(f"  Accuracy:    {best['accuracy']*100:.2f}%")
print(f"  ROC-AUC:     {best['roc_auc']:.4f}")
print(f"\n  All Results:")
for name, r in results.items():
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:20s}: "
          f"Acc={r['accuracy']*100:.2f}% | "
          f"AUC={r['roc_auc']:.4f}{marker}")
print(f"\n  Plots saved: plots/ folder")
print(f"  Model saved: models/generisk_model.pkl")
print("="*60)