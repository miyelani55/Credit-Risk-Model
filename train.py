"""
models/train.py
---------------
Trains three models:
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. XGBoost              (champion model)

Evaluates with AUC-ROC, precision-recall, KS statistic, and Gini.
Saves the best model + SHAP explainer.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    roc_auc_score, classification_report, roc_curve,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling  import SMOTE
import xgboost                as xgb
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42


# ─────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────
def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov stat — key metric in credit risk."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini(y_true, y_prob) -> float:
    """Gini = 2 * AUC - 1."""
    return 2 * roc_auc_score(y_true, y_prob) - 1


def evaluate(name, model, X_test, y_test) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    ks     = ks_statistic(y_test, y_prob)
    gi     = gini(y_test, y_prob)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  Avg Prec: {ap:.4f}")
    print(f"  KS Stat : {ks:.4f}")
    print(f"  Gini    : {gi:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Paid','Default']))

    return dict(name=name, model=model, auc=auc, ap=ap, ks=ks, gini=gi,
                y_prob=y_prob, y_pred=y_pred)


# ─────────────────────────────────────────────
# Plot helpers (saved to models/)
# ─────────────────────────────────────────────
def plot_roc_curves(results: list, save_path='models/roc_curves.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for r, c in zip(results, colors):
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
        ax.plot(fpr, tpr, color=c, lw=2,
                label=f"{r['name']}  (AUC={r['auc']:.3f}, Gini={r['gini']:.3f})")
    ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Credit Risk Models'); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"  📊 Saved {save_path}")


def plot_precision_recall(results: list, save_path='models/pr_curves.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for r, c in zip(results, colors):
        prec, rec, _ = precision_recall_curve(r['y_true'], r['y_prob'])
        ax.plot(rec, prec, color=c, lw=2,
                label=f"{r['name']}  (AP={r['ap']:.3f})")
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves'); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"  📊 Saved {save_path}")


def plot_shap(model, X_test, save_path='models/shap_summary.png'):
    print("\n  🔍 Computing SHAP values (this may take ~30s)...")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test.iloc[:500])      # sample for speed

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test.iloc[:500], show=False, plot_size=None)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved {save_path}")

    # Save explainer for Streamlit app
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    return explainer


# ─────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────
def train():
    os.makedirs('models', exist_ok=True)

    print("📂 Loading preprocessed data...")
    X_train = pd.read_parquet('data/X_train.parquet')
    X_test  = pd.read_parquet('data/X_test.parquet')
    y_train = pd.read_parquet('data/y_train.parquet').squeeze()
    y_test  = pd.read_parquet('data/y_test.parquet').squeeze()

    # Handle class imbalance with SMOTE
    print(f"\n⚖️  Class balance before SMOTE: {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"   After SMOTE          : {pd.Series(y_res).value_counts().to_dict()}")

    # ── Model 1: Logistic Regression ──────────
    print("\n🏋️  Training Logistic Regression...")
    lr = LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(X_res, y_res)

    # ── Model 2: Random Forest ─────────────────
    print("🏋️  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=50,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)   # RF handles imbalance natively

    # ── Model 3: XGBoost ──────────────────────
    print("🏋️  Training XGBoost...")
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,   # handles imbalance
        eval_metric='auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluate all ──────────────────────────
    print("\n📈 EVALUATION RESULTS")
    results = []
    for name, mdl in [('Logistic Regression', lr),
                       ('Random Forest',       rf),
                       ('XGBoost',             xgb_model)]:
        r = evaluate(name, mdl, X_test, y_test)
        r['y_true'] = y_test.values
        results.append(r)

    # ── Plots ─────────────────────────────────
    print("\n📊 Generating plots...")
    plot_roc_curves(results)
    plot_precision_recall(results)

    # SHAP on champion model (XGBoost)
    plot_shap(xgb_model, X_test)

    # ── Save champion model ───────────────────
    best = max(results, key=lambda r: r['auc'])
    print(f"\n🏆 Champion: {best['name']} (AUC={best['auc']:.4f})")
    joblib.dump(best['model'], 'models/champion_model.pkl')

    # Save feature names for app
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("✅ Saved models/champion_model.pkl")

    # ── Summary table ─────────────────────────
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ('model','y_prob','y_pred','y_true')}
        for r in results
    ]).set_index('name')
    summary = summary[['auc','ap','ks','gini']].round(4)
    print(f"\n{'═'*50}")
    print("MODEL COMPARISON SUMMARY")
    print('═'*50)
    print(summary.to_string())
    summary.to_csv('models/model_summary.csv')


if __name__ == '__main__':
    train()
