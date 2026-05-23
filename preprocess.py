"""
data/preprocess.py
------------------
Loads the Lending Club / Give Me Some Credit dataset and outputs
a clean train/test split ready for modelling.

Dataset: https://www.kaggle.com/datasets/wordsforthewise/lending-club
         OR use the synthetic generator below if you don't have the file yet.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 1. Synthetic data generator (runs without Kaggle)
# ─────────────────────────────────────────────
def generate_synthetic_data(n: int = 50_000) -> pd.DataFrame:
    """
    Generates a realistic synthetic credit dataset.
    Replace this with real Kaggle data for production use.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    age             = rng.integers(20, 70, n)
    income          = rng.lognormal(10.8, 0.6, n).astype(int)   # ~$50k median
    loan_amount     = rng.integers(1_000, 40_000, n)
    loan_term       = rng.choice([36, 60], n)
    int_rate        = rng.uniform(5.0, 28.0, n).round(2)
    dti             = rng.uniform(0, 45, n).round(2)             # debt-to-income
    open_acc        = rng.integers(1, 30, n)
    revol_util      = rng.uniform(0, 100, n).round(2)            # revolving util %
    delinq_2yrs     = rng.integers(0, 5, n)
    inq_last_6mths  = rng.integers(0, 8, n)
    pub_rec         = rng.integers(0, 3, n)
    emp_length      = rng.integers(0, 10, n)
    home_ownership  = rng.choice(['RENT', 'OWN', 'MORTGAGE'], n)
    loan_purpose    = rng.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement',
         'small_business', 'major_purchase', 'other'], n
    )

    # Default probability driven by real-world risk factors
    log_odds = (
        -3.5
        + 0.04  * (dti - 20)
        + 0.08  * (int_rate - 15)
        - 0.005 * (income / 1000 - 50)
        + 0.25  * delinq_2yrs
        + 0.15  * inq_last_6mths
        + 0.03  * (revol_util - 50)
        + 0.20  * pub_rec
        - 0.02  * emp_length
        + rng.normal(0, 0.5, n)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default      = (rng.random(n) < prob_default).astype(int)

    df = pd.DataFrame({
        'age':            age,
        'annual_inc':     income,
        'loan_amnt':      loan_amount,
        'term':           loan_term,
        'int_rate':       int_rate,
        'dti':            dti,
        'open_acc':       open_acc,
        'revol_util':     revol_util,
        'delinq_2yrs':    delinq_2yrs,
        'inq_last_6mths': inq_last_6mths,
        'pub_rec':         pub_rec,
        'emp_length':     emp_length,
        'home_ownership': home_ownership,
        'purpose':        loan_purpose,
        'loan_status':    default,          # 1 = default, 0 = paid
    })

    print(f"✅ Generated {n:,} synthetic records | Default rate: {default.mean():.2%}")
    return df


# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived ratios
    df['loan_to_income']      = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['monthly_payment_est'] = df['loan_amnt'] / df['term']
    df['payment_to_income']   = df['monthly_payment_est'] / (df['annual_inc'] / 12 + 1)
    df['risk_score']          = (df['int_rate'] * df['dti']) / 100  # crude composite

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=['home_ownership', 'purpose'], drop_first=True)

    return df


# ─────────────────────────────────────────────
# 3. Main pipeline
# ─────────────────────────────────────────────
def run_preprocessing(
    raw_path: str | None = None,
    save_dir: str = 'data'
) -> tuple:
    """
    Returns X_train, X_test, y_train, y_test and saves scaler.
    If raw_path is None, generates synthetic data.
    """
    os.makedirs(save_dir, exist_ok=True)

    if raw_path and os.path.exists(raw_path):
        print(f"📂 Loading data from {raw_path} ...")
        df = pd.read_csv(raw_path, low_memory=False)
        # Minimal clean for Lending Club format
        df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
        df['loan_status'] = (df['loan_status'] == 'Charged Off').astype(int)
        df.dropna(subset=['annual_inc', 'dti', 'int_rate'], inplace=True)
        df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '').astype(float)
        df['term']     = df['term'].astype(str).str.extract(r'(\d+)').astype(int)
    else:
        print("⚠️  No raw file found — using synthetic data (great for demos).")
        df = generate_synthetic_data()

    df = engineer_features(df)

    TARGET  = 'loan_status'
    FEATURES = [c for c in df.columns if c != TARGET and df[c].dtype != object]

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    joblib.dump(scaler,             f'{save_dir}/scaler.pkl')
    X_train_sc.to_parquet(          f'{save_dir}/X_train.parquet')
    X_test_sc.to_parquet(           f'{save_dir}/X_test.parquet')
    y_train.to_frame().to_parquet(  f'{save_dir}/y_train.parquet')
    y_test.to_frame().to_parquet(   f'{save_dir}/y_test.parquet')

    print(f"✅ Saved preprocessed data to '{save_dir}/'")
    print(f"   Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")
    print(f"   Default rate — Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    return X_train_sc, X_test_sc, y_train, y_test


if __name__ == '__main__':
    run_preprocessing()
