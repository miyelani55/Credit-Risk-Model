"""
tests/test_pipeline.py
----------------------
Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from data.preprocess import generate_synthetic_data, engineer_features


class TestDataGeneration:
    def test_synthetic_shape(self):
        df = generate_synthetic_data(n=1000)
        assert len(df) == 1000
        assert 'loan_status' in df.columns

    def test_default_rate_reasonable(self):
        df = generate_synthetic_data(n=5000)
        rate = df['loan_status'].mean()
        assert 0.05 < rate < 0.50, f"Default rate {rate:.2%} looks unrealistic"

    def test_no_nulls_in_numerics(self):
        df = generate_synthetic_data(n=1000)
        num_cols = df.select_dtypes(include=np.number).columns
        assert df[num_cols].isnull().sum().sum() == 0

    def test_income_positive(self):
        df = generate_synthetic_data(n=500)
        assert (df['annual_inc'] > 0).all()

    def test_int_rate_range(self):
        df = generate_synthetic_data(n=500)
        assert df['int_rate'].between(5, 28).all()


class TestFeatureEngineering:
    def setup_method(self):
        self.df = generate_synthetic_data(n=500)

    def test_derived_features_exist(self):
        out = engineer_features(self.df)
        for col in ['loan_to_income', 'monthly_payment_est',
                    'payment_to_income', 'risk_score']:
            assert col in out.columns, f"Missing feature: {col}"

    def test_loan_to_income_positive(self):
        out = engineer_features(self.df)
        assert (out['loan_to_income'] >= 0).all()

    def test_one_hot_encoding_done(self):
        out = engineer_features(self.df)
        assert 'home_ownership' not in out.columns
        assert 'purpose' not in out.columns
        # At least some dummy columns should appear
        ohe_cols = [c for c in out.columns if 'home_ownership_' in c or 'purpose_' in c]
        assert len(ohe_cols) > 0

    def test_no_inf_values(self):
        out = engineer_features(self.df)
        num = out.select_dtypes(include=np.number)
        assert not np.isinf(num.values).any()


class TestModelInterface:
    """Smoke tests for model interface — runs even without trained model."""

    def test_feature_count_stable(self):
        df = generate_synthetic_data(n=200)
        out = engineer_features(df)
        # Should have at least the base + derived features
        assert len(out.columns) >= 16

    def test_probability_range(self):
        """Logistic regression on tiny data just to check proba in [0,1]."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        df = generate_synthetic_data(n=500)
        out = engineer_features(df)
        TARGET = 'loan_status'
        X = out[[c for c in out.columns if c != TARGET and out[c].dtype != object]].fillna(0)
        y = out[TARGET]

        sc  = StandardScaler()
        Xs  = sc.fit_transform(X)
        lr  = LogisticRegression(max_iter=200)
        lr.fit(Xs, y)
        proba = lr.predict_proba(Xs)[:, 1]

        assert proba.min() >= 0.0
        assert proba.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
