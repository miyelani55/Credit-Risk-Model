# 🏦 Credit Risk Scoring Model

> An end-to-end machine learning pipeline that predicts the **Probability of Default (PD)** for loan applicants — a core component of bank credit decisioning systems.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Highlights

| Metric | Value |
|---|---|
| **Champion Model** | XGBoost |
| **AUC-ROC** | ~0.80 |
| **KS Statistic** | ~0.47 |
| **Gini Coefficient** | ~0.60 |
| **Class Imbalance Handling** | SMOTE + scale_pos_weight |
| **Explainability** | SHAP waterfall plots |
| **Deployment** | Streamlit interactive dashboard |

---

## 🏗️ Architecture

```
credit-risk-model/
├── data/
│   └── preprocess.py        # Synthetic data gen + feature engineering
├── models/
│   └── train.py             # LR + RF + XGBoost, SHAP, model selection
├── app/
│   └── streamlit_app.py     # Interactive scoring dashboard
├── tests/
│   └── test_pipeline.py     # Pytest unit tests
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-model.git
cd credit-risk-model
pip install -r requirements.txt
```

### 2. Preprocess data
```bash
python data/preprocess.py
# Generates synthetic loan data (no Kaggle account needed)
# Swap in real Lending Club data by passing raw_path="your_file.csv"
```

### 3. Train models
```bash
python models/train.py
# Trains Logistic Regression, Random Forest, XGBoost
# Saves champion model + SHAP explainer
```

### 4. Run the dashboard
```bash
streamlit run app/streamlit_app.py
```

### 5. Run tests
```bash
pytest tests/ -v
```

---

## 📊 Model Comparison

| Model | AUC-ROC | KS Stat | Gini |
|---|---|---|---|
| Logistic Regression | ~0.74 | ~0.38 | ~0.48 |
| Random Forest | ~0.78 | ~0.44 | ~0.56 |
| **XGBoost** ⭐ | **~0.80** | **~0.47** | **~0.60** |

---

## 🔍 Key Features

### Feature Engineering
- **Loan-to-Income Ratio** — loan amount / annual income
- **Payment-to-Income Ratio** — estimated monthly payment / monthly income
- **Risk Score** — composite `int_rate × dti / 100`
- One-hot encoding of home ownership & loan purpose

### Class Imbalance Handling
- **SMOTE** (Synthetic Minority Oversampling) on Logistic Regression
- **scale_pos_weight** for XGBoost
- **class_weight='balanced'** for Random Forest

### Explainability (SHAP)
SHAP (SHapley Additive exPlanations) values explain *why* each applicant received their score — critical for regulatory compliance (SR 11-7, model risk management).

---

## 📡 Using Real Data (Lending Club)

1. Download from [Kaggle: Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Save as `data/raw/lending_club.csv`
3. Update preprocess call:
```python
from data.preprocess import run_preprocessing
run_preprocessing(raw_path='data/raw/lending_club.csv')
```

---

## 💼 Why This Matters for Finance

Credit risk models are core to:
- **Loan underwriting** — approve/reject decisions
- **Pricing** — setting interest rates commensurate with risk
- **Regulatory capital** — Basel III requires PD, LGD, EAD estimates
- **IFRS 9 / CECL** — expected credit loss provisioning

---

## 🧰 Tech Stack

- **pandas / numpy** — data wrangling
- **scikit-learn** — LR, RF, preprocessing
- **xgboost** — gradient boosting
- **imbalanced-learn** — SMOTE
- **shap** — model explainability
- **streamlit + plotly** — dashboard
- **pytest** — unit testing

---

## 📄 License

MIT — free to use in your own portfolio.

---

*Built as a data science portfolio project. Not intended as financial advice.*
