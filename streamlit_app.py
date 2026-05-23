"""
app/streamlit_app.py
--------------------
Interactive Credit Risk Scoring Dashboard.
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shap
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-high   { color: #e74c3c; font-weight: 700; font-size: 2rem; }
    .risk-medium { color: #f39c12; font-weight: 700; font-size: 2rem; }
    .risk-low    { color: #27ae60; font-weight: 700; font-size: 2rem; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.dirname(__file__))
    model    = joblib.load(os.path.join(base, 'models/champion_model.pkl'))
    scaler   = joblib.load(os.path.join(base, 'data/scaler.pkl'))
    features = joblib.load(os.path.join(base, 'models/feature_names.pkl'))
    return model, scaler, features

try:
    model, scaler, feature_names = load_artifacts()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False


# ─── Sidebar: Loan application form ────────────────────────────────────────────
st.sidebar.header("📋 Loan Application")

age         = st.sidebar.slider("Age",                   18, 70,  35)
annual_inc  = st.sidebar.number_input("Annual Income ($)", 10_000, 500_000, 55_000, step=1_000)
loan_amnt   = st.sidebar.number_input("Loan Amount ($)",    1_000,  50_000, 12_000, step=500)
term        = st.sidebar.selectbox("Loan Term (months)", [36, 60])
int_rate    = st.sidebar.slider("Interest Rate (%)",         5.0,  30.0, 13.5, 0.1)
dti         = st.sidebar.slider("Debt-to-Income Ratio (%)",  0.0,  50.0, 18.0, 0.1)
open_acc    = st.sidebar.slider("Open Credit Accounts",       1,    30,   8)
revol_util  = st.sidebar.slider("Revolving Utilisation (%)",  0.0, 100.0, 45.0, 0.5)
delinq_2yrs = st.sidebar.slider("Delinquencies (2 yrs)",      0,    5,    0)
inq_last_6  = st.sidebar.slider("Credit Inquiries (6 mths)",  0,    8,    1)
pub_rec     = st.sidebar.slider("Public Records",             0,    3,    0)
emp_length  = st.sidebar.slider("Employment Length (yrs)",    0,    10,   5)
home_own    = st.sidebar.selectbox("Home Ownership",
                                   ['RENT', 'OWN', 'MORTGAGE'])
purpose     = st.sidebar.selectbox("Loan Purpose",
                                   ['debt_consolidation', 'credit_card',
                                    'home_improvement', 'small_business',
                                    'major_purchase', 'other'])

predict_btn = st.sidebar.button("🔍 Score This Applicant", type="primary")


# ─── Feature builder ────────────────────────────────────────────────────────────
def build_features(feature_names: list) -> pd.DataFrame:
    """Construct a 1-row dataframe matching training feature set."""
    raw = {
        'age':            age,
        'annual_inc':     annual_inc,
        'loan_amnt':      loan_amnt,
        'term':           term,
        'int_rate':       int_rate,
        'dti':            dti,
        'open_acc':       open_acc,
        'revol_util':     revol_util,
        'delinq_2yrs':    delinq_2yrs,
        'inq_last_6mths': inq_last_6,
        'pub_rec':         pub_rec,
        'emp_length':     emp_length,
    }
    raw['loan_to_income']      = loan_amnt / (annual_inc + 1)
    raw['monthly_payment_est'] = loan_amnt / term
    raw['payment_to_income']   = raw['monthly_payment_est'] / (annual_inc / 12 + 1)
    raw['risk_score']          = (int_rate * dti) / 100

    # One-hot encode (match training columns)
    for col in feature_names:
        if col not in raw:
            raw[col] = 0

    # Set the selected home ownership & purpose flags
    ho_col = f'home_ownership_{home_own}'
    pp_col = f'purpose_{purpose}'
    if ho_col in feature_names: raw[ho_col] = 1
    if pp_col in feature_names: raw[pp_col] = 1

    df = pd.DataFrame([raw])[feature_names]
    return df


# ─── Main layout ────────────────────────────────────────────────────────────────
st.title("🏦 Credit Risk Scoring Dashboard")
st.caption("ML-powered Probability of Default (PD) model | Built with XGBoost + SHAP")

if not MODEL_LOADED:
    st.error("⚠️  Model not found. Please run the training pipeline first:")
    st.code("python data/preprocess.py\npython models/train.py")
    st.stop()

# ─── Scoring ───────────────────────────────────────────────────────────────────
X_input = build_features(feature_names)
X_scaled = pd.DataFrame(scaler.transform(X_input),
                         columns=feature_names)

prob_default = float(model.predict_proba(X_scaled)[0, 1])
prob_pay     = 1 - prob_default

# Risk band
if prob_default < 0.15:
    risk_label = "LOW RISK ✅";  risk_class = "risk-low"
elif prob_default < 0.35:
    risk_label = "MEDIUM RISK ⚠️"; risk_class = "risk-medium"
else:
    risk_label = "HIGH RISK ❌";  risk_class = "risk-high"

# ─── Top KPIs ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Probability of Default", f"{prob_default:.1%}")
with col2:
    st.metric("Probability of Payoff",  f"{prob_pay:.1%}")
with col3:
    credit_score_est = int(850 - prob_default * 500)
    st.metric("Est. Credit Score",      str(credit_score_est))
with col4:
    st.markdown(f"<div class='metric-card'><p>Risk Band</p>"
                f"<p class='{risk_class}'>{risk_label}</p></div>",
                unsafe_allow_html=True)

st.divider()

# ─── Charts ────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Default Probability Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_default * 100,
        number={'suffix': '%', 'font': {'size': 32}},
        gauge={
            'axis':  {'range': [0, 100], 'ticksuffix': '%'},
            'bar':   {'color': "#2c3e50"},
            'steps': [
                {'range': [0,  15], 'color': '#2ecc71'},
                {'range': [15, 35], 'color': '#f39c12'},
                {'range': [35, 100],'color': '#e74c3c'},
            ],
            'threshold': {'line': {'color': '#2c3e50', 'width': 4},
                          'thickness': 0.75, 'value': prob_default * 100}
        },
        title={'text': "PD Score"},
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_b:
    st.subheader("Risk Factor Radar")
    categories = ['DTI', 'Int Rate', 'Revol Util',
                  'Delinquencies', 'Credit Inq.', 'Pub Records']
    vals = [
        min(dti / 50,        1.0),
        min((int_rate-5)/25, 1.0),
        min(revol_util/100,  1.0),
        min(delinq_2yrs/5,   1.0),
        min(inq_last_6/8,    1.0),
        min(pub_rec/3,       1.0),
    ]
    vals += [vals[0]]
    categories += [categories[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=vals, theta=categories, fill='toself',
        fillcolor='rgba(231,76,60,0.2)', line_color='#e74c3c'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        height=300, margin=dict(t=40, b=10)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─── SHAP Waterfall ────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explainability — Why this score?")
st.caption("SHAP values show how much each feature pushed the default probability up (red) or down (blue).")

try:
    explainer  = joblib.load('models/shap_explainer.pkl')
    shap_vals  = explainer.shap_values(X_scaled)
    shap_series = pd.Series(dict(zip(feature_names, shap_vals[0]))).sort_values()

    top_n = 12
    shap_top = shap_series.abs().nlargest(top_n).index
    shap_plot_data = shap_series[shap_top].sort_values()

    colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_plot_data.values]
    fig_shap = go.Figure(go.Bar(
        x=shap_plot_data.values,
        y=shap_plot_data.index,
        orientation='h',
        marker_color=colors,
    ))
    fig_shap.update_layout(
        xaxis_title="SHAP Value (impact on default probability)",
        height=400, margin=dict(l=20, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_shap, use_container_width=True)
except FileNotFoundError:
    st.info("SHAP explainer not found — run training first to enable explainability.")

# ─── Loan summary table ────────────────────────────────────────────────────────
with st.expander("📄 Full Application Details"):
    summary = {
        "Applicant Age":         age,
        "Annual Income":         f"${annual_inc:,}",
        "Loan Amount":           f"${loan_amnt:,}",
        "Term":                  f"{term} months",
        "Interest Rate":         f"{int_rate}%",
        "Debt-to-Income":        f"{dti}%",
        "Open Accounts":         open_acc,
        "Revolving Utilisation": f"{revol_util}%",
        "Delinquencies (2yr)":   delinq_2yrs,
        "Credit Inquiries (6m)": inq_last_6,
        "Public Records":        pub_rec,
        "Employment Length":     f"{emp_length} yrs",
        "Home Ownership":        home_own,
        "Loan Purpose":          purpose,
        "── SCORES ──":          "──────────",
        "PD Score":              f"{prob_default:.4f}",
        "Est. Credit Score":     credit_score_est,
        "Risk Band":             risk_label.replace("✅","").replace("⚠️","").replace("❌","").strip(),
    }
    st.table(pd.DataFrame.from_dict(summary, orient='index', columns=['Value']))

st.divider()
st.caption("Built as a portfolio project. Model trained on synthetic data. Not financial advice.")
