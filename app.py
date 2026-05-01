import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense · Customer Loyalty Predictor",
    page_icon="✈️",
    layout="centered",
)

# ── NEW UI CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

:root {
    --bg: linear-gradient(135deg, #0f172a, #020617);
    --glass: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.08);
    --accent: #6366f1;
    --accent2: #22d3ee;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --green: #22c55e;
    --red: #ef4444;
    --radius: 16px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

[data-testid="stHeader"], footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 750px !important;
    padding-top: 2rem;
}

/* Hero */
.hero {
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 2.7rem;
    font-weight: 700;
    background: linear-gradient(90deg, #fff, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
}

/* Stats */
.stats-strip {
    display: flex;
    gap: 10px;
    margin: 1.5rem 0;
}
.stat-box {
    flex: 1;
    backdrop-filter: blur(12px);
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    text-align: center;
}
.stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent2);
}
.stat-lbl {
    font-size: 0.7rem;
    color: var(--muted);
}

/* Section headers */
.sec-header {
    color: var(--accent2);
    font-size: 0.75rem;
    letter-spacing: 2px;
    margin-top: 2rem;
    margin-bottom: 0.6rem;
}

/* Inputs */
div[data-testid="stNumberInput"], 
div[data-testid="stSelectbox"] {
    backdrop-filter: blur(10px);
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 6px;
}
input {
    color: white !important;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 14px;
    border: none;
    padding: 0.9rem;
    font-weight: 600;
    font-size: 1rem;
    color: white;
    transition: 0.3s;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.3);
}

/* Result */
.res-card {
    backdrop-filter: blur(14px);
    background: var(--glass);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    padding: 1.5rem;
    margin-top: 1.5rem;
    display: flex;
    gap: 15px;
}
.res-title {
    font-size: 1.2rem;
    font-weight: 600;
}
.res-sub {
    font-size: 0.9rem;
    color: var(--muted);
}
.res-stay { border-left: 4px solid var(--green); }
.res-churn { border-left: 4px solid var(--red); }

/* Progress bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    height: 8px;
    border-radius: 100px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 100px;
}

/* Footer */
.app-footer {
    color: var(--muted);
    text-align: center;
    font-size: 0.7rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Customer Churn Predictor</h1>
    <p>Predict whether a customer will stay or churn using AI</p>
</div>
""", unsafe_allow_html=True)

# Stats strip
st.markdown("""
<div class="stats-strip">
    <div class="stat-box"><div class="stat-val">88%</div><div class="stat-lbl">Accuracy</div></div>
    <div class="stat-box"><div class="stat-val">953</div><div class="stat-lbl">Records</div></div>
    <div class="stat-box"><div class="stat-val">6</div><div class="stat-lbl">Features</div></div>
    <div class="stat-box"><div class="stat-val">100</div><div class="stat-lbl">Trees</div></div>
</div>
""", unsafe_allow_html=True)

# Inputs
st.markdown('<div class="sec-header">PERSONAL INFO</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    Age = st.number_input("Age", 18, 100, 30)
with c2:
    AnnualIncomeClass = st.selectbox("Income", ["Low Income", "Middle Income", "High Income"])

st.markdown('<div class="sec-header">TRAVEL</div>', unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    FrequentFlyer = st.selectbox("Frequent Flyer", ["No", "Yes"])
with c4:
    ServicesOpted = st.number_input("Services (1-10)", 1, 10, 3)

st.markdown('<div class="sec-header">ENGAGEMENT</div>', unsafe_allow_html=True)

c5, c6 = st.columns(2)
with c5:
    AccountSyncedToSocialMedia = st.selectbox("Social Sync", ["No", "Yes"])
with c6:
    BookedHotelOrNot = st.selectbox("Booked Hotel", ["No", "Yes"])

# Predict
predict = st.button("Predict Churn")

if predict:
    ff  = 1 if FrequentFlyer == "Yes" else 0
    asm = 1 if AccountSyncedToSocialMedia == "Yes" else 0
    bh  = 1 if BookedHotelOrNot == "Yes" else 0
    inc = {"Low Income": 0, "Middle Income": 1, "High Income": 2}[AnnualIncomeClass]

    features = np.array([[Age, ff, inc, ServicesOpted, asm, bh]])
    pred     = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]

    if pred == 1:
        pct = int(proba[1] * 100)
        st.markdown(f"""
        <div class="res-card res-churn">
            <div>
                <div class="res-title">⚠️ High Churn Risk</div>
                <div class="res-sub">Customer likely to leave. Take action.</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct}%;background:#ef4444;"></div>
                </div>
                <b>{pct}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pct = int(proba[0] * 100)
        st.markdown(f"""
        <div class="res-card res-stay">
            <div>
                <div class="res-title">✅ Customer Will Stay</div>
                <div class="res-sub">Strong loyalty detected.</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct}%;background:#22c55e;"></div>
                </div>
                <b>{pct}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    ChurnSense · AI Project · Streamlit App
</div>
""", unsafe_allow_html=True)
