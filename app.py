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

# ── CSS ───────────────────────────────────────────────────────────────────────

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
```



# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✈️ &nbsp;Travel Industry &nbsp;·&nbsp; Random Forest AI</div>
    <h1>Customer Churn<br>Predictor</h1>
    <p>Enter a customer profile and instantly discover whether<br>
       they are likely to stay loyal or churn.</p>
</div>
""", unsafe_allow_html=True)

# Stats strip
st.markdown("""
<div class="stats-strip">
    <div class="stat-box"><div class="stat-val">88%</div><div class="stat-lbl">Model Accuracy</div></div>
    <div class="stat-box"><div class="stat-val">953</div><div class="stat-lbl">Training Records</div></div>
    <div class="stat-box"><div class="stat-val">6</div><div class="stat-lbl">Input Features</div></div>
    <div class="stat-box"><div class="stat-val">100</div><div class="stat-lbl">Decision Trees</div></div>
</div>
""", unsafe_allow_html=True)


# ── Section 1 · Personal Info ────────────────────────────────────────────────
st.markdown('<div class="sec-header">👤 &nbsp;Personal Information</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="medium")
with c1:
    Age = st.number_input("Customer Age", min_value=18, max_value=100, value=30, step=1)
with c2:
    AnnualIncomeClass = st.selectbox(
        "Annual Income Class",
        ["Low Income", "Middle Income", "High Income"],
        index=1,
        help="Customer's household income bracket"
    )

# ── Section 2 · Travel Behaviour ─────────────────────────────────────────────
st.markdown('<div class="sec-header">✈️ &nbsp;Travel Behaviour</div>', unsafe_allow_html=True)

c3, c4 = st.columns(2, gap="medium")
with c3:
    FrequentFlyer = st.selectbox(
        "Frequent Flyer Member?",
        ["No", "Yes"],
        help="Is the customer enrolled in the frequent flyer programme?"
    )
with c4:
    ServicesOpted = st.number_input(
        "Services Opted (1 – 10)",
        min_value=1, max_value=10, value=3, step=1,
        help="Number of ancillary travel services the customer uses"
    )

# ── Section 3 · Digital Engagement ───────────────────────────────────────────
st.markdown('<div class="sec-header">📱 &nbsp;Digital Engagement</div>', unsafe_allow_html=True)

c5, c6 = st.columns(2, gap="medium")
with c5:
    AccountSyncedToSocialMedia = st.selectbox(
        "Social Media Account Synced?",
        ["No", "Yes"],
        help="Has the customer linked their account to social media?"
    )
with c6:
    BookedHotelOrNot = st.selectbox(
        "Booked Hotel via Platform?",
        ["No", "Yes"],
        help="Has the customer booked at least one hotel through the service?"
    )

# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 &nbsp; Predict Churn", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict:
    # Encode exactly as training
    ff  = 1 if FrequentFlyer == "Yes" else 0
    asm = 1 if AccountSyncedToSocialMedia == "Yes" else 0
    bh  = 1 if BookedHotelOrNot == "Yes" else 0
    inc = {"Low Income": 0, "Middle Income": 1, "High Income": 2}[AnnualIncomeClass]

    features = np.array([[Age, ff, inc, ServicesOpted, asm, bh]])
    pred     = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]   # [P(stay), P(churn)]

    if pred == 1:
        churn_pct = int(round(proba[1] * 100))
        bar_color = "#f87171"
        st.markdown(f"""
        <div class="res-card res-churn">
            <div class="res-icon">⚠️</div>
            <div style="flex:1">
                <div class="res-title">High Churn Risk Detected</div>
                <div class="res-sub">
                    This customer is at significant risk of churning.<br>
                    Recommended action: trigger a personalised retention campaign,
                    offer a loyalty reward, or escalate to the CRM team.
                </div>
                <div class="conf-wrap">
                    <div class="conf-label">Churn Probability</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{churn_pct}%;background:{bar_color};"></div>
                    </div>
                    <div class="conf-pct" style="color:{bar_color};">{churn_pct}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        stay_pct  = int(round(proba[0] * 100))
        bar_color = "#34d399"
        st.markdown(f"""
        <div class="res-card res-stay">
            <div class="res-icon">✅</div>
            <div style="flex:1">
                <div class="res-title">Customer Likely to Stay</div>
                <div class="res-sub">
                    This customer shows strong loyalty signals.<br>
                    Recommended action: nurture the relationship with a premium
                    upgrade offer or an exclusive loyalty tier invitation.
                </div>
                <div class="conf-wrap">
                    <div class="conf-label">Retention Probability</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{stay_pct}%;background:{bar_color};"></div>
                    </div>
                    <div class="conf-pct" style="color:{bar_color};">{stay_pct}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="app-footer">
    ChurnSense &nbsp;·&nbsp; Random Forest Classifier &nbsp;·&nbsp;
    B.Tech Gen AI — Final Project &nbsp;·&nbsp; For academic use only
</div>
""", unsafe_allow_html=True)
