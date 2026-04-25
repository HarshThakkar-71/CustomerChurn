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
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #151820;
    --card:    #1c2030;
    --border:  #2a2f42;
    --accent:  #4f8ef7;
    --accent2: #7c5cfc;
    --text:    #e8eaf2;
    --muted:   #7a80a0;
    --green:   #34d399;
    --red:     #f87171;
    --radius:  14px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer         { visibility: hidden !important; }

.block-container {
    max-width: 700px !important;
    padding: 2rem 1.8rem 4rem !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.4rem 0 1.6rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(79,142,247,.14), rgba(124,92,252,.14));
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 6px 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1.1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.13;
    background: linear-gradient(135deg, #ffffff 25%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 .55rem;
}
.hero p {
    color: var(--muted);
    font-size: .95rem;
    line-height: 1.65;
    margin: 0;
}

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    gap: .75rem;
    margin: 1.6rem 0 .4rem;
}
.stat-box {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .9rem 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.stat-lbl {
    font-size: .7rem;
    color: var(--muted);
    margin-top: 4px;
    letter-spacing: .06em;
    text-transform: uppercase;
}

/* ── Section headers ── */
.sec-header {
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.9rem 0 .85rem;
    display: flex;
    align-items: center;
    gap: 9px;
}
.sec-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Widget overrides ── */
div[data-testid="stNumberInput"] > div,
div[data-testid="stSelectbox"]   > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
div[data-testid="stNumberInput"] input { color: var(--text) !important; background: transparent !important; }
label { font-size: .83rem !important; font-weight: 500 !important; color: var(--muted) !important; }

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: .88rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: .05em;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    margin-top: .6rem;
    transition: opacity .2s;
}
div[data-testid="stButton"] > button:hover { opacity: .85; }

/* ── Result cards ── */
.res-card {
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    display: flex;
    align-items: flex-start;
    gap: 1.1rem;
    margin-top: 1.6rem;
    animation: pop .35s cubic-bezier(.22,1,.36,1);
}
@keyframes pop { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }

.res-stay  { background: linear-gradient(135deg,rgba(52,211,153,.09),rgba(52,211,153,.03)); border:1px solid rgba(52,211,153,.3); }
.res-churn { background: linear-gradient(135deg,rgba(248,113,113,.09),rgba(248,113,113,.03)); border:1px solid rgba(248,113,113,.3); }

.res-icon  { font-size: 2.5rem; line-height: 1; flex-shrink: 0; margin-top: 2px; }
.res-title { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; margin-bottom:6px; }
.res-sub   { font-size:.85rem; color:var(--muted); line-height:1.6; }
.res-stay  .res-title { color:var(--green); }
.res-churn .res-title { color:var(--red); }

/* ── Confidence meter ── */
.conf-wrap { margin-top: 1.1rem; }
.conf-label { font-size:.75rem; color:var(--muted); margin-bottom:5px; }
.conf-bar-bg { background:var(--border); border-radius:100px; height:8px; overflow:hidden; }
.conf-bar-fill { height:8px; border-radius:100px; transition:width .6s ease; }
.conf-pct { font-family:'Syne',sans-serif; font-size:.9rem; font-weight:700; margin-top:5px; }

/* ── Divider ── */
.divider { border:none; border-top:1px solid var(--border); margin:2.2rem 0; }

/* ── Footer ── */
.app-footer { text-align:center; font-size:.73rem; color:var(--muted); padding-top:1.5rem; }
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
