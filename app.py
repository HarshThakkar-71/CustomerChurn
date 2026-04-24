import streamlit as st
import pandas as pd
import joblib
import sklearn

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense · Predict Customer Loyalty",
    page_icon="✈️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

:root {
    --bg:#0d0f14;
    --surface:#151820;
    --card:#1c2030;
    --border:#2a2f42;
    --accent:#4f8ef7;
    --accent2:#7c5cfc;
    --text:#e8eaf2;
    --muted:#7a80a0;
    --success:#34d399;
    --danger:#f87171;
    --radius:14px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

#MainMenu, footer {visibility:hidden;}

.block-container {
    max-width: 680px !important;
    padding: 2.5rem 2rem 4rem !important;
}

.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}

.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
}

.section-label {
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2rem 0 .9rem;
}

div[data-testid="stButton"] > button {
    width: 100%;
    padding: .85rem 0 !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model (FIXED) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
<h1>Customer Churn Predictor</h1>
<p>Fill in details and predict churn</p>
</div>
""", unsafe_allow_html=True)

# ── Personal Info ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Customer Age", 18, 100, 30)

with col2:
    AnnualIncomeClass = st.selectbox(
        "Annual Income Class",
        ["Low Income","Middle Income","High Income"]
    )

# ── Travel ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Travel Behaviour</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    FrequentFlyer = st.selectbox(
        "Frequent Flyer?",
        ["No","Yes"]
    )

with col4:
    ServicesOpted = st.number_input(
        "Services Opted",
        1,10,3
    )

# ── Digital ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Digital Engagement</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    AccountSyncedToSocialMedia = st.selectbox(
        "Account Synced?",
        ["No","Yes"]
    )

with col6:
    BookedHotelOrNot = st.selectbox(
        "Booked Hotel?",
        ["No","Yes"]
    )

# ── Predict ───────────────────────────────────────────────────────────────────
predict_clicked = st.button("Run Churn Prediction")

# ── Prediction Logic ──────────────────────────────────────────────────────────
if predict_clicked:

    ff  = 1 if FrequentFlyer == "Yes" else 0
    asm = 1 if AccountSyncedToSocialMedia == "Yes" else 0
    bh  = 1 if BookedHotelOrNot == "Yes" else 0
    inc = {"Low Income":0,"Middle Income":1,"High Income":2}[AnnualIncomeClass]

    data = [[Age, ff, inc, ServicesOpted, asm, bh]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ High Churn Risk")
    else:
        st.success("✅ Customer Likely to Stay")
