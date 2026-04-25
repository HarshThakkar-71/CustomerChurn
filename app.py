import streamlit as st
import pickle
import pandas as pd

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

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8eaf2;
}

#MainMenu, footer {visibility:hidden;}

.block-container {
    max-width: 680px !important;
    padding: 2.5rem 2rem 4rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model (FINAL FIX) ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("✈️ Customer Churn Prediction")

Age = st.number_input("Customer Age", 18, 100, 30)

FrequentFlyer = st.selectbox(
    "Frequent Flyer?",
    ["No","Yes"]
)

AnnualIncomeClass = st.selectbox(
    "Annual Income Class",
    ["Low Income","Middle Income","High Income"]
)

ServicesOpted = st.number_input("Services Opted",1,10,3)

AccountSyncedToSocialMedia = st.selectbox(
    "Account Synced?",
    ["No","Yes"]
)

BookedHotelOrNot = st.selectbox(
    "Booked Hotel?",
    ["No","Yes"]
)

# ── Prediction ───────────────────────────────────────────────────────────────
if st.button("Predict"):

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
