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
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

/* ---------- Root palette ---------- */
:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --card:      #1c2030;
    --border:    #2a2f42;
    --accent:    #4f8ef7;
    --accent2:   #7c5cfc;
    --text:      #e8eaf2;
    --muted:     #7a80a0;
    --success:   #34d399;
    --danger:    #f87171;
    --radius:    14px;
}

/* ---------- Global ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

/* ---------- Hide Streamlit branding ---------- */
#MainMenu, footer { visibility: hidden; }

/* ---------- Main container ---------- */
.block-container {
    max-width: 680px !important;
    padding: 2.5rem 2rem 4rem !important;
}

/* ---------- Hero ---------- */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(79,142,247,.15), rgba(124,92,252,.15));
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #fff 30%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 .6rem;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
    line-height: 1.6;
}

/* ---------- Section label ---------- */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2rem 0 .9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ---------- Streamlit widget overrides ---------- */
div[data-testid="stNumberInput"] > div,
div[data-testid="stSelectbox"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
    background: transparent !important;
    color: var(--text) !important;
}

/* Label text */
label, .stSelectbox label, .stNumberInput label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    letter-spacing: .02em;
    margin-bottom: 4px !important;
}

/* ---------- Predict button ---------- */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 0.85rem 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: .04em;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    cursor: pointer;
    transition: opacity .2s;
    margin-top: .5rem;
}
div[data-testid="stButton"] > button:hover { opacity: .88; }

/* ---------- Result banners ---------- */
.result-stay {
    background: linear-gradient(135deg, rgba(52,211,153,.08), rgba(52,211,153,.04));
    border: 1px solid rgba(52,211,153,.35);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.5rem;
}
.result-churn {
    background: linear-gradient(135deg, rgba(248,113,113,.08), rgba(248,113,113,.04));
    border: 1px solid rgba(248,113,113,.35);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.5rem;
}
.result-icon { font-size: 2.4rem; line-height: 1; }
.result-title { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px; }
.result-sub { font-size: 0.85rem; color: var(--muted); line-height: 1.5; }
.result-stay  .result-title { color: var(--success); }
.result-churn .result-title { color: var(--danger); }

/* ---------- Divider ---------- */
hr.custom { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

/* ---------- Footer ---------- */
.app-footer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--muted);
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✈️ &nbsp;Travel Industry · AI Powered</div>
    <h1>Customer Churn<br>Predictor</h1>
    <p>Fill in the customer profile below and our model will<br>
       instantly tell you whether they're likely to stay or leave.</p>
</div>
""", unsafe_allow_html=True)


# ── Section 1 — Personal Info ─────────────────────────────────────────────────
st.markdown('<div class="section-label">👤 &nbsp;Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    Age = st.number_input("Customer Age", min_value=18, max_value=100, value=30, step=1)
with col2:
    AnnualIncomeClass = st.selectbox(
        "Annual Income Class",
        ["Low Income", "Middle Income", "High Income"],
        help="Select the customer's income bracket"
    )


# ── Section 2 — Travel Behaviour ──────────────────────────────────────────────
st.markdown('<div class="section-label">✈️ &nbsp;Travel Behaviour</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="medium")
with col3:
    FrequentFlyer = st.selectbox(
        "Frequent Flyer Member?",
        ["No", "Yes"],
        help="Is this customer enrolled in the frequent flyer program?"
    )
with col4:
    ServicesOpted = st.number_input(
        "Services Opted (1–10)",
        min_value=1, max_value=10, value=3, step=1,
        help="Number of ancillary services the customer has opted into"
    )


# ── Section 3 — Digital Engagement ───────────────────────────────────────────
st.markdown('<div class="section-label">📱 &nbsp;Digital Engagement</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2, gap="medium")
with col5:
    AccountSyncedToSocialMedia = st.selectbox(
        "Account Synced to Social Media?",
        ["No", "Yes"],
        help="Has the customer linked their account to a social media profile?"
    )
with col6:
    BookedHotelOrNot = st.selectbox(
        "Booked a Hotel via Us?",
        ["No", "Yes"],
        help="Has the customer made at least one hotel booking through the platform?"
    )


# ── Spacer + Predict ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍  Run Churn Prediction", use_container_width=True)


# ── Prediction Logic ──────────────────────────────────────────────────────────
if predict_clicked:

    ff  = 1 if FrequentFlyer == "Yes" else 0
    asm = 1 if AccountSyncedToSocialMedia == "Yes" else 0
    bh  = 1 if BookedHotelOrNot == "Yes" else 0
    inc = {"Low Income": 0, "Middle Income": 1, "High Income": 2}[AnnualIncomeClass]

    data = [[Age, ff, inc, ServicesOpted, asm, bh]]
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.markdown("""
        <div class="result-churn">
            <div class="result-icon">⚠️</div>
            <div>
                <div class="result-title">High Churn Risk</div>
                <div class="result-sub">This customer is likely to discontinue their relationship.<br>
                Consider a targeted retention offer or loyalty incentive.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-stay">
            <div class="result-icon">✅</div>
            <div>
                <div class="result-title">Customer Likely to Stay</div>
                <div class="result-sub">This customer shows strong loyalty indicators.<br>
                A great candidate for an upsell or premium tier offer.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="custom">', unsafe_allow_html=True)
st.markdown("""
<div class="app-footer">
    ChurnSense · Powered by Machine Learning &nbsp;·&nbsp; For internal analytics use only
</div>
""", unsafe_allow_html=True)