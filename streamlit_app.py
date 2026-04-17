"""
============================================================
Streamlit Deployment — Food Waste Risk Prediction App
============================================================
Run with:  streamlit run streamlit_app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Food Waste Risk Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styling ─────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E86AB;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .risk-high {
        background-color: #fde8e8;
        border-left: 6px solid #E84855;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #c0392b;
    }
    .risk-low {
        background-color: #e8f8f0;
        border-left: 6px solid #44BBA4;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a7a5e;
    }
    .metric-card {
        background-color: #f0f4ff;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .disclaimer {
        font-size: 0.82rem;
        color: #888;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────
st.markdown('<p class="main-title">🍽️ Food Waste Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict whether a meal-centre combination is at high or low risk '
            'of generating food waste, based on historical demand patterns.</p>', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar: App Info ──────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/fast-food.png", width=70)
    st.markdown("### About This App")
    st.info(
        "This application uses a **Random Forest classifier** trained on "
        "food service demand data to estimate food waste risk. Inputs "
        "include meal characteristics, centre details, and pricing information. "
        "Predictions are derived from a demand-based proxy, not measured waste."
    )
    st.markdown("### Model Performance")
    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | Accuracy | ~0.86 |
    | F1-Score | ~0.86 |
    | CV Folds | 3 (stratified) |
    """)
    st.markdown('<p class="disclaimer">Note: All predictions are probabilistic estimates '
                'based on a proxy variable derived from order volume. '
                'They do not represent measured waste.</p>', unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists("best_rf_model.pkl"):
        st.error("⚠️  Model file 'best_rf_model.pkl' not found. "
                 "Please run food_waste_prediction.py first to train and save the model.")
        st.stop()
    return joblib.load("best_rf_model.pkl")

model = load_model()

# ─── Input Form ─────────────────────────────────────────────
st.subheader("🔧 Enter Meal & Centre Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Temporal & Identifiers**")
    week          = st.number_input("Week Number", min_value=1, max_value=145, value=30)
    center_id     = st.number_input("Centre ID", min_value=10, max_value=200, value=55)
    meal_id       = st.number_input("Meal ID", min_value=1000, max_value=3000, value=1885)

with col2:
    st.markdown("**Pricing**")
    base_price     = st.number_input("Base Price (£)", min_value=10.0, max_value=500.0,
                                     value=150.0, step=0.5)
    checkout_price = st.number_input("Checkout Price (£)", min_value=10.0, max_value=500.0,
                                     value=136.0, step=0.5)
    op_area        = st.number_input("Operating Area (sq km)", min_value=1.0, max_value=15.0,
                                     value=4.0, step=0.1)

with col3:
    st.markdown("**Promotions & Category**")
    emailer_for_promotion = st.selectbox("Email Promotion Active?", [0, 1],
                                         format_func=lambda x: "Yes" if x else "No")
    homepage_featured     = st.selectbox("Homepage Featured?", [0, 1],
                                         format_func=lambda x: "Yes" if x else "No")
    category  = st.selectbox("Meal Category", [
        "Beverages", "Biryani", "Desert", "Extras", "Fish",
        "Other Snacks", "Pasta", "Pizza", "Rice Bowl",
        "Salads", "Sandwich", "Seafood", "Soup", "Starters"
    ])
    cuisine   = st.selectbox("Cuisine", ["Continental", "Indian", "Italian", "Thai"])

col4, col5 = st.columns(2)
with col4:
    center_type = st.selectbox("Centre Type", ["TYPE_A", "TYPE_B", "TYPE_C"])
with col5:
    city_code   = st.number_input("City Code", min_value=500, max_value=800, value=650)
    region_code = st.number_input("Region Code", min_value=10, max_value=100, value=34)

st.markdown("---")

# ─── Predict ────────────────────────────────────────────────
if st.button("🔍 Predict Waste Risk", use_container_width=True, type="primary"):

    discount_ratio = max(0, (base_price - checkout_price) / base_price)

    input_data = pd.DataFrame([{
        "week":                   week,
        "center_id":              center_id,
        "meal_id":                meal_id,
        "checkout_price":         checkout_price,
        "base_price":             base_price,
        "emailer_for_promotion":  emailer_for_promotion,
        "homepage_featured":      homepage_featured,
        "category":               category,
        "cuisine":                cuisine,
        "city_code":              city_code,
        "region_code":            region_code,
        "center_type":            center_type,
        "op_area":                op_area,
        "discount_ratio":         discount_ratio
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("📊 Prediction Result")

    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        st.metric("Waste Risk Class",
                  "🔴 High Waste" if prediction == 1 else "🟢 Low Waste")
    with rcol2:
        st.metric("High Waste Probability", f"{probability[1]*100:.1f}%")
    with rcol3:
        st.metric("Low Waste Probability", f"{probability[0]*100:.1f}%")

    # Risk Banner
    if prediction == 1:
        st.markdown(
            '<div class="risk-high">⚠️  HIGH WASTE RISK — This meal-centre combination '
            'has historically low demand. Consider reducing stock allocation, '
            'applying discounts, or running promotional campaigns.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="risk-low">✅  LOW WASTE RISK — Demand for this combination '
            'has historically been sufficient. Standard fulfilment strategy is appropriate.</div>',
            unsafe_allow_html=True
        )

    # Probability bar
    st.markdown("##### Confidence Breakdown")
    prob_df = pd.DataFrame({
        "Category":    ["Low Waste", "High Waste"],
        "Probability": [probability[0], probability[1]]
    })
    st.bar_chart(prob_df.set_index("Category"))

    # Input summary
    with st.expander("📋 Input Summary"):
        display_df = pd.DataFrame({
            "Feature": [
                "Week", "Centre ID", "Meal ID", "Base Price", "Checkout Price",
                "Discount Ratio", "Email Promotion", "Homepage Featured",
                "Category", "Cuisine", "Centre Type", "City Code", "Region Code", "Op Area"
            ],
            "Value": [
                week, center_id, meal_id, f"£{base_price:.2f}", f"£{checkout_price:.2f}",
                f"{discount_ratio*100:.1f}%",
                "Yes" if emailer_for_promotion else "No",
                "Yes" if homepage_featured else "No",
                category, cuisine, center_type, city_code, region_code, f"{op_area} sq km"
            ]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<p class="disclaimer">⚠️ Predictions are based on a proxy variable derived from historical '
        'order volumes. They should be used as decision-support tools only, not as definitive '
        'waste measurements. Model performance may degrade on data significantly different from the '
        'training distribution (model drift).</p>',
        unsafe_allow_html=True
    )

# ─── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>🔬 Trained on: Gurgaon Food Demand Dataset · Model: Tuned Random Forest · "
    "Pipeline: scikit-learn ColumnTransformer + GridSearchCV</small>",
    unsafe_allow_html=True
)
