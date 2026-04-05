# =============================================================================
# app.py — Streamlit UI for Behavioral Agent
# =============================================================================
# Run: streamlit run app.py
#
# Provides:
#   - Text input for user description
#   - Optional demographic fields (age, income, profession)
#   - Real-time profile output with charts
#   - Integration-ready JSON output
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import json

from main import generate_user_profile
from defaults import CATEGORIES  # IMPORTANT: Changing category names will break Simulation Agent

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Behavioral Agent — Credit Card Profiler",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Behavioral Agent")
st.subheader("Convert User Input → Structured Spend Profile")
st.caption("Agent 1 of 3 | Output feeds directly into Simulation Agent")

st.divider()

# =============================================================================
# INPUT SECTION
# =============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 User Description")
    user_text = st.text_area(
        label="What does the user say about their spending?",
        value="I use Swiggy a lot and shop on Amazon frequently. Sometimes I travel for work.",
        height=120,
        help="Free-form text about spending habits. Mention brands, platforms, or spending categories."
    )

    # Quick sample buttons
    st.markdown("**Quick Samples:**")
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    with sample_col1:
        if st.button("🍕 Foodie"):
            user_text = "I order Swiggy and Zomato daily. Huge foodie."
    with sample_col2:
        if st.button("✈️ Traveler"):
            user_text = "I fly frequently with Vistara and need lounge access."
    with sample_col3:
        if st.button("🛒 Shopper"):
            user_text = "I shop heavily on Flipkart and Amazon. Also Myntra."
    with sample_col4:
        if st.button("⛽ Driver"):
            user_text = "I drive every day. Fill BPCL petrol weekly."

with col2:
    st.markdown("### 👤 Demographics (Optional)")
    age = st.number_input("Age", min_value=18, max_value=80, value=28, step=1)
    income = st.number_input(
        "Monthly Income (₹)",
        min_value=0, max_value=500000, value=0, step=5000,
        help="Leave 0 if unknown. Used to estimate spend via 40% heuristic."
    )
    profession = st.selectbox(
        "Profession",
        options=["", "student", "salaried", "engineer", "doctor",
                 "professional", "business", "freelancer", "homemaker", "retired"],
        help="Optional. Affects default spend estimation."
    )

    # Reason: Build user_profile dict for generate_user_profile()
    user_profile = {}
    if age: user_profile["age"] = int(age)
    if income > 0: user_profile["income"] = int(income)
    if profession: user_profile["profession"] = profession
    if not user_profile: user_profile = None

st.divider()

# =============================================================================
# RUN AGENT
# =============================================================================
run_btn = st.button("🚀 Generate Profile", type="primary", use_container_width=True)

if run_btn or True:  # auto-run on load for demo feel; remove `or True` for manual-only
    with st.spinner("Running Behavioral Agent pipeline..."):
        output = generate_user_profile(user_text, user_profile)

    st.success("✅ Profile generated successfully")
    st.divider()

    # ── OUTPUT DISPLAY ─────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📊 Spend Vector (₹/month)")

        # Bar chart data
        spend_data = output["spend_vector"]
        categories = list(spend_data.keys())
        values     = list(spend_data.values())

        # Display as bar chart
        import pandas as pd
        df = pd.DataFrame({
            "Category": categories,
            "Monthly Spend (₹)": values
        }).sort_values("Monthly Spend (₹)", ascending=False)

        st.bar_chart(df.set_index("Category"))

        # Table
        st.dataframe(
            df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        total = sum(values)
        st.metric("Total Monthly Spend", f"₹{total:,}")

    with col_b:
        st.markdown("### ⚖️ Category Weights")

        weights_data = output["category_weights"]
        df_w = pd.DataFrame({
            "Category": list(weights_data.keys()),
            "Weight": [round(v, 4) for v in weights_data.values()]
        }).sort_values("Weight", ascending=False)

        st.dataframe(df_w.reset_index(drop=True), use_container_width=True, hide_index=True)

        st.markdown("### 🏷️ Tags")
        for tag in output["tags"]:
            st.badge(tag)

        st.markdown("### 🎯 Confidence Score")
        confidence = output["confidence"]
        color = "green" if confidence >= 0.75 else "orange" if confidence >= 0.50 else "red"
        st.progress(confidence)
        st.markdown(f"**{confidence:.0%}** confidence in this profile")
        if confidence < 0.5:
            st.warning("⚠️ Low confidence — profile may be mostly assumed from defaults. Provide more details.")
        elif confidence < 0.75:
            st.info("ℹ️ Medium confidence — profile partially inferred.")
        else:
            st.success("✅ High confidence — strong signals detected.")

    # ── RAW JSON OUTPUT (for integration) ────────────────────────────────────
    st.divider()
    st.markdown("### 🔌 Integration Output (JSON)")
    st.caption("This is the exact output consumed by the Simulation Agent (Agent 3)")
    st.json(output)

    # Download button
    st.download_button(
        label="⬇️ Download Profile JSON",
        data=json.dumps(output, indent=2),
        file_name="user_spend_profile.json",
        mime="application/json"
    )

# =============================================================================
# SIDEBAR: INFO
# =============================================================================
with st.sidebar:
    st.markdown("## ℹ️ Agent Info")
    st.markdown("""
    **Agent 1: Behavioral Profiler**

    **Pipeline:**
    ```
    Text Input
       ↓
    Keyword Detection
       ↓
    Brand → Category Mapping
       ↓
    Intensity Detection
       ↓
    Spend Estimation
       ↓
    Weight Normalization
       ↓
    Structured Profile Output
    ```
    """)

    st.markdown("## 🔗 Integration")
    st.markdown("""
    Output consumed by:
    - **Agent 2**: Knowledge + Retrieval
    - **Agent 3**: Simulation + Reasoning

    Fixed categories:
    """)
    for cat in CATEGORIES:
        # IMPORTANT: Changing category names will break Simulation Agent
        st.markdown(f"- `{cat}`")

    st.markdown("## 📦 Requirements")
    st.code("pip install streamlit pandas")
