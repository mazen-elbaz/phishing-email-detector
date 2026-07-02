"""
Streamlit demo — Phishing Email Detector

Paste a sender, subject, and body. The app extracts the same 10 features used in
training, runs the saved Random Forest model, and explains the prediction with SHAP.

Run with: streamlit run app.py
"""

import re
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Feature extraction — must exactly match 01_feature_extraction.py / the notebook
# ---------------------------------------------------------------------------

URGENCY_KEYWORDS = [
    "verify", "urgent", "suspend", "click here", "confirm your",
    "password", "account", "immediately", "act now", "limited time",
    "click below", "restricted", "unusual activity", "update your",
    "security alert", "unauthorized", "expire", "validate"
]

DISPLAY_NAME_BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "bank", "usaa",
    "chase", "wellsfargo", "netflix", "irs", "fedex", "ups", "dhl"
]

FEATURE_ORDER = [
    "subject_length", "body_length", "num_exclaim_subject",
    "urgency_keyword_count", "sender_display_mismatch",
    "sender_local_digit_count", "body_has_html_form", "body_has_href",
    "subject_has_re_fwd", "num_links_in_body_text",
]


def get_sender_domain(sender):
    if not sender:
        return ""
    match = re.search(r"@([\w\.-]+)", sender)
    return match.group(1).lower() if match else ""


def get_sender_display_name(sender):
    if not sender:
        return ""
    match = re.match(r'^"?([^"<]*)"?\s*<', sender)
    return match.group(1).strip().lower() if match else ""


def sender_local_part_digit_count(sender):
    if not sender:
        return 0
    cleaned = sender.strip().strip('"')
    match = re.search(r'^([^<@]+)@', cleaned)
    local = match.group(1) if match else cleaned.split("@")[0]
    return sum(c.isdigit() for c in local)


def display_name_mismatch(sender):
    display = get_sender_display_name(sender)
    domain = get_sender_domain(sender)
    if not display or not domain:
        return 0
    for brand in DISPLAY_NAME_BRANDS:
        if brand in display and brand not in domain:
            return 1
    return 0


def extract_features_single(sender, subject, body):
    combined_text = (subject + " " + body).lower()
    return {
        "subject_length": len(subject),
        "body_length": len(body),
        "num_exclaim_subject": subject.count("!"),
        "urgency_keyword_count": sum(kw in combined_text for kw in URGENCY_KEYWORDS),
        "sender_display_mismatch": display_name_mismatch(sender),
        "sender_local_digit_count": sender_local_part_digit_count(sender),
        "body_has_html_form": int("<form" in body.lower()),
        "body_has_href": int("href=" in body.lower()),
        "subject_has_re_fwd": int(bool(re.match(r"^(re|fwd):", subject.strip().lower()))),
        "num_links_in_body_text": len(re.findall(r"https?://[^\s\"'<>]+", body)),
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣", layout="centered")

st.title("🎣 Phishing Email Detector")
st.caption(
    "Random Forest classifier trained on the Nazario/Enron corpus · "
    "F1 = 0.89, ROC-AUC = 0.95 on held-out test data"
)

with st.expander("⚠️ Limitations — read before trusting the output", expanded=False):
    st.markdown(
        """
- No live URL reputation check (the training dataset's URL data was unusable — see README).
- Trained on 3,065 emails. Not validated against real-world production traffic.
- Legit examples come from corporate (Enron) email; very different inbox may behave differently.
- This is a portfolio/demo tool, not a production security product.
        """
    )

st.divider()

col1, col2 = st.columns(2)
with col1:
    sender = st.text_input("Sender", placeholder='"PayPal Support" <security@paypa1-verify.com>')
with col2:
    subject = st.text_input("Subject", placeholder="Urgent: Verify your account now!")

body = st.text_area(
    "Body",
    height=200,
    placeholder="Dear customer, we detected unusual activity on your account. "
                "Click here to verify your identity immediately: http://paypa1-verify.com/login",
)

analyze = st.button("Analyze Email", type="primary", use_container_width=True)

if analyze:
    if not (sender or subject or body):
        st.warning("Enter at least a subject or body to analyze.")
    else:
        model = joblib.load("best_model.pkl")

        feats = extract_features_single(sender, subject, body)
        X_input = pd.DataFrame([feats])[FEATURE_ORDER]

        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        st.divider()

        if pred == 1:
            st.error(f"### 🚨 Flagged as PHISHING — {prob:.0%} confidence")
        else:
            st.success(f"### ✅ Classified as LEGIT — {1 - prob:.0%} confidence")

        # SHAP explanation for this single prediction
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[:, :, 1][0] if shap_values.ndim == 3 else shap_values[0]

        explanation_df = pd.DataFrame({
            "feature": FEATURE_ORDER,
            "value": [feats[f] for f in FEATURE_ORDER],
            "impact": sv,
        }).sort_values("impact", key=abs, ascending=False)

        st.subheader("Why this prediction?")
        st.caption("Positive impact pushes toward PHISHING. Negative impact pushes toward LEGIT.")

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in explanation_df["impact"]]
        ax.barh(explanation_df["feature"], explanation_df["impact"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP contribution")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.dataframe(explanation_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Built as a dual ML/security portfolio project. "
    "See README for the dataset leakage investigation behind these features."
)
