"""Streamlit Live Demo for Project 5: Fake News Detection Using Transformer Models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import FakeNewsDetector

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

st.set_page_config(
    page_title="Fake News Detection | Transformer NLP",
    page_icon="📰",
    layout="wide",
)

SAMPLES = {
    "Custom Input": {
        "headline": "",
        "description": "",
        "content": "",
    },
    "Sample 1 — Credible Reporting": {
        "headline": "Central bank holds benchmark interest rate steady amid cooling inflation data",
        "description": "Policymakers cited steady labor market conditions and moderating consumer prices in their quarterly policy statement.",
        "content": (
            "The central bank's monetary policy committee voted unanimously on Wednesday to keep its benchmark "
            "lending rate unchanged at 4.75 percent, matching expectations from a survey of economists. In a written "
            "statement released after the two-day meeting, officials noted that headline inflation has eased over the "
            "past three quarters while employment growth remained resilient. Analysts said the committee will review "
            "upcoming wage and manufacturing figures before considering any rate adjustments later this year."
        ),
    },
    "Sample 2 — Potentially Misleading": {
        "headline": "SHOCKING: Secret government memo reveals tap water will be shut off nationwide this Friday!",
        "description": "Viral social media posts claim an unverified leaked directive orders citizens to stockpile supplies immediately.",
        "content": (
            "YOU WON'T SEE THIS ON MAINSTREAM MEDIA!! Anonymous insiders claim that a secret emergency order has "
            "already been signed behind closed doors to cut off public water supplies across every major city "
            "starting Friday midnight. Share this warning with everyone you know before they delete this page! "
            "Supposedly, unnamed officials are hiding the truth from the public to prevent panic, even though no "
            "agency has confirmed the rumor."
        ),
    },
}


@st.cache_resource
def load_detector() -> FakeNewsDetector | None:
    try:
        return FakeNewsDetector()
    except FileNotFoundError:
        return None


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def main() -> None:
    st.title("📰 Fake News Detection Using Transformer Models")
    st.caption(
        "Project 5 • NLP-based classification of online news articles (Headlines, Descriptions & Content) "
        "using contextual representations from fine-tuned DistilBERT."
    )

    with st.sidebar:
        st.header("⚙️ System Overview")
        st.markdown(
            "- **Backbone:** `distilbert-base-uncased`\n"
            "- **Baselines:** TF-IDF + Logistic Regression & Naive Bayes\n"
            "- **Input Fields:** Headline + Description + Full Content\n"
            "- **Classes:** `Credible` (0) vs `Misleading` (1)"
        )
        st.divider()
        sample_choice = st.selectbox("Load Demo Article", list(SAMPLES.keys()))
        if st.button("Apply Sample", use_container_width=True):
            st.session_state["headline"] = SAMPLES[sample_choice]["headline"]
            st.session_state["description"] = SAMPLES[sample_choice]["description"]
            st.session_state["content"] = SAMPLES[sample_choice]["content"]

    tab_predict, tab_metrics, tab_eda = st.tabs(
        ["🔍 Live Classifier", "📊 Model Experiments & Evaluation", "🗂️ Dataset & Error Analysis"]
    )

    with tab_predict:
        detector = load_detector()
        if detector is None:
            st.error(
                "Fine-tuned Transformer model not found in `models/distilbert_fakenews/`. "
                "Run `python -m src.train_transformer` first."
            )
        else:
            col_form, col_result = st.columns([3, 2], gap="large")
            with col_form:
                headline = st.text_input(
                    "Article Headline",
                    value=st.session_state.get("headline", SAMPLES["Sample 1 — Credible Reporting"]["headline"]),
                    placeholder="Enter the news headline...",
                )
                description = st.text_area(
                    "Article Description / Lead Summary",
                    value=st.session_state.get("description", SAMPLES["Sample 1 — Credible Reporting"]["description"]),
                    height=85,
                    placeholder="Enter the short article summary or sub-headline...",
                )
                content = st.text_area(
                    "Complete Article Content",
                    value=st.session_state.get("content", SAMPLES["Sample 1 — Credible Reporting"]["content"]),
                    height=220,
                    placeholder="Paste the full news article body here...",
                )
                analyze_clicked = st.button("🚀 Classify Article", type="primary", use_container_width=True)

            with col_result:
                st.subheader("Prediction Result")
                if analyze_clicked or (headline.strip() and content.strip()):
                    if not headline.strip() and not content.strip():
                        st.warning("Please provide at least a headline or article content.")
                    else:
                        res = detector.predict(headline, description, content)
                        if res["prediction"] == "Credible":
                            st.success(f"### ✅ {res['prediction']} ({res['confidence']:.1%} confidence)")
                        else:
                            st.error(f"### ⚠️ {res['prediction']} ({res['confidence']:.1%} confidence)")

                        st.markdown("**Class Probabilities**")
                        st.write(f"Credible: **{res['probabilities']['Credible']:.2%}**")
                        st.progress(res["probabilities"]["Credible"])
                        st.write(f"Misleading: **{res['probabilities']['Misleading']:.2%}**")
                        st.progress(res["probabilities"]["Misleading"])

                        word_count = len(f"{headline} {description} {content}".split())
                        m1, m2 = st.columns(2)
                        m1.metric("Word Count", word_count)
                        m2.metric("Transformer Tokens", res["token_count"])

                        st.info(
                            "**Limitation Note:** This prototype classifies articles based on contextual language "
                            "patterns learned from labelled training data, not live external fact-checking."
                        )

    with tab_metrics:
        comp = load_json(OUTPUTS_DIR / "comparison_metrics.json")
        if comp:
            st.subheader("Model Comparison on Held-Out Test Set")
            rows = []
            for model_name, data in comp.items():
                t = data["test"]
                rows.append(
                    {
                        "Model": model_name,
                        "Accuracy": t["accuracy"],
                        "Precision": t["precision"],
                        "Recall": t["recall"],
                        "F1-Score": t["f1_score"],
                        "ROC-AUC": t["roc_auc"],
                        "Train Time (s)": data.get("train_time_sec", "-"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            if (FIGURES_DIR / "model_comparison.png").exists():
                c1.image(str(FIGURES_DIR / "model_comparison.png"), caption="Baseline vs Transformer Performance")
            if (FIGURES_DIR / "roc_curves.png").exists():
                c2.image(str(FIGURES_DIR / "roc_curves.png"), caption="ROC Curves Across Models")

            c3, c4 = st.columns(2)
            if (FIGURES_DIR / "confusion_matrix_distilbert.png").exists():
                c3.image(str(FIGURES_DIR / "confusion_matrix_distilbert.png"), caption="DistilBERT Confusion Matrix")
            if (FIGURES_DIR / "training_history.png").exists():
                c4.image(str(FIGURES_DIR / "training_history.png"), caption="DistilBERT Training & Validation Curves")
        else:
            st.info("Run the training pipeline to populate evaluation metrics and figures.")

    with tab_eda:
        eda = load_json(OUTPUTS_DIR / "eda_summary.json")
        if eda:
            st.subheader("Dataset Preprocessing Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Raw Articles", f"{eda['raw_rows']:,}")
            s2.metric("After Deduplication", f"{eda['clean_rows']:,}")
            s3.metric("Duplicates Removed", f"{eda['duplicates_removed']:,}")
            s4.metric("Train / Val / Test", f"{eda['split_sizes']['train']:,} / {eda['split_sizes']['val']:,} / {eda['split_sizes']['test']:,}")

            c1, c2 = st.columns(2)
            if (FIGURES_DIR / "class_distribution.png").exists():
                c1.image(str(FIGURES_DIR / "class_distribution.png"), caption="Class Distribution After Cleaning")
            if (FIGURES_DIR / "text_length_distribution.png").exists():
                c2.image(str(FIGURES_DIR / "text_length_distribution.png"), caption="Article Word Length Distribution")

        err_path = OUTPUTS_DIR / "error_analysis.csv"
        if err_path.exists():
            st.subheader("Sample Misclassified Articles (Error Analysis)")
            err_df = pd.read_csv(err_path)
            st.dataframe(err_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
