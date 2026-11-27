"""Dataset download, cleaning, leakage removal, EDA, and stratified splitting."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Remove publisher datelines (e.g., "WASHINGTON (Reuters) - ") and social media footers to prevent shortcut learning.
REUTERS_DATELINE_RE = re.compile(r"^[A-Z][A-Z\s.,/]{1,40}\(Reuters\)\s*[-–—]?\s*")
TRAILING_BOILERPLATE_RE = re.compile(
    r"(Featured image via.*|Read more:.*|Photo by.*Getty Images.*)$",
    flags=re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+|reut\.rs/\S+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str, strip_dateline: bool = False) -> str:
    """Normalize encoding artifacts, URLs, boilerplate leaks, and whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.replace("\ufffd??", "'").replace("\ufffd", "'")
    if strip_dateline:
        text = REUTERS_DATELINE_RE.sub("", text.strip())
        text = TRAILING_BOILERPLATE_RE.sub("", text.strip())
    text = URL_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_description(content: str, max_chars: int = 220) -> str:
    """Extract the lead sentence / meta-description from the opening of an article."""
    if not content:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", content)
    lead = sentences[0] if sentences else content
    if len(lead) > max_chars:
        lead = lead[:max_chars].rsplit(" ", 1)[0].strip() + "..."
    return lead


def generate_eda_plots(df: pd.DataFrame, split_df: pd.DataFrame) -> None:
    """Save dataset class-balance and length-distribution figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Class distribution across splits
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_df = split_df.copy()
    plot_df["Class"] = plot_df["label"].map({0: "Credible (0)", 1: "Misleading (1)"})
    sns.countplot(data=plot_df, x="split", hue="Class", order=["train", "val", "test"], ax=ax)
    ax.set_title("Class Distribution Across Train / Validation / Test Splits")
    ax.set_xlabel("Dataset Split")
    ax.set_ylabel("Article Count")
    for container in ax.containers:
        ax.bar_label(container, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)

    # 2. Word-length distribution by class (clipped at 98th percentile for clarity)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cap = int(df["word_count"].quantile(0.98))
    len_df = df[df["word_count"] <= cap].copy()
    len_df["Class"] = len_df["label"].map({0: "Credible (0)", 1: "Misleading (1)"})
    sns.histplot(
        data=len_df,
        x="word_count",
        hue="Class",
        bins=40,
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.45,
        ax=ax,
    )
    ax.set_title("Article Word Count Distribution by Class (<= 98th Percentile)")
    ax.set_xlabel("Word Count (Headline + Content)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=180)
    plt.close(fig)


def prepare_dataset(max_samples: int = 18000, seed: int = 42) -> dict:
    """Download, clean, deduplicate, and split the news articles dataset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset `GonzaloA/fake_news` from Hugging Face...")
    ds = load_dataset("GonzaloA/fake_news")
    combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    df = combined.to_pandas()

    raw_rows = len(df)
    missing_before = {
        "title": int(df["title"].isna().sum()),
        "text": int(df["text"].isna().sum()),
        "label": int(df["label"].isna().sum()),
    }

    # Note: In GonzaloA/fake_news, raw label 0 = Fake/Misleading and raw label 1 = Real/Credible.
    # Remap to standard anomaly/detection convention: 0 = Credible, 1 = Misleading.
    df = df.dropna(subset=["title", "text", "label"]).copy()
    df["label"] = df["label"].astype(int).map({1: 0, 0: 1})

    df["headline"] = df["title"].apply(lambda x: clean_text(x, strip_dateline=False))
    df["content"] = df["text"].apply(lambda x: clean_text(x, strip_dateline=True))

    # Filter empty or trivially short articles
    before_short = len(df)
    df = df[(df["headline"].str.len() >= 10) & (df["content"].str.len() >= 50)].copy()
    short_removed = before_short - len(df)

    # Deduplicate globally on normalized headline + content prefix to avoid cross-split leakage
    df["dedup_key"] = (
        df["headline"].str.lower().str.strip() + "||" + df["content"].str[:200].str.lower().str.strip()
    )
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"]).reset_index(drop=True)
    duplicates_removed = before_dedup - len(df)

    df["description"] = df["content"].apply(extract_description)
    df["word_count"] = (df["headline"] + " " + df["content"]).str.split().str.len()

    clean_rows = len(df)

    # Optional stratified subsample for fast GPU/CPU experimentation
    if max_samples and len(df) > max_samples:
        df, _ = train_test_split(
            df[["headline", "description", "content", "label", "word_count"]],
            train_size=max_samples,
            stratify=df["label"],
            random_state=seed,
        )
        df = df.reset_index(drop=True)
    else:
        df = df[["headline", "description", "content", "label", "word_count"]].copy()

    # Stratified 70% train / 15% val / 15% test split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )

    cols = ["headline", "description", "content", "label", "word_count"]
    train_df[cols].to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df[cols].to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df[cols].to_csv(PROCESSED_DIR / "test.csv", index=False)

    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    generate_eda_plots(df, split_df)

    summary = {
        "dataset_source": "GonzaloA/fake_news (ISOT-derived online news corpus)",
        "label_mapping": {"0": "Credible", "1": "Misleading"},
        "raw_rows": int(raw_rows),
        "missing_before_cleaning": missing_before,
        "short_or_empty_removed": int(short_removed),
        "duplicates_removed": int(duplicates_removed),
        "clean_rows": int(clean_rows),
        "modeling_rows": int(len(df)),
        "class_balance": {
            "Credible (0)": int((df["label"] == 0).sum()),
            "Misleading (1)": int((df["label"] == 1).sum()),
        },
        "avg_word_count": {
            "Credible (0)": round(float(df.loc[df["label"] == 0, "word_count"].mean()), 1),
            "Misleading (1)": round(float(df.loc[df["label"] == 1, "word_count"].mean()), 1),
        },
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }

    with open(OUTPUTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Fake News dataset.")
    parser.add_argument("--max-samples", type=int, default=18000, help="Stratified sample size (0 = all).")
    args = parser.parse_args()
    prepare_dataset(max_samples=args.max_samples)
