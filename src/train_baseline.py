"""Baseline NLP models (TF-IDF + Logistic Regression & Naive Bayes) for comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def build_input_text(df: pd.DataFrame) -> pd.Series:
    """Combine headline, description, and article content into a unified text string."""
    return (
        "Headline: "
        + df["headline"].fillna("").astype(str)
        + " | Description: "
        + df["description"].fillna("").astype(str)
        + " | Content: "
        + df["content"].fillna("").astype(str)
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute standard binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_evaluate_baselines() -> dict:
    """Train TF-IDF baselines on processed splits and save metrics/artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X_train = build_input_text(train_df)
    y_train = train_df["label"].to_numpy()
    X_val = build_input_text(val_df)
    y_val = val_df["label"].to_numpy()
    X_test = build_input_text(test_df)
    y_test = test_df["label"].to_numpy()

    experiments = {
        "TF-IDF + Multinomial Naive Bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), stop_words="english")),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ),
        "TF-IDF + Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)),
            ]
        ),
    }

    results = {}
    test_probs_store = {}

    for name, pipeline in experiments.items():
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_sec = round(time.perf_counter() - start, 2)

        val_prob = pipeline.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)

        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        results[name] = {
            "train_time_sec": train_sec,
            "val": evaluate_predictions(y_val, val_pred, val_prob),
            "test": evaluate_predictions(y_test, test_pred, test_prob),
        }
        test_probs_store[name] = test_prob

        if "Logistic Regression" in name:
            joblib.dump(pipeline, MODELS_DIR / "baseline_logreg.joblib")

    with open(OUTPUTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    preds_df = pd.DataFrame(
        {
            "label": y_test,
            "logreg_prob": test_probs_store["TF-IDF + Logistic Regression"],
            "nb_prob": test_probs_store["TF-IDF + Multinomial Naive Bayes"],
        }
    )
    preds_df.to_csv(OUTPUTS_DIR / "baseline_test_preds.csv", index=False)

    print("Baseline training complete:")
    for model_name, metrics in results.items():
        t = metrics["test"]
        print(
            f"  {model_name}: Acc={t['accuracy']:.4f}, Prec={t['precision']:.4f}, "
            f"Rec={t['recall']:.4f}, F1={t['f1_score']:.4f}, ROC-AUC={t['roc_auc']:.4f} "
            f"({metrics['train_time_sec']}s)"
        )
    return results


if __name__ == "__main__":
    train_and_evaluate_baselines()
