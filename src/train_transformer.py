"""Fine-tune DistilBERT for Fake News Detection and generate evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.predict import format_article_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "distilbert_fakenews"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NewsDataset(Dataset):
    """Tokenized dataset combining headline, description, and article content."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256) -> None:
        texts = [
            format_article_input(h, d, c)
            for h, d, c in zip(
                df["headline"].fillna(""),
                df["description"].fillna(""),
                df["content"].fillna(""),
            )
        ]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


@torch.inference_mode()
def evaluate_loader(model, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    use_amp = device.type == "cuda"

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += float(outputs.loss.item()) * labels.size(0)
        probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.concatenate(all_labels), np.concatenate(all_probs)


def generate_evaluation_plots(
    history: list[dict],
    comparison: dict,
    y_test: np.ndarray,
    distil_probs: np.ndarray,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Training history
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], marker="s", label="Val Loss")
    axes[0].set_title("DistilBERT Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_xticks(hist_df["epoch"])
    axes[0].legend()

    axes[1].plot(hist_df["epoch"], hist_df["val_f1"], marker="o", color="#2ca02c", label="Val F1-Score")
    axes[1].plot(hist_df["epoch"], hist_df["val_roc_auc"], marker="s", color="#1f77b4", label="Val ROC-AUC")
    axes[1].set_title("Validation F1 & ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(max(0.5, float(hist_df[["val_f1", "val_roc_auc"]].min().min()) - 0.05), 1.01)
    axes[1].set_xticks(hist_df["epoch"])
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_history.png", dpi=180)
    plt.close(fig)

    # 2. Confusion matrix for DistilBERT
    preds = (distil_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Credible (0)", "Misleading (1)"],
        yticklabels=["Credible (0)", "Misleading (1)"],
        ax=ax,
    )
    ax.set_title("DistilBERT Test Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_distilbert.png", dpi=180)
    plt.close(fig)

    # 3. Model Comparison Bar Chart
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rows = []
    for model_name, data in comparison.items():
        for k, lbl in zip(metric_keys, metric_labels):
            rows.append({"Model": model_name, "Metric": lbl, "Score": data["test"][k]})
    comp_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.barplot(data=comp_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_title("Baseline vs. Transformer Test Set Comparison")
    ax.set_ylim(max(0.5, float(comp_df["Score"].min()) - 0.08), 1.03)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    # 4. ROC Curves across models
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    baseline_preds_path = OUTPUTS_DIR / "baseline_test_preds.csv"
    if baseline_preds_path.exists():
        b_df = pd.read_csv(baseline_preds_path)
        for col, label in [
            ("nb_prob", "TF-IDF + Naive Bayes"),
            ("logreg_prob", "TF-IDF + Logistic Regression"),
        ]:
            fpr, tpr, _ = roc_curve(b_df["label"], b_df[col])
            auc_val = roc_auc_score(b_df["label"], b_df[col])
            ax.plot(fpr, tpr, linestyle="--", label=f"{label} (AUC={auc_val:.4f})")

    fpr_d, tpr_d, _ = roc_curve(y_test, distil_probs)
    auc_d = roc_auc_score(y_test, distil_probs)
    ax.plot(fpr_d, tpr_d, linewidth=2.2, label=f"DistilBERT Transformer (AUC={auc_d:.4f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Classifier")
    ax.set_title("ROC Curves on Held-Out Test Set")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=180)
    plt.close(fig)


def save_error_analysis(test_df: pd.DataFrame, distil_probs: np.ndarray, top_k: int = 15) -> None:
    """Save the most confident misclassified examples for qualitative interpretation."""
    df = test_df.copy()
    df["pred_prob_misleading"] = np.round(distil_probs, 4)
    df["pred_label"] = (distil_probs >= 0.5).astype(int)
    errors = df[df["label"] != df["pred_label"]].copy()
    errors["error_type"] = np.where(
        errors["label"] == 1,
        "False Negative (Missed Misleading)",
        "False Positive (Flagged Credible)",
    )
    errors["model_confidence"] = np.where(
        errors["pred_label"] == 1,
        errors["pred_prob_misleading"],
        1.0 - errors["pred_prob_misleading"],
    )
    errors = errors.sort_values("model_confidence", ascending=False).head(top_k)
    out_cols = [
        "error_type",
        "label",
        "pred_label",
        "model_confidence",
        "word_count",
        "headline",
        "description",
    ]
    errors[out_cols].to_csv(OUTPUTS_DIR / "error_analysis.csv", index=False)


def train_transformer(
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Credible", 1: "Misleading"},
        label2id={"Credible": 0, "Misleading": 1},
    ).to(device)

    train_ds = NewsDataset(train_df, tokenizer, max_length=max_length)
    val_ds = NewsDataset(val_df, tokenizer, max_length=max_length)
    test_ds = NewsDataset(test_df, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_val_f1 = -1.0
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item()) * labels.size(0)
            if step % 100 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_ds)
        val_loss, val_y, val_probs = evaluate_loader(model, val_loader, device)
        val_metrics = compute_metrics(val_y, val_probs)
        epoch_sec = round(time.perf_counter() - epoch_start, 2)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_score"],
            "val_roc_auc": val_metrics["roc_auc"],
            "epoch_sec": epoch_sec,
        }
        history.append(row)
        print(
            f"Epoch {epoch} summary: train_loss={row['train_loss']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, val_f1={row['val_f1']:.4f}, "
            f"val_auc={row['val_roc_auc']:.4f} ({epoch_sec}s)"
        )

        if val_metrics["f1_score"] >= best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            model.save_pretrained(MODELS_DIR)
            tokenizer.save_pretrained(MODELS_DIR)

    total_train_sec = round(time.perf_counter() - train_start, 2)

    # Evaluate best saved checkpoint on test set
    best_model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR).to(device)
    _, y_val_best, val_probs_best = evaluate_loader(best_model, val_loader, device)
    _, y_test, test_probs = evaluate_loader(best_model, test_loader, device)

    distil_result = {
        "model_name": model_name,
        "train_time_sec": total_train_sec,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_length": max_length,
            "fp16": use_amp,
        },
        "history": history,
        "val": compute_metrics(y_val_best, val_probs_best),
        "test": compute_metrics(y_test, test_probs),
    }

    with open(OUTPUTS_DIR / "transformer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(distil_result, f, indent=2)

    comparison = {}
    baseline_path = OUTPUTS_DIR / "baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            comparison.update(json.load(f))
    comparison["DistilBERT (Fine-Tuned)"] = {
        "train_time_sec": total_train_sec,
        "val": distil_result["val"],
        "test": distil_result["test"],
    }

    with open(OUTPUTS_DIR / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    generate_evaluation_plots(history, comparison, y_test, test_probs)
    save_error_analysis(test_df, test_probs)

    t = distil_result["test"]
    print(
        f"\nDistilBERT Test Results: Acc={t['accuracy']:.4f}, Prec={t['precision']:.4f}, "
        f"Rec={t['recall']:.4f}, F1={t['f1_score']:.4f}, ROC-AUC={t['roc_auc']:.4f} "
        f"(total time: {total_train_sec}s)"
    )
    return distil_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT Fake News Classifier.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    train_transformer(epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length)
