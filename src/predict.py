"""Inference helper & CLI for the fine-tuned Transformer Fake News Detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_fakenews"

LABEL_MAP = {0: "Credible", 1: "Misleading"}


def format_article_input(headline: str, description: str, content: str) -> str:
    """Format headline, description, and content consistently with training."""
    h = (headline or "").strip()
    d = (description or "").strip()
    c = (content or "").strip()
    return f"Headline: {h} [SEP] Description: {d} [SEP] Content: {c}"


class FakeNewsDetector:
    """Wrapper around the fine-tuned Transformer classification model."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, max_length: int = 256) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_dir}. Run `python -m src.train_transformer` first."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.inference_mode()
    def predict(self, headline: str, description: str = "", content: str = "") -> dict[str, Any]:
        """Classify an article as Credible (0) or Misleading (1) with probabilities."""
        formatted = format_article_input(headline, description, content)
        encoded = self.tokenizer(
            formatted,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().tolist()
        credible_prob = float(probs[0])
        misleading_prob = float(probs[1])
        pred_idx = 1 if misleading_prob >= 0.5 else 0

        return {
            "prediction": LABEL_MAP[pred_idx],
            "label_id": pred_idx,
            "confidence": round(max(credible_prob, misleading_prob), 4),
            "probabilities": {
                "Credible": round(credible_prob, 4),
                "Misleading": round(misleading_prob, 4),
            },
            "token_count": int(encoded["input_ids"].shape[1]),
            "device": str(self.device),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a news article using the Transformer model.")
    parser.add_argument("--headline", required=True, help="News article headline")
    parser.add_argument("--description", default="", help="News article description/summary")
    parser.add_argument("--content", required=True, help="Complete article body text")
    args = parser.parse_args()

    detector = FakeNewsDetector()
    result = detector.predict(args.headline, args.description, args.content)
    print(f"Prediction : {result['prediction']} (confidence: {result['confidence']:.2%})")
    print(f"Credible   : {result['probabilities']['Credible']:.2%}")
    print(f"Misleading : {result['probabilities']['Misleading']:.2%}")


if __name__ == "__main__":
    main()
