"""Automated smoke and preprocessing tests for Fake News Detection."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import clean_text, extract_description
from src.predict import FakeNewsDetector, format_article_input


def test_clean_text_strips_reuters_leakage() -> None:
    raw = "WASHINGTON (Reuters) - Lawmakers passed the annual budget bill on Tuesday. https://reut.rs/abc123"
    cleaned = clean_text(raw, strip_dateline=True)
    assert "(Reuters)" not in cleaned
    assert "reut.rs" not in cleaned
    assert cleaned.startswith("Lawmakers passed")


def test_clean_text_strips_trailing_boilerplate() -> None:
    raw = "Viral claim spreads online without evidence. Featured image via Getty Images"
    cleaned = clean_text(raw, strip_dateline=True)
    assert "Featured image via" not in cleaned


def test_extract_description_and_format() -> None:
    content = "First sentence summarizes the core report clearly. Second sentence adds extra details."
    desc = extract_description(content)
    assert desc == "First sentence summarizes the core report clearly."
    formatted = format_article_input("Test Headline", desc, content)
    assert "Headline: Test Headline" in formatted
    assert "[SEP] Description:" in formatted
    assert "[SEP] Content:" in formatted


def test_model_inference_if_trained() -> None:
    model_dir = PROJECT_ROOT / "models" / "distilbert_fakenews"
    if not model_dir.exists():
        return
    detector = FakeNewsDetector(model_dir=model_dir)
    res = detector.predict(
        headline="Central bank leaves benchmark rate unchanged amid steady inflation",
        description="Officials cited stable labor data in their policy statement.",
        content="The monetary policy committee voted unanimously to maintain current rates.",
    )
    assert res["prediction"] in {"Credible", "Misleading"}
    assert abs(res["probabilities"]["Credible"] + res["probabilities"]["Misleading"] - 1.0) < 0.01


if __name__ == "__main__":
    test_clean_text_strips_reuters_leakage()
    test_clean_text_strips_trailing_boilerplate()
    test_extract_description_and_format()
    test_model_inference_if_trained()
    print("All unit tests passed!")
