# Project 5: Fake News Detection Using Transformer Models

**Problem Statement:** Given a collection of online news articles containing headlines, descriptions, and complete article content, develop an NLP-based system to classify news as potentially misleading or credible using contextual language representations.

**Expected Outcome:** A transformer-based news classification prototype that estimates whether an article resembles misleading or credible examples in its labelled dataset.

---

## 1. Project Structure

```text
fake_news_detection_transformer/
├── app.py                                      # Interactive Streamlit web demo
├── requirements.txt                            # Python dependencies
├── run_all.ps1                                 # One-click full pipeline runner
├── run_demo.ps1                                # One-click Streamlit demo launcher
├── TEAM_CONTRIBUTIONS.md                       # Team registration & contribution record
├── docs/
│   └── PRESENTATION_NOTES.md                   # Demo script & evaluator viva Q&A
├── notebooks/
│   └── Fake_News_Detection_Transformer.ipynb   # Executed end-to-end Jupyter Notebook
├── src/
│   ├── data_prep.py                            # Download, cleaning, leakage removal, EDA, splits
│   ├── train_baseline.py                       # TF-IDF + Naive Bayes & Logistic Regression
│   ├── train_transformer.py                    # DistilBERT GPU fine-tuning & evaluation plots
│   ├── predict.py                              # Inference wrapper (FakeNewsDetector) & CLI
│   └── build_notebook.py                       # Reproducible notebook generator
├── tests/
│   └── test_pipeline.py                        # Unit & inference smoke tests
├── data/processed/                             # Stratified train.csv, val.csv, test.csv
├── models/                                     # Saved DistilBERT & baseline checkpoints
└── outputs/
    ├── eda_summary.json                        # Dataset cleaning & split statistics
    ├── comparison_metrics.json                 # Baseline vs. DistilBERT test/val metrics
    ├── error_analysis.csv                      # High-confidence misclassified test articles
    └── figures/                                # EDA, training curves, confusion matrix, ROC
```

---

## 2. Dataset & Preprocessing (`src/data_prep.py`)

- **Source:** `GonzaloA/fake_news` (40,587 raw English news articles).
- **Label Normalization:** Mapped `0 = Credible` and `1 = Misleading` so positive-class metrics (Precision, Recall, F1) directly measure detection of misleading articles.
- **Data-Leakage Prevention:** Stripped leading wire-service datelines (e.g., `WASHINGTON (Reuters) - `), trailing social-media footers (`Featured image via ...`, `Read more: ...`), and URLs (`reut.rs/...`) so models learn contextual language rather than publisher watermarks.
- **Cleaning & Deduplication:** Removed 343 empty/short articles and 137 duplicates globally *before* splitting (`40,107` clean articles).
- **Structured Input & Stratified Split:** Constructed `headline`, `description` (lead summary sentence), and `content` (full body), then created a stratified 70% / 15% / 15% split on an 18,000-article modeling subset:
  - **Train:** `12,600` | **Validation:** `2,700` | **Test:** `2,700`
  - **Class Balance:** `9,740` Credible (`0`) vs. `8,260` Misleading (`1`)

---

## 3. Experimental Results (Held-Out Test Set, n = 2,700)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time (s) |
|---|---|---|---|---|---|---|
| TF-IDF + Multinomial Naive Bayes | 0.9374 | 0.9154 | 0.9516 | 0.9331 | 0.9809 | 15.64 |
| TF-IDF + Logistic Regression | 0.9648 | 0.9576 | 0.9661 | 0.9618 | 0.9925 | 14.20 |
| **DistilBERT (`distilbert-base-uncased`)** | **0.9822** | **0.9768** | **0.9847** | **0.9807** | **0.9991** | 252.49 |

- Fine-tuned **DistilBERT** reduced total test errors from **95** (`53 FP + 42 FN` with TF-IDF + Logistic Regression) down to **48** (`29 FP + 19 FN`) — nearly a **50% error reduction**.
- Generated visualizations in `outputs/figures/`: `class_distribution.png`, `text_length_distribution.png`, `training_history.png`, `model_comparison.png`, `confusion_matrix_distilbert.png`, and `roc_curves.png`.

---

## 4. Limitations & Error Analysis

- **Stylistic vs. Factual Classification:** The model predicts whether an article's contextual phrasing resembles credible or misleading examples in its labelled training set; it does **not** perform live external fact-checking.
- **Sequence Truncation:** Inputs are truncated at `max_length = 256` WordPiece tokens (`Headline [SEP] Description [SEP] Content`), which may omit claims appearing deep inside very long articles.
- **Label Noise in Public Corpora:** Qualitative inspection of `outputs/error_analysis.csv` shows occasional noisy ground-truth labels in the underlying public corpus (e.g., opinion columns or viral snippets).

---

## 5. How to Run

### Launch the Interactive Web Demo
```powershell
.\run_demo.bat
```
*(Or run directly without scripts: `.\.venv\Scripts\python.exe -m streamlit run app.py`)*

### Run CLI Inference
```powershell
.\.venv\Scripts\python.exe -m src.predict `
  --headline "Central bank holds benchmark interest rate steady" `
  --description "Policymakers cited steady labor data and cooling inflation." `
  --content "The monetary policy committee voted unanimously to leave its benchmark lending rate unchanged on Wednesday."
```

### Re-Run Full Pipeline & Notebook
```powershell
.\run_all.bat
```
