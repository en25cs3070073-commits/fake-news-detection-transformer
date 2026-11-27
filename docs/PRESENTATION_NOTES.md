# Final Demonstration & Viva Preparation Guide (Project 5)

This guide helps every team member explain the dataset, architecture, results, and limitations clearly during evaluation (Aspect 5 — 15 Marks).

---

## 1. 2-Minute Live Demo Script
1. **Problem Statement (20s):** Online news misinformation relies on misleading framing across headlines, summaries, and full article text. We built a Transformer-based NLP classifier (`distilbert-base-uncased`) that evaluates **Headline + Description + Content** jointly as `Credible (0)` vs. `Misleading (1)`.
2. **Data Cleaning & Anti-Leakage Insight (30s):** We cleaned the `GonzaloA/fake_news` corpus, remapped labels (`0 = Credible`, `1 = Misleading`), stripped publisher datelines (`CITY (Reuters) - `) and social footers (`Featured image via ...`), and deduplicated globally *before* splitting so the model learns language patterns rather than source tags.
3. **Experiments & Results (30s):** We benchmarked two classical TF-IDF baselines (Multinomial Naive Bayes and Logistic Regression) against fine-tuned DistilBERT using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
4. **Live App Demo (40s):** Launch `.\run_demo.bat` (`.\.venv\Scripts\python.exe -m streamlit run app.py`), test a credible article and a misleading article live, show the probability breakdown, and highlight the error analysis & limitations tab.

---

## 2. Likely Evaluator Viva Questions & Answers

- **Q: Why use a Transformer (DistilBERT) instead of only TF-IDF + Logistic Regression?**  
  **A:** TF-IDF treats words/n-grams as independent sparse counts and misses long-range context, negation, and tone shifts between headline and body. DistilBERT builds bidirectional contextual embeddings via self-attention while keeping 40% fewer parameters and ~60% faster inference than BERT-base.

- **Q: How did you prevent data leakage?**  
  **A:** First, we stripped `CITY (Reuters) - ` datelines, URLs, and `Featured image via` footers so the model cannot cheat using publisher watermarks. Second, we deduplicated articles globally on normalized `headline + content[:200]` before performing the stratified 70/15/15 split.

- **Q: How did you combine Headline, Description, and Complete Article Content?**  
  **A:** We formatted each example as `Headline: <headline> [SEP] Description: <description> [SEP] Content: <content>` so the Transformer's `[SEP]` tokens explicitly separate structural sections within a unified 256-token window.

- **Q: What are the key limitations of this prototype?**  
  **A:** (1) It classifies linguistic and stylistic patterns learned from its labelled historical dataset rather than performing live external fact-checking. (2) Articles exceeding 256 tokens are truncated. (3) Performance can degrade on new domains or emerging topics without periodic re-training or retrieval-augmented fact verification.
