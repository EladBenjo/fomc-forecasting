# FOMC Forecasting

Predicting inflation expectations from Federal Reserve communications using NLP features and
walk-forward machine learning models.

A portfolio project converting original research notebooks into a production-grade pipeline.

---

## What it does

1. **Ingests** Fed speeches (1996–present) and FOMC policy documents (statements, minutes)
2. **Extracts text features** — hawkish/dovish sentiment (FOMC-RoBERTa), meeting-over-meeting
   novelty (TF-IDF cosine distance), topic distributions (LDA)
3. **Joins** with inflation expectation targets from FRED (Michigan Survey, 5yr breakeven)
4. **Trains** walk-forward models — AR baselines → SARIMAX → XGBoost — with no lookahead leakage
5. **Serves** a Streamlit demo with pipeline status, feature explorer, model results, and a
   RAG chat interface grounded in source documents

---

## End deliverables

| Deliverable | Description |
| --- | --- |
| `notebooks/` | End-to-end pipeline narrative (EDA → features → models) |
| Streamlit app | 4-page interactive demo (status · features · models · RAG chat) |

---

## Architecture

```text
federalreserve.gov
    │
    ▼
Discovery → Fetch → Parse → Chunk
    │
    ▼
data/catalog/fedtext.db          ← unified SQLite (speeches + documents + chunks)
    │
    ├── text features             → data/features/doc_level/*.parquet
    │       sentiment (FOMC-RoBERTa)
    │       novelty   (TF-IDF cosine distance)
    │       topics    (LDA)
    │
    ├── FRED targets              → data/targets/*.parquet
    │
    └── embeddings (sqlite-vec)   ← RAG retrieval
            │
            ▼
        Streamlit app  /  Jupyter notebooks
```

---

## Feature engineering

### Sentiment — FOMC-RoBERTa

**Why sentence-level, not document-level:** The model has a 512-token limit. A typical Fed
speech is 5,000+ tokens. Passing a full document would silently truncate after the first few
paragraphs and score only the opening, discarding most of the signal.

**Why economic keyword pre-filtering:** ~80% of sentences in Fed documents are procedural
boilerplate ("The meeting was adjourned", "The vote was unanimous", footnotes). Filtering to
sentences that mention inflation, interest rates, labor market, or financial stability terms
cuts inference time ~5× and removes noise from the aggregate score.

**Score formula:** `(n_hawkish − n_dovish) / n_target_sentences ∈ [−1, 1]`
Labels: `LABEL_0` = Dovish, `LABEL_1` = Hawkish, `LABEL_2` = Neutral.

### Novelty — TF-IDF cosine distance

**Why TF-IDF, not embeddings:** We want lexical novelty — new vocabulary and new policy
phrases — not semantic similarity. Embedding-based cosine similarity would score a paraphrased
statement as nearly identical to the original even when new terminology was introduced. That
vocabulary shift is exactly the signal we want to capture.

**Computed per source type:** Each statement is compared to the previous statement; each speech
to the previous speech. Cross-type comparison (statement vs. speech) would be noise.

### Topics — intentionally omitted

The research notebook implemented both rule-based (keyword counting) and zero-shot
(BART-large-MNLI) topic classification. Both approaches produced poor class separation across
meetings, and topic features were excluded from the final model as a result. This is documented
here rather than silently absent.

---

## Current status

| Phase | Description | Status |
| ----- | ----------- | ------ |
| 1 | Ingest hardening (versioned migrations, validators, YAML config) | done |
| 1.5 | DB consolidation (`speeches.db` + `catalog.sqlite` → `fedtext.db`) | done |
| 2 | Feature engineering (sentiment, novelty, topics) | next |
| 3 | Target variable + dataset builder (FRED) | — |
| 4 | Baseline models (AR, SARIMAX) | — |
| 5 | ML models (XGBoost) | — |
| 6 | RAG layer (sentence-transformers + sqlite-vec) | — |
| 7 | Streamlit demo app | — |

**Data as of Phase 1.5:** 1,932 speeches · 534 FOMC documents · unified in `fedtext.db`

---

## Quickstart

```bash
# 1. Install (editable)
pip install -e .

# 2. Ingest speeches (1996–present)
python -m fedtext.ingest.speeches.pipeline

# 3. Ingest FOMC documents (statements + minutes)
python -m fedtext.ingest.documents.pipeline --categories St Mn

# 4. Validate data quality
python -m fedtext.ingest.validators.completeness

# 5. (One-time) consolidate legacy DBs if you have speeches.db / catalog.sqlite
python scripts/consolidate_dbs.py
```

For faster testing, add `--limit N` to any ingest command.

---

## Data sources

| Source | Content | URL |
| ------ | ------- | --- |
| Federal Reserve | Governor/President speeches | federalreserve.gov/newsevents/speech |
| Federal Reserve | FOMC statements (St) | federalreserve.gov/monetarypolicy |
| Federal Reserve | FOMC minutes (Mn) | federalreserve.gov/monetarypolicy |
| FRED | Michigan inflation expectations (`MICH`) | fred.stlouisfed.org |
| FRED | 5yr breakeven inflation (`T5YIE`) | fred.stlouisfed.org |

---

## Project layout

```text
src/fedtext/
├── common/          # db connection, path constants
├── ingest/
│   ├── speeches/    # discovery + fetch
│   ├── documents/   # discovery + fetch + parse
│   ├── storage/     # versioned SQL migrations
│   └── validators/  # data quality checks
└── text/            # chunker, embedder (stubs for Phase 6)

configs/             # sources.yaml — URLs, rate limits, categories
data/catalog/        # fedtext.db (SQLite)
scripts/             # one-off utilities (consolidate_dbs.py)
notebooks/           # research + portfolio notebooks
docs/                # roadmap
```

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the detailed phase breakdown.

---

## References

Shah, A., Papadopoulos, S., & Guo, T. (2023). *Trillion Dollar Words: A New Financial Dataset,
Task & Market Analysis*. Proceedings of ACL 2023. Georgia Tech FinTechLab.
Model: [gtfintechlab/FOMC-RoBERTa](https://huggingface.co/gtfintechlab/FOMC-RoBERTa)

---

## Data usage note

All data is scraped from public Federal Reserve websites (federalreserve.gov). US government works
are not subject to copyright (17 U.S.C. § 105). Scraping is rate-limited and non-commercial.
