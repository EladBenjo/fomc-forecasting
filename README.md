# FOMC Forecasting

Predicting inflation expectations from Federal Reserve communications using NLP features and walk-forward machine learning models.

A portfolio project converting original research notebooks into a production-grade pipeline.

---

## What it does

1. **Ingests** Fed speeches (1996-present) and FOMC policy documents (statements, minutes)
2. **Extracts text features** - hawkish/dovish sentiment via ZettaQuant central bank classifiers, meeting-over-meeting novelty (TF-IDF cosine distance), and topic diagnostics
3. **Joins** with inflation expectation targets from FRED (Michigan Survey, 5yr breakeven)
4. **Trains** walk-forward models - AR baselines -> SARIMAX -> XGBoost - with no lookahead leakage
5. **Serves** a Streamlit demo with pipeline status, feature explorer, model results, and a RAG chat interface grounded in source documents

---

## End deliverables

| Deliverable | Description |
| --- | --- |
| `notebooks/` | End-to-end pipeline narrative (EDA -> features -> models) |
| Streamlit app | 4-page interactive demo (status, features, models, RAG chat) |

---

## Architecture

```text
federalreserve.gov
    |
    v
Discovery -> Fetch -> Parse -> Chunk
    |
    v
data/catalog/fedtext.db          <- unified SQLite (speeches + documents + chunks)
    |
    |-- text features             -> data/features/doc_level/features.parquet
    |      sentiment (ZettaQuant API: relevancy + stance)
    |      novelty   (TF-IDF cosine distance)
    |      topics    (diagnostic only)
    |
    |-- FRED targets              -> data/targets/*.parquet
    |
    `-- embeddings (sqlite-vec)   <- RAG retrieval
            |
            v
        Streamlit app / Jupyter notebooks
```

---

## Feature engineering

### Sentiment - ZettaQuant API (refactored)

The sentiment stack was refactored from a local Hugging Face model path to ZettaQuant central bank classifiers served through the ZettaQuant API.

**Models used**
- `cb_inflation_relevancy_label` (Central Bank Inflation Relevancy Classifier)
- `cb_stance_label` (Central Bank Stance Classifier)

**Per-document flow (sentence-level)**
1. Normalize and split text into sentences.
2. Call ZettaQuant relevancy classifier in batches.
3. Keep only relevant sentences.
4. Call ZettaQuant stance classifier on relevant sentences.
5. Aggregate score as `(n_hawkish - n_dovish) / n_target_sentences`, where stance label `Irrelevant` is excluded from `n_target_sentences`.

**Why this refactor**
- Uses newer central-bank-specific models managed by ZettaQuant.
- Removes local model download/auth friction from the feature pipeline.
- Improves maintainability with a single API inference client and model-id routing.

**Why this is easier to extend with ZettaQuant Studio models**
- The sentiment module now routes inference by model ID through one shared API client.
- Adding a new Studio model is a narrow change: define model ID + label mapping + aggregation logic, without changing pipeline orchestration.

**Runtime configuration**
- Required: `ZQ_API_KEY`
- Optional: `ZQ_BATCH_SIZE` (default `10`), `ZQ_MAX_REQ_PER_MIN` (default `10`), `ZQ_TIMEOUT_SECONDS` (default `30`)

### Novelty - TF-IDF cosine distance

**Why TF-IDF, not embeddings:** We want lexical novelty - new vocabulary and new policy phrases - not semantic similarity. Embedding-based cosine similarity would score a paraphrased statement as nearly identical to the original even when new terminology was introduced. That vocabulary shift is exactly the signal we want to capture.

**Computed per source type:** Each statement is compared to the previous statement; each speech to the previous speech. Cross-type comparison (statement vs. speech) would be noise.

### Topics - intentionally omitted from final model

The research notebook implemented both rule-based (keyword counting) and zero-shot (BART-large-MNLI) topic classification. Both approaches produced poor class separation across meetings, and topic features were excluded from the final predictive model.

---

## Current status

| Phase | Description | Status |
| ----- | ----------- | ------ |
| 1 | Ingest hardening (versioned migrations, validators, YAML config) | done |
| 1.5 | DB consolidation (`speeches.db` + `catalog.sqlite` -> `fedtext.db`) | done |
| 2 | Feature engineering (ZettaQuant sentiment, novelty, topics) | done |
| 3 | Target variable + dataset builder (FRED) | next |
| 4 | Baseline models (AR, SARIMAX) | - |
| 5 | ML models (XGBoost) | - |
| 6 | RAG layer (sentence-transformers + sqlite-vec) | - |
| 7 | Streamlit demo app | - |

**Data as of Phase 1.5:** 1,932 speeches, 534 FOMC documents, unified in `fedtext.db`.

---

## Quickstart

```bash
# 1. Install (editable)
pip install -e .

# 2. Ingest speeches (1996-present)
python -m fedtext.ingest.speeches.pipeline

# 3. Ingest FOMC documents (statements + minutes)
python -m fedtext.ingest.documents.pipeline --categories St Mn

# 4. Validate data quality
python -m fedtext.ingest.validators.completeness

# 5. Configure ZettaQuant API key (required for sentiment features)
# PowerShell:
#   $env:ZQ_API_KEY="your_key_here"
# Bash:
#   export ZQ_API_KEY="your_key_here"

# 6. Build text features
python -m fedtext.text.features.pipeline
```

For faster testing, add `--limit N` to ingest/feature commands.

---

## Data sources

| Source | Content | URL |
| ------ | ------- | --- |
| Federal Reserve | Governor/President speeches | https://www.federalreserve.gov/newsevents/speech.htm |
| Federal Reserve | FOMC statements/minutes | https://www.federalreserve.gov/monetarypolicy.htm |
| FRED | Michigan inflation expectations (`MICH`) | https://fred.stlouisfed.org |
| FRED | 5yr breakeven inflation (`T5YIE`) | https://fred.stlouisfed.org |
| ZettaQuant | Central bank model inference API | https://api.zettaquant.ai |

---

## Project layout

```text
src/fedtext/
|-- common/          # db connection, path constants
|-- ingest/
|   |-- speeches/    # discovery + fetch
|   |-- documents/   # discovery + fetch + parse
|   |-- storage/     # versioned SQL migrations
|   `-- validators/  # data quality checks
`-- text/            # cleaning, chunker, features

configs/             # sources.yaml - URLs, rate limits, categories
data/catalog/        # fedtext.db (SQLite)
scripts/             # one-off utilities
notebooks/           # research + portfolio notebooks
docs/                # roadmap
```

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the detailed phase breakdown.

---

## References

- ZettaQuant API, model inference endpoint and central bank classifiers (`cb_inflation_relevancy_label`, `cb_stance_label`): https://api.zettaquant.ai
- ZettaQuant platform and Studio (for managed model lifecycle and deployment): https://api.zettaquant.ai/v1/models
- Shah, A., Papadopoulos, S., & Guo, T. (2023). *Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis*. Proceedings of ACL 2023. (historical context for earlier model choices)

---

## Data usage note

All data is scraped from public Federal Reserve websites (`federalreserve.gov`). US government works are not subject to copyright (17 U.S.C. Section 105). Scraping is rate-limited and non-commercial.
