# FOMC Forecasting

Forecast changes in inflation expectations from Federal Reserve communications.

This project turns public Fed speeches and FOMC documents into text features, joins them to FRED inflation-expectation targets, builds leak-free time-series datasets, and evaluates forecasting models.

Generated artifacts live under `data/` and are intentionally local.

## Business Problem

Fed communications often move before, or alongside, market inflation expectations. The project asks:

- Can policy tone, novelty, and event frequency improve forecasts of `T5YIE` changes?
- Which communication windows matter most?
- When should a simple univariate baseline remain the default?

## Architecture

```text
Federal Reserve pages -> ingest -> data/catalog/fedtext.db
Fed text              -> ZettaQuant + local features -> data/features/doc_level/*.parquet
FRED T5YIE            -> target pipeline -> data/targets/t5yie_*.parquet
features + target     -> dataset builders -> data/targets/model_dataset_*.parquet
datasets              -> SARIMAX / XGBoost -> data/models/**
artifacts             -> Streamlit dashboard
```

Key design rules:

- No-lookahead feature alignment.
- Versioned datasets and model runs with manifests/registries.
- Read-only dashboard over existing artifacts.
- Local-only generated data and model outputs.

## What Exists

- Fed speech and FOMC document ingest pipelines.
- Document-level text features:
  - ZettaQuant central-bank relevancy and stance labels.
  - Local novelty and event/count features.
  - SQLite checkpointing and sentence-level inference cache for resumable feature runs.
- FRED target pipeline for `T5YIE` raw and `diff1` outputs.
- Dataset builders:
  - `datasets.build_dataset.daily_builder`: current model-runner dataset schema.
  - `datasets.build_dataset.generic_builder`: config-driven experiment builder using `configs/datasets/t5yie_generic.yaml`.
  - `datasets.build_dataset.builder`: older monthly builder kept for compatibility with older artifacts/dashboard paths.
- Model runners:
  - SARIMAX baseline vs exogenous variants.
  - XGBoost daily model runner.
- Streamlit economist dashboard reading generated artifacts.
- Research notebooks and experiment reports under `notebooks/` and `reports/`.

## Run Locally

Install:

```powershell
pip install -e .
```

Ingest Fed communications:

```powershell
python -m fedtext.ingest.speeches.pipeline
python -m fedtext.ingest.documents.pipeline --categories St Mn
python -m fedtext.ingest.validators.completeness
```

Build text features:

```powershell
$env:ZQ_API_KEY="your_key_here"
python -m fedtext.text.features.pipeline
```

Build the target series:

```powershell
$env:FRED_API_KEY="your_key_here"
python -m fedtext.targets.pipeline --series-id T5YIE --transform diff1
```

Build modeling datasets:

```powershell
# Current model-runner schema
python -m datasets.build_dataset.daily_builder

# Config-driven experiment dataset
python -m datasets.build_dataset.generic_builder --config configs/datasets/t5yie_generic.yaml
```

Run models:

```powershell
python -m models.baselines.sarimax
python -m models.ml.xgboost
```

Launch the dashboard:

```powershell
streamlit run app/main.py
```

Run tests:

```powershell
python -m pytest -q
```

## Project Layout

```text
src/fedtext/          ingest, text features, FRED target pipeline
datasets/            dataset alignment, builders, manifests
models/              SARIMAX, XGBoost, metrics, run tracking
app/                 Streamlit dashboard
configs/             source and dataset configs
notebooks/           research and exploratory analysis
reports/             generated summaries and experiment notes
tests/               unit and integration tests
```

## Coming Next

- Compare generic-builder outputs against current daily model datasets.
- Decide whether the generic builder becomes the default modeling path.
- Align dashboard artifact loading with the promoted dataset path.
- Extend target coverage beyond `T5YIE`.
- Add retrieval/RAG over source Fed documents.

## Data Sources

- Federal Reserve speeches and FOMC documents from `federalreserve.gov`.
- FRED inflation-expectation series, currently `T5YIE`.
- ZettaQuant central-bank inference API for relevancy and stance labels.
