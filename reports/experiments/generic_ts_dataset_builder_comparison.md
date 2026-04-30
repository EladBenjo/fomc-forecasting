# Generic TS Dataset Builder Experiment (t5yie)

## 1) Audit of current state
- Existing dataset builders: `datasets/build_dataset/builder.py` (monthly) and `datasets/build_dataset/daily_builder.py` (daily).
- Existing datasets generated: `model_dataset_t5yie.parquet` (monthly) and `model_dataset_t5yie_daily.parquet` (daily) plus time split JSON artifacts.
- Existing feature families:
  - Monthly: aggregate text/event features in `alignment.py`.
  - Daily: target AR lag/rolling features + trailing communication windows in `daily_alignment.py`.
- No-lookahead handling:
  - AR features are shift-first.
  - Trailing exogenous features explicitly use trailing windows bounded by target cutoff (`target_date - lag_days`).
- Manifest/registry writing is centralized in `datasets/build_dataset/versioning.py` with JSON manifests and SQLite `dataset_registry` rows.
- Existing model runners:
  - XGBoost: `models/ml/xgboost.py`
  - SARIMAX benchmark: `models/baselines/sarimax.py`
- Metrics available for comparison: RMSE, MAE, directional accuracy, and paired comparison outputs for SARIMAX variants.

## 2) What changed
- Added a reusable feature module (`feature_generators.py`) with config-driven target and trailing exogenous feature generation.
- Added a new generic config-driven builder (`generic_builder.py`) with YAML config loading, split generation, and manifest/registry integration through existing `write_manifest`.
- Added a dataset config (`configs/datasets/t5yie_generic.yaml`) expressing lags/windows/aggs/cutoff/splits/outputs.
- Added tests for generic lag/rolling/trailing/no-lookahead semantics and versioned writes.

## 3) Config used
- `configs/datasets/t5yie_generic.yaml`

## 4) Generated feature groups
- Target AR: configurable lags/rolling mean/rolling std + optional momentum pairs.
- Fed communication trailing windows: configurable windows and aggs (`mean`, `sum`, `max`, `last`, `count`).
- Missingness/event presence:
  - `has_text_signal_{window}d`
  - `event_count_{window}d`
  - `days_since_last_fed_event`

## 5) Leakage checks performed
- Target AR features are built from shifted target history only.
- Exogenous features are computed from strictly trailing windows ending at `target_date - lag_days`.
- Split boundaries are time-based and deterministic via config.

## 6) Dataset + model comparison status
- In this environment, Python dependencies (notably `pandas`) are unavailable for execution, so full artifact generation and model reruns were not completed here.
- The implementation is ready to run locally where dependencies/data artifacts exist.

## 7) Recommendation
- **Modify/keep for experimentation**: keep this generic builder for controlled experiments and tune config recipes.
- Next steps:
  1. Run generic builder locally to generate `model_dataset_t5yie_generic.parquet` and split/manifest artifacts.
  2. Reuse existing XGBoost and SARIMAX runners on the new dataset.
  3. Compare RMSE/MAE/directional accuracy against latest existing runs and decide promote/iterate/discard.
