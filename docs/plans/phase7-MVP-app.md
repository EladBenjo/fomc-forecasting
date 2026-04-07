## Phase 7 App MVP: Streamlit Integration of Existing Pipeline Outputs

### Summary
- Build a **read-only Streamlit app** now, wired to current artifacts.
- Immediate scope: **Pipeline Status + Feature Explorer + Phase 4 Model Results**.
- Keep **XGBoost (Phase 5)** and **RAG (Phase 6)** explicitly deferred for now, with placeholders/guidance in app UX.
- App must never auto-run training/embedding pipelines; it should show status and exact CLI commands when artifacts are missing.

### Key Implementation Changes
- **App shell and navigation**
  - Add Streamlit app entrypoint and page routing with pages for:
    - Status
    - Features
    - Models
    - RAG (disabled placeholder: "not implemented in this milestone")
- **Artifact-driven data layer**
  - Add a small app data-access layer to read:
    - feature artifact(s): `data/features/doc_level/features.parquet`
    - Phase 4 runs: `data/models/baselines/t5yie/<run_version>/...`
    - optional manifest registry: `data/models/baselines/run_registry.sqlite3`
  - Implement deterministic "latest run" selection and explicit run selection.
  - Centralize missing-artifact checks and return user-facing guidance commands.
- **Page behavior**
  - **Status page:** show availability/freshness of key artifacts and ready/missing states.
  - **Features page:** show basic time-series summaries from existing feature parquet.
  - **Models page:** show `results_table`, `paired_comparison`, run summary, and prediction chart from selected run.
  - **RAG page:** placeholder only (no retrieval/generation yet), with roadmap note.
- **Dependencies and docs**
  - Add Streamlit dependency and any minimal plotting dependency if needed.
  - Update README app run instructions and app section to reflect MVP behavior and deferred Phase 5/6 integration.

### Public Interfaces / Contracts
- App launch command: `streamlit run app/main.py`
- App expects the existing artifact contracts already in repo:
  - Phase 4 run folder files (`predictions.parquet`, `results_table.json`, `paired_comparison.json`, `run_summary.json`, `run_config.json`)
  - Feature parquet for feature views.
- App is **non-mutating**:
  - no pipeline execution from UI
  - no writes to model/data artifacts

### Test Plan
- Add app-focused tests for the data-access layer:
  - latest run discovery
  - artifact parsing/validation
  - missing-artifact behavior and guidance text
- Add lightweight page smoke tests (imports + render helpers) to catch regressions.
- Add one integration-style test fixture with a synthetic Phase 4 run directory to verify end-to-end model page data loading.
- Acceptance checks:
  - app boots with no crashes when artifacts are missing
  - missing sections show actionable CLI guidance
  - with artifacts present, model metrics/predictions display correctly

### Assumptions and Defaults
- Framework: **Streamlit**
- Delivery order: **staged**, but this milestone is **app MVP only**
- Current milestone excludes:
  - Phase 5 XGBoost implementation/integration
  - Phase 6 RAG implementation/integration
- Deferred work will be added in subsequent slices to the existing app pages rather than creating a second app.

