# FOMC Documents Ingestion Design Spec (MVP)

## A) Assumptions & constraints
- **Single source entry page:** `https://www.federalreserve.gov/monetarypolicy/materials/` is the only discovery entry point.
- **MVP scope:** only two document categories are ingested:
  - `STATEMENT` (site label examples: “Policy Statements”)
  - `MINUTES` (site label examples: “Minutes (1993-Present)”)
- **No private/hidden APIs:** extraction is from HTML on the materials page and linked document pages/files only.
- **Deterministic, restartable pipeline:** each stage is idempotent and writes status + timestamps.
- **Stable identity required:** every logical document gets a stable `doc_id` independent of run time.
- **Content formats expected:** HTML pages and PDFs are primary; other file types are allowed but flagged.
- **Compatibility constraint:** artifacts must support later integration with time-series ingestion and modeling, so schemas must include run metadata, provenance, and validation flags.

## B) Folder layout (inside larger repo)
- `src/fedtext/ingest/documents/`
  - `discovery/` — materials-page crawling, link extraction, category mapping
  - `fetch/` — conditional HTTP retrieval and raw storage
  - `parse/` — content normalization into canonical JSON
  - `validators/` — schema and business-rule QC checks
- `src/fedtext/common/`
  - shared HTTP client config, hashing, date normalization, enums, logging helpers
- `configs/`
  - source config (`sources.yaml`) and run mode config
- `data/`
  - `catalog/` (SQLite DB)
  - `raw/documents/` (downloaded files)
  - `normalized/documents/` (parsed JSON)
  - `quarantine/documents/` (poison/failing docs)
- `reports/scraping/`
  - run-level markdown summaries and quality snapshots
- `docs/`
  - design specs and operational notes
- `tests/`
  - stage-level contract tests and regression fixtures

## C) Data artifacts and schemas

### 1) Catalog storage (SQLite preferred)
**DB path:** `data/catalog/documents_catalog.sqlite`

**Table: `documents`** (one row per logical doc)
- `doc_id` TEXT PRIMARY KEY — stable identifier (`sha256(category + '|' + canonical_url)`)
- `category` TEXT NOT NULL — internal enum: `STATEMENT` / `MINUTES`
- `raw_document_type` TEXT NOT NULL — exact site label
- `source_url` TEXT NOT NULL — URL found during discovery
- `canonical_url` TEXT NOT NULL — normalized URL used for identity/dedup
- `title` TEXT — latest known title
- `published_date` TEXT — ISO date (`YYYY-MM-DD`) when available
- `meeting_date` TEXT — ISO date when available
- `content_type` TEXT — HTTP/media type (e.g., `application/pdf`)
- `file_ext` TEXT — normalized extension (`pdf`, `html`, etc.)
- `etag` TEXT — last seen server ETag
- `last_modified` TEXT — last seen HTTP Last-Modified
- `raw_sha256` TEXT — hash of last fetched bytes
- `parser_name` TEXT — parser used to produce normalized output
- `parser_version` TEXT — semantic parser version string
- `discovery_status` TEXT — `PENDING|SUCCESS|FAILED|SKIPPED`
- `fetch_status` TEXT — `PENDING|SUCCESS|FAILED|SKIPPED`
- `parse_status` TEXT — `PENDING|SUCCESS|FAILED|SKIPPED`
- `validate_status` TEXT — `PENDING|SUCCESS|FAILED|SKIPPED`
- `discovery_run_id` TEXT
- `fetch_run_id` TEXT
- `parse_run_id` TEXT
- `validate_run_id` TEXT
- `first_seen_at` TEXT — ISO-8601 timestamp
- `last_seen_at` TEXT — ISO-8601 timestamp
- `updated_at` TEXT — ISO-8601 timestamp

**Table: `document_runs`** (optional but recommended; one row per stage execution per doc)
- `run_id` TEXT
- `doc_id` TEXT
- `stage` TEXT (`discovery|fetch|parse|validate`)
- `status` TEXT
- `attempt` INTEGER
- `error_code` TEXT
- `error_message` TEXT
- `started_at` TEXT
- `finished_at` TEXT

### 2) Raw artifacts
**Storage path pattern:**
- `data/raw/documents/{category}/{yyyy}/{doc_id}/{fetch_ts_utc}.{file_ext}`

**Table: `raw_artifacts`**
- `artifact_id` TEXT PRIMARY KEY (e.g., `sha256(doc_id + '|' + fetch_ts_utc)`)
- `doc_id` TEXT NOT NULL
- `fetch_run_id` TEXT NOT NULL
- `storage_path` TEXT NOT NULL
- `source_url` TEXT NOT NULL
- `http_status` INTEGER
- `content_type` TEXT
- `file_ext` TEXT
- `etag` TEXT
- `last_modified` TEXT
- `byte_size` INTEGER
- `raw_sha256` TEXT NOT NULL
- `fetched_at` TEXT NOT NULL

### 3) Normalized artifact JSON schema
**Storage path pattern:**
- `data/normalized/documents/{category}/{yyyy}/{doc_id}.json`

**JSON object fields (minimum):**
- `doc_id` (string)
- `category` (string enum)
- `raw_document_type` (string)
- `source_url` (string)
- `canonical_url` (string)
- `title` (string)
- `published_date` (string, `YYYY-MM-DD`, nullable)
- `meeting_date` (string, `YYYY-MM-DD`, nullable)
- `content_type` (string)
- `file_ext` (string)
- `raw_sha256` (string)
- `parser_name` (string)
- `parser_version` (string)
- `extracted_text` (string, nullable)
- `sections` (array of `{heading: string, text: string}`; may be empty)
- `metadata` (object: additional extracted keys)
- `pipeline_status` (object with `discovery|fetch|parse|validate` status values)
- `run_ids` (object with `discovery_run_id|fetch_run_id|parse_run_id|validate_run_id`)
- `timestamps` (object with `discovered_at|fetched_at|parsed_at|validated_at` ISO-8601 values)

### 4) Validation flags schema
**Table: `validation_flags`** (many flags per doc)
- `flag_id` TEXT PRIMARY KEY
- `doc_id` TEXT NOT NULL
- `validate_run_id` TEXT NOT NULL
- `severity` TEXT (`INFO|WARN|ERROR`)
- `flag_code` TEXT (e.g., `MISSING_TITLE`, `BAD_DATE`, `EMPTY_TEXT`)
- `flag_message` TEXT
- `rule_name` TEXT
- `created_at` TEXT

## D) Stage-by-stage I/O contract

### Stage 1: Discovery
- **Inputs:**
  - `configs/sources.yaml` (base URL + category mapping rules)
  - HTML of materials page
- **Outputs:**
  - Upsert rows in `documents` with identity/provenance fields and discovery status
  - `discovery_run_id` and timestamps populated
- **Output location:**
  - `data/catalog/documents_catalog.sqlite` (`documents`, optionally `document_runs`)
- **Must NOT do:**
  - Download full document binaries
  - Parse full document text
  - Execute validation rules beyond basic link sanity

### Stage 2: Fetch
- **Inputs:**
  - `documents` rows eligible for fetch (by status/mode/change detection)
- **Outputs:**
  - Raw file saved in `data/raw/documents/...`
  - `raw_artifacts` row inserted
  - `documents` updated with `raw_sha256`, HTTP metadata, fetch status/run/timestamp
- **Output location:**
  - Files: `data/raw/documents/...`
  - DB: `documents`, `raw_artifacts`, optional `document_runs`
- **Must NOT do:**
  - Interpret document semantics beyond MIME/ext detection
  - Produce normalized JSON

### Stage 3: Parse (normalize)
- **Inputs:**
  - Latest successful raw artifact for each target doc
  - Parser config/version
- **Outputs:**
  - Canonical JSON artifact per doc
  - `documents` parse fields/status/run/timestamp updated
- **Output location:**
  - `data/normalized/documents/...`
  - DB: `documents`, optional parse entries in `document_runs`
- **Must NOT do:**
  - Reach back to network except for referenced local artifacts
  - Make policy/business QC pass/fail decisions (leave to validators)

### Stage 4: Validators (QC)
- **Inputs:**
  - Normalized JSON + catalog row + optional raw metadata
- **Outputs:**
  - Validation flags in `validation_flags`
  - `validate_status` and `validate_run_id` on `documents`
  - Run-level markdown report
- **Output location:**
  - DB: `validation_flags`, `documents`, optional `document_runs`
  - Report: `reports/scraping/{run_id}_summary.md`
- **Must NOT do:**
  - Mutate raw files
  - Re-parse documents
  - Invent missing core fields; only flag issues

## E) Incremental update strategy
- **Change detection priority:**
  1. Use conditional requests with stored `etag` and/or `last_modified`.
  2. If unavailable/unreliable, compare `raw_sha256` of newly fetched content.
- **When to refetch:**
  - New `doc_id` discovered.
  - Existing doc with changed `etag`/`last_modified`.
  - Forced refetch mode.
  - Previous fetch failed and retry window allows.
- **When to reparse:**
  - `raw_sha256` changed.
  - Parser version changed.
  - Previous parse failed.
- **Rerun modes:**
  - `full_backfill`: discover + fetch + parse + validate for all in-scope docs.
  - `incremental`: discover all; fetch/parse/validate only new/changed/invalidated docs.
  - `failed_only`: only docs with stage status `FAILED` or quarantined retry-eligible.

## F) Failure handling & logging
- **Network robustness:**
  - Request timeout (connect/read) enforced.
  - Retries with exponential backoff + jitter for transient errors (`429`, `5xx`, timeouts).
  - Respectful rate limiting (fixed minimum interval + burst cap).
- **Poison-document policy:**
  - Track consecutive failures per stage.
  - After max attempts (e.g., 3), set status `FAILED`, emit `POISON_DOCUMENT` flag, move/mark in `data/quarantine/documents/` (metadata marker or copied artifact).
  - Quarantined docs excluded from normal incremental mode unless `failed_only` or explicit override.
- **Structured logging:**
  - Per-event structured logs include `run_id`, `stage`, `doc_id`, `url`, `attempt`, `status`, `latency_ms`, `error_code`.
- **Run summary report:**
  - Markdown at `reports/scraping/{run_id}_summary.md` with:
    - totals by stage/status
    - new vs updated vs unchanged docs
    - failure table with top error codes
    - quarantine additions
    - validation WARN/ERROR counts by rule

## G) Definition of Done (MVP checklist)

### Phase 1: Policy Statements (`STATEMENT`)
- [ ] Discovery identifies statement links from materials page and writes deterministic `doc_id` rows.
- [ ] Fetch stores raw artifacts with hash + HTTP metadata.
- [ ] Parse produces one normalized JSON per fetched statement with required core fields.
- [ ] Validators emit flags and set `validate_status`.
- [ ] Incremental rerun skips unchanged docs.
- [ ] Run summary markdown generated.

### Phase 2: Minutes (`MINUTES`)
- [ ] Category mapping expanded to minutes links from same materials page.
- [ ] Same contracts (discovery/fetch/parse/validate) pass for minutes.
- [ ] Meeting date extraction supported when present; missing values flagged (not silently dropped).
- [ ] Full + incremental + failed-only modes verified for both categories.
- [ ] Catalog and artifacts remain backward-compatible with downstream modeling pipelines.
