"""Canonical path constants for the fedtext package.

All modules should import paths from here — never construct DB or data
paths inline in business logic.
"""

from pathlib import Path

# Repository root (four parents up from this file: common → fedtext → src → repo)
REPO_ROOT = Path(__file__).parents[3]

DATA_DIR    = REPO_ROOT / "data"
CATALOG_DIR = DATA_DIR / "catalog"
FEATURES_DIR = DATA_DIR / "features"
TARGETS_DIR  = DATA_DIR / "targets"
SPLITS_DIR   = DATA_DIR / "splits"
RAW_DIR      = DATA_DIR / "raw"

FEDTEXT_DB   = CATALOG_DIR / "fedtext.db"

# Legacy paths — kept for reference during migration only
SPEECHES_DB  = CATALOG_DIR / "speeches.db"
DOCUMENTS_DB = CATALOG_DIR / "catalog.sqlite"

CONFIGS_DIR  = REPO_ROOT / "configs"
REPORTS_DIR  = REPO_ROOT / "reports"
