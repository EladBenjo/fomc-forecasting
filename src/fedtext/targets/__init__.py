"""Target-series pipelines and versioning helpers."""

from fedtext.targets.pipeline import apply_transform, fetch_series, run
from fedtext.targets.versioning import TargetDatasetManifest, default_dataset_version, write_manifest

__all__ = [
    "TargetDatasetManifest",
    "apply_transform",
    "default_dataset_version",
    "fetch_series",
    "run",
    "write_manifest",
]

