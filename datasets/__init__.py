"""Dataset pipeline package."""

__all__ = ["BuildDatasetConfig", "build_model_dataset"]


def __getattr__(name: str):
    if name in {"BuildDatasetConfig", "build_model_dataset"}:
        from datasets.build_dataset.builder import BuildDatasetConfig, build_model_dataset

        return {
            "BuildDatasetConfig": BuildDatasetConfig,
            "build_model_dataset": build_model_dataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
