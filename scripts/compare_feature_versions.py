"""Compare two feature dataset manifests and report diffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "hawkish_score",
    "n_hawkish",
    "n_dovish",
    "n_neutral",
    "n_target_sentences",
    "novelty",
]


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_stats(df: pd.DataFrame, col: str) -> dict[str, float | None]:
    if col not in df.columns or df.empty:
        return {"mean": None, "std": None, "min": None, "max": None}
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def compare(manifest_a: Path, manifest_b: Path, top_changed: int = 20) -> str:
    a = _load_manifest(manifest_a)
    b = _load_manifest(manifest_b)

    path_a = Path(a["output_path"])
    path_b = Path(b["output_path"])

    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    merged = df_a.merge(
        df_b,
        on=["doc_id", "source_type"],
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
    )

    changed = merged[merged["_merge"] == "both"].copy()
    if "hawkish_score_a" in changed.columns and "hawkish_score_b" in changed.columns:
        changed["abs_hawkish_delta"] = (
            changed["hawkish_score_b"].astype(float) - changed["hawkish_score_a"].astype(float)
        ).abs()
        changed = changed.sort_values("abs_hawkish_delta", ascending=False)

    lines: list[str] = []
    lines.append(f"# Feature dataset comparison: {a['dataset_version']} vs {b['dataset_version']}")
    lines.append("")
    lines.append("## Manifest-level metadata")
    lines.append(f"- A created_at: {a.get('created_at_utc')}")
    lines.append(f"- B created_at: {b.get('created_at_utc')}")
    lines.append(f"- A git_sha: {a.get('git_sha')}")
    lines.append(f"- B git_sha: {b.get('git_sha')}")
    lines.append(f"- A cleaning_version: {a.get('cleaning_version')}")
    lines.append(f"- B cleaning_version: {b.get('cleaning_version')}")
    lines.append(f"- A sentence_split_version: {a.get('sentence_split_version')}")
    lines.append(f"- B sentence_split_version: {b.get('sentence_split_version')}")
    lines.append("")

    lines.append("## Row-level deltas")
    lines.append(f"- Rows A: {len(df_a)}")
    lines.append(f"- Rows B: {len(df_b)}")
    lines.append(f"- Added rows in B: {(merged['_merge'] == 'right_only').sum()}")
    lines.append(f"- Removed rows from A: {(merged['_merge'] == 'left_only').sum()}")
    lines.append("")

    lines.append("## Numeric feature distribution summaries")
    for col in NUMERIC_COLUMNS:
        sa = _safe_stats(df_a, col)
        sb = _safe_stats(df_b, col)
        lines.append(f"### {col}")
        lines.append(f"- A mean/std/min/max: {sa['mean']} / {sa['std']} / {sa['min']} / {sa['max']}")
        lines.append(f"- B mean/std/min/max: {sb['mean']} / {sb['std']} / {sb['min']} / {sb['max']}")

    lines.append("")
    lines.append(f"## Top {top_changed} docs by absolute hawkish score delta")
    if "abs_hawkish_delta" in changed.columns and not changed.empty:
        head = changed[["doc_id", "source_type", "hawkish_score_a", "hawkish_score_b", "abs_hawkish_delta"]].head(top_changed)
        lines.append(head.to_markdown(index=False))
    else:
        lines.append("No overlapping rows with comparable hawkish scores.")

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two feature dataset manifests")
    p.add_argument("manifest_a", type=Path)
    p.add_argument("manifest_b", type=Path)
    p.add_argument("--out", type=Path, default=None, help="Optional output Markdown path")
    p.add_argument("--top-changed", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = compare(args.manifest_a, args.manifest_b, top_changed=args.top_changed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote comparison report to {args.out}")
    else:
        print(report)
