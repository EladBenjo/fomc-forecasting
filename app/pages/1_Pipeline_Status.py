"""Pipeline status page for Streamlit MVP."""

from __future__ import annotations

import streamlit as st

from app.lib.artifacts import build_status_snapshot

st.title("Pipeline Status")
st.caption("Artifact readiness and generation guidance for the current repo.")

snapshot = build_status_snapshot()
checks = snapshot["checks"]
runs = snapshot["runs"]

for check in checks:
    if check["ready"]:
        st.success(f"{check['label']}: ready")
    else:
        st.warning(f"{check['label']}: missing")
    st.caption(check["path"])
    if not check["ready"]:
        st.code(check["remediation_command"])

st.subheader("Phase 4 Run Discovery")
st.write(f"Discovered runs: {len(runs['all'])}")
st.write(f"Complete runs: {len(runs['complete'])}")
st.write(f"Latest complete run: `{runs['latest_complete']}`" if runs["latest_complete"] else "Latest complete run: none")

if runs["latest_any"] and runs["latest_any_missing_files"]:
    st.info(
        f"Latest discovered run `{runs['latest_any']}` is incomplete. "
        f"Missing files: {', '.join(runs['latest_any_missing_files'])}"
    )

