from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _StubCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


class _StreamlitStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")

    def columns(self, spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_StubCtx() for _ in range(count)]

    def container(self, *args, **kwargs):
        return _StubCtx()

    def expander(self, *args, **kwargs):
        return _StubCtx()

    def selectbox(self, label, options, index=0, **kwargs):
        if not options:
            return None
        return options[index]

    def stop(self):
        raise RuntimeError("streamlit.stop() invoked in smoke test")

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


def _import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_streamlit_pages_import_without_crashing(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", _StreamlitStub())

    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "app" / "main.py",
        repo_root / "app" / "pages" / "1_Pipeline_Status.py",
        repo_root / "app" / "pages" / "2_Feature_Explorer.py",
        repo_root / "app" / "pages" / "3_Model_Results.py",
        repo_root / "app" / "pages" / "4_RAG_Chat.py",
    ]

    for idx, path in enumerate(targets):
        try:
            _import_from_path(f"app_smoke_module_{idx}", path)
        except RuntimeError as exc:
            # streamlit.stop() is acceptable in smoke tests for missing artifacts.
            assert "streamlit.stop()" in str(exc)

