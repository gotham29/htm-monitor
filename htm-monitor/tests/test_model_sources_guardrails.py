# tests/test_model_sources_guardrails.py

from __future__ import annotations

import pytest

from htm_monitor.utils.config import _model_sources


def test_model_sources_accepts_source_string():
    assert _model_sources("m", {"source": "s1"}) == ["s1"]


def test_model_sources_accepts_sources_list():
    assert _model_sources("m", {"sources": ["s1", "s2"]}) == ["s1", "s2"]


def test_model_sources_rejects_both_present():
    with pytest.raises(ValueError, match="exactly one of"):
        _model_sources("m", {"source": "s1", "sources": ["s2"]})


def test_model_sources_rejects_neither_present():
    with pytest.raises(ValueError, match="exactly one of"):
        _model_sources("m", {"features": ["timestamp", "x"]})


def test_model_sources_rejects_empty_source():
    with pytest.raises(ValueError, match="non-empty string"):
        _model_sources("m", {"source": ""})


def test_model_sources_rejects_empty_sources():
    with pytest.raises(ValueError, match="non-empty list"):
        _model_sources("m", {"sources": []})


def test_model_sources_rejects_bad_sources_entries():
    with pytest.raises(ValueError, match="entries must be non-empty strings"):
        _model_sources("m", {"sources": ["ok", ""]})


def test_model_sources_rejects_non_string_source():
    with pytest.raises(ValueError, match="non-empty string"):
        _model_sources("m", {"source": 123})  # type: ignore[arg-type]


def test_model_sources_rejects_sources_not_a_list():
    with pytest.raises(ValueError, match="non-empty list"):
        _model_sources("m", {"sources": "s1"})  # type: ignore[arg-type]


def test_model_sources_rejects_sources_with_non_string_entries():
    with pytest.raises(ValueError, match="entries must be non-empty strings"):
        _model_sources("m", {"sources": ["ok", 5]})  # type: ignore[list-item]
