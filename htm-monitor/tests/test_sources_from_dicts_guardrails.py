#tests/test_sources_from_dicts_guardrails.py

from __future__ import annotations

import pytest

from demo.make_usecase_config import sources_from_dicts


def test_sources_from_dicts_rejects_missing_required_key():
    with pytest.raises(ValueError, match="missing required key"):
        sources_from_dicts(
            [
                {
                    "name": "s",
                    # missing path
                    "timestamp_col": "timestamp",
                    "timestamp_format": "%Y-%m-%d %H:%M:%S",
                    "fields": {"feat": "value"},
                }
            ]
        )


def test_sources_from_dicts_rejects_empty_fields():
    with pytest.raises(ValueError, match=r"\.fields must be a non-empty mapping"):
        sources_from_dicts(
            [
                {
                    "name": "s",
                    "path": "/tmp/x.csv",
                    "timestamp_col": "timestamp",
                    "timestamp_format": "%Y-%m-%d %H:%M:%S",
                    "fields": {},
                }
            ]
        )


def test_sources_from_dicts_rejects_non_string_field_keys():
    with pytest.raises(ValueError, match="fields must have string keys"):
        sources_from_dicts(
            [
                {
                    "name": "s",
                    "path": "/tmp/x.csv",
                    "timestamp_col": "timestamp",
                    "timestamp_format": "%Y-%m-%d %H:%M:%S",
                    "fields": {1: "value"},  # YAML can produce this
                }
            ]
        )