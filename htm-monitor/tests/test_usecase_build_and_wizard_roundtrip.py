#tests/test_usecase_build_and_wizard_roundtrip.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from demo.make_usecase_config import build_usecase_config, sources_from_dicts, sources_to_dicts, SourceSpec


def _write_csv(tmp_path: Path, name: str, values):
    p = tmp_path / name
    df = pd.DataFrame(
        {
            "timestamp": [f"2011-01-01 00:00:{i:02d}" for i in range(len(values))],
            "value": list(values),
        }
    )
    df.to_csv(p, index=False)
    return str(p)


def test_build_spec_roundtrip_builds_same_config(tmp_path: Path):
    csv_a = _write_csv(tmp_path, "a.csv", [0.0, 1.0, 2.0, 3.0, 100.0])
    csv_b = _write_csv(tmp_path, "b.csv", [10.0, 11.0, 12.0, 13.0, 14.0])

    sources = [
        SourceSpec(
            name="sa",
            path=csv_a,
            timestamp_col="timestamp",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            fields={"feat_a": "value"},
        ),
        SourceSpec(
            name="sb",
            path=csv_b,
            timestamp_col="timestamp",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            fields={"feat_b": "value"},
        ),
    ]

    params: Dict[str, Any] = dict(
        low_q=0.01,
        high_q=0.99,
        margin=0.03,
        seed_base=42,
        model_layout="separate",
    )

    cfg_direct = build_usecase_config("my_usecase", sources, **params)

    spec = {
        "usecase": "my_usecase",
        "sources": sources_to_dicts(sources),
        "params": params,
    }

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))

    loaded = yaml.safe_load(spec_path.read_text())
    sources2 = sources_from_dicts(loaded["sources"])
    cfg_from_spec = build_usecase_config(loaded["usecase"], sources2, **loaded["params"])

    assert cfg_from_spec == cfg_direct