"""Robust parquet/CSV I/O and input discovery manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ResolvedSource:
    """One logical artifact and the concrete path + format used."""

    logical_name: str
    resolved_path: str
    format: str  # "parquet" | "csv" | "json" | "missing"
    exists: bool
    note: str = ""


@dataclass
class InputManifest:
    """All sources discovered for a preprocessing run."""

    production_root: str
    sources: list[ResolvedSource] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "production_root": self.production_root,
            "sources": [asdict(s) for s in self.sources],
            "extra": self.extra,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def read_table(path: Path, **read_csv_kw: Any) -> pd.DataFrame:
    """Load parquet or CSV based on suffix."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, **read_csv_kw)
    raise ValueError(f"Unsupported format: {path}")


def read_json_if_exists(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; have sample {list(df.columns)[:30]}...")
