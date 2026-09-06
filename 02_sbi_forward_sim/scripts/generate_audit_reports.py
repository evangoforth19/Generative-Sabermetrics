#!/usr/bin/env python3
"""Write preprocessing audit markdown under reports/ (no dataset rebuild)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src.audit_reports import write_all_audit_reports  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--production-root",
        type=Path,
        default=_MMC2_ROOT / "Outputs" / "refactor_exact_root_rhh" / "production",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    p.add_argument(
        "--manifest-extra-json",
        type=Path,
        default=None,
        help="Optional path to JSON dict with fair-wedge row counts (from a prior build log).",
    )
    args = p.parse_args()
    extra = {}
    if args.manifest_extra_json and args.manifest_extra_json.is_file():
        extra = json.loads(args.manifest_extra_json.read_text(encoding="utf-8"))
    write_all_audit_reports(
        args.production_root.resolve(),
        args.project_root.resolve(),
        extra,
    )
    print("Wrote reports under", args.project_root / "reports")


if __name__ == "__main__":
    main()
