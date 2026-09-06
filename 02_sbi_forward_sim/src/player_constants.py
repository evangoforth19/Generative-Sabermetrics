"""One row per player: bat / COM / inertia constants for decoder and SBI."""

from __future__ import annotations

import pandas as pd

from .schema import P_COLUMNS, SELECTED_EVENTS_BAT_RENAME


def build_player_constants_from_selected_events(selected_events: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate exogenous bat constants from production selected_events (one row per event).

    Uses the same physical fields the inverse pipeline wrote onto each event; one row
    per batter_name via first non-null per column within group.
    """
    if "batter_name" not in selected_events.columns:
        raise ValueError("selected_events must contain batter_name")

    sub = selected_events.copy()
    sub["batter_name"] = sub["batter_name"].astype(str).str.strip().str.lower()

    # Rename L_in, W_oz → canonical p names when present
    rename_map = {k: v for k, v in SELECTED_EVENTS_BAT_RENAME.items() if k in sub.columns}
    sub = sub.rename(columns=rename_map)

    needed = ["batter_name"] + [c for c in P_COLUMNS if c in sub.columns]
    sub = sub[needed].copy()

    # One row per player: first complete-ish row — use median for numeric stability
    rows = []
    for name, g in sub.groupby("batter_name"):
        row: dict = {"batter_name": name}
        for c in P_COLUMNS:
            if c not in g.columns:
                row[c] = float("nan")
                continue
            s = pd.to_numeric(g[c], errors="coerce").dropna()
            row[c] = float(s.median()) if len(s) else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    for c in P_COLUMNS:
        if c not in out.columns:
            out[c] = float("nan")
    cols = ["batter_name"] + P_COLUMNS
    return out[cols].sort_values("batter_name").reset_index(drop=True)
