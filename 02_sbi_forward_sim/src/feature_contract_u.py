"""Frozen, strictly validated feature contract for stage-u p(u|g,p)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT_RELPATH = "configs/feature_contracts/p_u_given_g_feature_contract.json"


def _resolve_project_root(project_root: Path | None) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[1]
    return Path(project_root).resolve()


def resolve_feature_contract_path(cfg: dict[str, Any], project_root: Path | None = None) -> Path:
    root = _resolve_project_root(project_root)
    rel = cfg.get("paths", {}).get("feature_contract", DEFAULT_CONTRACT_RELPATH)
    p = (root / str(rel)).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Stage-u feature contract not found: {p}")
    return p


def load_feature_contract(cfg: dict[str, Any], project_root: Path | None = None) -> dict[str, Any]:
    path = resolve_feature_contract_path(cfg, project_root)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_stage_u_config_against_contract(cfg: dict[str, Any], contract: dict[str, Any]) -> None:
    cid = contract.get("contract_id", "unknown")

    def _fail(msg: str) -> None:
        raise ValueError(f"Stage-u feature contract violation ({cid}): {msg}")

    if list(cfg.get("targets", [])) != list(contract["targets"]):
        _fail(f"cfg targets {cfg.get('targets')!r} != contract {contract['targets']!r}")

    if list(cfg.get("numeric_features", [])) != list(contract["numeric_features"]):
        _fail(f"cfg numeric_features != contract.numeric_features")

    if list(cfg.get("player_constant_features", [])) != list(contract["player_constant_features"]):
        _fail(f"cfg player_constant_features != contract.player_constant_features")

    if cfg.get("weight_column") != contract["weight_column"]:
        _fail(f"cfg weight_column {cfg.get('weight_column')!r} != contract {contract['weight_column']!r}")

    exp_order = list(contract["x_numeric_zscore_column_order"])
    got_order = list(cfg["numeric_features"]) + list(cfg["player_constant_features"])
    if got_order != exp_order:
        _fail(f"numeric+player column order {got_order!r} != contract x_numeric_zscore_column_order")

    yaml_cat_order = list(cfg.get("categorical_features", {}).keys())
    if yaml_cat_order != list(contract["categorical_yaml_key_order"]):
        _fail(
            f"categorical_features key order in cfg YAML {yaml_cat_order!r} != "
            f"contract categorical_yaml_key_order"
        )

    cfg_cat: dict[str, Any] = cfg["categorical_features"]
    con_cat: dict[str, Any] = contract["categorical_features"]
    if set(cfg_cat.keys()) != set(con_cat.keys()):
        _fail(f"categorical feature keys {set(cfg_cat.keys())} != contract {set(con_cat.keys())}")

    for name in sorted(con_cat.keys()):
        ed_cfg = int(cfg_cat[name]["embedding_dim"])
        ed_con = int(con_cat[name]["embedding_dim"])
        if ed_cfg != ed_con:
            _fail(f"{name}: embedding_dim {ed_cfg} != contract {ed_con}")

    stats_rel = str(cfg.get("paths", {}).get("standardization_stats", ""))
    exp_stats = str(contract["standardization_stats_relpath"])
    if stats_rel != exp_stats:
        _fail(f"paths.standardization_stats {stats_rel!r} != contract {exp_stats!r}")

    alpha = list(contract["categorical_model_forward_order_alphabetical"])
    if alpha != sorted(alpha):
        _fail("contract categorical_model_forward_order_alphabetical must be sorted alphabetically")
    if alpha != sorted(con_cat.keys()):
        _fail("categorical_model_forward_order_alphabetical must list all categorical keys")


def validate_preprocessing_stats_against_contract(
    stats: dict[str, dict[str, float]],
    contract: dict[str, Any],
) -> None:
    cid = contract.get("contract_id", "unknown")

    def _fail(msg: str) -> None:
        raise ValueError(f"Stage-u feature contract violation ({cid}): {msg}")

    for k in contract["preprocessing_stats_must_contain"]:
        if k not in stats:
            _fail(f"standardization_stats.json missing required key {k!r}")

    if contract.get("player_constants_must_not_be_in_preprocessing_stats", True):
        for c in contract["player_constant_features"]:
            if c in stats:
                _fail(
                    f"player constant {c!r} must not appear in preprocessing standardization_stats.json "
                    f"(train-only z-scoring on the player-constant block)"
                )


def validate_vocab_sizes_against_contract(
    vocabs: dict[str, dict[str, int]],
    contract: dict[str, Any],
) -> None:
    cid = contract.get("contract_id", "unknown")

    def _fail(msg: str) -> None:
        raise ValueError(f"Stage-u feature contract violation ({cid}): {msg}")

    for name, spec in contract["categorical_features"].items():
        exp_sz = int(spec["vocab_size"])
        act_sz = len(vocabs[name])
        if act_sz != exp_sz:
            _fail(
                f"categorical {name!r}: vocab size {act_sz} != frozen contract {exp_sz} "
                f"(data or token encoding drift)"
            )


def assert_vocabs_identical(a: dict[str, dict[str, int]], b: dict[str, dict[str, int]], *, context: str) -> None:
    if set(a.keys()) != set(b.keys()):
        raise ValueError(f"{context}: vocab keys differ: {set(a.keys())} vs {set(b.keys())}")
    for k in sorted(a.keys()):
        if a[k] != b[k]:
            raise ValueError(
                f"{context}: vocab for {k!r} differs from checkpoint (strict eval). "
                f"Re-evaluate with the same training data contract, or retrain."
            )
