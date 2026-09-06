"""Strict feature contract for stage-z p(z|u,g,p)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT_RELPATH = "configs/feature_contracts/p_z_given_u_g_feature_contract.json"


def _root(project_root: Path | None) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[1]
    return Path(project_root).resolve()


def resolve_feature_contract_path_z(cfg: dict[str, Any], project_root: Path | None = None) -> Path:
    root = _root(project_root)
    rel = cfg.get("paths", {}).get("feature_contract", DEFAULT_CONTRACT_RELPATH)
    p = (root / str(rel)).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Stage-z feature contract not found: {p}")
    return p


def load_feature_contract_z(cfg: dict[str, Any], project_root: Path | None = None) -> dict[str, Any]:
    return json.loads(resolve_feature_contract_path_z(cfg, project_root).read_text(encoding="utf-8"))


def reorder_cfg_categorical_features_for_contract(cfg: dict[str, Any], contract: dict[str, Any]) -> None:
    """Mutate ``cfg['categorical_features']`` key order to match ``contract['categorical_yaml_key_order']``.

    Resolved YAML may serialize map keys in a different order than the frozen contract; stage-z
    validation and ``nn.ModuleDict(sorted(keys))`` expect the contract order.
    """
    cat = cfg.get("categorical_features")
    if not isinstance(cat, dict):
        return
    order = list(contract["categorical_yaml_key_order"])
    if list(cat.keys()) == order:
        return
    missing = set(order) - set(cat.keys())
    if missing:
        raise ValueError(f"categorical_features missing keys for reorder: {sorted(missing)}")
    extra = set(cat.keys()) - set(order)
    if extra:
        raise ValueError(f"categorical_features has unknown keys vs contract: {sorted(extra)}")
    cfg["categorical_features"] = {k: cat[k] for k in order}


def validate_stage_z_config_against_contract(cfg: dict[str, Any], contract: dict[str, Any]) -> None:
    reorder_cfg_categorical_features_for_contract(cfg, contract)
    cid = contract.get("contract_id", "unknown")

    def _fail(msg: str) -> None:
        raise ValueError(f"Stage-z feature contract violation ({cid}): {msg}")

    if list(cfg["targets"]) != list(contract["targets"]):
        _fail(f"cfg targets != contract.targets")
    if list(cfg["upstream_u_features"]) != list(contract["upstream_u_features"]):
        _fail("cfg upstream_u_features mismatch")
    if list(cfg["numeric_context_features"]) != list(contract["numeric_context_features"]):
        _fail("cfg numeric_context_features mismatch")
    if list(cfg["player_constant_features"]) != list(contract["player_constant_features"]):
        _fail("cfg player_constant_features mismatch")
    if cfg.get("weight_column") != contract["weight_column"]:
        _fail("weight_column mismatch")

    exp_concat = (
        list(contract["upstream_u_features"])
        + list(contract["numeric_context_features"])
        + list(contract["player_constant_features"])
    )
    if exp_concat != list(contract["x_numeric_zscore_column_order"]):
        _fail("contract x_numeric_zscore_column_order != u|g_num|p")

    yaml_cat = list(cfg["categorical_features"].keys())
    if yaml_cat != list(contract["categorical_yaml_key_order"]):
        _fail("categorical YAML key order mismatch")

    cfg_cat = cfg["categorical_features"]
    con_cat = contract["categorical_features"]
    if set(cfg_cat.keys()) != set(con_cat.keys()):
        _fail("categorical keys mismatch")
    for name in sorted(con_cat.keys()):
        if int(cfg_cat[name]["embedding_dim"]) != int(con_cat[name]["embedding_dim"]):
            _fail(f"{name} embedding_dim mismatch")

    stats_rel = str(cfg.get("paths", {}).get("standardization_stats", ""))
    if stats_rel != str(contract["standardization_stats_relpath"]):
        _fail("paths.standardization_stats must match contract")

    alpha = list(contract["categorical_model_forward_order_alphabetical"])
    if alpha != sorted(alpha) or sorted(alpha) != sorted(con_cat.keys()):
        _fail("categorical_model_forward_order_alphabetical invalid")


def validate_vocab_sizes_z(vocabs: dict[str, dict[str, int]], contract: dict[str, Any]) -> None:
    cid = contract.get("contract_id", "unknown")

    def _fail(msg: str) -> None:
        raise ValueError(f"Stage-z feature contract violation ({cid}): {msg}")

    for name, spec in contract["categorical_features"].items():
        exp = int(spec["vocab_size"])
        act = len(vocabs[name])
        if act != exp:
            _fail(f"categorical {name!r} vocab size {act} != contract {exp}")


def validate_preprocessing_stats_z(
    stats: dict[str, dict[str, float]],
    contract: dict[str, Any],
) -> None:
    cid = contract.get("contract_id", "unknown")

    def _fail(msg: str) -> None:
        raise ValueError(f"Stage-z feature contract violation ({cid}): {msg}")

    pset = set(contract["player_constant_features"])
    for k in contract["x_numeric_zscore_column_order"]:
        if k in pset:
            continue
        if k not in stats:
            _fail(f"standardization_stats.json missing key {k!r} (required for z-stage inputs)")
    for t in contract["targets"]:
        if t not in stats:
            _fail(f"standardization_stats.json missing target key {t!r}")

    if contract.get("player_constants_must_not_be_in_preprocessing_stats", True):
        for c in contract["player_constant_features"]:
            if c in stats:
                _fail(
                    f"player constant {c!r} must not appear in preprocessing stats "
                    f"(train-only z-score at load)"
                )


def assert_vocabs_identical_z(
    a: dict[str, dict[str, int]], b: dict[str, dict[str, int]], *, context: str
) -> None:
    if set(a.keys()) != set(b.keys()):
        raise ValueError(f"{context}: vocab keys differ")
    for k in sorted(a.keys()):
        if a[k] != b[k]:
            raise ValueError(f"{context}: vocab for {k!r} differs from checkpoint")
