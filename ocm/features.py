"""Feature engineering: flatten params JSON to a fixed-order numeric vector."""

from __future__ import annotations

from typing import Any

from ocm.keys import make_feature_order_key


def derive_feature_order_key_from_params(
    params: dict[str, Any], *, merge_derived: bool = True
) -> str:
    """
    与训练管线一致：可选合并衍生特征后，对单层特征名做有序并集，再生成稳定 key。
    用于录入时自动写入 records.feature_order_key，无需手填。
    """
    p = dict(params)
    if merge_derived:
        for k, v in optional_derived_features(p).items():
            p.setdefault(k, v)
    names = union_feature_names_from_params_list([p])
    return make_feature_order_key(names)


def flatten_params_for_export(params: dict[str, Any]) -> dict[str, Any]:
    """One-level flatten for CSV export (includes stride columns and bool as 0/1)."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k == "memory_stride" and isinstance(v, list):
            for i, x in enumerate(v):
                out[f"memory_stride_{i}"] = x
        elif k == "is_contiguous" and isinstance(v, bool):
            out["is_contiguous"] = 1 if v else 0
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v
        elif isinstance(v, bool):
            out[k] = 1 if v else 0
    return out


def params_to_feature_dict(params: dict[str, Any]) -> dict[str, float]:
    """Turn a single params dict into scalar features (names -> float)."""
    feats: dict[str, float] = {}
    stride: list[Any] | None = None
    for k, v in params.items():
        if k == "memory_stride":
            if isinstance(v, list):
                stride = v
            continue
        if k == "is_contiguous":
            feats["is_contiguous"] = 1.0 if bool(v) else 0.0
            continue
        if isinstance(v, bool):
            feats[k] = 1.0 if v else 0.0
            continue
        if isinstance(v, (int, float)):
            feats[k] = float(v)
            continue
        # skip nested dicts / strings for tree features
    if stride is not None:
        for i, x in enumerate(stride):
            try:
                feats[f"memory_stride_{i}"] = float(x)
            except (TypeError, ValueError):
                feats[f"memory_stride_{i}"] = 0.0
    return feats


def params_to_feature_row(
    params: dict[str, Any], feature_order: list[str]
) -> list[float]:
    """Map params to a row aligned with feature_order (missing -> 0.0)."""
    d = params_to_feature_dict(params)
    return [float(d.get(name, 0.0)) for name in feature_order]


def union_feature_names_from_params_list(
    params_list: list[dict[str, Any]],
) -> list[str]:
    """Collect union of feature names from multiple param dicts, sorted for stability."""
    names: set[str] = set()
    for p in params_list:
        names.update(params_to_feature_dict(p).keys())
    return sorted(names)


def build_training_matrix(
    params_list: list[dict[str, Any]], latencies: list[float]
) -> tuple[list[str], list[list[float]], list[float]]:
    """
    Build X (rows), y, and the canonical feature_order from raw records.
    feature_order is the sorted union of all keys appearing in the batch.
    """
    if len(params_list) != len(latencies):
        raise ValueError("params_list and latencies length mismatch")
    feature_order = union_feature_names_from_params_list(params_list)
    X = [params_to_feature_row(p, feature_order) for p in params_list]
    y = [float(t) for t in latencies]
    return feature_order, X, y


def optional_derived_features(params: dict[str, Any]) -> dict[str, float]:
    """
    If params contain recognizable matmul / conv keys, add FLOPs / memory hints.
    Caller may merge into params before training.
    """
    extra: dict[str, float] = {}
    # Matmul: M, N, K -> 2*M*N*K FLOPs (rough)
    if all(k in params for k in ("M", "N", "K")):
        try:
            m, n, k = float(params["M"]), float(params["N"]), float(params["K"])
            extra["flops_matmul"] = 2.0 * m * n * k
            extra["memory_bytes_matmul"] = 4.0 * (m * k + k * n + m * n)  # fp32 bytes
        except (TypeError, ValueError):
            pass
    return extra
