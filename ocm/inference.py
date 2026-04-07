"""Load booster from JSON text and predict latency."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import xgboost as xgb

from ocm.database import find_exact_match_record_latency, get_model_row
from ocm.features import (
    derive_feature_order_key_from_params,
    optional_derived_features,
    params_to_feature_row,
)


def _merge_derived(params: dict[str, Any], merge_derived: bool) -> dict[str, Any]:
    p = dict(params)
    if merge_derived:
        for k, v in optional_derived_features(p).items():
            p.setdefault(k, v)
    return p


def resolve_model_row_for_prediction(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    params: dict[str, Any],
    merge_derived: bool = True,
    feature_order_key: str | None = None,
) -> dict[str, Any] | None:
    """
    Resolve a model row for prediction.
    - If feature_order_key is given, use it directly.
    - Else first try the feature_order_key derived from params.
    - Else fall back to the legacy "exactly one model exists" behavior.
    """
    if feature_order_key is not None:
        return get_model_row(conn, op_name, device, feature_order_key)

    inferred_key = derive_feature_order_key_from_params(
        params,
        merge_derived=merge_derived,
    )
    row = get_model_row(conn, op_name, device, inferred_key)
    if row is not None:
        return row
    return get_model_row(conn, op_name, device, None)


def predict_latency_details(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    params: dict[str, Any],
    merge_derived: bool = True,
    feature_order_key: str | None = None,
    *,
    use_exact_record_if_match: bool = True,
) -> dict[str, Any] | None:
    """
    若 `use_exact_record_if_match` 为 True，先在 records 中按与入库相同的 params JSON 精确匹配；
    命中则直接返回该条实测 latency，否则再尝试 XGBoost 模型。
    """
    if use_exact_record_if_match:
        rec_lat, rec_id = find_exact_match_record_latency(conn, op_name, device, params)
        if rec_lat is not None and rec_id is not None:
            return {
                "predicted_latency_ms": rec_lat,
                "source": "record",
                "record_id": rec_id,
                "op_name": op_name,
                "device": device,
            }

    row = resolve_model_row_for_prediction(
        conn,
        op_name,
        device,
        params,
        merge_derived=merge_derived,
        feature_order_key=feature_order_key,
    )
    if row is None:
        return None

    p = _merge_derived(params, merge_derived=merge_derived)
    feature_order: list[str] = row["feature_order"]
    vec = params_to_feature_row(p, feature_order)
    X = np.asarray([vec], dtype=np.float64)

    booster = xgb.Booster()
    payload = row["model_payload"]
    if isinstance(payload, str):
        booster.load_model(bytearray(payload.encode("utf-8")))
    else:
        booster.load_model(bytearray(payload))

    dm = xgb.DMatrix(X, feature_names=feature_order)
    pred = booster.predict(dm)
    return {
        "predicted_latency_ms": float(pred[0]),
        "source": "model",
        "feature_order_key": row["feature_order_key"],
        "feature_order": feature_order,
        "op_name": row["op_name"],
        "device": row["device"],
    }


def predict_latency(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    params: dict[str, Any],
    merge_derived: bool = True,
    feature_order_key: str | None = None,
    *,
    use_exact_record_if_match: bool = True,
) -> float | None:
    """
    Return latency (ms) or None if neither a matching record nor a model exists.
    By default, if a record exists with the same canonical params JSON as stored
    at insert time, returns that measured latency; otherwise uses the XGBoost model.
    """
    details = predict_latency_details(
        conn,
        op_name,
        device,
        params,
        merge_derived=merge_derived,
        feature_order_key=feature_order_key,
        use_exact_record_if_match=use_exact_record_if_match,
    )
    if details is None:
        return None
    return float(details["predicted_latency_ms"])


def predict_with_booster_json(
    model_payload: str,
    feature_order: list[str],
    params: dict[str, Any],
    merge_derived: bool = True,
) -> float:
    """Predict without DB (e.g. after manual paste override)."""
    p = _merge_derived(params, merge_derived=merge_derived)
    vec = params_to_feature_row(p, feature_order)
    X = np.asarray([vec], dtype=np.float64)
    booster = xgb.Booster()
    booster.load_model(bytearray(model_payload.encode("utf-8")))
    dm = xgb.DMatrix(X, feature_names=feature_order)
    return float(booster.predict(dm)[0])
