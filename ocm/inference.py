"""Load booster from JSON text and predict latency."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import xgboost as xgb

from ocm.database import get_model_row
from ocm.features import optional_derived_features, params_to_feature_row


def predict_latency(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    params: dict[str, Any],
    merge_derived: bool = True,
    feature_order_key: str | None = None,
) -> float | None:
    """
    Return predicted latency (ms) or None if no model exists.
    If feature_order_key is None and multiple models exist for (op_name, device), returns None.
    """
    row = get_model_row(conn, op_name, device, feature_order_key)
    if row is None:
        return None

    p = dict(params)
    if merge_derived:
        for k, v in optional_derived_features(p).items():
            p.setdefault(k, v)

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
    return float(pred[0])


def predict_with_booster_json(
    model_payload: str,
    feature_order: list[str],
    params: dict[str, Any],
    merge_derived: bool = True,
) -> float:
    """Predict without DB (e.g. after manual paste override)."""
    p = dict(params)
    if merge_derived:
        for k, v in optional_derived_features(p).items():
            p.setdefault(k, v)
    vec = params_to_feature_row(p, feature_order)
    X = np.asarray([vec], dtype=np.float64)
    booster = xgb.Booster()
    booster.load_model(bytearray(model_payload.encode("utf-8")))
    dm = xgb.DMatrix(X, feature_names=feature_order)
    return float(booster.predict(dm)[0])
