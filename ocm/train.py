"""Train XGBoost regressor and persist model as JSON text in SQLite."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import xgboost as xgb

from ocm.database import fetch_records, upsert_model
from ocm.features import build_training_matrix, optional_derived_features

MIN_SAMPLES_DEFAULT = 2


def _merge_derived_into_params(p: dict[str, Any]) -> dict[str, Any]:
    out = dict(p)
    for k, v in optional_derived_features(p).items():
        out.setdefault(k, v)
    return out


def fit_and_store_model(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    xgb_params: dict[str, Any] | None = None,
    merge_derived: bool = True,
    feature_order_key: str | None = None,
    *,
    unlabeled_only: bool = False,
) -> tuple[bool, str]:
    """
    Load records for (op_name, device).
    - unlabeled_only=True: only rows with NULL feature_order_key.
    - elif feature_order_key is set: only matching key.
    - else: all rows for that op+device.
    """
    recs = fetch_records(
        conn, op_name, device, feature_order_key, unlabeled_only=unlabeled_only
    )
    if len(recs) < min_samples:
        return False, f"需要至少 {min_samples} 条样本，当前 {len(recs)} 条。"

    params_list = [r["params"] for r in recs]
    if merge_derived:
        params_list = [_merge_derived_into_params(p) for p in params_list]
    latencies = [r["latency"] for r in recs]

    feature_order, X, y = build_training_matrix(params_list, latencies)
    if not feature_order:
        return False, "没有可用的数值特征。"

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    defaults: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }
    if xgb_params:
        defaults.update(xgb_params)

    reg = xgb.XGBRegressor(**defaults)
    reg.fit(X_arr, y_arr)

    booster = reg.get_booster()
    raw = booster.save_raw("json")
    if isinstance(raw, str):
        payload = raw
    else:
        payload = bytes(raw).decode("utf-8")

    fk = upsert_model(conn, op_name, device, payload, feature_order)
    return True, (
        f"已训练并保存模型：样本 {len(recs)}，特征 {len(feature_order)}，"
        f"feature_order_key={fk}"
    )
