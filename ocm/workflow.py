"""High-level: insert record and optionally auto-train."""

from __future__ import annotations

import sqlite3
from typing import Any

from ocm.database import insert_record
from ocm.train import MIN_SAMPLES_DEFAULT, fit_and_store_model


def add_record_maybe_autofit(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    params: dict[str, Any],
    latency: float,
    auto_fit: bool = False,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    feature_order_key: str | None = None,
) -> tuple[int, tuple[bool, str] | None]:
    """
    Insert one benchmark row. If auto_fit is True, refit model when enough samples exist.
    Training uses the same feature_order_key filter as the inserted row when that key is set.
    Returns (record_id, fit_result_or_none).
    """
    rid = insert_record(conn, op_name, device, params, latency, feature_order_key=feature_order_key)
    if not auto_fit:
        return rid, None
    ok, msg = fit_and_store_model(
        conn, op_name, device, min_samples=min_samples, feature_order_key=feature_order_key
    )
    return rid, (ok, msg)
