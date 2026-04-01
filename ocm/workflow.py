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
    auto_fit: bool = True,
    min_samples: int = MIN_SAMPLES_DEFAULT,
) -> tuple[int, tuple[bool, str] | None]:
    """
    Insert one benchmark row. If auto_fit and enough samples exist, refit model.
    Returns (record_id, fit_result_or_none).
    """
    rid = insert_record(conn, op_name, device, params, latency)
    if not auto_fit:
        return rid, None
    ok, msg = fit_and_store_model(conn, op_name, device, min_samples=min_samples)
    return rid, (ok, msg)
