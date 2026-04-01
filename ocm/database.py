"""SQLite schema and accessors for records + models."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ocm.sqlite3"


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            op_name TEXT NOT NULL,
            device TEXT NOT NULL,
            params TEXT NOT NULL,
            latency REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_records_op_device ON records (op_name, device);

        CREATE TABLE IF NOT EXISTS models (
            op_name TEXT NOT NULL,
            device TEXT NOT NULL,
            model_payload TEXT NOT NULL,
            feature_order TEXT NOT NULL,
            PRIMARY KEY (op_name, device)
        );

        CREATE TABLE IF NOT EXISTS param_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            params TEXT NOT NULL
        );
        """
    )
    conn.commit()


def insert_record(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    params: dict[str, Any],
    latency: float,
) -> int:
    cur = conn.execute(
        "INSERT INTO records (op_name, device, params, latency) VALUES (?, ?, ?, ?)",
        (op_name, device, json.dumps(params, ensure_ascii=False, sort_keys=True), float(latency)),
    )
    conn.commit()
    return int(cur.lastrowid)


def fetch_records(conn: sqlite3.Connection, op_name: str, device: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, op_name, device, params, latency FROM records WHERE op_name = ? AND device = ? ORDER BY id",
        (op_name, device),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "op_name": r["op_name"],
                "device": r["device"],
                "params": json.loads(r["params"]),
                "latency": r["latency"],
            }
        )
    return out


def list_op_device_pairs(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    cur = conn.execute(
        "SELECT DISTINCT op_name, device FROM records ORDER BY op_name, device"
    )
    return [(str(r[0]), str(r[1])) for r in cur.fetchall()]


def get_model_row(
    conn: sqlite3.Connection, op_name: str, device: str
) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT op_name, device, model_payload, feature_order FROM models WHERE op_name = ? AND device = ?",
        (op_name, device),
    ).fetchone()
    if r is None:
        return None
    return {
        "op_name": r["op_name"],
        "device": r["device"],
        "model_payload": r["model_payload"],
        "feature_order": json.loads(r["feature_order"]),
    }


def upsert_model(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    model_payload: str,
    feature_order: list[str],
) -> None:
    conn.execute(
        """
        INSERT INTO models (op_name, device, model_payload, feature_order)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(op_name, device) DO UPDATE SET
            model_payload = excluded.model_payload,
            feature_order = excluded.feature_order
        """,
        (op_name, device, model_payload, json.dumps(feature_order, ensure_ascii=False)),
    )
    conn.commit()


def list_param_templates(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All saved param templates: id, name, params (dict)."""
    rows = conn.execute(
        "SELECT id, name, params FROM param_templates ORDER BY name COLLATE NOCASE"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "name": r["name"],
                "params": json.loads(r["params"]),
            }
        )
    return out


def get_param_template_by_name(
    conn: sqlite3.Connection, name: str
) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT id, name, params FROM param_templates WHERE name = ?",
        (name,),
    ).fetchone()
    if r is None:
        return None
    return {
        "id": r["id"],
        "name": r["name"],
        "params": json.loads(r["params"]),
    }


def save_param_template(
    conn: sqlite3.Connection, name: str, params: dict[str, Any]
) -> int:
    """
    Insert or replace template by name. Returns row id.
    """
    nm = name.strip()
    payload = json.dumps(params, ensure_ascii=False, sort_keys=True)
    conn.execute(
        """
        INSERT INTO param_templates (name, params) VALUES (?, ?)
        ON CONFLICT(name) DO UPDATE SET params = excluded.params
        """,
        (nm, payload),
    )
    conn.commit()
    r = conn.execute("SELECT id FROM param_templates WHERE name = ?", (nm,)).fetchone()
    return int(r["id"]) if r else 0


def delete_param_template(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute("DELETE FROM param_templates WHERE name = ?", (name,))
    conn.commit()
    return cur.rowcount > 0


def export_records_flat_csv_rows(
    conn: sqlite3.Connection, op_name: str, device: str
) -> tuple[list[str], list[list[Any]]]:
    """Return header + rows for CSV export (flattened params + latency)."""
    from ocm.features import flatten_params_for_export

    recs = fetch_records(conn, op_name, device)
    if not recs:
        return [], []
    rows_out: list[list[Any]] = []
    all_keys: set[str] = set()
    flattened: list[dict[str, Any]] = []
    for rec in recs:
        flat = flatten_params_for_export(rec["params"])
        flat["latency"] = rec["latency"]
        flattened.append(flat)
        all_keys.update(flat.keys())
    header = sorted(all_keys)
    if "latency" in header:
        header.remove("latency")
    header.append("latency")
    for flat in flattened:
        rows_out.append([flat.get(k, "") for k in header])
    return header, rows_out
