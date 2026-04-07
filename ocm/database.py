"""SQLite schema and accessors for records + models."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ocm.keys import make_feature_order_key

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ocm.sqlite3"


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Upgrade old DBs: records.feature_order_key; models PK includes feature_order_key."""
    rec_cols = {r[1] for r in conn.execute("PRAGMA table_info(records)").fetchall()}
    if rec_cols and "feature_order_key" not in rec_cols:
        conn.execute("ALTER TABLE records ADD COLUMN feature_order_key TEXT")
        conn.commit()

    m_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='models'"
    ).fetchone()
    if not m_exists:
        return
    m_cols = {r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()}
    if "feature_order_key" in m_cols:
        return

    conn.execute("ALTER TABLE models RENAME TO models_legacy")
    conn.execute(
        """
        CREATE TABLE models (
            op_name TEXT NOT NULL,
            device TEXT NOT NULL,
            feature_order_key TEXT NOT NULL,
            model_payload TEXT NOT NULL,
            feature_order TEXT NOT NULL,
            PRIMARY KEY (op_name, device, feature_order_key)
        )
        """
    )
    for r in conn.execute("SELECT * FROM models_legacy").fetchall():
        fo = json.loads(r["feature_order"])
        fk = make_feature_order_key(fo)
        conn.execute(
            """
            INSERT INTO models (op_name, device, feature_order_key, model_payload, feature_order)
            VALUES (?, ?, ?, ?, ?)
            """,
            (r["op_name"], r["device"], fk, r["model_payload"], r["feature_order"]),
        )
    conn.execute("DROP TABLE models_legacy")
    conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            op_name TEXT NOT NULL,
            device TEXT NOT NULL,
            params TEXT NOT NULL,
            latency REAL NOT NULL,
            feature_order_key TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_records_op_device ON records (op_name, device);

        CREATE TABLE IF NOT EXISTS models (
            op_name TEXT NOT NULL,
            device TEXT NOT NULL,
            feature_order_key TEXT NOT NULL,
            model_payload TEXT NOT NULL,
            feature_order TEXT NOT NULL,
            PRIMARY KEY (op_name, device, feature_order_key)
        );

        CREATE TABLE IF NOT EXISTS param_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            params TEXT NOT NULL
        );
        """
    )
    conn.commit()
    _migrate_schema(conn)
    # 必须在迁移后为旧库补上 feature_order_key 列，再建索引，否则 legacy records 表无此列会报错
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_fok ON records (op_name, device, feature_order_key)"
    )
    conn.commit()


def insert_record(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    params: dict[str, Any],
    latency: float,
    feature_order_key: str | None = None,
    *,
    auto_key_from_params: bool = True,
) -> tuple[int, str | None]:
    """
    写入一条 record。默认根据 params 自动计算 feature_order_key（与训练特征展开一致）；
    若显式传入 feature_order_key 则使用该值；若传入 auto_key_from_params=False 且未传 key，则存 NULL（未标注）。
    返回 (row_id, 实际写入的 feature_order_key)。
    """
    from ocm.features import derive_feature_order_key_from_params

    if feature_order_key is not None:
        fk: str | None = feature_order_key
    elif auto_key_from_params:
        fk = derive_feature_order_key_from_params(params)
    else:
        fk = None
    cur = conn.execute(
        """
        INSERT INTO records (op_name, device, params, latency, feature_order_key)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            op_name,
            device,
            json.dumps(params, ensure_ascii=False, sort_keys=True),
            float(latency),
            fk,
        ),
    )
    conn.commit()
    return int(cur.lastrowid), fk


def find_exact_match_record_latency(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    params: dict[str, Any],
) -> tuple[float | None, int | None]:
    """
    若存在与插入时相同规范 JSON 的 params（sort_keys + ensure_ascii），
    返回最新一条（id 最大）的 latency 与 record id；否则 (None, None)。
    """
    blob = json.dumps(params, ensure_ascii=False, sort_keys=True)
    r = conn.execute(
        """
        SELECT id, latency FROM records
        WHERE op_name = ? AND device = ? AND params = ?
        ORDER BY id DESC LIMIT 1
        """,
        (op_name, device, blob),
    ).fetchone()
    if r is None:
        return None, None
    return float(r["latency"]), int(r["id"])


def fetch_records(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    feature_order_key: str | None = None,
    *,
    unlabeled_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Rows for (op_name, device).
    - feature_order_key None and unlabeled_only False: all rows.
    - unlabeled_only True: only rows where feature_order_key IS NULL.
    - feature_order_key set: only rows with that exact key.
    """
    if unlabeled_only:
        rows = conn.execute(
            """
            SELECT id, op_name, device, params, latency, feature_order_key
            FROM records
            WHERE op_name = ? AND device = ? AND feature_order_key IS NULL
            ORDER BY id
            """,
            (op_name, device),
        ).fetchall()
    elif feature_order_key is None:
        rows = conn.execute(
            """
            SELECT id, op_name, device, params, latency, feature_order_key
            FROM records WHERE op_name = ? AND device = ? ORDER BY id
            """,
            (op_name, device),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, op_name, device, params, latency, feature_order_key
            FROM records
            WHERE op_name = ? AND device = ? AND feature_order_key = ?
            ORDER BY id
            """,
            (op_name, device, feature_order_key),
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
                "feature_order_key": r["feature_order_key"],
            }
        )
    return out


def get_record_by_id(
    conn: sqlite3.Connection, record_id: int
) -> dict[str, Any] | None:
    r = conn.execute(
        """
        SELECT id, op_name, device, params, latency, feature_order_key
        FROM records WHERE id = ?
        """,
        (record_id,),
    ).fetchone()
    if r is None:
        return None
    return {
        "id": r["id"],
        "op_name": r["op_name"],
        "device": r["device"],
        "params": json.loads(r["params"]),
        "latency": r["latency"],
        "feature_order_key": r["feature_order_key"],
    }


def list_records(
    conn: sqlite3.Connection,
    *,
    op_name: str | None = None,
    device: str | None = None,
    limit: int = 2000,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List records with optional filters, newest id first."""
    q = """
        SELECT id, op_name, device, params, latency, feature_order_key
        FROM records WHERE 1=1
    """
    args: list[Any] = []
    if op_name is not None and str(op_name).strip() != "":
        q += " AND op_name = ?"
        args.append(op_name.strip())
    if device is not None and str(device).strip() != "":
        q += " AND device = ?"
        args.append(device.strip())
    q += " ORDER BY id DESC LIMIT ? OFFSET ?"
    args.extend([limit, offset])
    rows = conn.execute(q, args).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "op_name": r["op_name"],
                "device": r["device"],
                "params": json.loads(r["params"]),
                "latency": r["latency"],
                "feature_order_key": r["feature_order_key"],
            }
        )
    return out


def update_record(
    conn: sqlite3.Connection,
    record_id: int,
    op_name: str,
    device: str,
    params: dict[str, Any],
    latency: float,
    feature_order_key: str | None = None,
    *,
    auto_key_from_params: bool = True,
) -> str | None:
    """Update a row; returns stored feature_order_key."""
    from ocm.features import derive_feature_order_key_from_params

    if feature_order_key is not None:
        fk: str | None = feature_order_key
    elif auto_key_from_params:
        fk = derive_feature_order_key_from_params(params)
    else:
        fk = None
    conn.execute(
        """
        UPDATE records
        SET op_name = ?, device = ?, params = ?, latency = ?, feature_order_key = ?
        WHERE id = ?
        """,
        (
            op_name,
            device,
            json.dumps(params, ensure_ascii=False, sort_keys=True),
            float(latency),
            fk,
            record_id,
        ),
    )
    conn.commit()
    return fk


def delete_record(conn: sqlite3.Connection, record_id: int) -> bool:
    cur = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
    conn.commit()
    return cur.rowcount > 0


def list_op_device_pairs(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    cur = conn.execute(
        "SELECT DISTINCT op_name, device FROM records ORDER BY op_name, device"
    )
    return [(str(r[0]), str(r[1])) for r in cur.fetchall()]


def list_record_export_keys(
    conn: sqlite3.Connection, op_name: str, device: str
) -> list[str | None]:
    """
    Distinct feature_order_key values for export UI. None represents 未标注 (NULL in DB).
    """
    rows = conn.execute(
        """
        SELECT DISTINCT feature_order_key FROM records
        WHERE op_name = ? AND device = ?
        ORDER BY feature_order_key IS NULL DESC, feature_order_key
        """,
        (op_name, device),
    ).fetchall()
    keys: list[str | None] = []
    for (k,) in rows:
        keys.append(k)
    return keys


def list_models_for_op_device(
    conn: sqlite3.Connection, op_name: str, device: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT op_name, device, feature_order_key, model_payload, feature_order
        FROM models WHERE op_name = ? AND device = ?
        ORDER BY feature_order_key
        """,
        (op_name, device),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "op_name": r["op_name"],
                "device": r["device"],
                "feature_order_key": r["feature_order_key"],
                "model_payload": r["model_payload"],
                "feature_order": json.loads(r["feature_order"]),
            }
        )
    return out


def get_model_row(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    feature_order_key: str | None = None,
) -> dict[str, Any] | None:
    """
    One model row. If feature_order_key is None and exactly one model exists for
    (op_name, device), return it; if zero or multiple, return None.
    """
    if feature_order_key is not None:
        r = conn.execute(
            """
            SELECT op_name, device, feature_order_key, model_payload, feature_order
            FROM models WHERE op_name = ? AND device = ? AND feature_order_key = ?
            """,
            (op_name, device, feature_order_key),
        ).fetchone()
        if r is None:
            return None
        return {
            "op_name": r["op_name"],
            "device": r["device"],
            "feature_order_key": r["feature_order_key"],
            "model_payload": r["model_payload"],
            "feature_order": json.loads(r["feature_order"]),
        }
    rows = conn.execute(
        "SELECT op_name, device, feature_order_key, model_payload, feature_order FROM models WHERE op_name = ? AND device = ?",
        (op_name, device),
    ).fetchall()
    if len(rows) != 1:
        return None
    r = rows[0]
    return {
        "op_name": r["op_name"],
        "device": r["device"],
        "feature_order_key": r["feature_order_key"],
        "model_payload": r["model_payload"],
        "feature_order": json.loads(r["feature_order"]),
    }


def upsert_model(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    model_payload: str,
    feature_order: list[str],
) -> str:
    fk = make_feature_order_key(feature_order)
    conn.execute(
        """
        INSERT INTO models (op_name, device, feature_order_key, model_payload, feature_order)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(op_name, device, feature_order_key) DO UPDATE SET
            model_payload = excluded.model_payload,
            feature_order = excluded.feature_order
        """,
        (
            op_name,
            device,
            fk,
            model_payload,
            json.dumps(feature_order, ensure_ascii=False),
        ),
    )
    conn.commit()
    return fk


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
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    feature_order_key: str | None = None,
    *,
    unlabeled_only: bool = False,
) -> tuple[list[str], list[list[Any]]]:
    """Return header + rows for CSV export (flattened params + latency + meta columns)."""
    from ocm.features import flatten_params_for_export

    recs = fetch_records(
        conn, op_name, device, feature_order_key, unlabeled_only=unlabeled_only
    )
    if not recs:
        return [], []
    flattened: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    for rec in recs:
        flat = flatten_params_for_export(rec["params"])
        flat["record_id"] = rec["id"]
        flat["feature_order_key"] = rec["feature_order_key"] or ""
        flat["latency"] = rec["latency"]
        flattened.append(flat)
        all_keys.update(flat.keys())
    param_keys = sorted(
        k for k in all_keys if k not in ("record_id", "feature_order_key", "latency")
    )
    header = ["record_id", "feature_order_key"] + param_keys + ["latency"]
    rows_out: list[list[Any]] = []
    for flat in flattened:
        rows_out.append([flat.get(k, "") for k in header])
    return header, rows_out


def export_filename_suffix(op_name: str, device: str, segment: str) -> str:
    """
    Build a short filename segment. segment is one of:
    'all', 'unlabeled', or the literal feature_order_key string (hashed for length).
    """
    safe_op = op_name.replace("::", "_")
    if segment == "all":
        return f"{safe_op}_{device}_all"
    if segment == "unlabeled":
        return f"{safe_op}_{device}_unlabeled"
    h = hash(segment) & 0xFFFFFFFF
    return f"{safe_op}_{device}_fok_{h:08x}"
