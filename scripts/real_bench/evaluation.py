from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from .common import DEFAULT_DB_PATH, REAL_DEVICE, REAL_SOURCE, ROOT
from ocm.database import get_connection, init_db
from ocm.features import build_training_matrix, optional_derived_features, params_to_feature_row
from ocm.train import fit_and_store_model

SMALL_SAMPLE_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 24,
    "max_depth": 3,
    "learning_rate": 0.12,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": 1,
    "random_state": 42,
}


def load_real_records(conn: sqlite3.Connection, *, device: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, op_name, device, params, latency, feature_order_key
        FROM records
        WHERE device = ?
        ORDER BY id
        """,
        (device,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        params = json.loads(row["params"])
        meta = params.get("benchmark_meta") if isinstance(params, dict) else None
        if not (isinstance(meta, dict) and meta.get("source") == REAL_SOURCE):
            continue
        out.append(
            {
                "id": int(row["id"]),
                "op_name": row["op_name"],
                "device": row["device"],
                "params": params,
                "latency": float(row["latency"]),
                "feature_order_key": row["feature_order_key"],
            }
        )
    return out


def merge_derived(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    for key, value in optional_derived_features(params).items():
        out.setdefault(key, value)
    return out


def group_records(records: list[dict[str, Any]]) -> dict[tuple[str, str, str | None], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["op_name"]),
            str(record["device"]),
            record["feature_order_key"],
        )
        grouped[key].append(record)
    return grouped


def split_group(records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda x: int(x["id"]))
    if len(ordered) < 3:
        raise ValueError("至少需要 3 条样本才能划分 train/test")

    rng = random.Random(seed)
    idxs = list(range(len(ordered)))
    rng.shuffle(idxs)
    test_count = max(1, len(ordered) // 4)
    if len(ordered) - test_count < 2:
        test_count = len(ordered) - 2
    test_idx = set(sorted(idxs[:test_count]))

    train = [row for i, row in enumerate(ordered) if i not in test_idx]
    test = [row for i, row in enumerate(ordered) if i in test_idx]
    return train, test


def train_regressor(
    train_records: list[dict[str, Any]],
    xgb_params: dict[str, Any] | None = None,
) -> tuple[xgb.XGBRegressor, list[str]]:
    params_list = [merge_derived(row["params"]) for row in train_records]
    latencies = [float(row["latency"]) for row in train_records]
    feature_order, X, y = build_training_matrix(params_list, latencies)
    if not feature_order:
        raise ValueError("没有可用特征")

    defaults: dict[str, Any] = dict(SMALL_SAMPLE_XGB_PARAMS)
    if xgb_params:
        defaults.update(xgb_params)

    reg = xgb.XGBRegressor(**defaults)
    reg.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return reg, feature_order


def evaluate_group(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    reg, feature_order = train_regressor(train_records)
    preds: list[float] = []
    actuals: list[float] = []
    rows: list[dict[str, Any]] = []

    for row in test_records:
        params = merge_derived(row["params"])
        vec = params_to_feature_row(params, feature_order)
        pred = float(reg.predict(np.asarray([vec], dtype=np.float64))[0])
        actual = float(row["latency"])
        preds.append(pred)
        actuals.append(actual)
        ape = abs(pred - actual) / actual if actual > 0 else 0.0
        rows.append(
            {
                "id": int(row["id"]),
                "actual_ms": round(actual, 6),
                "pred_ms": round(pred, 6),
                "abs_err_ms": round(abs(pred - actual), 6),
                "ape": round(ape, 6),
            }
        )

    mae = sum(abs(p - a) for p, a in zip(preds, actuals)) / len(actuals)
    rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(actuals))
    mape = sum(abs(p - a) / a for p, a in zip(preds, actuals) if a > 0) / len(actuals)

    return {
        "feature_order": feature_order,
        "train_ids": [int(row["id"]) for row in train_records],
        "test_ids": [int(row["id"]) for row in test_records],
        "test_rows": rows,
        "metrics": {
            "mae_ms": round(mae, 6),
            "rmse_ms": round(rmse, 6),
            "mape": round(mape, 6),
        },
    }


def aggregate_metrics(group_reports: list[dict[str, Any]]) -> dict[str, float]:
    all_rows = [row for report in group_reports for row in report["test_rows"]]
    if not all_rows:
        return {"mae_ms": 0.0, "rmse_ms": 0.0, "mape": 0.0}

    actuals = [row["actual_ms"] for row in all_rows]
    preds = [row["pred_ms"] for row in all_rows]
    mae = sum(abs(p - a) for p, a in zip(preds, actuals)) / len(actuals)
    rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(actuals))
    mape = sum(abs(p - a) / a for p, a in zip(preds, actuals) if a > 0) / len(actuals)
    return {
        "mae_ms": round(mae, 6),
        "rmse_ms": round(rmse, 6),
        "mape": round(mape, 6),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate real benchmark records with train/test split and optionally store models."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--device", default=REAL_DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--store-models",
        action="store_true",
        help="评估后，对可评估组用全量样本训练并写入 models 表",
    )
    parser.add_argument(
        "--report-path",
        default=str(ROOT / "reports" / "real_train_test_report.json"),
        help="将 train/test 划分和指标写到 JSON 报告文件",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    conn = get_connection(args.db_path)
    init_db(conn)
    records = load_real_records(conn, device=args.device)
    grouped = group_records(records)

    eligible_reports: list[dict[str, Any]] = []
    skipped_groups: list[dict[str, Any]] = []
    stored_models: list[dict[str, Any]] = []

    for (op_name, device, fok), rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary = {
            "op_name": op_name,
            "device": device,
            "feature_order_key": fok,
            "count": len(rows),
        }
        if len(rows) < 3:
            summary["reason"] = "insufficient_samples_for_split"
            summary["record_ids"] = [int(r["id"]) for r in rows]
            skipped_groups.append(summary)
            continue

        train_rows, test_rows = split_group(rows, seed=args.seed)
        report = evaluate_group(train_rows, test_rows)
        summary.update(report)
        eligible_reports.append(summary)

        if args.store_models:
            ok, msg = fit_and_store_model(
                conn,
                op_name,
                device,
                feature_order_key=fok,
                xgb_params=SMALL_SAMPLE_XGB_PARAMS,
            )
            stored_models.append(
                {
                    "op_name": op_name,
                    "device": device,
                    "feature_order_key": fok,
                    "ok": ok,
                    "message": msg,
                }
            )

    overall = aggregate_metrics(eligible_reports)
    report = {
        "device": args.device,
        "seed": args.seed,
        "overall_metrics": overall,
        "evaluated_groups": eligible_reports,
        "skipped_groups": skipped_groups,
        "stored_models": stored_models,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"device={args.device}")
    print(f"real_records={len(records)}")
    print(f"evaluated_groups={len(eligible_reports)}")
    print(f"skipped_groups={len(skipped_groups)}")
    print(f"overall_metrics={overall}")
    print(f"report_path={report_path}")
    print("\nEvaluated groups:")
    for group in eligible_reports:
        print(
            f"- {group['op_name']} | count={group['count']} "
            f"| train={group['train_ids']} | test={group['test_ids']} "
            f"| metrics={group['metrics']}"
        )
    if skipped_groups:
        print("\nSkipped groups:")
        for group in skipped_groups:
            print(
                f"- {group['op_name']} | count={group['count']} "
                f"| ids={group['record_ids']} | reason={group['reason']}"
            )
    if stored_models:
        print("\nStored models:")
        for model in stored_models:
            print(f"- {model['op_name']} | ok={model['ok']} | message={model['message']}")
    return 0
