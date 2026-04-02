from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import DEFAULT_DB_PATH, REAL_DEVICE, REAL_SOURCE
from ocm.database import (
    fetch_records,
    get_connection,
    get_record_by_id,
    init_db,
    list_models_for_op_device,
)
from ocm.inference import predict_latency_details
from ocm.train import MIN_SAMPLES_DEFAULT, fit_and_store_model


def resolve_single_operator(conn, raw_op: str) -> str:
    token = raw_op.strip()
    if not token:
        raise ValueError("算子名不能为空")
    if "::" in token:
        return token

    rows = conn.execute(
        """
        SELECT DISTINCT op_name FROM (
            SELECT op_name FROM records
            UNION
            SELECT op_name FROM models
        )
        WHERE op_name = ? OR op_name LIKE ?
        ORDER BY op_name
        """,
        (token, f"%::{token}"),
    ).fetchall()
    matches = [str(row[0]) for row in rows]
    if not matches:
        raise ValueError(f"未找到算子: {token}")
    if len(matches) > 1:
        raise ValueError(f"算子别名不唯一: {token} -> {matches}")
    return matches[0]


def load_real_operator_records(
    conn,
    *,
    op_name: str,
    device: str,
) -> list[dict[str, Any]]:
    rows = fetch_records(conn, op_name, device)
    out: list[dict[str, Any]] = []
    for row in rows:
        params = row["params"]
        meta = params.get("benchmark_meta") if isinstance(params, dict) else None
        if not (isinstance(meta, dict) and meta.get("source") == REAL_SOURCE):
            continue
        out.append(row)
    return out


def summarize_record_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str | None, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["feature_order_key"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for feature_order_key, rows in sorted(
        grouped.items(),
        key=lambda item: ((item[0] is None), str(item[0])),
    ):
        summaries.append(
            {
                "feature_order_key": feature_order_key,
                "count": len(rows),
                "record_ids": [int(row["id"]) for row in rows],
            }
        )
    return summaries


def print_record_groups(op_name: str, device: str, groups: list[dict[str, Any]]) -> None:
    print(f"算子: {op_name}")
    print(f"设备: {device}")
    print(f"feature_order 分组数: {len(groups)}")
    for idx, group in enumerate(groups, start=1):
        print(
            f"  [{idx}] count={group['count']} "
            f"feature_order_key={group['feature_order_key']} "
            f"record_ids={group['record_ids']}"
        )


def write_json_report(report_path: str | None, payload: dict[str, Any]) -> None:
    if not report_path:
        return
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 JSON: {path}")


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and store model(s) for one selected operator.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--op", required=True, help="算子 key 或 alias")
    parser.add_argument("--device", default=REAL_DEVICE)
    parser.add_argument("--feature-order-key", default=None, help="只训练指定 feature_order_key 的一组样本")
    parser.add_argument("--list-groups", action="store_true", help="只列出该算子的样本分组，不训练")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_DEFAULT)
    parser.add_argument("--report-path", default=None, help="将训练结果写成 JSON")
    return parser


def run_train_cli() -> int:
    parser = build_train_parser()
    args = parser.parse_args()

    conn = get_connection(args.db_path)
    init_db(conn)
    try:
        op_name = resolve_single_operator(conn, args.op)
        records = load_real_operator_records(conn, op_name=op_name, device=args.device)
        groups = summarize_record_groups(records)
        if not groups:
            print(f"未找到真实 benchmark 样本: op={op_name} device={args.device}")
            return 1

        print_record_groups(op_name, args.device, groups)
        if args.list_groups:
            return 0

        selected_groups = groups
        if args.feature_order_key is not None:
            selected_groups = [g for g in groups if g["feature_order_key"] == args.feature_order_key]
            if not selected_groups:
                raise ValueError(f"未找到指定 feature_order_key 对应的样本组: {args.feature_order_key}")

        results: list[dict[str, Any]] = []
        print("\n开始训练:")
        for group in selected_groups:
            record_feature_order_key = group["feature_order_key"]
            if record_feature_order_key is None:
                ok, msg = fit_and_store_model(
                    conn,
                    op_name,
                    args.device,
                    min_samples=args.min_samples,
                    unlabeled_only=True,
                )
            else:
                ok, msg = fit_and_store_model(
                    conn,
                    op_name,
                    args.device,
                    min_samples=args.min_samples,
                    feature_order_key=record_feature_order_key,
                )
            result = {
                "op_name": op_name,
                "device": args.device,
                "record_feature_order_key": record_feature_order_key,
                "record_count": group["count"],
                "record_ids": group["record_ids"],
                "ok": ok,
                "message": msg,
            }
            results.append(result)
            print(
                f"  ok={ok} count={group['count']} "
                f"record_feature_order_key={record_feature_order_key} "
                f"message={msg}"
            )

        stored_models = list_models_for_op_device(conn, op_name, args.device)
        report = {
            "op_name": op_name,
            "device": args.device,
            "selected_group_count": len(selected_groups),
            "trained_groups": results,
            "stored_models": [
                {
                    "feature_order_key": model["feature_order_key"],
                    "feature_order": model["feature_order"],
                }
                for model in stored_models
            ],
        }
        write_json_report(args.report_path, report)
    finally:
        conn.close()
    return 0


def parse_params_input(params_json: str | None, params_file: str | None) -> dict[str, Any]:
    if params_json:
        return json.loads(params_json)
    if params_file:
        return json.loads(Path(params_file).read_text(encoding="utf-8"))
    raise ValueError("请提供 --params-json、--params-file 或 --record-id")


def build_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict latency for one selected operator.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--op", default=None, help="算子 key 或 alias；若给 --record-id 可省略")
    parser.add_argument("--device", default=REAL_DEVICE)
    parser.add_argument("--feature-order-key", default=None, help="显式指定使用哪个模型")
    parser.add_argument("--record-id", type=int, default=None, help="直接用某条 records 样本做预测")
    parser.add_argument("--params-json", default=None, help="直接传 params JSON 字符串")
    parser.add_argument("--params-file", default=None, help="从 JSON 文件加载 params")
    parser.add_argument("--list-models", action="store_true", help="列出该算子当前已存模型")
    parser.add_argument("--report-path", default=None, help="将预测结果写成 JSON")
    return parser


def run_predict_cli() -> int:
    parser = build_predict_parser()
    args = parser.parse_args()

    conn = get_connection(args.db_path)
    init_db(conn)
    try:
        record = None
        if args.record_id is not None:
            record = get_record_by_id(conn, args.record_id)
            if record is None:
                raise ValueError(f"record_id 不存在: {args.record_id}")

        if args.op is not None:
            op_name = resolve_single_operator(conn, args.op)
        elif record is not None:
            op_name = str(record["op_name"])
        else:
            raise ValueError("请提供 --op，或者通过 --record-id 让脚本自动推断算子")

        device = args.device
        if record is not None:
            if args.op is None:
                device = str(record["device"])
            elif str(record["device"]) != args.device:
                print(
                    f"警告: record_id={args.record_id} 的 device={record['device']}，"
                    f"将按命令行指定的 device={args.device} 查找模型"
                )

        if args.list_models:
            models = list_models_for_op_device(conn, op_name, device)
            print(f"模型数: {len(models)}")
            for idx, model in enumerate(models, start=1):
                print(
                    f"  [{idx}] feature_order_key={model['feature_order_key']} "
                    f"feature_order={model['feature_order']}"
                )
            return 0

        params = record["params"] if record is not None else parse_params_input(args.params_json, args.params_file)
        details = predict_latency_details(
            conn,
            op_name,
            device,
            params,
            feature_order_key=args.feature_order_key,
        )
        if details is None:
            print(f"未找到可用模型: op={op_name} device={device}")
            return 1

        report = {
            "op_name": op_name,
            "device": device,
            "record_id": int(record["id"]) if record is not None else None,
            "actual_latency_ms": float(record["latency"]) if record is not None else None,
            "predicted_latency_ms": round(float(details["predicted_latency_ms"]), 6),
            "feature_order_key": details["feature_order_key"],
            "feature_order": details["feature_order"],
            "params": params,
        }
        if report["actual_latency_ms"] is not None:
            actual = float(report["actual_latency_ms"])
            pred = float(report["predicted_latency_ms"])
            report["abs_err_ms"] = round(abs(pred - actual), 6)
            report["ape"] = round(abs(pred - actual) / actual, 6) if actual > 0 else 0.0

        print(json.dumps(report, ensure_ascii=False, indent=2))
        write_json_report(args.report_path, report)
    finally:
        conn.close()
    return 0
