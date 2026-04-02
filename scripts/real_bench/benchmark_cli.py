from __future__ import annotations

import argparse
import sqlite3

import torch

from .benchmark_ops import (
    build_cases_for_sample_ids,
    plan_sample_ids_for_specs,
    print_available_ops,
    resolve_operator_keys,
)
from .common import (
    DEFAULT_DB_PATH,
    DEFAULT_LIMIT_PER_OP,
    DEFAULT_MAX_CV,
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    choose_device_index,
    collect_results,
    existing_keys,
    fetch_existing_benchmark_sample_ids,
    fetch_record_summary,
    init_db,
    insert_results,
    normalize_device_name,
    print_collection_summary,
    semantic_record_key,
    verify_inserted,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure real CUDA operator benchmarks and write records.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-cv", type=float, default=DEFAULT_MAX_CV)
    parser.add_argument("--op", action="append", default=[], help="指定要测的算子 key 或 alias，可重复或逗号分隔")
    parser.add_argument("--all-ops", action="store_true", help="显式跑所有已注册算子")
    parser.add_argument("--list-ops", action="store_true", help="列出所有可用算子并退出")
    parser.add_argument("--limit-per-op", type=int, default=DEFAULT_LIMIT_PER_OP, help="每个算子最多采样多少条 case")
    parser.add_argument("--top-up-to", type=int, default=None, help="只补当前算子缺失样本，直到库内总数达到该值")
    parser.add_argument("--rerun-existing", action="store_true", help="即使 sample_id 已存在，也重新测量这些 case")
    parser.add_argument("--dry-run", action="store_true", help="只测量和校验，不写数据库")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_ops:
        print_available_ops()
        return 0
    if args.warmup < 1 or args.repeats < 5:
        raise ValueError("warmup 至少 1，repeats 至少 5")
    if args.limit_per_op < 1:
        raise ValueError("limit-per-op 至少 1")
    if args.top_up_to is not None and args.top_up_to < 1:
        raise ValueError("top-up-to 至少 1")

    selected_ops = resolve_operator_keys(args.op, allow_all=args.all_ops)

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    device_index = choose_device_index(args.device_index)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    gpu_name = torch.cuda.get_device_name(device_index)
    device_name = normalize_device_name(gpu_name)

    conn = sqlite3.connect(args.db_path)
    try:
        init_db(conn)
        existing_sample_ids = fetch_existing_benchmark_sample_ids(
            conn,
            device=device_name,
            op_keys=selected_ops,
        )
        existing_semantic_keys = existing_keys(conn)
    finally:
        conn.close()

    sample_ids_by_op = plan_sample_ids_for_specs(
        selected_ops,
        limit_per_op=args.limit_per_op,
        existing_sample_ids_by_op=existing_sample_ids,
        top_up_to=args.top_up_to,
        rerun_existing=args.rerun_existing,
    )
    cases = build_cases_for_sample_ids(device, sample_ids_by_op)
    if not args.rerun_existing:
        cases = [
            case
            for case in cases
            if semantic_record_key(case.op_name, device_name, case.params) not in existing_semantic_keys
        ]
    results, skipped = collect_results(
        cases,
        device_index=device_index,
        gpu_name=gpu_name,
        warmup=args.warmup,
        repeats=args.repeats,
        max_cv=args.max_cv,
    )
    print_collection_summary(
        gpu_name=gpu_name,
        device_index=device_index,
        selected_ops=selected_ops,
        cases=cases,
        results=results,
        skipped=skipped,
    )
    print("\n计划采样的 sample_id:")
    for op_key in selected_ops:
        existing_count = len(existing_sample_ids.get(op_key, set()))
        scheduled = sample_ids_by_op.get(op_key, [])
        print(
            f"  {op_key}: existing={existing_count}, scheduled={len(scheduled)}, "
            f"sample_ids={scheduled}"
        )

    if args.dry_run:
        print("\nDry run 完成，未写数据库。")
        return 0

    conn = sqlite3.connect(args.db_path)
    try:
        init_db(conn)
        inserted_ids, inserted_results = insert_results(conn, results)
        verify_inserted(conn, inserted_ids)
        print(f"\n实际写入 records: {len(inserted_ids)}")
        by_inserted: dict[str, int] = {}
        for result in inserted_results:
            by_inserted[result.op_key] = by_inserted.get(result.op_key, 0) + 1
        for op_key, count in sorted(by_inserted.items()):
            print(f"  inserted {op_key}: {count}")
        if inserted_results:
            print("\n新写入样本与 record id:")
            for row_id, result in zip(inserted_ids, inserted_results):
                print(f"  record_id={row_id} <- {result.op_key}#{result.sample_id:02d} ({result.note})")
        print("\n当前库内 id 概览:")
        for op_name, count, min_id, max_id in fetch_record_summary(conn, selected_ops, device_name):
            print(f"  {op_name}: count={count}, id_range=[{min_id}, {max_id}]")
    finally:
        conn.close()

    return 0
