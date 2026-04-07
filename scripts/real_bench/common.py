from __future__ import annotations

import json
import math
import sqlite3
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DB_PATH = ROOT / "data" / "ocm.sqlite3"
DEFAULT_WARMUP = 20
DEFAULT_REPEATS = 50
DEFAULT_MAX_CV = 0.10
DEFAULT_LIMIT_PER_OP = 20
REAL_SOURCE = "real_pytorch_cuda_event"
REAL_DEVICE = "NVIDIA_A100_80GB_PCIe"


@dataclass(frozen=True)
class BenchCase:
    op_key: str
    sample_id: int
    op_name: str
    params: dict[str, Any]
    run: Callable[[], torch.Tensor]
    output_shape: tuple[int, ...]
    note: str
    inner_loops: int = 1


@dataclass(frozen=True)
class BenchResult:
    op_key: str
    sample_id: int
    op_name: str
    device: str
    params: dict[str, Any]
    latency_ms: float
    stats: dict[str, float]
    note: str


@dataclass(frozen=True)
class OperatorSpec:
    key: str
    aliases: tuple[str, ...]
    builder: Callable[..., Any]
    configs: tuple[dict[str, Any], ...]


def normalize_device_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def choose_device_index(requested: int | None) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA，无法收集真实 GPU benchmark")
    count = torch.cuda.device_count()
    if requested is not None:
        if requested < 0 or requested >= count:
            raise ValueError(f"device-index 超出范围: {requested}")
        return requested

    best_idx = 0
    best_free = -1
    for idx in range(count):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx
    return best_idx


def dtype_name(dtype: torch.dtype) -> str:
    import torch

    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    if dtype not in mapping:
        raise ValueError(f"暂不支持的 dtype: {dtype}")
    return mapping[dtype]


def tensor_stride_list(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.stride()]


def conv_output_size(size: int, kernel: int, stride: int, pad: int, dilation: int) -> int:
    out = math.floor((size + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1)
    if out <= 0:
        raise ValueError("非法卷积输出尺寸")
    return out


def params_to_feature_dict(params: dict[str, Any]) -> dict[str, float]:
    feats: dict[str, float] = {}
    stride: list[Any] | None = None
    for key, value in params.items():
        if key == "memory_stride":
            if isinstance(value, list):
                stride = value
            continue
        if key == "is_contiguous":
            feats["is_contiguous"] = 1.0 if bool(value) else 0.0
            continue
        if isinstance(value, bool):
            feats[key] = 1.0 if value else 0.0
            continue
        if isinstance(value, (int, float)):
            feats[key] = float(value)
    if stride is not None:
        for i, value in enumerate(stride):
            feats[f"memory_stride_{i}"] = float(value)
    return feats


def derive_feature_order_key_from_params(params: dict[str, Any]) -> str:
    names = sorted(params_to_feature_dict(params).keys())
    return json.dumps(names, ensure_ascii=False, separators=(",", ":"))


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            op_name TEXT NOT NULL,
            device TEXT NOT NULL,
            params TEXT NOT NULL,
            latency REAL NOT NULL,
            feature_order_key TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_records_op_device ON records (op_name, device);
        CREATE INDEX IF NOT EXISTS idx_records_fok ON records (op_name, device, feature_order_key);
        """
    )
    conn.commit()


def insert_record(
    conn: sqlite3.Connection,
    op_name: str,
    device: str,
    params: dict[str, Any],
    latency: float,
) -> tuple[int, str]:
    payload = json.dumps(params, ensure_ascii=False, sort_keys=True)
    fok = derive_feature_order_key_from_params(params)
    cur = conn.execute(
        """
        INSERT INTO records (op_name, device, params, latency, feature_order_key)
        VALUES (?, ?, ?, ?, ?)
        """,
        (op_name, device, payload, float(latency), fok),
    )
    conn.commit()
    return int(cur.lastrowid), fok


def semantic_params_payload(params: dict[str, Any]) -> str:
    core = dict(params)
    core.pop("benchmark_meta", None)
    return json.dumps(core, ensure_ascii=False, sort_keys=True)


def semantic_record_key(op_name: str, device: str, params: dict[str, Any]) -> tuple[str, str, str]:
    return (op_name, device, semantic_params_payload(params))


def existing_keys(conn: sqlite3.Connection) -> set[tuple[str, str, str]]:
    rows = conn.execute("SELECT op_name, device, params FROM records").fetchall()
    keys: set[tuple[str, str, str]] = set()
    for op_name, device, payload in rows:
        params = json.loads(payload)
        keys.add(semantic_record_key(str(op_name), str(device), params))
    return keys


def benchmark_run(
    run: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
    *,
    inner_loops: int = 1,
) -> tuple[list[float], torch.Tensor]:
    import torch

    if inner_loops < 1:
        raise ValueError("inner_loops 至少为 1")

    out: torch.Tensor | None = None
    for _ in range(warmup):
        for _ in range(inner_loops):
            out = run()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_loops):
            out = run()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)) / inner_loops)

    if out is None:
        raise RuntimeError("benchmark 未产生输出")
    return timings, out


def summarize_timings(timings: list[float]) -> dict[str, float]:
    if not timings:
        raise ValueError("没有 timing 数据")
    median = statistics.median(timings)
    mean = statistics.mean(timings)
    stdev = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    cv = stdev / mean if mean > 0 else 0.0
    return {
        "median_ms": round(median, 6),
        "mean_ms": round(mean, 6),
        "std_ms": round(stdev, 6),
        "min_ms": round(min(timings), 6),
        "max_ms": round(max(timings), 6),
        "cv": round(cv, 6),
    }


def validate_result_shape(output: torch.Tensor, expected: tuple[int, ...]) -> None:
    import torch

    if tuple(output.shape) != expected:
        raise ValueError(f"输出 shape 不匹配: got={tuple(output.shape)} expected={expected}")
    if not torch.isfinite(output).all().item():
        raise ValueError("输出存在 NaN/Inf")


def enrich_params(
    params: dict[str, Any],
    *,
    op_key: str,
    sample_id: int,
    device_index: int,
    gpu_name: str,
    case_note: str,
    warmup: int,
    repeats: int,
    inner_loops: int,
    stats: dict[str, float],
) -> dict[str, Any]:
    import torch

    out = dict(params)
    out["benchmark_meta"] = {
        "source": REAL_SOURCE,
        "op_key": op_key,
        "sample_id": sample_id,
        "sample_label": f"{op_key}#{sample_id:02d}",
        "device_index": device_index,
        "gpu_name": gpu_name,
        "case_note": case_note,
        "warmup_iters": warmup,
        "measure_iters": repeats,
        "inner_loops": inner_loops,
        "stats": stats,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    return out


def collect_results(
    cases: list[BenchCase],
    *,
    device_index: int,
    gpu_name: str,
    warmup: int,
    repeats: int,
    max_cv: float,
) -> tuple[list[BenchResult], list[tuple[str, int, str]]]:
    results: list[BenchResult] = []
    skipped: list[tuple[str, int, str]] = []

    for case in cases:
        timings, output = benchmark_run(
            case.run,
            warmup=warmup,
            repeats=repeats,
            inner_loops=case.inner_loops,
        )
        validate_result_shape(output, case.output_shape)
        stats = summarize_timings(timings)

        if stats["cv"] > max_cv:
            timings, output = benchmark_run(
                case.run,
                warmup=max(1, warmup // 2),
                repeats=repeats * 2,
                inner_loops=case.inner_loops,
            )
            validate_result_shape(output, case.output_shape)
            stats = summarize_timings(timings)

        if stats["cv"] > max_cv:
            skipped.append((case.op_key, case.sample_id, f"{case.note}: cv={stats['cv']:.4f}"))
            continue

        params = enrich_params(
            case.params,
            op_key=case.op_key,
            sample_id=case.sample_id,
            device_index=device_index,
            gpu_name=gpu_name,
            case_note=case.note,
            warmup=warmup,
            repeats=repeats,
            inner_loops=case.inner_loops,
            stats=stats,
        )
        results.append(
            BenchResult(
                op_key=case.op_key,
                sample_id=case.sample_id,
                op_name=case.op_name,
                device=normalize_device_name(gpu_name),
                params=params,
                latency_ms=stats["median_ms"],
                stats=stats,
                note=case.note,
            )
        )

    return results, skipped


def insert_results(conn: sqlite3.Connection, results: list[BenchResult]) -> tuple[list[int], list[BenchResult]]:
    seen = existing_keys(conn)
    inserted_ids: list[int] = []
    inserted_results: list[BenchResult] = []

    for result in results:
        key = semantic_record_key(result.op_name, result.device, result.params)
        if key in seen:
            continue
        row_id, _ = insert_record(conn, result.op_name, result.device, result.params, result.latency_ms)
        inserted_ids.append(row_id)
        inserted_results.append(result)
        seen.add(key)

    return inserted_ids, inserted_results


def verify_inserted(conn: sqlite3.Connection, inserted_ids: list[int]) -> None:
    for row_id in inserted_ids:
        row = conn.execute(
            "SELECT params, latency, feature_order_key FROM records WHERE id = ?",
            (row_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"记录缺失: id={row_id}")
        params = json.loads(row[0])
        expected = derive_feature_order_key_from_params(params)
        if row[2] != expected:
            raise RuntimeError(f"feature_order_key 错误: id={row_id}")
        if not isinstance(row[1], float) or row[1] <= 0:
            raise RuntimeError(f"latency 非法: id={row_id}")


def fetch_existing_benchmark_sample_ids(
    conn: sqlite3.Connection,
    *,
    device: str,
    op_keys: list[str],
) -> dict[str, set[int]]:
    sample_ids_by_op = {op_key: set() for op_key in op_keys}
    if not op_keys:
        return sample_ids_by_op

    placeholders = ",".join("?" for _ in op_keys)
    rows = conn.execute(
        f"""
        SELECT op_name, params
        FROM records
        WHERE device = ? AND op_name IN ({placeholders})
        ORDER BY id
        """,
        [device, *op_keys],
    ).fetchall()
    for op_name, payload in rows:
        try:
            params = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue
        meta = params.get("benchmark_meta") if isinstance(params, dict) else None
        if not isinstance(meta, dict):
            continue
        sample_id = meta.get("sample_id")
        if isinstance(sample_id, int):
            sample_ids_by_op.setdefault(str(op_name), set()).add(int(sample_id))
    return sample_ids_by_op


def fetch_record_summary(conn: sqlite3.Connection, op_keys: list[str], device: str) -> list[tuple[str, int, int, int]]:
    if not op_keys:
        return []
    placeholders = ",".join("?" for _ in op_keys)
    rows = conn.execute(
        f"""
        SELECT op_name, COUNT(*), MIN(id), MAX(id)
        FROM records
        WHERE device = ? AND op_name IN ({placeholders})
        GROUP BY op_name
        ORDER BY op_name
        """,
        [device, *op_keys],
    ).fetchall()
    return [(str(op_name), int(count), int(min_id), int(max_id)) for op_name, count, min_id, max_id in rows]


def print_collection_summary(
    *,
    gpu_name: str,
    device_index: int,
    selected_ops: list[str],
    cases: list[BenchCase],
    results: list[BenchResult],
    skipped: list[tuple[str, int, str]],
) -> None:
    print(f"GPU: {gpu_name} (index={device_index})")
    print(f"目标算子: {', '.join(selected_ops)}")
    print(f"候选 case 数: {len(cases)}")
    print(f"通过稳定性校验的 case 数: {len(results)}")
    print(f"跳过的 case 数: {len(skipped)}")
    if skipped:
        for op_key, sample_id, reason in skipped:
            print(f"  skipped {op_key}#{sample_id:02d}: {reason}")

    by_op = Counter(result.op_key for result in results)
    print("\n通过校验的分布:")
    for op_key, count in sorted(by_op.items()):
        print(f"  {op_key}: {count}")

    print("\n样本预览:")
    for result in results[: min(8, len(results))]:
        print(
            "  "
            f"{result.op_key}#{result.sample_id:02d} "
            f"latency={result.latency_ms:.6f}ms "
            f"cv={result.stats['cv']:.4f} "
            f"note={result.note}"
        )
