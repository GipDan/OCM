#!/usr/bin/env python3
"""Collect real GPU benchmark records and insert only validated records."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ocm.sqlite3"
DEFAULT_WARMUP = 20
DEFAULT_REPEATS = 50
DEFAULT_MAX_CV = 0.10


@dataclass(frozen=True)
class BenchCase:
    op_name: str
    params: dict[str, Any]
    run: Callable[[], torch.Tensor]
    output_shape: tuple[int, ...]
    note: str


@dataclass(frozen=True)
class BenchResult:
    op_name: str
    device: str
    params: dict[str, Any]
    latency_ms: float
    stats: dict[str, float]


def normalize_device_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def choose_device_index(requested: int | None) -> int:
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
    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    if dtype not in mapping:
        raise ValueError(f"暂不支持的 dtype: {dtype}")
    return mapping[dtype]


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


def tensor_stride_list(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.stride()]


def conv_output_size(size: int, kernel: int, stride: int, pad: int, dilation: int) -> int:
    out = math.floor((size + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1)
    if out <= 0:
        raise ValueError("非法卷积输出尺寸")
    return out


def benchmark_run(run: Callable[[], torch.Tensor], warmup: int, repeats: int) -> tuple[list[float], torch.Tensor]:
    out: torch.Tensor | None = None
    for _ in range(warmup):
        out = run()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = run()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))

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
    if tuple(output.shape) != expected:
        raise ValueError(f"输出 shape 不匹配: got={tuple(output.shape)} expected={expected}")
    if not torch.isfinite(output).all().item():
        raise ValueError("输出存在 NaN/Inf")


def make_conv2d_case(
    device: torch.device,
    *,
    layout: str,
    dtype: torch.dtype,
    n: int,
    c: int,
    h: int,
    w: int,
    k: int,
    r: int,
    s: int,
    stride_hw: tuple[int, int],
    pad_hw: tuple[int, int],
    groups: int = 1,
    note: str = "",
) -> BenchCase:
    stride_h, stride_w = stride_hw
    pad_h, pad_w = pad_hw
    memory_format = torch.channels_last if layout == "nhwc" else torch.contiguous_format
    x = torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous(memory_format=memory_format)
    weight = torch.randn(k, c // groups, r, s, device=device, dtype=dtype)
    bias = torch.randn(k, device=device, dtype=dtype)

    out_h = conv_output_size(h, r, stride_h, pad_h, 1)
    out_w = conv_output_size(w, s, stride_w, pad_w, 1)
    op_name = f"nn::conv2d_{layout}_{dtype_name(dtype)}"
    is_contiguous = bool(
        x.is_contiguous(memory_format=torch.channels_last)
        if layout == "nhwc"
        else x.is_contiguous()
    )

    params = {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "K": k,
        "R": r,
        "S": s,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": groups,
        "is_contiguous": is_contiguous,
        "memory_stride": tensor_stride_list(x),
    }

    def run() -> torch.Tensor:
        return F.conv2d(x, weight, bias, stride=stride_hw, padding=pad_hw, groups=groups)

    return BenchCase(
        op_name=op_name,
        params=params,
        run=run,
        output_shape=(n, k, out_h, out_w),
        note=note,
    )


def make_depthwise_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    n: int,
    c: int,
    h: int,
    w: int,
    kernel: int,
    stride_hw: tuple[int, int],
    pad_hw: tuple[int, int],
    note: str = "",
) -> BenchCase:
    stride_h, stride_w = stride_hw
    pad_h, pad_w = pad_hw
    x = torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)
    weight = torch.randn(c, 1, kernel, kernel, device=device, dtype=dtype)

    out_h = conv_output_size(h, kernel, stride_h, pad_h, 1)
    out_w = conv_output_size(w, kernel, stride_w, pad_w, 1)
    params = {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "K": c,
        "R": kernel,
        "S": kernel,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": c,
        "is_contiguous": bool(x.is_contiguous(memory_format=torch.channels_last)),
        "memory_stride": tensor_stride_list(x),
    }

    def run() -> torch.Tensor:
        return F.conv2d(x, weight, None, stride=stride_hw, padding=pad_hw, groups=c)

    return BenchCase(
        op_name=f"nn::depthwise_conv2d_nhwc_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=(n, c, out_h, out_w),
        note=note,
    )


def make_matmul_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    non_contiguous: bool,
    note: str = "",
) -> BenchCase:
    if non_contiguous:
        a = torch.randn(m, k * 2, device=device, dtype=dtype)[:, ::2]
        b = torch.randn(k * 2, n, device=device, dtype=dtype)[::2, :]
    else:
        a = torch.randn(m, k, device=device, dtype=dtype).contiguous()
        b = torch.randn(k, n, device=device, dtype=dtype).contiguous()

    params = {
        "M": m,
        "N": n,
        "K": k,
        "is_contiguous": bool(a.is_contiguous() and b.is_contiguous()),
        "memory_stride": tensor_stride_list(a),
    }

    def run() -> torch.Tensor:
        return torch.matmul(a, b)

    return BenchCase(
        op_name=f"nn::matmul_row_major_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=(m, n),
        note=note or ("non_contiguous" if non_contiguous else "contiguous"),
    )


def make_batch_matmul_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    bsz: int,
    m: int,
    n: int,
    k: int,
    note: str = "",
) -> BenchCase:
    a = torch.randn(bsz, m, k, device=device, dtype=dtype).contiguous()
    b = torch.randn(bsz, k, n, device=device, dtype=dtype).contiguous()
    params = {
        "B": bsz,
        "M": m,
        "N": n,
        "K": k,
        "is_contiguous": True,
        "memory_stride": tensor_stride_list(a),
    }

    def run() -> torch.Tensor:
        return torch.bmm(a, b)

    return BenchCase(
        op_name=f"nn::batch_matmul_row_major_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=(bsz, m, n),
        note=note,
    )


def make_reduce_sum_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    n: int,
    c: int,
    h: int,
    w: int,
    axis: int,
    keepdim: bool,
    non_contiguous: bool,
    note: str = "",
) -> BenchCase:
    if non_contiguous:
        x = torch.randn(n, c, h, w * 2, device=device, dtype=dtype)[:, :, :, ::2]
    else:
        x = torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous()
    out_shape = list(x.shape)
    if keepdim:
        out_shape[axis] = 1
    else:
        del out_shape[axis]

    params = {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "axis": axis,
        "keepdim": keepdim,
        "is_contiguous": bool(x.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    def run() -> torch.Tensor:
        return x.sum(dim=axis, keepdim=keepdim)

    return BenchCase(
        op_name=f"math::reduce_sum_nchw_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=tuple(out_shape),
        note=note or ("non_contiguous" if non_contiguous else "contiguous"),
    )


def make_pool2d_case(
    device: torch.device,
    *,
    pool_kind: str,
    dtype: torch.dtype,
    n: int,
    c: int,
    h: int,
    w: int,
    kernel: int,
    stride_hw: tuple[int, int],
    pad_hw: tuple[int, int],
    note: str = "",
) -> BenchCase:
    stride_h, stride_w = stride_hw
    pad_h, pad_w = pad_hw
    x = torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous()
    out_h = conv_output_size(h, kernel, stride_h, pad_h, 1)
    out_w = conv_output_size(w, kernel, stride_w, pad_w, 1)
    params = {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "kernel_h": kernel,
        "kernel_w": kernel,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "is_contiguous": bool(x.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    if pool_kind == "avg":
        def run() -> torch.Tensor:
            return F.avg_pool2d(x, kernel_size=kernel, stride=stride_hw, padding=pad_hw)

        op_name = f"nn::avg_pool2d_nchw_{dtype_name(dtype)}"
    elif pool_kind == "max":
        def run() -> torch.Tensor:
            return F.max_pool2d(x, kernel_size=kernel, stride=stride_hw, padding=pad_hw)

        op_name = f"nn::max_pool2d_nchw_{dtype_name(dtype)}"
    else:
        raise ValueError(f"未知 pool_kind: {pool_kind}")

    return BenchCase(
        op_name=op_name,
        params=params,
        run=run,
        output_shape=(n, c, out_h, out_w),
        note=note,
    )


def make_adaptive_avg_pool2d_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    n: int,
    c: int,
    h: int,
    w: int,
    out_h: int,
    out_w: int,
    note: str = "",
) -> BenchCase:
    x = torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous()
    params = {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "out_h": out_h,
        "out_w": out_w,
        "is_contiguous": bool(x.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    def run() -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, output_size=(out_h, out_w))

    return BenchCase(
        op_name=f"nn::adaptive_avg_pool2d_nchw_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=(n, c, out_h, out_w),
        note=note,
    )


def make_resize_case(
    device: torch.device,
    *,
    mode: str,
    dtype: torch.dtype,
    n: int,
    c: int,
    h: int,
    w: int,
    out_h: int,
    out_w: int,
    note: str = "",
) -> BenchCase:
    x = torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous()
    params = {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "out_h": out_h,
        "out_w": out_w,
        "is_contiguous": bool(x.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    if mode == "bilinear":
        def run() -> torch.Tensor:
            return F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)

        op_name = f"vision::resize_bilinear_nchw_{dtype_name(dtype)}"
    elif mode == "nearest":
        def run() -> torch.Tensor:
            return F.interpolate(x, size=(out_h, out_w), mode="nearest")

        op_name = f"vision::resize_nearest_nchw_{dtype_name(dtype)}"
    else:
        raise ValueError(f"未知 resize mode: {mode}")

    return BenchCase(
        op_name=op_name,
        params=params,
        run=run,
        output_shape=(n, c, out_h, out_w),
        note=note,
    )


def make_unary_nchw_case(
    device: torch.device,
    *,
    op_base: str,
    dtype: torch.dtype,
    n: int,
    c: int,
    h: int,
    w: int,
    note: str = "",
) -> BenchCase:
    x = torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous()
    params = {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "is_contiguous": bool(x.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    if op_base == "relu":
        def run() -> torch.Tensor:
            return F.relu(x)
    elif op_base == "silu":
        def run() -> torch.Tensor:
            return F.silu(x)
    else:
        raise ValueError(f"未知 unary op: {op_base}")

    return BenchCase(
        op_name=f"nn::{op_base}_nchw_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=(n, c, h, w),
        note=note,
    )


def make_gelu_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    m: int,
    n: int,
    note: str = "",
) -> BenchCase:
    x = torch.randn(m, n, device=device, dtype=dtype).contiguous()
    params = {
        "M": m,
        "N": n,
        "is_contiguous": bool(x.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    def run() -> torch.Tensor:
        return F.gelu(x)

    return BenchCase(
        op_name=f"nn::gelu_row_major_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=(m, n),
        note=note,
    )


def make_softmax_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    m: int,
    n: int,
    dim: int,
    note: str = "",
) -> BenchCase:
    x = torch.randn(m, n, device=device, dtype=dtype).contiguous()
    params = {
        "M": m,
        "N": n,
        "dim": dim,
        "is_contiguous": bool(x.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    def run() -> torch.Tensor:
        return F.softmax(x, dim=dim)

    return BenchCase(
        op_name=f"nn::softmax_row_major_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=(m, n),
        note=note,
    )


def make_layernorm_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    m: int,
    n: int,
    note: str = "",
) -> BenchCase:
    x = torch.randn(m, n, device=device, dtype=dtype).contiguous()
    weight = torch.randn(n, device=device, dtype=dtype)
    bias = torch.randn(n, device=device, dtype=dtype)
    params = {
        "M": m,
        "N": n,
        "normalized_dim": n,
        "is_contiguous": bool(x.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    def run() -> torch.Tensor:
        return F.layer_norm(x, (n,), weight=weight, bias=bias)

    return BenchCase(
        op_name=f"nn::layernorm_row_major_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=(m, n),
        note=note,
    )


def make_binary_nchw_case(
    device: torch.device,
    *,
    op_base: str,
    dtype: torch.dtype,
    n: int,
    c: int,
    h: int,
    w: int,
    note: str = "",
) -> BenchCase:
    x = torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous()
    y = torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous()
    params = {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "is_contiguous": bool(x.is_contiguous() and y.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    if op_base == "add":
        def run() -> torch.Tensor:
            return x + y
    elif op_base == "mul":
        def run() -> torch.Tensor:
            return x * y
    else:
        raise ValueError(f"未知 binary op: {op_base}")

    return BenchCase(
        op_name=f"math::{op_base}_nchw_{dtype_name(dtype)}",
        params=params,
        run=run,
        output_shape=(n, c, h, w),
        note=note,
    )


def build_cases(device: torch.device) -> list[BenchCase]:
    cases: list[BenchCase] = []

    for cfg in [
        dict(layout="nchw", dtype=torch.float32, n=1, c=64, h=56, w=56, k=64, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="resnet_stage1"),
        dict(layout="nchw", dtype=torch.float32, n=8, c=64, h=56, w=56, k=128, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="resnet_stage2"),
        dict(layout="nchw", dtype=torch.float32, n=16, c=128, h=28, w=28, k=256, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="resnet_stage3"),
        dict(layout="nchw", dtype=torch.float32, n=32, c=256, h=14, w=14, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="projection"),
        dict(layout="nchw", dtype=torch.float32, n=4, c=32, h=112, w=112, k=64, r=7, s=7, stride_hw=(2, 2), pad_hw=(3, 3), note="stem_conv"),
        dict(layout="nchw", dtype=torch.float32, n=8, c=64, h=56, w=56, k=64, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="bottleneck_reduce"),
        dict(layout="nchw", dtype=torch.float32, n=8, c=128, h=28, w=28, k=128, r=3, s=3, stride_hw=(2, 2), pad_hw=(1, 1), note="downsample_conv"),
        dict(layout="nchw", dtype=torch.float32, n=16, c=128, h=28, w=28, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="bottleneck_expand"),
        dict(layout="nchw", dtype=torch.float32, n=16, c=256, h=14, w=14, k=256, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="stage4_mid"),
        dict(layout="nchw", dtype=torch.float32, n=32, c=512, h=7, w=7, k=1024, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="stage5_proj"),
        dict(layout="nchw", dtype=torch.float32, n=8, c=64, h=56, w=56, k=192, r=5, s=5, stride_hw=(1, 1), pad_hw=(2, 2), note="wide_kernel"),
    ]:
        cases.append(make_conv2d_case(device, **cfg))

    for cfg in [
        dict(layout="nhwc", dtype=torch.float16, n=1, c=64, h=56, w=56, k=64, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_small"),
        dict(layout="nhwc", dtype=torch.float16, n=8, c=64, h=56, w=56, k=128, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_medium"),
        dict(layout="nhwc", dtype=torch.float16, n=16, c=128, h=28, w=28, k=256, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_large"),
        dict(layout="nhwc", dtype=torch.float16, n=32, c=256, h=14, w=14, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="channels_last_projection"),
        dict(layout="nhwc", dtype=torch.float16, n=4, c=32, h=112, w=112, k=64, r=7, s=7, stride_hw=(2, 2), pad_hw=(3, 3), note="channels_last_stem"),
        dict(layout="nhwc", dtype=torch.float16, n=8, c=64, h=56, w=56, k=64, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="channels_last_reduce"),
        dict(layout="nhwc", dtype=torch.float16, n=8, c=128, h=28, w=28, k=128, r=3, s=3, stride_hw=(2, 2), pad_hw=(1, 1), note="channels_last_downsample"),
        dict(layout="nhwc", dtype=torch.float16, n=16, c=128, h=28, w=28, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="channels_last_expand"),
        dict(layout="nhwc", dtype=torch.float16, n=16, c=256, h=14, w=14, k=256, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_stage4"),
        dict(layout="nhwc", dtype=torch.float16, n=32, c=512, h=7, w=7, k=1024, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="channels_last_stage5"),
        dict(layout="nhwc", dtype=torch.float16, n=8, c=64, h=56, w=56, k=192, r=5, s=5, stride_hw=(1, 1), pad_hw=(2, 2), note="channels_last_wide_kernel"),
    ]:
        cases.append(make_conv2d_case(device, **cfg))

    for cfg in [
        dict(dtype=torch.float16, n=1, c=64, h=56, w=56, kernel=3, stride_hw=(1, 1), pad_hw=(1, 1), note="depthwise_small"),
        dict(dtype=torch.float16, n=8, c=128, h=28, w=28, kernel=3, stride_hw=(1, 1), pad_hw=(1, 1), note="depthwise_medium"),
        dict(dtype=torch.float16, n=16, c=256, h=14, w=14, kernel=5, stride_hw=(1, 1), pad_hw=(2, 2), note="depthwise_large"),
    ]:
        cases.append(make_depthwise_case(device, **cfg))

    for cfg in [
        dict(dtype=torch.float32, m=512, n=512, k=512, non_contiguous=False, note="ffn_small"),
        dict(dtype=torch.float32, m=1024, n=1024, k=512, non_contiguous=False, note="ffn_medium"),
        dict(dtype=torch.float32, m=2048, n=1024, k=1024, non_contiguous=False, note="ffn_large"),
        dict(dtype=torch.float32, m=1024, n=1024, k=512, non_contiguous=True, note="ffn_medium_strided"),
        dict(dtype=torch.float32, m=256, n=2048, k=512, non_contiguous=False, note="skinny_wide"),
        dict(dtype=torch.float32, m=1536, n=768, k=512, non_contiguous=False, note="transformer_mlp"),
        dict(dtype=torch.float32, m=2048, n=2048, k=256, non_contiguous=False, note="square_large"),
        dict(dtype=torch.float32, m=3072, n=768, k=768, non_contiguous=False, note="decoder_proj"),
        dict(dtype=torch.float32, m=1536, n=1536, k=768, non_contiguous=True, note="square_strided"),
        dict(dtype=torch.float32, m=4096, n=512, k=1024, non_contiguous=False, note="tall_skinny"),
        dict(dtype=torch.float32, m=2048, n=1536, k=1024, non_contiguous=False, note="wide_out"),
    ]:
        cases.append(make_matmul_case(device, **cfg))

    for cfg in [
        dict(dtype=torch.float16, m=1024, n=1024, k=1024, non_contiguous=False, note="gemm_fp16_small"),
        dict(dtype=torch.float16, m=2048, n=1024, k=1024, non_contiguous=False, note="gemm_fp16_medium"),
        dict(dtype=torch.float16, m=4096, n=1024, k=1024, non_contiguous=False, note="gemm_fp16_large"),
        dict(dtype=torch.float16, m=2048, n=1024, k=1024, non_contiguous=True, note="gemm_fp16_strided"),
    ]:
        cases.append(make_matmul_case(device, **cfg))

    for cfg in [
        dict(dtype=torch.float16, bsz=8, m=128, n=128, k=128, note="attention_small"),
        dict(dtype=torch.float16, bsz=16, m=256, n=256, k=128, note="attention_medium"),
        dict(dtype=torch.float16, bsz=32, m=512, n=512, k=256, note="attention_large"),
    ]:
        cases.append(make_batch_matmul_case(device, **cfg))

    for cfg in [
        dict(dtype=torch.float32, n=8, c=64, h=56, w=56, axis=1, keepdim=True, non_contiguous=False, note="channel_reduce"),
        dict(dtype=torch.float32, n=16, c=128, h=28, w=28, axis=2, keepdim=False, non_contiguous=False, note="height_reduce"),
        dict(dtype=torch.float32, n=16, c=256, h=14, w=14, axis=3, keepdim=False, non_contiguous=True, note="width_reduce_strided"),
    ]:
        cases.append(make_reduce_sum_case(device, **cfg))

    for cfg in [
        dict(pool_kind="avg", dtype=torch.float32, n=8, c=64, h=56, w=56, kernel=2, stride_hw=(2, 2), pad_hw=(0, 0), note="avg_pool_stage1"),
        dict(pool_kind="avg", dtype=torch.float32, n=16, c=128, h=28, w=28, kernel=3, stride_hw=(2, 2), pad_hw=(1, 1), note="avg_pool_stage2"),
        dict(pool_kind="max", dtype=torch.float32, n=8, c=64, h=56, w=56, kernel=2, stride_hw=(2, 2), pad_hw=(0, 0), note="max_pool_stage1"),
        dict(pool_kind="max", dtype=torch.float32, n=16, c=128, h=28, w=28, kernel=3, stride_hw=(2, 2), pad_hw=(1, 1), note="max_pool_stage2"),
    ]:
        cases.append(make_pool2d_case(device, **cfg))

    for cfg in [
        dict(dtype=torch.float32, n=8, c=128, h=28, w=28, out_h=7, out_w=7, note="globalish_pool"),
        dict(dtype=torch.float32, n=16, c=256, h=14, w=14, out_h=1, out_w=1, note="global_pool"),
    ]:
        cases.append(make_adaptive_avg_pool2d_case(device, **cfg))

    for cfg in [
        dict(mode="bilinear", dtype=torch.float32, n=4, c=64, h=56, w=56, out_h=112, out_w=112, note="upsample_bilinear"),
        dict(mode="nearest", dtype=torch.float32, n=4, c=64, h=56, w=56, out_h=112, out_w=112, note="upsample_nearest"),
    ]:
        cases.append(make_resize_case(device, **cfg))

    for cfg in [
        dict(op_base="relu", dtype=torch.float32, n=16, c=128, h=28, w=28, note="relu_featuremap"),
        dict(op_base="silu", dtype=torch.float32, n=16, c=128, h=28, w=28, note="silu_featuremap"),
    ]:
        cases.append(make_unary_nchw_case(device, **cfg))

    for cfg in [
        dict(dtype=torch.float32, m=4096, n=1024, note="gelu_ffn"),
    ]:
        cases.append(make_gelu_case(device, **cfg))

    for cfg in [
        dict(dtype=torch.float32, m=2048, n=1024, dim=1, note="softmax_logits"),
    ]:
        cases.append(make_softmax_case(device, **cfg))

    for cfg in [
        dict(dtype=torch.float32, m=4096, n=1024, note="layernorm_ffn"),
    ]:
        cases.append(make_layernorm_case(device, **cfg))

    for cfg in [
        dict(op_base="add", dtype=torch.float32, n=32, c=256, h=56, w=56, note="residual_add"),
        dict(op_base="mul", dtype=torch.float32, n=32, c=256, h=56, w=56, note="elementwise_mul"),
    ]:
        cases.append(make_binary_nchw_case(device, **cfg))

    return cases


def enrich_params(
    params: dict[str, Any],
    *,
    device_index: int,
    gpu_name: str,
    case_note: str,
    warmup: int,
    repeats: int,
    stats: dict[str, float],
) -> dict[str, Any]:
    out = dict(params)
    out["benchmark_meta"] = {
        "source": "real_pytorch_cuda_event",
        "device_index": device_index,
        "gpu_name": gpu_name,
        "case_note": case_note,
        "warmup_iters": warmup,
        "measure_iters": repeats,
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
) -> tuple[list[BenchResult], list[tuple[str, str]]]:
    results: list[BenchResult] = []
    skipped: list[tuple[str, str]] = []

    for case in cases:
        timings, output = benchmark_run(case.run, warmup=warmup, repeats=repeats)
        validate_result_shape(output, case.output_shape)
        stats = summarize_timings(timings)

        if stats["cv"] > max_cv:
            timings, output = benchmark_run(case.run, warmup=warmup // 2, repeats=repeats * 2)
            validate_result_shape(output, case.output_shape)
            stats = summarize_timings(timings)

        if stats["cv"] > max_cv:
            skipped.append((case.op_name, f"{case.note}: cv={stats['cv']:.4f}"))
            continue

        params = enrich_params(
            case.params,
            device_index=device_index,
            gpu_name=gpu_name,
            case_note=case.note,
            warmup=warmup,
            repeats=repeats,
            stats=stats,
        )
        results.append(
            BenchResult(
                op_name=case.op_name,
                device=normalize_device_name(gpu_name),
                params=params,
                latency_ms=stats["median_ms"],
                stats=stats,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure real CUDA operator benchmarks and write records.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-cv", type=float, default=DEFAULT_MAX_CV)
    parser.add_argument("--dry-run", action="store_true", help="只测量和校验，不写数据库")
    args = parser.parse_args()

    if args.warmup < 1 or args.repeats < 5:
        raise ValueError("warmup 至少 1，repeats 至少 5")

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    device_index = choose_device_index(args.device_index)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    gpu_name = torch.cuda.get_device_name(device_index)

    cases = build_cases(device)
    results, skipped = collect_results(
        cases,
        device_index=device_index,
        gpu_name=gpu_name,
        warmup=args.warmup,
        repeats=args.repeats,
        max_cv=args.max_cv,
    )

    print(f"GPU: {gpu_name} (index={device_index})")
    print(f"候选 case 数: {len(cases)}")
    print(f"通过稳定性校验的 case 数: {len(results)}")
    print(f"跳过的 case 数: {len(skipped)}")
    if skipped:
        for op_name, reason in skipped:
            print(f"  skipped {op_name}: {reason}")

    by_op = Counter(result.op_name for result in results)
    print("\n通过校验的分布:")
    for op_name, count in sorted(by_op.items()):
        print(f"  {op_name}: {count}")

    for result in results[:8]:
        print(
            "  "
            f"{result.op_name} latency={result.latency_ms:.6f}ms "
            f"cv={result.stats['cv']:.4f} device={result.device}"
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
        by_inserted = Counter(result.op_name for result in inserted_results)
        for op_name, count in sorted(by_inserted.items()):
            print(f"  inserted {op_name}: {count}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
