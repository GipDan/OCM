from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from .common import BenchCase, OperatorSpec, conv_output_size, dtype_name, tensor_stride_list


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
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return op_name, params, run, (n, k, out_h, out_w), note


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
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return f"nn::depthwise_conv2d_nhwc_{dtype_name(dtype)}", params, run, (n, c, out_h, out_w), note


def make_matmul_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    non_contiguous: bool,
    note: str = "",
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    sample_note = note or ("non_contiguous" if non_contiguous else "contiguous")
    return f"nn::matmul_row_major_{dtype_name(dtype)}", params, run, (m, n), sample_note


def make_batch_matmul_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    bsz: int,
    m: int,
    n: int,
    k: int,
    note: str = "",
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return f"nn::batch_matmul_row_major_{dtype_name(dtype)}", params, run, (bsz, m, n), note


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
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    sample_note = note or ("non_contiguous" if non_contiguous else "contiguous")
    return f"math::reduce_sum_nchw_{dtype_name(dtype)}", params, run, tuple(out_shape), sample_note


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
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return op_name, params, run, (n, c, out_h, out_w), note


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
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return f"nn::adaptive_avg_pool2d_nchw_{dtype_name(dtype)}", params, run, (n, c, out_h, out_w), note


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
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return op_name, params, run, (n, c, out_h, out_w), note


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
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return f"nn::{op_base}_nchw_{dtype_name(dtype)}", params, run, (n, c, h, w), note


def make_gelu_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    m: int,
    n: int,
    note: str = "",
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
    x = torch.randn(m, n, device=device, dtype=dtype).contiguous()
    params = {
        "M": m,
        "N": n,
        "is_contiguous": bool(x.is_contiguous()),
        "memory_stride": tensor_stride_list(x),
    }

    def run() -> torch.Tensor:
        return F.gelu(x)

    return f"nn::gelu_row_major_{dtype_name(dtype)}", params, run, (m, n), note


def make_softmax_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    m: int,
    n: int,
    dim: int,
    note: str = "",
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return f"nn::softmax_row_major_{dtype_name(dtype)}", params, run, (m, n), note


def make_layernorm_case(
    device: torch.device,
    *,
    dtype: torch.dtype,
    m: int,
    n: int,
    note: str = "",
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return f"nn::layernorm_row_major_{dtype_name(dtype)}", params, run, (m, n), note


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
) -> tuple[str, dict[str, Any], Callable[[], torch.Tensor], tuple[int, ...], str]:
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

    return f"math::{op_base}_nchw_{dtype_name(dtype)}", params, run, (n, c, h, w), note


CONV2D_NCHW_FP32_CONFIGS = (
    dict(layout="nchw", dtype=torch.float32, n=1, c=3, h=224, w=224, k=64, r=7, s=7, stride_hw=(2, 2), pad_hw=(3, 3), note="stem_rgb"),
    dict(layout="nchw", dtype=torch.float32, n=4, c=32, h=112, w=112, k=64, r=7, s=7, stride_hw=(2, 2), pad_hw=(3, 3), note="stem_conv"),
    dict(layout="nchw", dtype=torch.float32, n=8, c=32, h=112, w=112, k=64, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="early_stage_small"),
    dict(layout="nchw", dtype=torch.float32, n=1, c=64, h=56, w=56, k=64, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="resnet_stage1"),
    dict(layout="nchw", dtype=torch.float32, n=8, c=64, h=56, w=56, k=64, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="bottleneck_reduce"),
    dict(layout="nchw", dtype=torch.float32, n=8, c=64, h=56, w=56, k=128, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="resnet_stage2"),
    dict(layout="nchw", dtype=torch.float32, n=16, c=64, h=56, w=56, k=128, r=3, s=3, stride_hw=(2, 2), pad_hw=(1, 1), note="stride2_downsample"),
    dict(layout="nchw", dtype=torch.float32, n=8, c=96, h=56, w=56, k=192, r=5, s=5, stride_hw=(1, 1), pad_hw=(2, 2), note="wide_kernel_56"),
    dict(layout="nchw", dtype=torch.float32, n=16, c=128, h=28, w=28, k=128, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="resnet_stage3"),
    dict(layout="nchw", dtype=torch.float32, n=16, c=128, h=28, w=28, k=256, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="bottleneck_expand"),
    dict(layout="nchw", dtype=torch.float32, n=8, c=128, h=28, w=28, k=128, r=3, s=3, stride_hw=(2, 2), pad_hw=(1, 1), note="downsample_conv"),
    dict(layout="nchw", dtype=torch.float32, n=32, c=128, h=28, w=28, k=256, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="stage3_wide_batch"),
    dict(layout="nchw", dtype=torch.float32, n=8, c=192, h=28, w=28, k=192, r=5, s=5, stride_hw=(1, 1), pad_hw=(2, 2), note="wide_kernel_28"),
    dict(layout="nchw", dtype=torch.float32, n=16, c=256, h=14, w=14, k=256, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="stage4_mid"),
    dict(layout="nchw", dtype=torch.float32, n=16, c=256, h=14, w=14, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="projection"),
    dict(layout="nchw", dtype=torch.float32, n=32, c=256, h=14, w=14, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="projection_batched"),
    dict(layout="nchw", dtype=torch.float32, n=8, c=256, h=14, w=14, k=256, r=5, s=5, stride_hw=(1, 1), pad_hw=(2, 2), note="wide_kernel_14"),
    dict(layout="nchw", dtype=torch.float32, n=16, c=512, h=7, w=7, k=512, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="stage5_mid"),
    dict(layout="nchw", dtype=torch.float32, n=32, c=512, h=7, w=7, k=1024, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="stage5_proj"),
    dict(layout="nchw", dtype=torch.float32, n=8, c=768, h=7, w=7, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="late_fusion"),
)

CONV2D_NHWC_FP16_CONFIGS = (
    dict(layout="nhwc", dtype=torch.float16, n=1, c=64, h=56, w=56, k=64, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_small"),
    dict(layout="nhwc", dtype=torch.float16, n=4, c=32, h=112, w=112, k=64, r=7, s=7, stride_hw=(2, 2), pad_hw=(3, 3), note="channels_last_stem"),
    dict(layout="nhwc", dtype=torch.float16, n=8, c=32, h=112, w=112, k=64, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_early"),
    dict(layout="nhwc", dtype=torch.float16, n=8, c=64, h=56, w=56, k=64, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="channels_last_reduce"),
    dict(layout="nhwc", dtype=torch.float16, n=8, c=64, h=56, w=56, k=128, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_medium"),
    dict(layout="nhwc", dtype=torch.float16, n=16, c=64, h=56, w=56, k=128, r=3, s=3, stride_hw=(2, 2), pad_hw=(1, 1), note="channels_last_stride2"),
    dict(layout="nhwc", dtype=torch.float16, n=8, c=64, h=56, w=56, k=192, r=5, s=5, stride_hw=(1, 1), pad_hw=(2, 2), note="channels_last_wide_kernel"),
    dict(layout="nhwc", dtype=torch.float16, n=16, c=96, h=56, w=56, k=160, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_mobilenetish"),
    dict(layout="nhwc", dtype=torch.float16, n=8, c=128, h=28, w=28, k=128, r=3, s=3, stride_hw=(2, 2), pad_hw=(1, 1), note="channels_last_downsample"),
    dict(layout="nhwc", dtype=torch.float16, n=16, c=128, h=28, w=28, k=256, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_large"),
    dict(layout="nhwc", dtype=torch.float16, n=16, c=128, h=28, w=28, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="channels_last_expand"),
    dict(layout="nhwc", dtype=torch.float16, n=32, c=128, h=28, w=28, k=256, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_batchy"),
    dict(layout="nhwc", dtype=torch.float16, n=16, c=256, h=14, w=14, k=256, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_stage4"),
    dict(layout="nhwc", dtype=torch.float16, n=16, c=256, h=14, w=14, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="channels_last_projection"),
    dict(layout="nhwc", dtype=torch.float16, n=8, c=256, h=14, w=14, k=256, r=5, s=5, stride_hw=(1, 1), pad_hw=(2, 2), note="channels_last_wide14"),
    dict(layout="nhwc", dtype=torch.float16, n=8, c=384, h=14, w=14, k=384, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_dense14"),
    dict(layout="nhwc", dtype=torch.float16, n=16, c=512, h=7, w=7, k=512, r=3, s=3, stride_hw=(1, 1), pad_hw=(1, 1), note="channels_last_stage5"),
    dict(layout="nhwc", dtype=torch.float16, n=32, c=512, h=7, w=7, k=1024, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="channels_last_stage5_proj"),
    dict(layout="nhwc", dtype=torch.float16, n=8, c=768, h=7, w=7, k=512, r=1, s=1, stride_hw=(1, 1), pad_hw=(0, 0), note="channels_last_late_fusion"),
    dict(layout="nhwc", dtype=torch.float16, n=4, c=128, h=112, w=112, k=64, r=3, s=3, stride_hw=(2, 2), pad_hw=(1, 1), note="channels_last_large_map"),
)

MATMUL_FP32_CONFIGS = (
    dict(dtype=torch.float32, m=256, n=256, k=256, non_contiguous=False, note="square_256"),
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
    dict(dtype=torch.float32, m=768, n=3072, k=768, non_contiguous=False, note="mlp_expand"),
    dict(dtype=torch.float32, m=3072, n=768, k=3072, non_contiguous=False, note="mlp_contract"),
    dict(dtype=torch.float32, m=1024, n=4096, k=256, non_contiguous=False, note="wider_output"),
    dict(dtype=torch.float32, m=4096, n=1024, k=256, non_contiguous=False, note="taller_input"),
    dict(dtype=torch.float32, m=896, n=896, k=448, non_contiguous=True, note="square_mid_strided"),
    dict(dtype=torch.float32, m=640, n=1536, k=320, non_contiguous=False, note="attention_ffn_mix"),
    dict(dtype=torch.float32, m=1280, n=512, k=2560, non_contiguous=False, note="projection_heavy_k"),
    dict(dtype=torch.float32, m=512, n=1280, k=2560, non_contiguous=True, note="projection_heavy_k_strided"),
    dict(dtype=torch.float32, m=1536, n=768, k=512, non_contiguous=True, note="transformer_mlp_strided"),
    dict(dtype=torch.float32, m=768, n=1536, k=512, non_contiguous=False, note="transformer_mlp_wide"),
    dict(dtype=torch.float32, m=1024, n=2048, k=256, non_contiguous=False, note="mid_wide_output"),
    dict(dtype=torch.float32, m=2048, n=1024, k=256, non_contiguous=False, note="mid_tall_input"),
    dict(dtype=torch.float32, m=512, n=2048, k=2048, non_contiguous=False, note="heavy_k_wide"),
    dict(dtype=torch.float32, m=2048, n=512, k=2048, non_contiguous=False, note="heavy_k_tall"),
    dict(dtype=torch.float32, m=1536, n=1024, k=768, non_contiguous=False, note="decoder_backproj"),
    dict(dtype=torch.float32, m=1024, n=1536, k=768, non_contiguous=False, note="decoder_proj_mid"),
    dict(dtype=torch.float32, m=1024, n=4096, k=256, non_contiguous=True, note="wider_output_strided"),
    dict(dtype=torch.float32, m=4096, n=1024, k=256, non_contiguous=True, note="taller_input_strided"),
)

MATMUL_FP16_CONFIGS = (
    dict(dtype=torch.float16, m=512, n=512, k=512, non_contiguous=False, note="fp16_square_512"),
    dict(dtype=torch.float16, m=1024, n=1024, k=1024, non_contiguous=False, note="gemm_fp16_small"),
    dict(dtype=torch.float16, m=2048, n=1024, k=1024, non_contiguous=False, note="gemm_fp16_medium"),
    dict(dtype=torch.float16, m=4096, n=1024, k=1024, non_contiguous=False, note="gemm_fp16_large"),
    dict(dtype=torch.float16, m=2048, n=1024, k=1024, non_contiguous=True, note="gemm_fp16_strided"),
    dict(dtype=torch.float16, m=1024, n=4096, k=512, non_contiguous=False, note="fp16_wide_out"),
    dict(dtype=torch.float16, m=4096, n=512, k=1024, non_contiguous=False, note="fp16_tall_skinny"),
    dict(dtype=torch.float16, m=1536, n=1536, k=768, non_contiguous=False, note="fp16_square_transformer"),
    dict(dtype=torch.float16, m=1536, n=1536, k=768, non_contiguous=True, note="fp16_square_transformer_strided"),
    dict(dtype=torch.float16, m=768, n=3072, k=768, non_contiguous=False, note="fp16_mlp_expand"),
    dict(dtype=torch.float16, m=3072, n=768, k=3072, non_contiguous=False, note="fp16_mlp_contract"),
    dict(dtype=torch.float16, m=256, n=2048, k=512, non_contiguous=False, note="fp16_skinny_wide"),
    dict(dtype=torch.float16, m=2048, n=1536, k=1024, non_contiguous=False, note="fp16_wide_out_large"),
    dict(dtype=torch.float16, m=2048, n=2048, k=256, non_contiguous=False, note="fp16_square_large"),
    dict(dtype=torch.float16, m=896, n=896, k=448, non_contiguous=True, note="fp16_square_mid_strided"),
    dict(dtype=torch.float16, m=640, n=1536, k=320, non_contiguous=False, note="fp16_attention_ffn_mix"),
    dict(dtype=torch.float16, m=1280, n=512, k=2560, non_contiguous=False, note="fp16_projection_heavy_k"),
    dict(dtype=torch.float16, m=512, n=1280, k=2560, non_contiguous=True, note="fp16_projection_heavy_k_strided"),
    dict(dtype=torch.float16, m=1024, n=1536, k=768, non_contiguous=False, note="fp16_decoder_proj"),
    dict(dtype=torch.float16, m=1536, n=1024, k=768, non_contiguous=False, note="fp16_decoder_backproj"),
    dict(dtype=torch.float16, m=1536, n=768, k=512, non_contiguous=True, note="fp16_transformer_mlp_strided"),
    dict(dtype=torch.float16, m=768, n=1536, k=512, non_contiguous=False, note="fp16_transformer_mlp_wide"),
    dict(dtype=torch.float16, m=1024, n=2048, k=256, non_contiguous=False, note="fp16_mid_wide_output"),
    dict(dtype=torch.float16, m=2048, n=1024, k=256, non_contiguous=False, note="fp16_mid_tall_input"),
    dict(dtype=torch.float16, m=512, n=2048, k=2048, non_contiguous=False, note="fp16_heavy_k_wide"),
    dict(dtype=torch.float16, m=2048, n=512, k=2048, non_contiguous=False, note="fp16_heavy_k_tall"),
    dict(dtype=torch.float16, m=1536, n=1024, k=768, non_contiguous=True, note="fp16_decoder_backproj_strided"),
    dict(dtype=torch.float16, m=1024, n=1536, k=768, non_contiguous=True, note="fp16_decoder_proj_mid_strided"),
    dict(dtype=torch.float16, m=1024, n=4096, k=256, non_contiguous=True, note="fp16_wider_output_strided"),
    dict(dtype=torch.float16, m=4096, n=1024, k=256, non_contiguous=True, note="fp16_taller_input_strided"),
)

BATCH_MATMUL_FP16_CONFIGS = (
    dict(dtype=torch.float16, bsz=4, m=64, n=64, k=64, note="attention_tiny"),
    dict(dtype=torch.float16, bsz=8, m=128, n=128, k=128, note="attention_small"),
    dict(dtype=torch.float16, bsz=16, m=256, n=256, k=128, note="attention_medium"),
    dict(dtype=torch.float16, bsz=32, m=512, n=512, k=256, note="attention_large"),
    dict(dtype=torch.float16, bsz=8, m=128, n=64, k=128, note="kv_project"),
    dict(dtype=torch.float16, bsz=8, m=64, n=128, k=128, note="qk_small"),
    dict(dtype=torch.float16, bsz=16, m=128, n=64, k=64, note="value_small"),
    dict(dtype=torch.float16, bsz=16, m=256, n=128, k=64, note="value_medium"),
    dict(dtype=torch.float16, bsz=32, m=128, n=128, k=64, note="many_heads"),
    dict(dtype=torch.float16, bsz=32, m=256, n=64, k=128, note="decoder_cross"),
    dict(dtype=torch.float16, bsz=40, m=128, n=128, k=128, note="long_context"),
    dict(dtype=torch.float16, bsz=64, m=64, n=64, k=64, note="micro_heads"),
)

REDUCE_SUM_FP32_CONFIGS = (
    dict(dtype=torch.float32, n=8, c=64, h=56, w=56, axis=1, keepdim=True, non_contiguous=False, note="channel_reduce_keepdim"),
    dict(dtype=torch.float32, n=16, c=128, h=28, w=28, axis=2, keepdim=False, non_contiguous=False, note="height_reduce"),
    dict(dtype=torch.float32, n=16, c=256, h=14, w=14, axis=3, keepdim=False, non_contiguous=True, note="width_reduce_strided"),
    dict(dtype=torch.float32, n=4, c=32, h=112, w=112, axis=1, keepdim=False, non_contiguous=False, note="channel_reduce_large_map"),
    dict(dtype=torch.float32, n=8, c=96, h=56, w=56, axis=2, keepdim=True, non_contiguous=False, note="height_keepdim"),
    dict(dtype=torch.float32, n=8, c=96, h=56, w=56, axis=3, keepdim=False, non_contiguous=False, note="width_reduce"),
    dict(dtype=torch.float32, n=16, c=64, h=56, w=56, axis=0, keepdim=False, non_contiguous=False, note="batch_reduce"),
    dict(dtype=torch.float32, n=2, c=512, h=28, w=28, axis=1, keepdim=False, non_contiguous=True, note="channel_reduce_strided"),
    dict(dtype=torch.float32, n=4, c=256, h=14, w=14, axis=2, keepdim=False, non_contiguous=True, note="height_reduce_strided"),
    dict(dtype=torch.float32, n=4, c=256, h=14, w=14, axis=3, keepdim=True, non_contiguous=False, note="width_keepdim"),
    dict(dtype=torch.float32, n=16, c=32, h=128, w=128, axis=1, keepdim=True, non_contiguous=False, note="big_map_channel"),
    dict(dtype=torch.float32, n=32, c=64, h=28, w=28, axis=0, keepdim=True, non_contiguous=False, note="batch_keepdim"),
)

AVG_POOL_FP32_CONFIGS = (
    dict(pool_kind="avg", dtype=torch.float32, n=8, c=64, h=56, w=56, kernel=2, stride_hw=(2, 2), pad_hw=(0, 0), note="avg_pool_stage1"),
    dict(pool_kind="avg", dtype=torch.float32, n=16, c=128, h=28, w=28, kernel=3, stride_hw=(2, 2), pad_hw=(1, 1), note="avg_pool_stage2"),
    dict(pool_kind="avg", dtype=torch.float32, n=4, c=32, h=112, w=112, kernel=3, stride_hw=(2, 2), pad_hw=(1, 1), note="avg_pool_stem"),
    dict(pool_kind="avg", dtype=torch.float32, n=16, c=256, h=14, w=14, kernel=2, stride_hw=(2, 2), pad_hw=(0, 0), note="avg_pool_stage4"),
    dict(pool_kind="avg", dtype=torch.float32, n=8, c=512, h=7, w=7, kernel=2, stride_hw=(1, 1), pad_hw=(0, 0), note="avg_pool_small_map"),
    dict(pool_kind="avg", dtype=torch.float32, n=32, c=64, h=28, w=28, kernel=3, stride_hw=(1, 1), pad_hw=(1, 1), note="avg_pool_same"),
)

MAX_POOL_FP32_CONFIGS = (
    dict(pool_kind="max", dtype=torch.float32, n=8, c=64, h=56, w=56, kernel=2, stride_hw=(2, 2), pad_hw=(0, 0), note="max_pool_stage1"),
    dict(pool_kind="max", dtype=torch.float32, n=16, c=128, h=28, w=28, kernel=3, stride_hw=(2, 2), pad_hw=(1, 1), note="max_pool_stage2"),
    dict(pool_kind="max", dtype=torch.float32, n=4, c=32, h=112, w=112, kernel=3, stride_hw=(2, 2), pad_hw=(1, 1), note="max_pool_stem"),
    dict(pool_kind="max", dtype=torch.float32, n=16, c=256, h=14, w=14, kernel=2, stride_hw=(2, 2), pad_hw=(0, 0), note="max_pool_stage4"),
    dict(pool_kind="max", dtype=torch.float32, n=8, c=512, h=7, w=7, kernel=2, stride_hw=(1, 1), pad_hw=(0, 0), note="max_pool_small_map"),
    dict(pool_kind="max", dtype=torch.float32, n=32, c=64, h=28, w=28, kernel=3, stride_hw=(1, 1), pad_hw=(1, 1), note="max_pool_same"),
)

AVAILABLE_SPECS = (
    OperatorSpec("nn::conv2d_nchw_fp32", ("conv2d_nchw_fp32",), make_conv2d_case, CONV2D_NCHW_FP32_CONFIGS),
    OperatorSpec("nn::conv2d_nhwc_fp16", ("conv2d_nhwc_fp16",), make_conv2d_case, CONV2D_NHWC_FP16_CONFIGS),
    OperatorSpec("nn::depthwise_conv2d_nhwc_fp16", ("depthwise_conv2d_nhwc_fp16",), make_depthwise_case, (
        dict(dtype=torch.float16, n=1, c=64, h=56, w=56, kernel=3, stride_hw=(1, 1), pad_hw=(1, 1), note="depthwise_small"),
        dict(dtype=torch.float16, n=8, c=128, h=28, w=28, kernel=3, stride_hw=(1, 1), pad_hw=(1, 1), note="depthwise_medium"),
        dict(dtype=torch.float16, n=16, c=256, h=14, w=14, kernel=5, stride_hw=(1, 1), pad_hw=(2, 2), note="depthwise_large"),
        dict(dtype=torch.float16, n=32, c=256, h=14, w=14, kernel=3, stride_hw=(2, 2), pad_hw=(1, 1), note="depthwise_stride2"),
        dict(dtype=torch.float16, n=8, c=384, h=14, w=14, kernel=3, stride_hw=(1, 1), pad_hw=(1, 1), note="depthwise_dense"),
        dict(dtype=torch.float16, n=16, c=512, h=7, w=7, kernel=3, stride_hw=(1, 1), pad_hw=(1, 1), note="depthwise_stage5"),
    )),
    OperatorSpec("nn::matmul_row_major_fp32", ("matmul_row_major_fp32",), make_matmul_case, MATMUL_FP32_CONFIGS),
    OperatorSpec("nn::matmul_row_major_fp16", ("matmul_row_major_fp16",), make_matmul_case, MATMUL_FP16_CONFIGS),
    OperatorSpec("nn::batch_matmul_row_major_fp16", ("batch_matmul_row_major_fp16",), make_batch_matmul_case, BATCH_MATMUL_FP16_CONFIGS),
    OperatorSpec("math::reduce_sum_nchw_fp32", ("reduce_sum_nchw_fp32",), make_reduce_sum_case, REDUCE_SUM_FP32_CONFIGS),
    OperatorSpec("nn::avg_pool2d_nchw_fp32", ("avg_pool2d_nchw_fp32",), make_pool2d_case, AVG_POOL_FP32_CONFIGS),
    OperatorSpec("nn::max_pool2d_nchw_fp32", ("max_pool2d_nchw_fp32",), make_pool2d_case, MAX_POOL_FP32_CONFIGS),
    OperatorSpec("nn::adaptive_avg_pool2d_nchw_fp32", ("adaptive_avg_pool2d_nchw_fp32",), make_adaptive_avg_pool2d_case, (
        dict(dtype=torch.float32, n=8, c=128, h=28, w=28, out_h=7, out_w=7, note="globalish_pool"),
        dict(dtype=torch.float32, n=16, c=256, h=14, w=14, out_h=1, out_w=1, note="global_pool"),
        dict(dtype=torch.float32, n=8, c=64, h=56, w=56, out_h=1, out_w=1, note="stage1_global_pool"),
        dict(dtype=torch.float32, n=4, c=512, h=7, w=7, out_h=1, out_w=1, note="stage5_global_pool"),
    )),
    OperatorSpec("vision::resize_bilinear_nchw_fp32", ("resize_bilinear_nchw_fp32",), make_resize_case, (
        dict(mode="bilinear", dtype=torch.float32, n=4, c=64, h=56, w=56, out_h=112, out_w=112, note="upsample_bilinear"),
        dict(mode="bilinear", dtype=torch.float32, n=8, c=128, h=28, w=28, out_h=56, out_w=56, note="featuremap_x2"),
        dict(mode="bilinear", dtype=torch.float32, n=2, c=32, h=112, w=112, out_h=224, out_w=224, note="input_x2"),
        dict(mode="bilinear", dtype=torch.float32, n=16, c=256, h=14, w=14, out_h=28, out_w=28, note="stage4_x2"),
    )),
    OperatorSpec("vision::resize_nearest_nchw_fp32", ("resize_nearest_nchw_fp32",), make_resize_case, (
        dict(mode="nearest", dtype=torch.float32, n=4, c=64, h=56, w=56, out_h=112, out_w=112, note="upsample_nearest"),
        dict(mode="nearest", dtype=torch.float32, n=8, c=128, h=28, w=28, out_h=56, out_w=56, note="featuremap_x2"),
        dict(mode="nearest", dtype=torch.float32, n=2, c=32, h=112, w=112, out_h=224, out_w=224, note="input_x2"),
        dict(mode="nearest", dtype=torch.float32, n=16, c=256, h=14, w=14, out_h=28, out_w=28, note="stage4_x2"),
    )),
    OperatorSpec("nn::relu_nchw_fp32", ("relu_nchw_fp32",), make_unary_nchw_case, (
        dict(op_base="relu", dtype=torch.float32, n=16, c=128, h=28, w=28, note="relu_featuremap"),
        dict(op_base="relu", dtype=torch.float32, n=32, c=256, h=14, w=14, note="relu_stage4"),
        dict(op_base="relu", dtype=torch.float32, n=8, c=64, h=56, w=56, note="relu_stage2"),
        dict(op_base="relu", dtype=torch.float32, n=4, c=512, h=7, w=7, note="relu_stage5"),
    )),
    OperatorSpec("nn::silu_nchw_fp32", ("silu_nchw_fp32",), make_unary_nchw_case, (
        dict(op_base="silu", dtype=torch.float32, n=16, c=128, h=28, w=28, note="silu_featuremap"),
        dict(op_base="silu", dtype=torch.float32, n=32, c=256, h=14, w=14, note="silu_stage4"),
        dict(op_base="silu", dtype=torch.float32, n=8, c=64, h=56, w=56, note="silu_stage2"),
        dict(op_base="silu", dtype=torch.float32, n=4, c=512, h=7, w=7, note="silu_stage5"),
    )),
    OperatorSpec("nn::gelu_row_major_fp32", ("gelu_row_major_fp32",), make_gelu_case, (
        dict(dtype=torch.float32, m=4096, n=1024, note="gelu_ffn"),
        dict(dtype=torch.float32, m=2048, n=4096, note="gelu_expand"),
        dict(dtype=torch.float32, m=1024, n=3072, note="gelu_mid"),
        dict(dtype=torch.float32, m=512, n=8192, note="gelu_long_vector"),
    )),
    OperatorSpec("nn::softmax_row_major_fp32", ("softmax_row_major_fp32",), make_softmax_case, (
        dict(dtype=torch.float32, m=2048, n=1024, dim=1, note="softmax_logits"),
        dict(dtype=torch.float32, m=4096, n=512, dim=1, note="softmax_tall"),
        dict(dtype=torch.float32, m=512, n=4096, dim=1, note="softmax_wide"),
        dict(dtype=torch.float32, m=4096, n=128, dim=0, note="softmax_batch_dim"),
    )),
    OperatorSpec("nn::layernorm_row_major_fp32", ("layernorm_row_major_fp32",), make_layernorm_case, (
        dict(dtype=torch.float32, m=4096, n=1024, note="layernorm_ffn"),
        dict(dtype=torch.float32, m=2048, n=4096, note="layernorm_expand"),
        dict(dtype=torch.float32, m=1024, n=3072, note="layernorm_mid"),
        dict(dtype=torch.float32, m=512, n=8192, note="layernorm_long_vector"),
    )),
    OperatorSpec("math::add_nchw_fp32", ("add_nchw_fp32",), make_binary_nchw_case, (
        dict(op_base="add", dtype=torch.float32, n=32, c=256, h=56, w=56, note="residual_add"),
        dict(op_base="add", dtype=torch.float32, n=16, c=128, h=28, w=28, note="feature_add"),
        dict(op_base="add", dtype=torch.float32, n=8, c=64, h=112, w=112, note="large_map_add"),
        dict(op_base="add", dtype=torch.float32, n=4, c=512, h=7, w=7, note="late_add"),
    )),
    OperatorSpec("math::mul_nchw_fp32", ("mul_nchw_fp32",), make_binary_nchw_case, (
        dict(op_base="mul", dtype=torch.float32, n=32, c=256, h=56, w=56, note="elementwise_mul"),
        dict(op_base="mul", dtype=torch.float32, n=16, c=128, h=28, w=28, note="feature_mul"),
        dict(op_base="mul", dtype=torch.float32, n=8, c=64, h=112, w=112, note="large_map_mul"),
        dict(op_base="mul", dtype=torch.float32, n=4, c=512, h=7, w=7, note="late_mul"),
    )),
)

SPEC_BY_KEY = {spec.key: spec for spec in AVAILABLE_SPECS}
SPEC_BY_ALIAS = {alias: spec.key for spec in AVAILABLE_SPECS for alias in (spec.key, *spec.aliases)}


def split_requested_ops(raw_ops: list[str]) -> list[str]:
    tokens: list[str] = []
    for raw in raw_ops:
        for piece in raw.split(","):
            value = piece.strip()
            if value:
                tokens.append(value)
    return tokens


def resolve_operator_keys(raw_ops: list[str], *, allow_all: bool) -> list[str]:
    tokens = split_requested_ops(raw_ops)
    if allow_all:
        return [spec.key for spec in AVAILABLE_SPECS]
    if not tokens:
        raise ValueError("请通过 --op 指定要采样的算子，或使用 --all-ops")

    resolved: list[str] = []
    unknown: list[str] = []
    for token in tokens:
        key = SPEC_BY_ALIAS.get(token)
        if key is None:
            unknown.append(token)
            continue
        if key not in resolved:
            resolved.append(key)
    if unknown:
        raise ValueError(f"未知算子: {', '.join(unknown)}")
    return resolved


def plan_sample_ids_for_specs(
    op_keys: list[str],
    *,
    limit_per_op: int,
    existing_sample_ids_by_op: dict[str, set[int]] | None = None,
    top_up_to: int | None = None,
    rerun_existing: bool = False,
) -> dict[str, list[int]]:
    sample_ids_by_op: dict[str, list[int]] = {}
    for op_key in op_keys:
        spec = SPEC_BY_KEY[op_key]
        existing = existing_sample_ids_by_op.get(op_key, set()) if existing_sample_ids_by_op else set()
        if top_up_to is not None and not rerun_existing:
            desired_runs = max(0, top_up_to - len(existing))
        else:
            desired_runs = top_up_to if top_up_to is not None else limit_per_op
        desired_runs = max(0, min(desired_runs, len(spec.configs)))
        selected: list[int] = []
        for sample_id in range(1, len(spec.configs) + 1):
            if len(selected) >= desired_runs:
                break
            if not rerun_existing and sample_id in existing:
                continue
            selected.append(sample_id)
        sample_ids_by_op[op_key] = selected
    return sample_ids_by_op


def build_cases_for_sample_ids(
    device: torch.device,
    sample_ids_by_op: dict[str, list[int]],
) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for op_key, sample_ids in sample_ids_by_op.items():
        spec = SPEC_BY_KEY[op_key]
        for sample_id in sample_ids:
            if sample_id < 1 or sample_id > len(spec.configs):
                raise ValueError(f"sample_id 超出范围: {op_key}#{sample_id:02d}")
            cfg = spec.configs[sample_id - 1]
            op_name, params, run, output_shape, note = spec.builder(device, **cfg)
            if op_name != op_key:
                raise RuntimeError(f"算子注册与 builder 返回不一致: expected={op_key} got={op_name}")
            cases.append(
                BenchCase(
                    op_key=op_key,
                    sample_id=sample_id,
                    op_name=op_name,
                    params=params,
                    run=run,
                    output_shape=output_shape,
                    note=note or f"sample_{sample_id:02d}",
                )
            )
    return cases


def build_cases_for_specs(device: torch.device, op_keys: list[str], limit_per_op: int) -> list[BenchCase]:
    sample_ids_by_op = plan_sample_ids_for_specs(
        op_keys,
        limit_per_op=limit_per_op,
        rerun_existing=True,
    )
    return build_cases_for_sample_ids(device, sample_ids_by_op)


def print_available_ops() -> None:
    print("可用算子:")
    for spec in AVAILABLE_SPECS:
        aliases = ", ".join(spec.aliases)
        print(f"  {spec.key} | aliases=[{aliases}] | cases={len(spec.configs)}")
