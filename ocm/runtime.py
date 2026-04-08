"""Runtime OCM integration for a whitelist of aten ops."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ocm.database import (
    DEFAULT_DB_PATH,
    count_records_for_group,
    get_connection,
    init_db,
    insert_record,
)
from ocm.features import derive_feature_order_key_from_params
from ocm.inference import predict_latency_details
from ocm.train import fit_and_store_model


def normalize_device_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def dtype_name(dtype: torch.dtype) -> str | None:
    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    return mapping.get(dtype)


def tensor_stride_list(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.stride()]


@dataclass(frozen=True)
class RuntimeOCMConfig:
    db_path: str | Path = DEFAULT_DB_PATH
    enabled: bool = True
    write_records: bool = True
    predict_before_run: bool = True
    use_exact_match: bool = True
    use_stats_fallback: bool = True
    auto_fit: bool = False
    min_samples_for_fit: int = 20
    retrain_every: int = 10
    whitelist: tuple[str, ...] = (
        "aten.add.Tensor",
        "aten.mul.Tensor",
        "aten.mm.default",
    )

    @classmethod
    def from_env(cls, db_path: str | Path = DEFAULT_DB_PATH) -> "RuntimeOCMConfig":
        import os

        def flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() not in {"0", "false", "off", "no"}

        def integer(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return default
            return int(raw)

        return cls(
            db_path=db_path,
            enabled=flag("OCM_RUNTIME_ENABLED", True),
            write_records=flag("OCM_RUNTIME_WRITE", True),
            predict_before_run=flag("OCM_RUNTIME_PREDICT", True),
            use_exact_match=flag("OCM_RUNTIME_USE_EXACT", True),
            use_stats_fallback=flag("OCM_RUNTIME_USE_STATS", True),
            auto_fit=flag("OCM_RUNTIME_AUTO_FIT", False),
            min_samples_for_fit=integer("OCM_RUNTIME_MIN_SAMPLES", 20),
            retrain_every=integer("OCM_RUNTIME_RETRAIN_EVERY", 10),
        )


@dataclass(frozen=True)
class ResolvedOCMOp:
    aten_name: str
    op_name: str
    device_name: str
    device_index: int
    params: dict[str, Any]


@dataclass
class RuntimeOCMEvent:
    index: int
    aten_name: str
    op_name: str
    device: str
    params: dict[str, Any]
    actual_latency_ms: float
    predicted_latency_ms: float | None = None
    prediction_source: str | None = None
    predicted_feature_order_key: str | None = None
    record_id: int | None = None
    stored_feature_order_key: str | None = None
    fit_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_supported_dtype(dtype: torch.dtype) -> bool:
    return dtype_name(dtype) is not None


def _device_name_for_tensor(tensor: torch.Tensor) -> str:
    if not tensor.is_cuda:
        raise ValueError("runtime OCM 目前只支持 CUDA tensor")
    return normalize_device_name(torch.cuda.get_device_name(tensor.device))


def _resolve_add_like(
    aten_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    base_name: str,
) -> ResolvedOCMOp | None:
    del kwargs
    if len(args) < 2:
        return None
    lhs, rhs = args[0], args[1]
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        return None
    if not lhs.is_cuda or not rhs.is_cuda:
        return None
    if lhs.device != rhs.device or lhs.dtype != rhs.dtype or lhs.shape != rhs.shape:
        return None
    if lhs.dim() != 4 or not _is_supported_dtype(lhs.dtype):
        return None

    suffix = dtype_name(lhs.dtype)
    if suffix is None:
        return None
    params = {
        "N": int(lhs.shape[0]),
        "C": int(lhs.shape[1]),
        "H": int(lhs.shape[2]),
        "W": int(lhs.shape[3]),
        "is_contiguous": bool(lhs.is_contiguous()),
        "memory_stride": tensor_stride_list(lhs),
    }
    return ResolvedOCMOp(
        aten_name=aten_name,
        op_name=f"math::{base_name}_nchw_{suffix}",
        device_name=_device_name_for_tensor(lhs),
        device_index=int(lhs.device.index),
        params=params,
    )


def _resolve_mm(
    aten_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ResolvedOCMOp | None:
    del kwargs
    if len(args) < 2:
        return None
    lhs, rhs = args[0], args[1]
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        return None
    if not lhs.is_cuda or not rhs.is_cuda:
        return None
    if lhs.device != rhs.device or lhs.dtype != rhs.dtype:
        return None
    if lhs.dim() != 2 or rhs.dim() != 2 or lhs.shape[1] != rhs.shape[0]:
        return None
    if not _is_supported_dtype(lhs.dtype):
        return None

    suffix = dtype_name(lhs.dtype)
    if suffix is None:
        return None
    params = {
        "M": int(lhs.shape[0]),
        "N": int(rhs.shape[1]),
        "K": int(lhs.shape[1]),
        "is_contiguous": bool(lhs.is_contiguous()),
        "memory_stride": tensor_stride_list(lhs),
    }
    return ResolvedOCMOp(
        aten_name=aten_name,
        op_name=f"nn::matmul_row_major_{suffix}",
        device_name=_device_name_for_tensor(lhs),
        device_index=int(lhs.device.index),
        params=params,
    )


def resolve_ocm_op(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> ResolvedOCMOp | None:
    kwargs = kwargs or {}
    aten_name = str(func)
    if aten_name == "aten.add.Tensor":
        alpha = kwargs.get("alpha", args[2] if len(args) >= 3 else 1)
        if alpha != 1:
            return None
        return _resolve_add_like(aten_name, args, kwargs, base_name="add")
    if aten_name == "aten.mul.Tensor":
        return _resolve_add_like(aten_name, args, kwargs, base_name="mul")
    if aten_name == "aten.mm.default":
        return _resolve_mm(aten_name, args, kwargs)
    return None


class RuntimeOCMMode(TorchDispatchMode):
    """Intercept a small whitelist of aten ops and sync them to OCM."""

    def __init__(self, config: RuntimeOCMConfig | None = None) -> None:
        super().__init__()
        self.config = config or RuntimeOCMConfig.from_env()
        self.conn = get_connection(self.config.db_path)
        init_db(self.conn)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout = 5000")
        except sqlite3.DatabaseError:
            pass
        self.events: list[RuntimeOCMEvent] = []

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RuntimeOCMMode":
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()

    def summary(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def _maybe_predict(self, resolved: ResolvedOCMOp) -> dict[str, Any] | None:
        if not self.config.predict_before_run:
            return None
        return predict_latency_details(
            self.conn,
            resolved.op_name,
            resolved.device_name,
            resolved.params,
            use_exact_record_if_match=self.config.use_exact_match,
            use_stats_fallback=self.config.use_stats_fallback,
        )

    def _maybe_store(
        self,
        resolved: ResolvedOCMOp,
        latency_ms: float,
    ) -> tuple[int | None, str | None]:
        if not self.config.write_records:
            return None, None
        return insert_record(
            self.conn,
            resolved.op_name,
            resolved.device_name,
            resolved.params,
            latency_ms,
        )

    def _maybe_fit(self, resolved: ResolvedOCMOp, feature_order_key: str | None) -> str | None:
        if not self.config.auto_fit or feature_order_key is None:
            return None
        count = count_records_for_group(
            self.conn,
            resolved.op_name,
            resolved.device_name,
            feature_order_key,
        )
        if count < self.config.min_samples_for_fit:
            return None
        if self.config.retrain_every > 0 and count % self.config.retrain_every != 0:
            return None
        ok, msg = fit_and_store_model(
            self.conn,
            resolved.op_name,
            resolved.device_name,
            min_samples=self.config.min_samples_for_fit,
            feature_order_key=feature_order_key,
        )
        prefix = "fit_ok" if ok else "fit_skip"
        return f"{prefix}: {msg}"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        kwargs = kwargs or {}
        if not self.config.enabled:
            return func(*args, **kwargs)
        if str(func) not in self.config.whitelist:
            return func(*args, **kwargs)

        resolved = resolve_ocm_op(func, args, kwargs)
        if resolved is None:
            return func(*args, **kwargs)

        prediction = self._maybe_predict(resolved)
        if resolved.device_index >= 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(resolved.device_index)
            start.record(stream)
            result = func(*args, **kwargs)
            end.record(stream)
            end.synchronize()
            latency_ms = float(start.elapsed_time(end))
        else:
            started = perf_counter()
            result = func(*args, **kwargs)
            latency_ms = (perf_counter() - started) * 1000.0

        record_id, stored_key = self._maybe_store(resolved, latency_ms)
        if stored_key is None:
            stored_key = derive_feature_order_key_from_params(resolved.params)
        fit_message = self._maybe_fit(resolved, stored_key)
        event = RuntimeOCMEvent(
            index=len(self.events),
            aten_name=resolved.aten_name,
            op_name=resolved.op_name,
            device=resolved.device_name,
            params=resolved.params,
            actual_latency_ms=round(latency_ms, 6),
            predicted_latency_ms=(
                round(float(prediction["predicted_latency_ms"]), 6)
                if prediction is not None
                else None
            ),
            prediction_source=prediction["source"] if prediction is not None else None,
            predicted_feature_order_key=(
                str(prediction.get("feature_order_key"))
                if prediction is not None and prediction.get("feature_order_key") is not None
                else None
            ),
            record_id=record_id,
            stored_feature_order_key=stored_key,
            fit_message=fit_message,
        )
        self.events.append(event)
        return result


__all__ = [
    "ResolvedOCMOp",
    "RuntimeOCMConfig",
    "RuntimeOCMEvent",
    "RuntimeOCMMode",
    "resolve_ocm_op",
]
