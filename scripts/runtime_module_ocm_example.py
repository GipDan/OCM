from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocm.runtime import RuntimeOCMConfig, RuntimeOCMMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a more complex nn.Module under RuntimeOCMMode."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=ROOT / "data" / "ocm.sqlite3",
        help="Path to the OCM SQLite database.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=2,
        help="How many timed forward passes to execute.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for the image-like activation tensor.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Channel size for the module input/output.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=56,
        help="Spatial height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=56,
        help="Spatial width.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=1024,
        help="Hidden width for the matmul core.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not append runtime samples to the DB.",
    )
    parser.add_argument(
        "--auto-fit",
        action="store_true",
        help="Periodically retrain when runtime samples cross the configured threshold.",
    )
    return parser


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This sample requires CUDA.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device.index)
    return device


class ResidualRuntimeBlock(torch.nn.Module):
    def __init__(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        hidden: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.channels = channels
        self.height = height
        self.width = width
        self.hidden = hidden
        self.patch_hw = 4

        self.residual_bias = torch.nn.Parameter(
            torch.randn(1, channels, height, width, device=device, dtype=torch.float32) * 0.05
        )
        self.gate = torch.nn.Parameter(
            torch.randn(1, channels, height, width, device=device, dtype=torch.float32) * 0.02
        )

        self.weight_in = torch.nn.Parameter(
            torch.randn(channels, hidden, device=device, dtype=torch.float32) * 0.01
        )
        self.weight_out = torch.nn.Parameter(
            torch.randn(hidden, channels, device=device, dtype=torch.float32) * 0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.residual_bias.expand_as(x)
        gate = self.gate.expand_as(x)

        y = x + bias
        y = y * gate

        patch = y[:, :, : self.patch_hw, : self.patch_hw]
        flat = patch.permute(0, 2, 3, 1).reshape(-1, self.channels)
        core = flat @ self.weight_in
        core = core @ self.weight_out

        restored_patch = (
            core.reshape(y.shape[0], self.patch_hw, self.patch_hw, self.channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        restored = x.clone()
        restored[:, :, : self.patch_hw, : self.patch_hw] = restored_patch
        out = restored + x
        out = out * gate
        return out


def make_config(base_config: RuntimeOCMConfig, args: argparse.Namespace) -> RuntimeOCMConfig:
    return RuntimeOCMConfig(
        db_path=base_config.db_path,
        enabled=base_config.enabled,
        write_records=not args.no_write if base_config.write_records else False,
        predict_before_run=base_config.predict_before_run,
        use_exact_match=base_config.use_exact_match,
        use_stats_fallback=base_config.use_stats_fallback,
        auto_fit=args.auto_fit or base_config.auto_fit,
        min_samples_for_fit=base_config.min_samples_for_fit,
        retrain_every=base_config.retrain_every,
        whitelist=base_config.whitelist,
    )


def summarize_events(events: list[dict[str, object]]) -> dict[str, object]:
    actual_total = 0.0
    predicted_total = 0.0
    predicted_count = 0
    source_counts: dict[str, int] = {}

    for event in events:
        actual_total += float(event["actual_latency_ms"])
        predicted = event.get("predicted_latency_ms")
        if predicted is not None:
            predicted_total += float(predicted)
            predicted_count += 1
        source = str(event.get("prediction_source") or "none")
        source_counts[source] = source_counts.get(source, 0) + 1

    return {
        "event_count": len(events),
        "actual_total_ms": round(actual_total, 6),
        "predicted_total_ms": round(predicted_total, 6),
        "predicted_event_count": predicted_count,
        "prediction_sources": source_counts,
    }


def run_forward(
    module: ResidualRuntimeBlock,
    x: torch.Tensor,
    iters: int,
) -> float:
    out = x
    for _ in range(iters):
        out = module(out)
    torch.cuda.synchronize(x.device)
    return float(out.mean().item())


def main() -> None:
    args = build_parser().parse_args()
    device = ensure_cuda()
    config = make_config(RuntimeOCMConfig.from_env(db_path=args.db_path), args)

    module = ResidualRuntimeBlock(
        batch=args.batch,
        channels=args.channels,
        height=args.height,
        width=args.width,
        hidden=args.hidden,
        device=device,
    ).eval()

    x = torch.randn(
        args.batch,
        args.channels,
        args.height,
        args.width,
        device=device,
        dtype=torch.float32,
    )

    run_forward(module, x, 1)
    with RuntimeOCMMode(config) as mode:
        final_mean = run_forward(module, x, args.iters)

    events = mode.summary()
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"db_path={args.db_path}")
    print(f"iters={args.iters}")
    print(f"module=ResidualRuntimeBlock")
    print(f"final_mean={final_mean:.6f}")
    print("summary=")
    print(json.dumps(summarize_events(events), ensure_ascii=False, indent=2))
    print("runtime_events=")
    print(json.dumps(events, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
