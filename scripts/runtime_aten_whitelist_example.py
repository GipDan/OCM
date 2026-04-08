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
        description="Run a small CUDA workload under RuntimeOCMMode."
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
        default=3,
        help="How many iterations of the sample workload to execute.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not append runtime samples to the DB.",
    )
    parser.add_argument(
        "--auto-fit",
        action="store_true",
        help="Periodically retrain a model after enough runtime samples exist.",
    )
    return parser


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This sample requires a CUDA-capable pytorch environment.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device.index)
    return device


def run_sample_workload(iters: int, device: torch.device) -> float:
    add_lhs = torch.randn(32, 64, 56, 56, device=device, dtype=torch.float32)
    add_rhs = torch.randn_like(add_lhs)
    mul_rhs = torch.randn_like(add_lhs)

    matmul_lhs = torch.randn(1024, 512, device=device, dtype=torch.float32)
    matmul_rhs = torch.randn(512, 1024, device=device, dtype=torch.float32)

    final_value = 0.0
    for _ in range(iters):
        y = add_lhs + add_rhs
        z = y * mul_rhs
        mm = matmul_lhs @ matmul_rhs
        final_value = float((z.reshape(-1)[0] + mm[0, 0]).item())
        matmul_lhs = mm[:, :512].contiguous()
    torch.cuda.synchronize(device)
    return final_value


def main() -> None:
    args = build_parser().parse_args()
    device = ensure_cuda()
    config = RuntimeOCMConfig.from_env(db_path=args.db_path)
    config = RuntimeOCMConfig(
        db_path=config.db_path,
        enabled=config.enabled,
        write_records=not args.no_write if config.write_records else False,
        predict_before_run=config.predict_before_run,
        use_exact_match=config.use_exact_match,
        use_stats_fallback=config.use_stats_fallback,
        auto_fit=args.auto_fit or config.auto_fit,
        min_samples_for_fit=config.min_samples_for_fit,
        retrain_every=config.retrain_every,
        whitelist=config.whitelist,
    )

    run_sample_workload(1, device)
    with RuntimeOCMMode(config) as mode:
        final_value = run_sample_workload(args.iters, device)

    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"db_path={args.db_path}")
    print(f"iters={args.iters}")
    print(f"final_value={final_value:.6f}")
    print("runtime_events=")
    print(json.dumps(mode.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
