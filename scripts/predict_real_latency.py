#!/usr/bin/env python3
"""Thin CLI wrapper for single-operator latency prediction."""

from real_bench.model_cli import run_predict_cli


if __name__ == "__main__":
    raise SystemExit(run_predict_cli())
