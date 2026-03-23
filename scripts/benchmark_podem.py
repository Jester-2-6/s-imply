"""
Benchmark: Classic PODEM vs AI-PODEM on a single fault.

Each run parses a fresh copy of the circuit to avoid state pollution across repeats.

Usage:
    python -m scripts.benchmark_podem \\
        --bench data/bench/ISCAS85/c432.bench \\
        --model checkpoints/reconv_max_occupancy/best_model.pth \\
        --gate 259 --sa 1 --repeats 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from typing import TextIO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.ai_podem import (
    AiPodemConfig,
    HierarchicalReconvSolver,
    ModelPairPredictor,
    ai_podem,
)
from src.atpg.podem import get_statistics, podem, reset_statistics
from src.util.io import parse_bench_file
from src.util.struct import Fault, LogicValue


class _TeeStream:
    """Mirror writes to multiple file-like streams."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def _maybe_log_output(log_path: str | None):
    """Mirror stdout and stderr to a log file when requested."""
    if not log_path:
        yield
        return

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee_stdout = _TeeStream(sys.stdout, log_file)
        tee_stderr = _TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            yield


@contextlib.contextmanager
def _silence():
    """Suppress all stdout and stderr inside the block."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _maybe_silence(enabled: bool):
    """Return a context manager that suppresses output only when requested."""
    return _silence() if enabled else contextlib.nullcontext()


def _fresh_circuit(bench_path: str):
    """Return a freshly parsed (or deep-copied) circuit each call to avoid state pollution."""
    return parse_bench_file(bench_path)


def run_classic_podem(bench_path: str, fault: Fault) -> tuple[bool, float, int]:
    """Classic PODEM on a fresh circuit copy, timed, silent."""
    import src.atpg.podem as podem_mod

    circuit, total_gates = _fresh_circuit(bench_path)
    # Force full re-init (SCOAP, topological order) for this circuit copy
    podem_mod.scoap_calculated = False
    reset_statistics()

    t0 = time.perf_counter()
    with _silence():
        result = podem(circuit, fault, total_gates)
    elapsed = time.perf_counter() - t0
    stats = get_statistics()
    return bool(result), elapsed, stats["backtrack_count"]


def run_ai_podem(
    bench_path: str,
    fault: Fault,
    circuit_path: str,
    device: str,
    model_path: str,
    pre_loaded_model=None,
    verbose: bool = False,
) -> tuple[bool, float, int]:
    """AI-PODEM (AI activation, classic PODEM propagation) on a fresh circuit, timed, silent."""
    circuit, total_gates = _fresh_circuit(bench_path)

    config = AiPodemConfig(
        model_path=model_path,
        device=device,
        enable_ai_activation=True,
        enable_ai_propagation=False,
        verbose=verbose,
    )

    with _maybe_silence(not verbose):
        predictor = ModelPairPredictor(
            circuit,
            circuit_path,
            config,
            pre_loaded_model=pre_loaded_model,
        )
    solver = HierarchicalReconvSolver(
        circuit,
        predictor,
        verbose=verbose,
        circuit_path=circuit_path,
    )
    reset_statistics()

    t0 = time.perf_counter()
    with _maybe_silence(not verbose):
        result = ai_podem(
            circuit,
            fault,
            total_gates,
            predictor=predictor,
            solver=solver,
            enable_ai_activation=True,
            enable_ai_propagation=False,
            verbose=verbose,
            no_fallback=True,
        )
    elapsed = time.perf_counter() - t0
    stats = get_statistics()
    if not result:
        print(
            "\n[AI-PODEM] FAILED: AI could not solve fault "
            f"(gate={fault.gate_id}, val={fault.value}). No fallback."
        )
        sys.exit(1)
    return bool(result), elapsed, stats["backtrack_count"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Classic PODEM vs AI-PODEM")
    parser.add_argument(
        "--bench",
        type=str,
        default="data/bench/ISCAS85/c432.bench",
        help="Path to .bench file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/reconv_max_occupancy/best_model.pth",
        help="Path to AI model checkpoint",
    )
    parser.add_argument("--gate", type=int, default=329, help="Fault gate ID")
    parser.add_argument("--sa", type=int, default=1, choices=[0, 1], help="Stuck-at value (0 or 1)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed repeats per method")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print AI model queries, predictions, and solver decisions",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional file path to capture benchmark stdout and stderr",
    )
    args = parser.parse_args()

    device = "cuda"

    with _maybe_log_output(args.log_file):
        # SA0 → LogicValue.D (good machine output is 1, D)
        # SA1 → LogicValue.DB (good machine output is 0, D̄)
        fault_val = LogicValue.DB if args.sa == 1 else LogicValue.D
        fault = Fault(args.gate, fault_val)

        print(f"Circuit  : {args.bench}")
        print(f"Fault    : Gate {args.gate}  stuck-at-{args.sa}  ({fault_val})")
        print(f"Device   : {device}")
        print(f"Repeats  : {args.repeats}")
        print(f"Verbose  : {args.verbose}")
        if args.log_file:
            print(f"Log file : {args.log_file}")
        print()

        # ------------------------------------------------------------------ #
        #  Pre-load model once, avoids redundant torch.load + embedding
        #  extraction on every repeat.
        # ------------------------------------------------------------------ #
        print("Loading model...", end=" ", flush=True)
        _t_load = time.perf_counter()
        _cfg_probe = AiPodemConfig(model_path=args.model, device=device, verbose=args.verbose)
        _probe_circuit, _ = _fresh_circuit(args.bench)
        with _maybe_silence(not args.verbose):
            _probe_predictor = ModelPairPredictor(_probe_circuit, args.bench, _cfg_probe)
        pre_loaded_model = _probe_predictor.model
        print(f"done ({time.perf_counter() - _t_load:.2f}s)")

        # ------------------------------------------------------------------ #
        #  Timed runs  (each run gets a fresh circuit to eliminate state bugs)
        # ------------------------------------------------------------------ #
        classic_times: list[float] = []
        classic_ok: list[bool] = []
        classic_bt: list[int] = []
        ai_times: list[float] = []
        ai_ok: list[bool] = []
        ai_bt: list[int] = []

        for r in range(args.repeats):
            ok_c, t_c, bt_c = run_classic_podem(args.bench, fault)
            classic_times.append(t_c)
            classic_ok.append(ok_c)
            classic_bt.append(bt_c)

            ok_a, t_a, bt_a = run_ai_podem(
                args.bench,
                fault,
                args.bench,
                device,
                args.model,
                pre_loaded_model=pre_loaded_model,
                verbose=args.verbose,
            )
            ai_times.append(t_a)
            ai_ok.append(ok_a)
            ai_bt.append(bt_a)

            print(
                f"  Run {r + 1}/{args.repeats}"
                f"  |  Classic PODEM : {'OK  ' if ok_c else 'FAIL'}  {t_c:.4f}s  BT: {bt_c}"
                f"  |  AI-PODEM : {'OK  ' if ok_a else 'FAIL'}  {t_a:.4f}s  BT: {bt_a}"
            )

        # ------------------------------------------------------------------ #
        #  Summary
        # ------------------------------------------------------------------ #
        avg_c = sum(classic_times) / len(classic_times)
        avg_a = sum(ai_times) / len(ai_times)
        avg_bt_c = sum(classic_bt) / len(classic_bt)
        avg_bt_a = sum(ai_bt) / len(ai_bt)
        speedup = avg_c / avg_a if avg_a > 0 else float("inf")

        c_pass = sum(classic_ok)
        a_pass = sum(ai_ok)

        print()
        print("=" * 72)
        print(
            f"{'Method':<20} {'Pass':>6} {'Avg time (s)':>14} "
            f"{'Avg BT':>10}  {'vs Classic':>10}"
        )
        print("-" * 72)
        print(
            f"{'Classic PODEM':<20} {c_pass}/{args.repeats:>3} {avg_c:>14.4f} "
            f"{avg_bt_c:>10.1f}  {'1.00×':>10}"
        )
        print(
            f"{'AI-PODEM':<20} {a_pass}/{args.repeats:>3} {avg_a:>14.4f} "
            f"{avg_bt_a:>10.1f}  {speedup:>9.2f}×"
        )
        print("=" * 72)


if __name__ == "__main__":
    main()
