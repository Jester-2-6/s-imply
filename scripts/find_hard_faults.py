#!/usr/bin/env python
"""
Find hard faults in c432 that vanilla PODEM cannot solve quickly.
"""

import os
import signal
import sys
import time
from contextlib import contextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.podem import get_all_faults, initialize, podem
from src.util.io import parse_bench_file
from src.util.struct import LogicValue


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to enforce a time limit."""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def find_hard_faults(circuit_path, timeout_seconds=10):
    """Find faults that take longer than timeout_seconds to solve."""

    print(f"Loading circuit from {circuit_path}")
    circuit, total_gates = parse_bench_file(circuit_path)

    print("Getting all faults...")
    all_faults = get_all_faults(circuit, total_gates)
    print(f"Total faults: {len(all_faults)}")

    hard_faults = []
    solved_faults = []
    unsolvable_faults = []

    for i, fault in enumerate(all_faults):
        # Reset circuit
        initialize(circuit, total_gates)

        fault_desc = f"Gate {fault.gate_id} SA{0 if fault.value == LogicValue.D else 1}"

        try:
            start_time = time.time()
            with time_limit(timeout_seconds):
                result = podem(circuit, fault, total_gates)
                elapsed = time.time() - start_time

                if result:
                    solved_faults.append((fault, elapsed))
                    status = f"✓ Solved in {elapsed:.2f}s"
                else:
                    unsolvable_faults.append(fault)
                    status = f"✗ Unsolvable (checked in {elapsed:.2f}s)"

        except TimeoutException:
            elapsed = time.time() - start_time
            hard_faults.append(fault)
            status = f"⏱ TIMEOUT after {timeout_seconds}s"
        except Exception as e:
            status = f"⚠ Error: {e}"

        print(f"[{i+1}/{len(all_faults)}] {fault_desc}: {status}")

    return hard_faults, solved_faults, unsolvable_faults


def main():
    circuit_path = "data/bench/ISCAS85/c432.bench"
    timeout_seconds = 10

    print("=" * 80)
    print(f"Finding Hard Faults in c432 (timeout: {timeout_seconds}s)")
    print("=" * 80)
    print()

    hard_faults, solved_faults, unsolvable_faults = find_hard_faults(circuit_path, timeout_seconds)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        "Total faults tested: " f"{len(hard_faults) + len(solved_faults) + len(unsolvable_faults)}"
    )
    print(f"Solved quickly (< {timeout_seconds}s): {len(solved_faults)}")
    print(f"Unsolvable: {len(unsolvable_faults)}")
    print(f"Hard faults (timeout): {len(hard_faults)}")

    if hard_faults:
        print(f"\n{'='*80}")
        print("HARD FAULTS (did not converge within timeout):")
        print("=" * 80)
        for fault in hard_faults:
            fault_type = "SA0" if fault.value == LogicValue.D else "SA1"
            print(f"  Gate {fault.gate_id} {fault_type}")

    if solved_faults:
        # Show slowest solved faults
        solved_faults.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{'='*80}")
        print("TOP 10 SLOWEST SOLVED FAULTS:")
        print("=" * 80)
        for fault, elapsed in solved_faults[:10]:
            fault_type = "SA0" if fault.value == LogicValue.D else "SA1"
            print(f"  Gate {fault.gate_id} {fault_type}: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
