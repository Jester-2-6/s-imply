#!/usr/bin/env python
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.util.io import parse_bench_file
from src.atpg.podem import podem, get_all_faults, initialize, print_pi, SUCCESS, UNTESTABLE, TIMEOUT, BACKTRACK_LIMIT
from src.util.struct import LogicValue

def generate_patterns():
    circuit_path = 'data/bench/ISCAS85/c432.bench'
    print(f"Loading circuit from {circuit_path}")
    circuit, total_gates = parse_bench_file(circuit_path)
    
    print("Getting all faults...")
    all_faults = get_all_faults(circuit, total_gates)
    print(f"Total faults: {len(all_faults)}")
    
    testable_count = 0
    untestable_count = 0
    timeout_count = 0
    backtrack_limit_count = 0
    
    output_file = "c432_patterns.log"
    
    start_time = time.time()
    
    with open(output_file, "w") as f:
        f.write("================================================================================\n")
        f.write("Generating Test Patterns for C432\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write("================================================================================\n\n")
        
        for i, fault in enumerate(all_faults):
            # Reset circuit
            initialize(circuit, total_gates)
            
            fault_desc = f"Gate {fault.gate_id} SA{0 if fault.value == LogicValue.D else 1}"
            
            # Run PODEM with 10s timeout
            try:
                result = podem(circuit, fault, total_gates, timeout=10.0)
                
                if result == SUCCESS:
                    pattern = print_pi(circuit, total_gates)
                    status_line = f"[{i+1}/{len(all_faults)}] {fault_desc}: ✓ Pattern: {pattern}"
                    print(status_line)
                    f.write(status_line + "\n")
                    testable_count += 1
                elif result == TIMEOUT:
                    status_line = f"[{i+1}/{len(all_faults)}] {fault_desc}: ⏰ TIMEOUT"
                    print(status_line)
                    f.write(status_line + "\n")
                    timeout_count += 1
                elif result == BACKTRACK_LIMIT:
                    status_line = f"[{i+1}/{len(all_faults)}] {fault_desc}: 🚫 BACKTRACK LIMIT"
                    print(status_line)
                    f.write(status_line + "\n")
                    backtrack_limit_count += 1
                else: # UNTESTABLE or FAILURE
                    status_line = f"[{i+1}/{len(all_faults)}] {fault_desc}: ✗ Untestable"
                    print(status_line)
                    f.write(status_line + "\n")
                    untestable_count += 1
                    
            except Exception as e:
                status_line = f"[{i+1}/{len(all_faults)}] {fault_desc}: ⚠ Error {e}"
                print(status_line)
                f.write(status_line + "\n")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Summary
        summary = (
            "\n================================================================================\n"
            "SUMMARY\n"
            "================================================================================\n"
            f"Total faults: {len(all_faults)}\n"
            f"Testable: {testable_count}\n"
            f"Untestable: {untestable_count}\n"
            f"Timeouts: {timeout_count}\n"
            f"Backtrack Limits: {backtrack_limit_count}\n"
            f"Total Time: {duration:.2f}s\n"
            f"Fault coverage: {(testable_count / len(all_faults) * 100):.2f}%\n"
        )
        print(summary)
        f.write(summary)
        
    print(f"\nPatterns written to: {output_file}")

if __name__ == "__main__":
    generate_patterns()
