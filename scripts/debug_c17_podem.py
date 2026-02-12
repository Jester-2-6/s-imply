#!/usr/bin/env python
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.util.io import parse_bench_file
from src.atpg.podem import podem, get_all_faults, initialize, print_pi
from src.util.struct import LogicValue

def debug_c17():
    circuit_path = 'data/bench/ISCAS85/c17.bench'
    print(f"Loading {circuit_path}")
    circuit, total_gates = parse_bench_file(circuit_path)
    
    all_faults = get_all_faults(circuit, total_gates)
    print(f"Total faults: {len(all_faults)}")
    
    for i, fault in enumerate(all_faults):
        initialize(circuit, total_gates)
        fault_desc = f"Gate {fault.gate_id} SA{0 if fault.value == LogicValue.D else 1}"
        
        try:
            result = podem(circuit, fault, total_gates)
            if result:
                pattern = print_pi(circuit, total_gates)
                print(f"Fault {fault_desc}: Pattern = {pattern}")
                if '3' in pattern or '4' in pattern:
                    print(f"  --> FAILURE: Pattern contains D/D' (3/4)!")
        except Exception as e:
            print(f"Fault {fault_desc}: Error {e}")

if __name__ == "__main__":
    debug_c17()
