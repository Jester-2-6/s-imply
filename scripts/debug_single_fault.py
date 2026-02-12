import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.atpg.podem as index_podem
from src.util.io import parse_bench_file
from src.atpg.podem import podem, initialize, print_pi
from src.util.struct import Fault, LogicValue

def debug_fault():
    # Enable verbose
    index_podem.VERBOSE = True
    index_podem.TRACE_DECISIONS = False
    
    circuit_path = 'data/bench/ISCAS85/c432.bench'
    print(f"Loading {circuit_path}")
    circuit, total_gates = parse_bench_file(circuit_path)
    
    # Fault: Gate 8 SA0 (needs 1) (LogicValue.D: Good=1, Faulty=0)
    # Gate 8 is input.
    fault = Fault(8, LogicValue.D)
    
    print(f"Testing Fault Gate {fault.gate_id} SA0")
    initialize(circuit, total_gates)
    
    result = podem(circuit, fault, total_gates)
    print(f"Result: {result}")
    
    if result:
        print("Pattern:", print_pi(circuit, total_gates))

if __name__ == "__main__":
    debug_fault()
