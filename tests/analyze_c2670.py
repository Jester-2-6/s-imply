import os
import sys
import random
from typing import List, Dict, Any
from src.util.struct import Gate, GateType, LogicValue
from src.util.io import parse_bench_file
from src.atpg.reconv_podem import check_path_pair_consistency, find_all_reconv_pairs, PathConsistencySolver

def verify_assignment(circuit: List[Gate], assignment: Dict[int, LogicValue], reconv_node: int, target_val: LogicValue) -> bool:
    """Verify if the assignment produces the target value at reconv_node."""
    # We need to simulate the circuit with the assignment.
    # Since we are checking LOCAL consistency, we only care if the assigned values are consistent with each other
    # and lead to the target.
    # However, `logic_sim` simulates the whole circuit.
    # If we set the values in `circuit` and run `logic_sim`, it will overwrite them unless they are PIs.
    # But we want to check if the assignment is *internally* consistent.
    # i.e. for every gate in assignment, does its computed value match the assigned value?
    
    # Let's check consistency of the assignment itself.
    for gate_idx, val in assignment.items():
        gate = circuit[gate_idx]
        if not gate.fin: # Leaf/Input
            continue
            
        # Get input values from assignment (or default to XD if not in assignment)
        # Note: The solver should have assigned all necessary inputs for the path gates.
        # Side inputs might be assigned.
        
        input_vals = []
        for fin in gate.fin:
            if fin in assignment:
                input_vals.append(assignment[fin])
            else:
                # If not in assignment, it must be irrelevant? 
                # Or maybe it's a side input that wasn't assigned?
                # If it wasn't assigned, it means it's a side input that can be anything?
                # No, solver assigns side inputs if they matter.
                # If it's not assigned, maybe it's XD?
                input_vals.append(LogicValue.XD)
        
        # Compute expected output
        # We need a helper to compute gate logic.
        # We can reuse PathConsistencySolver._compute_gate if we make it static or public.
        # Or just re-implement simple logic here.
        computed = compute_gate_logic(gate.type, input_vals)
        
        if computed != val and computed != LogicValue.XD:
            # If computed is XD, it means some inputs are unknown, so we can't verify this gate fully,
            # but it doesn't contradict.
            # If computed is 0/1 and differs from val, it's a conflict.
            # However, if the solver says "Possible", it should have assigned enough inputs to make it non-XD.
            return False

    # Also check if R matches target
    if assignment.get(reconv_node) != target_val:
        return False
        
    return True

def compute_gate_logic(gtype: int, inputs: List[LogicValue]) -> LogicValue:
    if gtype == GateType.AND:
        return LogicValue.ONE if all(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO if any(i == LogicValue.ZERO for i in inputs) else LogicValue.XD
    elif gtype == GateType.NAND:
        return LogicValue.ZERO if all(i == LogicValue.ONE for i in inputs) else LogicValue.ONE if any(i == LogicValue.ZERO for i in inputs) else LogicValue.XD
    elif gtype == GateType.OR:
        return LogicValue.ONE if any(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO if all(i == LogicValue.ZERO for i in inputs) else LogicValue.XD
    elif gtype == GateType.NOR:
        return LogicValue.ZERO if any(i == LogicValue.ONE for i in inputs) else LogicValue.ONE if all(i == LogicValue.ZERO for i in inputs) else LogicValue.XD
    elif gtype == GateType.XOR:
        if any(i == LogicValue.XD for i in inputs): return LogicValue.XD
        ones = sum(1 for i in inputs if i == LogicValue.ONE)
        return LogicValue.ONE if ones % 2 == 1 else LogicValue.ZERO
    elif gtype == GateType.XNOR:
        if any(i == LogicValue.XD for i in inputs): return LogicValue.XD
        ones = sum(1 for i in inputs if i == LogicValue.ONE)
        return LogicValue.ONE if ones % 2 == 0 else LogicValue.ZERO
    elif gtype == GateType.BUFF:
        return inputs[0]
    elif gtype == GateType.NOT:
        return LogicValue.ZERO if inputs[0] == LogicValue.ONE else LogicValue.ONE if inputs[0] == LogicValue.ZERO else LogicValue.XD
    return LogicValue.XD

def analyze_c2670():
    bench_path = "data/bench/ISCAS85/c2670.bench"
    if not os.path.exists(bench_path):
        print(f"Error: {bench_path} not found.")
        return

    print(f"Loading {bench_path}...")
    circuit, _ = parse_bench_file(bench_path)
    
    print("Finding reconvergent pairs (limit 100)...")
    # Increase beam width/depth to ensure we find enough
    pairs = find_all_reconv_pairs(circuit, beam_width=16, max_depth=50, max_pairs=100)
    print(f"Found {len(pairs)} pairs.")
    
    if len(pairs) < 100:
        print("Warning: Found fewer than 100 pairs.")

    total_checks = 0
    consistent_count = 0
    verified_count = 0
    
    for i, info in enumerate(pairs):
        # print(f"Analyzing Pair {i+1}...")
        results = check_path_pair_consistency(circuit, info)
        
        for target in [0, 1]:
            assignment = results[target]
            total_checks += 1
            
            if assignment is not None:
                consistent_count += 1
                # Verify
                if verify_assignment(circuit, assignment, info['reconv'], LogicValue(target)):
                    verified_count += 1
                else:
                    print(f"FAIL: Pair {i+1} Target {target} - Assignment verification failed.")

    print("\nAnalysis Results:")
    print(f"Total Pairs: {len(pairs)}")
    print(f"Total Checks (Target 0 & 1): {total_checks}")
    print(f"Consistent Cases Found: {consistent_count}")
    print(f"Verified Correct: {verified_count}")
    
    if consistent_count > 0:
        accuracy = (verified_count / consistent_count) * 100.0
        print(f"Accuracy (Verified / Consistent): {accuracy:.2f}%")
    else:
        print("Accuracy: N/A (No consistent cases found)")

if __name__ == "__main__":
    analyze_c2670()
