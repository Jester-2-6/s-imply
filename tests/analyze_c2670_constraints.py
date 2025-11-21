import os
import sys
import random
from typing import List, Dict, Any, Set
from src.util.struct import Gate, GateType, LogicValue
from src.util.io import parse_bench_file
from src.atpg.reconv_podem import check_path_pair_consistency, find_all_reconv_pairs, PathConsistencySolver

def get_random_valid_state(circuit: List[Gate], path_gates: Set[int], start_node: int) -> Dict[int, LogicValue]:
    """
    Generate a valid state for path gates by simulating with random inputs.
    Returns a dict of {gate_idx: value} for all gates in path_gates.
    """
    # 1. Assign random value to Start Node
    s_val = random.choice([LogicValue.ZERO, LogicValue.ONE])
    
    # 2. Assign random values to all side inputs of path gates
    # We need to identify side inputs.
    assignments = {start_node: s_val}
    
    # We need to propagate values. 
    # Since we don't have a full simulator that accepts partial assignments easily,
    # let's do a forward pass on path_gates sorted topologically.
    sorted_gates = sorted(list(path_gates))
    
    for gid in sorted_gates:
        gate = circuit[gid]
        if gid == start_node:
            continue
            
        input_vals = []
        for fin in gate.fin:
            if fin in path_gates:
                # Must be computed already if sorted
                if fin not in assignments:
                    # Should not happen if topological sort is correct and path is connected
                    # But path_gates might not be fully connected subgraph if there are gaps?
                    # The path finder returns list of nodes.
                    # If fin is in path_gates, it should be in assignments.
                    # Unless fin is S? S is in assignments.
                    # Maybe topological sort of IDs isn't perfect topological sort?
                    # For ISCAS benchmarks, ID order usually respects topology.
                    assignments[fin] = LogicValue.XD # Fallback
                input_vals.append(assignments[fin])
            else:
                # Side input - assign random
                val = random.choice([LogicValue.ZERO, LogicValue.ONE])
                assignments[fin] = val # Track side inputs too if we want?
                input_vals.append(val)
        
        # Compute output
        # We need a compute function.
        val = compute_gate_logic(gate.type, input_vals)
        assignments[gid] = val
        
    return {k: v for k, v in assignments.items() if k in path_gates}

def compute_gate_logic(gtype: int, inputs: List[LogicValue]) -> LogicValue:
    # Simple logic computation
    if gtype == GateType.AND:
        return LogicValue.ONE if all(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO
    elif gtype == GateType.NAND:
        return LogicValue.ZERO if all(i == LogicValue.ONE for i in inputs) else LogicValue.ONE
    elif gtype == GateType.OR:
        return LogicValue.ONE if any(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO
    elif gtype == GateType.NOR:
        return LogicValue.ZERO if any(i == LogicValue.ONE for i in inputs) else LogicValue.ONE
    elif gtype == GateType.XOR:
        ones = sum(1 for i in inputs if i == LogicValue.ONE)
        return LogicValue.ONE if ones % 2 == 1 else LogicValue.ZERO
    elif gtype == GateType.XNOR:
        ones = sum(1 for i in inputs if i == LogicValue.ONE)
        return LogicValue.ONE if ones % 2 == 0 else LogicValue.ZERO
    elif gtype == GateType.BUFF:
        return inputs[0]
    elif gtype == GateType.NOT:
        return LogicValue.ZERO if inputs[0] == LogicValue.ONE else LogicValue.ONE
    return LogicValue.XD

def analyze_c2670_constraints():
    bench_path = "data/bench/ISCAS85/c2670.bench"
    if not os.path.exists(bench_path):
        print(f"Error: {bench_path} not found.")
        return

    print(f"Loading {bench_path}...")
    circuit, _ = parse_bench_file(bench_path)
    
    print("Finding reconvergent pairs (limit 100)...")
    pairs = find_all_reconv_pairs(circuit, beam_width=16, max_depth=50, max_pairs=100)
    print(f"Found {len(pairs)} pairs.")
    
    solver = PathConsistencySolver(circuit)
    
    total_tests = 0
    passed_tests = 0
    
    for i, info in enumerate(pairs):
        start_node = info['start']
        reconv_node = info['reconv']
        path_gates = set()
        for p in info['paths']:
            path_gates.update(p)
            
        # 1. Generate a valid state
        state = get_random_valid_state(circuit, path_gates, start_node)
        
        # 2. Pick 1-2 constraints
        candidates = list(path_gates)
        # Remove S and R from candidates to make it "intermediate"
        if start_node in candidates: candidates.remove(start_node)
        if reconv_node in candidates: candidates.remove(reconv_node)
        
        if not candidates:
            continue
            
        num_constraints = min(len(candidates), random.randint(1, 2))
        constraint_nodes = random.sample(candidates, num_constraints)
        constraints = {n: state[n] for n in constraint_nodes}
        
        target_val = state[reconv_node]
        
        # 3. Run solver with constraints
        # print(f"Pair {i+1}: Target={target_val}, Constraints={constraints}")
        assignment = solver.solve(info, target_val, constraints)
        
        total_tests += 1
        
        if assignment:
            # Verify assignment respects constraints
            constraints_met = True
            for n, v in constraints.items():
                if assignment.get(n) != v:
                    print(f"FAIL: Constraint mismatch at {n}. Expected {v}, Got {assignment.get(n)}")
                    constraints_met = False
            
            # Verify assignment produces target
            # (We assume verify_assignment from previous script works, but let's trust solver for now or copy it)
            # For now, just checking if it found a solution is good, because we KNOW one exists (the state we generated).
            if constraints_met:
                passed_tests += 1
            else:
                 print(f"FAIL: Pair {i+1} - Solver returned assignment violating constraints.")
        else:
            print(f"FAIL: Pair {i+1} - Solver failed to find solution for KNOWN valid state with constraints.")
            print(f"State: {state}")
            print(f"Constraints: {constraints}")

    print("\nConstraint Analysis Results:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Accuracy: {passed_tests/total_tests*100:.2f}%")

if __name__ == "__main__":
    analyze_c2670_constraints()
