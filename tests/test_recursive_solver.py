import os
import sys
import random
from typing import List, Dict, Any, Set
from src.util.struct import Gate, GateType, LogicValue
from src.util.io import parse_bench_file
from src.atpg.reconv_podem import RecursiveStructureSolver
from src.atpg.logic_sim_three import logic_sim, reset_gates

def verify_assignment_sim(circuit: List[Gate], assignment: Dict[int, LogicValue], target_node: int, target_val: LogicValue) -> bool:
    """
    Verify assignment by simulating the circuit.
    We set PIs from assignment, simulate, and check if target_node has target_val.
    """
    reset_gates(circuit, len(circuit)-1)
    
    # Apply PIs
    for gid, val in assignment.items():
        gate = circuit[gid]
        if not gate.fin: # PI
            gate.val = val
        # Note: We only apply PIs. Internal gates should be computed.
        # But if assignment contains internal gates, we ignore them for simulation inputs,
        # but we can check if they match simulation results.
        
    # Run simulation
    logic_sim(circuit, len(circuit)-1)
    
    # Check target
    if circuit[target_node].val != target_val:
        # print(f"Mismatch at target {target_node}. Expected {target_val}, Got {circuit[target_node].val}")
        return False
        
    # Check consistency of all assigned internal nodes
    for gid, val in assignment.items():
        if circuit[gid].val != val and circuit[gid].val != LogicValue.XD:
            # print(f"Mismatch at internal {gid}. Expected {val}, Got {circuit[gid].val}")
            return False
            
    return True

def test_recursive_solver():
    bench_path = "data/bench/ISCAS85/c2670.bench"
    if not os.path.exists(bench_path):
        print(f"Error: {bench_path} not found.")
        return

    print(f"Loading {bench_path}...")
    circuit, _ = parse_bench_file(bench_path)
    
    solver = RecursiveStructureSolver(circuit)
    
    # Pick 500 random nodes
    # Filter for nodes that are not PIs (or include PIs? PIs are trivial)
    # Let's pick internal gates.
    candidates = [i for i, g in enumerate(circuit) if g.fin and g.type != GateType.INPT]
    
    if len(candidates) < 500:
        test_nodes = candidates
    else:
        test_nodes = random.sample(candidates, 500)
        
    print(f"Testing {len(test_nodes)} random nodes...")
    
    passed = 0
    total_checks = 0
    justified = 0
    
    for node in test_nodes:
        for val in [LogicValue.ZERO, LogicValue.ONE]:
            # Try to justify node=val
            # print(f"Justifying Node {node} = {val}...")
            assignment = solver.solve(node, val)
            
            total_checks += 1
            
            if assignment:
                justified += 1
                if verify_assignment_sim(circuit, assignment, node, val):
                    passed += 1
                else:
                    print(f"FAIL: Node {node} Val {val} - Assignment verification failed.")
            else:
                # Impossible or solver failed.
                # We can't easily distinguish without exhaustive search.
                # But we count it as "not justified".
                # print(f"Could not justify Node {node} = {val}")
                pass
                
    print("\nRecursive Solver Results:")
    print(f"Total Checks: {total_checks}")
    print(f"Justified (Solution Found): {justified}")
    print(f"Verified Correct: {passed}")
    if justified > 0:
        print(f"Accuracy (Verified/Justified): {passed/justified*100:.2f}%")
    else:
        print("Accuracy: N/A")

if __name__ == "__main__":
    test_recursive_solver()
