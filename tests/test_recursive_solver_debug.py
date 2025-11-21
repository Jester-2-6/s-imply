import os
import sys
import random
from typing import List, Dict, Any, Set
from src.util.struct import Gate, GateType, LogicValue
from src.util.io import parse_bench_file
from src.atpg.reconv_podem import RecursiveStructureSolver
from src.atpg.logic_sim_three import logic_sim, reset_gates

def verify_assignment_sim(circuit: List[Gate], assignment: Dict[int, LogicValue], target_node: int, target_val: LogicValue) -> tuple:
    """
    Verify assignment by simulating the circuit.
    Returns (success, errors_dict)
    """
    reset_gates(circuit, len(circuit)-1)
    
    # Apply PIs
    for gid, val in assignment.items():
        gate = circuit[gid]
        if not gate.fin: # PI
            gate.val = val
        
    # Run simulation
    logic_sim(circuit, len(circuit)-1)
    
    errors = {}
    
    # Check target
    if circuit[target_node].val != target_val:
        errors['target'] = f"Target {target_node}: expected {target_val}, got {circuit[target_node].val}"
        
    # Check consistency of all assigned internal nodes
    for gid, val in assignment.items():
        if circuit[gid].val != val and circuit[gid].val != LogicValue.XD:
            errors[gid] = f"Node {gid}: expected {val}, got {circuit[gid].val}"
            
    return (len(errors) == 0, errors)

def test_recursive_solver_debug():
    bench_path = "data/bench/ISCAS85/c2670.bench"
    circuit, _ = parse_bench_file(bench_path)
    
    solver = RecursiveStructureSolver(circuit)
    
    # Test just a few nodes to debug
    candidates = [i for i, g in enumerate(circuit) if g.fin and g.type != GateType.INPT]
    test_nodes = random.sample(candidates, min(10, len(candidates)))
        
    print(f"Testing {len(test_nodes)} random nodes for debugging...")
    
    for node in test_nodes:
        for val in [LogicValue.ZERO, LogicValue.ONE]:
            print(f"\n=== Node {node} = {val} ===")
            assignment = solver.solve(node, val)
            
            if assignment:
                print(f"Found assignment with {len(assignment)} gates")
                success, errors = verify_assignment_sim(circuit, assignment, node, val)
                if success:
                    print("✓ VERIFIED")
                else:
                    print(f"✗ FAILED - {len(errors)} errors:")
                    for k, v in list(errors.items())[:5]:  # Show first 5 errors
                        print(f"  {v}")
            else:
                print("No solution found")

if __name__ == "__main__":
    test_recursive_solver_debug()
