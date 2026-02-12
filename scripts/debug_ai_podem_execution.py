import os
import sys
import torch
import time
import argparse
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.util.struct import Gate, GateType, LogicValue, Fault
from src.util.io import parse_bench_file
from src.atpg.ai_podem import ai_podem, ModelPairPredictor, AiPodemConfig

def main():
    parser = argparse.ArgumentParser(description="Debug AI PODEM Execution logic")
    parser.add_argument('circuit', type=str, help="Path to circuit .bench file")
    parser.add_argument('fault', type=str, help="Fault string (e.g. '1-0' for gate 1 stuck-at-0)")
    parser.add_argument('--model', type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    print(f"Loading circuit: {args.circuit}")
    circuit, total_gates = parse_bench_file(args.circuit)
    circuit_path = args.circuit
    
    # Parse fault
    try:
        gate_id, val = map(int, args.fault.split('-'))
        fault = Fault(gate_id, LogicValue(val))
        print(f"Target Fault: Gate {gate_id} s-a-{val}")
    except:
        print(f"Invalid fault format: {args.fault}. Use gate-val (e.g. 1-0)")
        return

    # Initialize Predictor
    print("Loading Model...")
    config = AiPodemConfig(
        model_path=args.model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_ai_activation=True,
        enable_ai_propagation=True
    )
    
    predictor = ModelPairPredictor(circuit, circuit_path, config)
    
    # Monkey patch the solver to trace execution
    import src.atpg.recursive_reconv_solver as solver_module
    
    original_solve_recursive = solver_module.HierarchicalReconvSolver._solve_recursive
    recursion_depth = [0]
    call_count = [0]
    
    def traced_solve_recursive(self, pair_idx, pairs, current_constraints, seed=None):
        if pair_idx < len(pairs):
            pair = pairs[pair_idx]
            stem = pair.get('start', pair.get('stem'))
            reconv = pair.get('reconv')
            depth = recursion_depth[0]
            indent = "  " * depth
            
            seed_info = f" [Seed: {seed}]" if seed is not None else ""
            print(f"\n{indent}┌─ [pair_idx={pair_idx}] Solving Pair: {stem} -> {reconv}{seed_info}")
            recursion_depth[0] += 1
            
        # Call original
        result = original_solve_recursive(self, pair_idx, pairs, current_constraints, seed)
        
        if pair_idx < len(pairs):
            recursion_depth[0] -= 1
            indent = "  " * recursion_depth[0]
            
            if result:
                print(f"{indent}└─ ✓ SOLVED (Pair {pair_idx})")
            else:
                print(f"{indent}└─ ✗ FAILED (Pair {pair_idx})")
                
        return result
    
    solver_module.HierarchicalReconvSolver._solve_recursive = traced_solve_recursive
    
    # Add verification after solve
    original_solve_main = solver_module.HierarchicalReconvSolver.solve
    
    def traced_solve_with_verification(self, target_node, target_val, constraints=None, seed=None):
        # Call original traced solve - ensure seed is passed
        # Note: solve() calls _solve_recursive, so we don't need to manually trace here if _solve_recursive is patched
        # BUT we need to call the ORIGINAL solve method logic, or just let it run.
        # Since we patched _solve_recursive, calling original_solve_main will use the traced recursive method.
        
        # However, original_solve_main is unbound, so we need to bind it or call it on self
        # Actually, we can't easily call 'original_solve_main' if it was a method.
        # Better approach: We are replacing the method on the class.
        
        # Let's just reimplement the wrapper logic around the original method.
        # But wait, we can't easily call the original method conformantly if we replaced it on the class 
        # unless we saved it. We did: original_solve_main.
        
        result = original_solve_main(self, target_node, target_val, constraints, seed)
        
        if result:
            print(f"\n{'─'*60}")
            print("LOGIC CONSISTENCY VERIFICATION")
            print(f"{'─'*60}")
            
            from src.atpg.logic_sim_three import compute_gate_value
            
            # Check if each assigned gate value is consistent with its inputs
            inconsistencies = []
            partial_assignments = []
            
            for gid, expected_val in result.items():
                if gid > total_gates:
                    continue
                    
                gate = circuit[gid]
                
                # Skip primary inputs - they can be set arbitrarily
                if gate.type == GateType.INPT:
                    continue
                
                # Check if all inputs are assigned
                all_inputs_assigned = all(
                    fin in result for fin in gate.fin
                )
                
                if not all_inputs_assigned:
                    # Some inputs are not assigned - can't verify
                    partial_assignments.append(gid)
                    continue
                
                # Temporarily set gate values to check consistency
                saved_vals = {}
                for fin in gate.fin:
                    saved_vals[fin] = circuit[fin].val
                    if fin in result:
                        circuit[fin].val = result[fin]
                
                # Compute what this gate should be based on its inputs
                computed_val = compute_gate_value(circuit, gate)
                
                # Restore
                for fin, val in saved_vals.items():
                    circuit[fin].val = val
                
                # Check consistency
                if computed_val != expected_val:
                    inconsistencies.append({
                        'gate': gid,
                        'expected': expected_val,
                        'computed': computed_val,
                        'inputs': {fin: result.get(fin, 'X') for fin in gate.fin}
                    })
            
            # Report results
            print(f"Checked {len(result)} gate assignments")
            print(f"Primary inputs: {sum(1 for gid in result if gid <= total_gates and circuit[gid].type == GateType.INPT)}")
            print(f"Internal gates: {len(result) - sum(1 for gid in result if gid <= total_gates and circuit[gid].type == GateType.INPT)}")
            
            if partial_assignments:
                print(f"\n⚠ {len(partial_assignments)} gates have incomplete input assignments (cannot verify)")
            
            if inconsistencies:
                print(f"\n✗ LOGIC INCONSISTENCIES FOUND: {len(inconsistencies)}")
                for inc in inconsistencies[:5]:
                    print(f"\n  Gate {inc['gate']}:")
                    print(f"    Expected value: {inc['expected']}")
                    print(f"    Computed from inputs: {inc['computed']}")
                    print(f"    Input values: {inc['inputs']}")
                if len(inconsistencies) > 5:
                    print(f"\n  ... +{len(inconsistencies) - 5} more inconsistencies")
            else:
                print(f"\n✓ ALL ASSIGNMENTS ARE LOGICALLY CONSISTENT")
                print(f"  Every gate's value matches what its inputs produce")
            
            # Check target specifically
            print(f"\nTarget Gate {target_node}:")
            print(f"  Required: {target_val}")
            print(f"  Assigned: {result.get(target_node, 'NOT ASSIGNED')}")
            if result.get(target_node) == target_val:
                print(f"  ✓ Target requirement satisfied")
            else:
                print(f"  ✗ Target requirement NOT satisfied")
            
            print(f"{'─'*60}\n")
        
        return result
    
    # Replace with verification wrapper
    solver_module.HierarchicalReconvSolver.solve = traced_solve_with_verification
    
    
    # Also trace AIBacktracer decisions
    import src.atpg.ai_podem as ai_podem_module
    
    original_backtracer_call = ai_podem_module.AIBacktracer.__call__
    backtracer_call_count = [0]
    
    def traced_backtracer_call(self, objective, circuit):
        backtracer_call_count[0] += 1
        bt_id = backtracer_call_count[0]
        
        print(f"\n  ╔═══ [Backtracer {bt_id}] PROPAGATION OBJECTIVE ═══")
        print(f"  ║   Target: Gate {objective.gate_id} = {objective.value}")
        print(f"  ║   (Finding PI assignment to justify this gate during propagation)")
        print(f"  ╚═══")
        
        result = original_backtracer_call(self, objective, circuit)
        
        if result and result.gate_id != -1:
            print(f"  ╚═ ✓ PI Assignment: Gate {result.gate_id} = {result.value}")
        else:
            print(f"  ╚═ ✗ Failed (no PI assignment found)")
        
        return result
    
    ai_podem_module.AIBacktracer.__call__ = traced_backtracer_call
    
    print("="*80)
    print("RUNNING 100 AI PODEM CYCLES (AI Activation ONLY - Detailed Trace)")
    print("="*80 + "\n")
    
    success_count = 0
    failure_count = 0
    # patterns = [] # Unused
    
    # Reset circuit
    from src.atpg.podem import initialize
    initialize(circuit, total_gates)
    
    call_count[0] = 0
    
    try:
        result = ai_podem(
            circuit, 
            fault, 
            total_gates,
            circuit_path=circuit_path,
            predictor=predictor,
            enable_ai_activation=True,
            enable_ai_propagation=False,  # AI activation only
            verbose=False
        )
        
        if result:
            print("\n" + "="*80)
            print(f"FINAL RESULT: SUCCESS - Found pattern")
            print(f"Pattern: {result}")
            print("="*80)
        else:
            print("\n" + "="*80)
            print(f"FINAL RESULT: FAILURE - No pattern found")
            print("="*80)
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

if __name__ == "__main__":
    f = open('debug_ai_podem_execution.log', 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    try:
        main()
    finally:
        sys.stdout = original_stdout
        f.close()
