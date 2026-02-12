#!/usr/bin/env python
"""
Debug script for path picker - visualizes reconvergent path pairs
"""
import sys
import os
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.util.io import parse_bench_file
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver
from src.atpg.ai_podem import ModelPairPredictor
from src.util.struct import LogicValue, Fault, GateType

def print_gate_info(circuit, gate_id):
    """Print detailed gate information"""
    if gate_id >= len(circuit):
        return f"Gate {gate_id}: OUT OF BOUNDS"
    
    gate = circuit[gate_id]
    
    # Use actual GateType enum
    gate_type_names = {
        0: 'UNINITIALIZED',
        GateType.INPT: 'INPUT',
        GateType.FROM: 'BRANCH',
        GateType.BUFF: 'BUFF',
        GateType.NOT: 'NOT',
        GateType.AND: 'AND',
        GateType.NAND: 'NAND',
        GateType.OR: 'OR',
        GateType.NOR: 'NOR',
        GateType.XOR: 'XOR',
        GateType.XNOR: 'XNOR'
    }
    type_name = gate_type_names.get(gate.type, f'UNKNOWN({gate.type})')
    
    fanins = gate.fin if hasattr(gate, 'fin') and gate.fin is not None else []
    fanouts = gate.fot if hasattr(gate, 'fot') and gate.fot is not None else []
    
    return f"Gate {gate_id}: {type_name} (fanins: {fanins}, fanouts: {fanouts})"

def print_path(circuit, path, indent="  "):
    """Print a path in human-readable format"""
    print(f"{indent}Path length: {len(path)}")
    for i, node_id in enumerate(path):
        arrow = " -> " if i < len(path) - 1 else ""
        print(f"{indent}  [{i}] {print_gate_info(circuit, node_id)}{arrow}")

def debug_path_picker(args=None, output_file=None):
    """Debug the path picker for c432 circuit, fault sa0 on node 296"""
    
    # Redirect output to file if specified
    if output_file:
        sys.stdout = open(output_file, 'w')
    
    print("="*80)
    print("DEBUG: Path Picker for c432, Fault sa0 on Node 296")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Parse circuit
    default_circuit = 'data/bench/ISCAS85/c432.bench'
    circuit_path = args.circuit if args and hasattr(args, 'circuit') else default_circuit
    circuit, total_gates = parse_bench_file(circuit_path)
    
    print(f"\nCircuit: c432")
    print(f"Total gates: {len(circuit)}")
    print(f"Total gates (including PIs): {total_gates}")
    
    # Print circuit structure
    print("\n" + "="*80)
    print("CIRCUIT STRUCTURE")
    print("="*80)
    for i in range(len(circuit)):
        gate = circuit[i]
        # Only print initialized nodes
        if gate.type != 0:
            print(print_gate_info(circuit, i))
    
    # Create fault: sa0 on node 22
    fault_str = args.fault if args and hasattr(args, 'fault') else "296-1" # Default: 296 s-a-0 (D=1/0)
    try:
        gate_id, val = map(int, fault_str.split('-'))
        fault_val = LogicValue.D if val == 0 else LogicValue.D_BAR 
        # Actually, for ATPG input, usually it's stuck-at value, so we justify the opposite
        # But here let's assume gate-s_a_val format.
        # If stuck-at-0, we need D (1/0). If stuck-at-1, we need D_BAR (0/1).
        fault = Fault(gate_id=gate_id, value=fault_val)
    except:
        print(f"Invalid fault format: {fault_str}. Using default 296-0")
        fault = Fault(gate_id=296, value=LogicValue.D)
        val = 0
        
    print(f"\n" + "="*80)
    print(f"FAULT: Gate {fault.gate_id} s-a-{val} (Value: {fault.value})")
    print(print_gate_info(circuit, fault.gate_id))
    print("="*80)
    
    # Create predictor and solver
    print("\nInitializing predictor and solver...")
    try:
        from src.atpg.ai_podem import AiPodemConfig
        
        # Determine device
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Use model path from args or default
        default_model = 'checkpoints/reconv_rl_model.pt'
        model_path = args.model if args and hasattr(args, 'model') else default_model
        
        config = AiPodemConfig(
            model_path=model_path,
            device=device,
            enable_ai_activation=True,
            enable_ai_propagation=True
        )
        
        predictor = ModelPairPredictor(
            circuit, 
            circuit_path, 
            config
        )
        solver = HierarchicalReconvSolver(circuit, predictor)
        print("✓ Predictor and solver initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get transitive fanin
    print(f"\n" + "="*80)
    print(f"TRANSITIVE FANIN of Gate {fault.gate_id}")
    print("="*80)
    fanin_set = solver._get_transitive_fanin(fault.gate_id)
    print(f"Fanin nodes: {sorted(fanin_set)}")
    print(f"Total fanin nodes: {len(fanin_set)}")
    
    # Find reconvergent pairs
    print(f"\n" + "="*80)
    print("RECONVERGENT PAIRS")
    print("="*80)
    pairs = solver._find_pairs_in_set(fanin_set)
    print(f"Found {len(pairs)} reconvergent pair(s)")
    
    if not pairs:
        print("\n⚠ No reconvergent pairs found!")
        print("This means the circuit has no reconvergent fanout structure")
        print("in the transitive fanin cone of the fault node.")
        return
    
    # Print each pair in detail
    for idx, pair in enumerate(pairs):
        print(f"\n{'-'*80}")
        print(f"PAIR #{idx + 1}")
        print(f"{'-'*80}")
        
        # Check which keys are available
        stem_key = 'start' if 'start' in pair else 'stem'
        stem_node = pair[stem_key]
        
        print(f"Stem/Start node: {stem_node}")
        print(print_gate_info(circuit, stem_node))
        print(f"\nReconvergence node: {pair['reconv']}")
        print(print_gate_info(circuit, pair['reconv']))
        
        if 'branches' in pair:
            print(f"\nBranch nodes: {pair['branches']}")
        
        print(f"\nPaths ({len(pair['paths'])} total):")
        for path_idx, path in enumerate(pair['paths']):
            print(f"\n  Path {path_idx + 1}:")
            print_path(circuit, path, indent="    ")
    
    # Collect and sort all pairs
    print(f"\n" + "="*80)
    print("SORTED PAIRS (by priority)")
    print("="*80)
    sorted_pairs = solver._collect_and_sort_pairs(fault.gate_id)
    print(f"Total pairs after sorting: {len(sorted_pairs)}")
    
    # Compute distances from fault node for display
    from collections import deque
    distances = {fault.gate_id: 0}
    queue = deque([fault.gate_id])
    fanin_set = solver._get_transitive_fanin(fault.gate_id)
    
    while queue:
        curr = queue.popleft()
        curr_dist = distances[curr]
        gate = circuit[curr]
        fanins = getattr(gate, 'fin', []) or []
        
        for fin in fanins:
            if fin in fanin_set and fin not in distances:
                distances[fin] = curr_dist + 1
                queue.append(fin)
    
    for idx, pair in enumerate(sorted_pairs):
        stem_key = 'start' if 'start' in pair else 'stem'
        reconv_node = pair['reconv']
        
        # Calculate metrics
        path1_len = len(pair['paths'][0])
        path2_len = len(pair['paths'][1])
        total_path_len = path1_len + path2_len
        dist_to_fault = distances.get(reconv_node, 9999)
        combined_score = total_path_len + dist_to_fault
        
        print(f"\n{idx + 1}. Stem: {pair[stem_key]}, Reconv: {reconv_node} | "
              f"Paths: [{path1_len}, {path2_len}] (Total: {total_path_len}) | "
              f"Dist: {dist_to_fault} | Score: {combined_score}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Debug path picker for ATPG')
    parser.add_argument('--output', '-o', default='debug_path_picker_output.log',
                        help='Output file path (default: debug_path_picker_output.log)')
    parser.add_argument('--model', type=str, default='checkpoints/reconv_rl_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--circuit', type=str, default='data/bench/ISCAS85/c432.bench',
                        help='Path to circuit .bench file')
    parser.add_argument('--fault', type=str, default='296-0',
                        help='Fault string (gate-val)')
    args = parser.parse_args()
    
    debug_path_picker(args=args, output_file=args.output)
    print(f"\nOutput written to: {args.output}", file=sys.stderr)
