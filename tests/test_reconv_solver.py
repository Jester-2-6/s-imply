import os
import sys
from src.util.struct import Gate, GateType, LogicValue
from src.util.io import parse_bench_file
from src.atpg.reconv_podem import check_path_pair_consistency, find_all_reconv_pairs

def test_c17():
    print("Testing c17.bench...")
    # Load c17
    # Path might be relative to where we run it
    bench_path = "data/bench/ISCAS85/c17.bench"
    if not os.path.exists(bench_path):
        print(f"Error: {bench_path} not found.")
        return

    circuit, _ = parse_bench_file(bench_path)
    
    # Find pairs
    pairs = find_all_reconv_pairs(circuit, beam_width=16, max_depth=10)
    print(f"Found {len(pairs)} reconvergent pairs.")
    
    for i, info in enumerate(pairs):
        print(f"\nPair {i+1}: Start={info['start']}, Reconv={info['reconv']}")
        print(f"  Branches: {info['branches']}")
        print(f"  Paths: {info['paths']}")
        
        results = check_path_pair_consistency(circuit, info)
        print(f"  Target 0: {'Possible' if results[0] else 'Impossible'}")
        if results[0]:
            print(f"    Assignment: {results[0]}")
            
        print(f"  Target 1: {'Possible' if results[1] else 'Impossible'}")
        if results[1]:
            print(f"    Assignment: {results[1]}")

        # Verification logic (manual check for c17)
        # c17 structure:
        # 10 = NAND(1, 3)
        # 11 = NAND(3, 6)
        # 16 = NAND(2, 11)
        # 19 = NAND(11, 7)
        # 22 = NAND(19, 16)
        # 23 = NAND(16, 10)
        
        # Pair: Start=3, Reconv=23
        # Path 1: 3 -> 10 -> 23
        # Path 2: 3 -> 11 -> 16 -> 23
        
        if info['start'] == 3 and info['reconv'] == 23:
            # Let's analyze manually:
            # Target 23=0:
            # NAND(16, 10) = 0 => 16=1 AND 10=1
            # 10=1 => NAND(1, 3)=1 => (1=0 OR 3=0)
            # 16=1 => NAND(2, 11)=1 => (2=0 OR 11=0)
            # 11=0 => NAND(3, 6)=0 => 3=1 AND 6=1
            # So if 11=0, then 3=1.
            # If 3=1, then for 10=1, we need 1=0.
            # So S=3 must be 1.
            # Path 1 (3->10->23): 3=1, 10=1 (requires 1=0).
            # Path 2 (3->11->16->23): 3=1, 11=0 (requires 6=1), 16=1 (requires 2=X).
            # Consistent? Yes. S=1, Side inputs: 1=0, 6=1.
            # So Target 0 should be Possible.
            
            # Target 23=1:
            # NAND(16, 10) = 1 => (16=0 OR 10=0)
            # Case A: 10=0 => NAND(1, 3)=0 => 1=1 AND 3=1. (S=3 must be 1)
            # Case B: 16=0 => NAND(2, 11)=0 => 2=1 AND 11=1.
            #   11=1 => NAND(3, 6)=1 => (3=0 OR 6=0).
            #   If 3=0 (S=0), then 10=1 (since 3=0). So we satisfy 23=1 via 16=0.
            # So Target 1 is Possible (with S=0 or S=1).
            
            if results[0] and results[1]:
                print("PASS: Pair (3, 23) allows both 0 and 1.")
            else:
                print("FAIL: Pair (3, 23) should allow both 0 and 1.")

if __name__ == "__main__":
    test_c17()
