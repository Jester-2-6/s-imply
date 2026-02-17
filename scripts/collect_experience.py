import argparse
import os
import random
import resource
import sys

import torch
from tqdm import tqdm

from src.atpg.ai_podem import ModelPairPredictor, ai_podem
from src.atpg.logic_sim_three import reset_gates
from src.atpg.podem import get_all_faults
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver
from src.ml.rl.rl_recorder import ExperienceRecorder
from src.util.io import parse_bench_file

# Increase recursion limit and stack size for deep circuit solving
sys.setrecursionlimit(100000)
try:
    # Increase stack size to 256MB
    resource.setrlimit(resource.RLIMIT_STACK, (256 * 1024 * 1024, resource.RLIM_INFINITY))
except Exception:
    pass  # May fail on some systems


def collect_experience(
    bench_dir: str = "data/bench/ISCAS85",
    model_path: str = "checkpoints/reconv_minimal_model.pt",
    output_dir: str = "data/rl_experience",
    max_faults_per_circuit: int = 50,
    gpu_id: int = 0,
):
    """
    Run AI-PODEM on benchmarks to collect RL experience.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Benchmarks to process
    benchmarks = []
    print(f"Checking bench_dir: {bench_dir}")
    if os.path.exists(bench_dir):
        # Walk to find .bench files or just list
        if os.path.isdir(bench_dir):
            benchmarks = [f for f in os.listdir(bench_dir) if f.endswith(".bench")]
            benchmarks.sort()
            print(f"Found {len(benchmarks)} benchmarks: {benchmarks[:5]}...")
        else:
            print(f"{bench_dir} is not a directory.")
    else:
        print(f"Benchmark directory {bench_dir} not found.")
        return

    recorder = ExperienceRecorder(save_dir=output_dir)

    # Global stats
    total_episodes = 0
    total_success = 0

    for bench_file in benchmarks:
        print(f"Processing {bench_file}...")
        circuit_path = os.path.join(bench_dir, bench_file)

        try:
            circuit, total_gates = parse_bench_file(circuit_path)
            print(f"  Parsed {len(circuit)} gates.")
        except Exception as e:
            print(f"Failed to parse {bench_file}: {e}")
            continue

        faults = get_all_faults(circuit, total_gates)
        print(f"  Found {len(faults)} faults.")
        if not faults:
            print(f"No faults found in {bench_file}")
            continue

        # Shuffle and limit faults
        random.shuffle(faults)
        selected_faults = faults[:max_faults_per_circuit]
        print(f"  Selected {len(selected_faults)} faults for processing.")

        # Load Predictor & Solver once per circuit
        # (Model loaded onto GPU/CPU inside predictor)
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        try:
            from src.atpg.ai_podem import AiPodemConfig

            cfg = AiPodemConfig(
                model_path=model_path,
                device=device,
                enable_ai_activation=True,
                enable_ai_propagation=True,
                verbose=False,
            )
            predictor = ModelPairPredictor(circuit, circuit_path, cfg)
        except Exception as e:
            print(f"Failed to load predictor for {bench_file}: {e}")
            continue

        # Create solver with recorder
        solver = HierarchicalReconvSolver(circuit, predictor, recorder=recorder)

        pbar = tqdm(selected_faults, desc=f"Faults ({bench_file})")
        for fault in pbar:
            try:
                # Reset circuit state before each fault
                reset_gates(circuit, total_gates)

                # Start Episode
                recorder.start_episode(f"{bench_file}_{fault.gate_id}_{fault.value}")

                # Run AI-PODEM
                success = ai_podem(
                    circuit=circuit,
                    fault=fault,
                    total_gates=total_gates,
                    circuit_path=circuit_path,
                    predictor=predictor,
                    solver=solver,
                    enable_ai_activation=True,
                    enable_ai_propagation=True,
                    verbose=False,
                )

                # End Episode
                final_reward = 10.0 if success else -5.0
                recorder.finish_episode(final_reward=final_reward)

                total_episodes += 1
                if success:
                    total_success += 1

            except Exception as e:
                print(f"  Error processing fault {fault.gate_id}: {e}")
                recorder.finish_episode(final_reward=-5.0)  # Mark as failure
                total_episodes += 1

            # Periodic save
            if total_episodes % 10 == 0:
                recorder.save_buffer()

            pbar.set_postfix({"succ": total_success, "eps": total_episodes})

    # Final save
    recorder.save_buffer()
    print(f"Collection complete. {total_episodes} episodes, {total_success} successful.")


if __name__ == "__main__":
    # Add CLI later if needed

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench_dir", default="data/bench/ISCAS85", help="Directory with .bench files"
    )
    parser.add_argument(
        "--model",
        default="checkpoints/reconv_minimal_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--max_faults", type=int, default=50, help="Max faults per circuit")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    collect_experience(
        bench_dir=args.bench_dir,
        model_path=args.model,
        max_faults_per_circuit=args.max_faults,
        gpu_id=args.gpu,
    )
