#!/usr/bin/env python
"""
Unified RL Training Pipeline
Allows running individual stages or the full pipeline with CLI arguments
"""
import sys
import os
import argparse
import subprocess
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_experience_collection(args):
    """Stage 1: Collect experience data from AI-PODEM runs"""
    print_section("STAGE 1: Experience Collection")
    
    cmd = [
        "python", "scripts/collect_experience.py",
        "--max_faults", str(args.max_faults)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    if result.returncode != 0:
        print(f"❌ Experience collection failed with code {result.returncode}")
        return False
    
    print("✓ Experience collection completed")
    return True

def run_training(args):
    """Stage 2: Train the RL model"""
    print_section("STAGE 2: Model Training")
    
    cmd = [
        "python", "scripts/train_rl.py",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr)
    ]
    
    if args.model:
        cmd.extend(["--model", args.model])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    if result.returncode != 0:
        print(f"❌ Training failed with code {result.returncode}")
        return False
    
    print("✓ Training completed")
    return True

def run_benchmark(args):
    """Stage 3: Benchmark the trained model"""
    print_section("STAGE 3: Benchmarking")
    
    model_path = args.output_model or "checkpoints/reconv_rl_model.pt"
    
    cmd = [
        "python", "scripts/benchmark_c432_compare.py",
        "--ai_model", model_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    if result.returncode != 0:
        print(f"❌ Benchmarking failed with code {result.returncode}")
        return False
    
    print("✓ Benchmarking completed")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Unified RL Training Pipeline for AI-PODEM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_rl_pipeline.py --all
  
  # Run only experience collection
  python scripts/run_rl_pipeline.py --collect
  
  # Run training and benchmarking
  python scripts/run_rl_pipeline.py --train --benchmark
  
  # Custom parameters
  python scripts/run_rl_pipeline.py --all --max_faults 200 --epochs 30
        """
    )
    
    # Stage selection
    stage_group = parser.add_argument_group('Stage Selection')
    stage_group.add_argument('--all', action='store_true',
                            help='Run all stages (collect, train, benchmark)')
    stage_group.add_argument('--collect', action='store_true',
                            help='Run experience collection')
    stage_group.add_argument('--train', action='store_true',
                            help='Run model training')
    stage_group.add_argument('--benchmark', action='store_true',
                            help='Run benchmarking')
    
    # Experience collection parameters
    collect_group = parser.add_argument_group('Experience Collection Parameters')
    collect_group.add_argument('--max_faults', type=int, default=100,
                              help='Maximum faults per circuit (default: 100)')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs', type=int, default=20,
                            help='Number of training epochs (default: 20)')
    train_group.add_argument('--batch_size', type=int, default=32,
                            help='Training batch size (default: 32)')
    train_group.add_argument('--lr', type=float, default=1e-4,
                            help='Learning rate (default: 1e-4)')
    train_group.add_argument('--model', type=str,
                            help='Pretrained model path to continue training')
    
    # Output parameters
    output_group = parser.add_argument_group('Output Parameters')
    output_group.add_argument('--output_model', type=str,
                             help='Output model path (default: checkpoints/reconv_rl_model.pt)')
    
    args = parser.parse_args()
    
    # If no stage specified, show help
    if not (args.all or args.collect or args.train or args.benchmark):
        parser.print_help()
        return 1
    
    # Determine which stages to run
    run_collect = args.all or args.collect
    run_train = args.all or args.train
    run_bench = args.all or args.benchmark
    
    # Print pipeline configuration
    print_section("RL Training Pipeline Configuration")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nStages to run:")
    print(f"  - Experience Collection: {'✓' if run_collect else '✗'}")
    print(f"  - Model Training:        {'✓' if run_train else '✗'}")
    print(f"  - Benchmarking:          {'✓' if run_bench else '✗'}")
    print(f"\nParameters:")
    print(f"  - Max faults per circuit: {args.max_faults}")
    print(f"  - Training epochs:        {args.epochs}")
    print(f"  - Batch size:             {args.batch_size}")
    print(f"  - Learning rate:          {args.lr}")
    
    # Run stages
    success = True
    
    if run_collect:
        if not run_experience_collection(args):
            success = False
            if args.all:
                print("\n❌ Pipeline aborted due to collection failure")
                return 1
    
    if run_train and success:
        if not run_training(args):
            success = False
            if args.all:
                print("\n❌ Pipeline aborted due to training failure")
                return 1
    
    if run_bench and success:
        if not run_benchmark(args):
            success = False
    
    # Final summary
    print_section("Pipeline Summary")
    if success:
        print("✓ All requested stages completed successfully!")
        return 0
    else:
        print("❌ Some stages failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
