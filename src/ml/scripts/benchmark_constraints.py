import os
import time

import torch

from src.ml.train import generate_constraints


def benchmark():
    # Setup dummy data
    B = 128
    files = ["data/circuits/b14.bench"] * B

    # Check if files exist
    if not os.path.exists(files[0]):
        # Try finding one
        found = False
        for root, dirs, fs in os.walk("data"):
            for f in fs:
                if f.endswith(".bench"):
                    files = [os.path.join(root, f)] * B
                    found = True
                    break
            if found:
                break
        if not found:
            print("No .bench files found in data/")
            return

    print(f"Benchmarking with {files[0]}")

    # Dummy node IDs [B, P, L]
    P = 2
    L = 10
    node_ids = torch.randint(1, 100, (B, P, L))

    start = time.time()
    c_mask, c_vals = generate_constraints(node_ids, files, prob=0.5)
    end = time.time()

    print(f"Time for B={B}: {end-start:.4f}s")
    print(f"Per sample: {(end-start)/B*1000:.2f}ms")


if __name__ == "__main__":
    benchmark()
