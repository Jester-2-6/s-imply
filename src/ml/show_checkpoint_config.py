"""Utility to print saved TrainConfig from a reconv checkpoint."""

from __future__ import annotations

import argparse
from typing import Any

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Print TrainConfig stored in a reconv checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint .pth file")
    args = parser.parse_args()

    state: Any = torch.load(args.checkpoint, map_location="cpu")
    config = state.get("config") if isinstance(state, dict) else None
    if config is None:
        print("No config dictionary found in checkpoint")
        return

    for key in sorted(config.keys()):
        print(f"{key}: {config[key]}")


if __name__ == "__main__":
    main()
