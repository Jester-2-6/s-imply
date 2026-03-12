"""
Patch existing processed shards to add anchor supervision fields.

The reconvergence node is always the LAST valid position on every path
(paths are ordered [stem → intermediate → reconv]).  We extract this from
the attn_mask already present in each shard — no solver or raw pickle needed.

Adds/overwrites:
  anchor_p   [N] int64  — path index of anchor (always 0; all paths share terminal)
  anchor_l   [N] int64  — position index of the last valid node on path 0
  anchor_v   [N] int64  — target value (--prefer-value, default 1)
  solvability [N] int64 — 0 (SAT) for all; kept for format compatibility

Also injects the anchor's target value into paths_emb logic dims so that
the model sees the "given" target at the terminal node when anchor_hint is
active:
  paths_emb[b, 0, anchor_l[b], D-3] = 0   (val=0 indicator)
  paths_emb[b, 0, anchor_l[b], D-2] = 1   (val=1 indicator if prefer_value=1)
  paths_emb[b, 0, anchor_l[b], D-1] = 0   (unknown indicator cleared)

Usage:
    python scripts/patch_shard_anchors.py \\
        --shard-dir /home/local1/cache-cw/processed_reconv/ \\
        [--prefer-value 1] \\
        [--output-dir /path/to/output]   # omit to patch in-place
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def patch_shard(shard_path: str, prefer_value: int, out_path: str) -> dict:
    shard = torch.load(shard_path, weights_only=False, map_location="cpu")

    am: torch.Tensor = shard["attn_mask"]   # [N, P, L]
    pe: torch.Tensor = shard["paths_emb"]   # [N, P, L, D]  float16

    N, P, L = am.shape
    D = pe.shape[-1]

    # Length of each path: [N, P]
    path_lens = am.long().sum(dim=2)   # [N, P]
    # Terminal position on path 0 (same terminal across all paths by construction)
    term_pos = (path_lens[:, 0] - 1).clamp(min=0)  # [N]
    valid = path_lens[:, 0] >= 1                     # [N] bool

    anchor_p = torch.zeros(N, dtype=torch.long)
    anchor_l = term_pos.clone()
    # Mixed (-1): alternate 0/1 by sample index for balanced supervision signal.
    # Constant (0 or 1): assign the same target to every sample.
    if prefer_value == -1:
        anchor_v = torch.where(
            torch.arange(N) % 2 == 0,
            torch.ones(N, dtype=torch.long),
            torch.zeros(N, dtype=torch.long),
        )
    else:
        anchor_v = torch.full((N,), prefer_value, dtype=torch.long)
    solvability = torch.zeros(N, dtype=torch.long)

    # Invalidate zero-length paths (shouldn't happen but be defensive)
    anchor_p[~valid] = -1
    anchor_l[~valid] = -1

    # Inject anchor value into paths_emb logic dims at the terminal node.
    # D-3: val=0 indicator, D-2: val=1 indicator, D-1: unknown indicator
    b_idx = torch.arange(N)[valid]
    l_idx = anchor_l[valid]
    av_at_valid = anchor_v[valid]

    pe_f = pe.float()
    # Clear previous logic dims first
    pe_f[b_idx, 0, l_idx, D - 3] = 0.0
    pe_f[b_idx, 0, l_idx, D - 2] = 0.0
    pe_f[b_idx, 0, l_idx, D - 1] = 0.0
    is_one = av_at_valid == 1
    pe_f[b_idx[is_one],  0, l_idx[is_one],  D - 2] = 1.0   # val=1: [0,1,0]
    pe_f[b_idx[~is_one], 0, l_idx[~is_one], D - 3] = 1.0   # val=0: [1,0,0]

    shard["paths_emb"] = pe_f.half()
    shard["anchor_p"] = anchor_p
    shard["anchor_l"] = anchor_l
    shard["anchor_v"] = anchor_v
    shard["solvability"] = solvability

    # Add empty constraint fields for format compatibility
    if "constraint_mask" not in shard:
        shard["constraint_mask"] = torch.zeros(N, P, L, dtype=torch.bool)
    if "constraint_vals" not in shard:
        shard["constraint_vals"] = torch.zeros(N, P, L, dtype=torch.long)

    torch.save(shard, out_path)

    n_valid = int(valid.sum().item())
    return {"n_valid": n_valid, "n_total": N}


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch shard anchors for training")
    parser.add_argument(
        "--shard-dir",
        type=str,
        default="/home/local1/cache-cw/processed_reconv/",
        help="Directory containing shard_NNNNN.pt files",
    )
    parser.add_argument(
        "--prefer-value",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help=(
            "Target value for the reconvergence (terminal) node. "
            "0 or 1 assigns the same value to all samples. "
            "-1 (default) alternates 0/1 by sample index for balanced training."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Omit or empty to patch in-place.",
    )
    args = parser.parse_args()

    shard_dir = args.shard_dir
    out_dir = args.output_dir or shard_dir

    if out_dir != shard_dir:
        os.makedirs(out_dir, exist_ok=True)

    shards = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt"))
    if not shards:
        print(f"No .pt files found in {shard_dir}")
        sys.exit(1)

    print(f"Patching {len(shards)} shards in '{shard_dir}' → '{out_dir}'")
    print(f"  prefer_value={args.prefer_value}  (anchor target at terminal node)")
    print()

    total_valid = total_n = 0
    for i, fname in enumerate(shards):
        in_path = os.path.join(shard_dir, fname)
        out_path = os.path.join(out_dir, fname)
        stats = patch_shard(in_path, args.prefer_value, out_path)
        total_valid += stats["n_valid"]
        total_n += stats["n_total"]
        print(
            f"  [{i + 1:3d}/{len(shards)}] {fname}  "
            f"valid_anchors={stats['n_valid']}/{stats['n_total']}"
        )

    print()
    print(f"Done. {total_valid}/{total_n} samples have valid anchors.")


if __name__ == "__main__":
    main()
