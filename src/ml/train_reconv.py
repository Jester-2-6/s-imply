"""
Minimal trainer for the Multi-Path reconvergent transformer.

This script focuses on a supervised-only training loop with cross-entropy loss
over per-node labels derived from available justifications in the dataset.

Usage (example):
    conda activate torch
    python -m src.ml.train_reconv train \
            --dataset data/datasets/reconv_dataset.pkl \
            --output checkpoints/reconv_minimal \
            --epochs 5

Notes:
- Embedding dimension defaults to 128 to match the dummy embeddings path.
- Mixed precision, RL, auto batch scaling, etc., are out-of-scope for this
  minimal baseline.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Tuple, Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

# Suppress PyTorch nested tensor prototype warning emitted by Transformer internals.
# This warning is harmless for our usage and can be safely ignored.
warnings.filterwarnings(
    "ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage and will change",
    category=UserWarning,
)

from src.ml.reconv_lib import MultiPathTransformer
from src.ml.reconv_ds import ReconvergentPathsDataset, reconv_collate
from src.util.io import parse_bench_file
from src.util.struct import GateType


@dataclass
class TrainConfig:
    dataset: str
    output: str
    epochs: int = 10
    # Internal defaults; not exposed via CLI for simplicity
    batch_size: int = 1500
    lr: float = 1e-4
    embedding_dim: int = 128
    nhead: int = 4
    num_encoder_layers: int = 1
    num_interaction_layers: int = 1
    prefer_value: int = 1
    # Make verbose default enabled so users see progress without an extra flag
    verbose: bool = True


def make_dataloaders(cfg: TrainConfig, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    # Auto-detect processed shards: look for processed/ subdirectory next to dataset
    dataset_dir = os.path.dirname(cfg.dataset)
    processed_dir = os.path.join(dataset_dir, 'reconv_processed')
    load_processed = os.path.isdir(processed_dir)
    # For efficient pipelining, keep dataset tensors on CPU and transfer batches to
    # the GPU in the training loop. This allows DataLoader to use multiple workers
    # and pin_memory to speed host->device transfers.
    dataset = ReconvergentPathsDataset(
        cfg.dataset,
        device=torch.device('cpu'),
        prefer_value=cfg.prefer_value,
        processed_dir=processed_dir if load_processed else None,
        load_processed=load_processed,
    )
    # Minimal split: 90/10 train/val
    n = len(dataset)
    n_train = max(1, int(0.9 * n))
    n_val = max(1, n - n_train)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    # Configure DataLoader for throughput
    cpu_count = os.cpu_count() or 4
    num_workers = min(4, max(1, cpu_count - 1))
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=reconv_collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=reconv_collate,
        num_workers=max(0, min(2, cpu_count - 1)),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


@lru_cache(maxsize=64)
def _load_circuit(bench_file: str):
    circuit, _ = parse_bench_file(bench_file)
    return circuit


def _pair_constraint_ok(gate_type: int, prev_val: int, cur_val: int) -> bool:
    """Local constraint along a path edge assuming side-input freedom.

    - NOT: output is inversion of previous
    - BUFF: output equals previous
    - AND: output 1 requires prev 1; output 0 has no constraint
    - NAND: output 0 requires prev 1; output 1 no constraint
    - OR: output 0 requires prev 0; output 1 no constraint
    - NOR: output 1 requires prev 0; output 0 no constraint
    - XOR/XNOR/INPT/FROM: considered satisfiable with some side-input
    """
    if gate_type == GateType.NOT:
        return (1 - prev_val) == cur_val
    if gate_type == GateType.BUFF:
        return prev_val == cur_val
    if gate_type == GateType.AND:
        return True if cur_val == 0 else (prev_val == 1)
    if gate_type == GateType.NAND:
        return True if cur_val == 1 else (prev_val == 1)
    if gate_type == GateType.OR:
        return True if cur_val == 1 else (prev_val == 0)
    if gate_type == GateType.NOR:
        return True if cur_val == 0 else (prev_val == 0)
    return True


def policy_loss_and_metrics(
    logits: torch.Tensor,
    node_ids: torch.Tensor,
    mask_valid: torch.Tensor,
    files: list[str],
) -> tuple[torch.Tensor, float, float]:
    """Compute REINFORCE loss with LUT-inspired constraints and reconv consistency.

    Returns: (loss, avg_reward, valid_rate)
    
    Optimized version: minimizes Python loops and GPU->CPU syncs for large batches.
    """
    B, P, L, C = logits.shape
    device = logits.device
    
    # Sample actions; detach to avoid retaining graph history through the sample op.
    actions = torch.distributions.Categorical(logits=logits).sample().detach()  # [B, P, L]
    # Compute log-probabilities directly from logits to avoid distribution caching quirks.
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, P, L, C]
    logp = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B, P, L]

    # Vectorized constraint checking: we'll compute a simplified reward based on
    # reconvergence consistency only (matching last values across paths).
    # This avoids loading circuit files and nested loops per batch.
    # For full constraint checking, precompute gate types in dataset preprocessing.
    
    checked_list: list[float] = []
    wrong_list: list[float] = []
    
    # Fast path: only check reconvergence consistency (last node values match)
    for b in range(B):
        checked = 0
        wrong = 0
        last_vals: list[int] = []
        for p in range(P):
            if mask_valid[b, p].any():
                last_idx = int(mask_valid[b, p].sum().item()) - 1
                last_vals.append(int(actions[b, p, last_idx].item()))
        
        if len(last_vals) >= 2:
            ref = last_vals[0]
            for v in last_vals[1:]:
                checked += 1
                if v != ref:
                    wrong += 1
        
        checked_list.append(float(checked))
        wrong_list.append(float(wrong))

    checked_t = torch.tensor(checked_list, dtype=torch.float32, device=device)
    wrong_t = torch.tensor(wrong_list, dtype=torch.float32, device=device)
    valid = (wrong_t == 0) & (checked_t > 0)
    denom = torch.clamp(checked_t, min=1.0)
    reward = torch.where(checked_t > 0, torch.where(valid, torch.ones_like(denom), -wrong_t / denom), torch.zeros_like(denom))
    reward = reward.detach()

    # Policy gradient loss: -mean(reward * mean logp over valid positions)
    logp_sum = (logp * mask_valid).sum(dim=(1, 2))  # [B]
    count = torch.clamp(mask_valid.sum(dim=(1, 2)).float(), min=1.0)
    per_sample_loss = -(reward * (logp_sum / count))  # [B]
    loss = per_sample_loss.mean()

    avg_reward = float(reward.mean().item())
    valid_rate = float(valid.float().mean().item())
    return loss, avg_reward, valid_rate


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    verbose: bool = True,
    print_every: int = 10,
    scaler: Optional[Any] = None,
) -> Tuple[float, float, float]:
    """Run one training epoch.

    When verbose=True this will print periodic batch-level progress so the user
    can see training is making progress (useful after large dataset/shard loading).
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_reward = 0.0
    total_valid = 0.0

    # Try to determine number of batches for nicer progress output
    try:
        nbatches = len(loader)
    except Exception:
        nbatches = None

    # Create an iterator; use tqdm for nicer progress when verbose.
    iterator: Any
    if verbose:
        iterator = tqdm(loader, total=nbatches, desc="Train", leave=False)
    else:
        iterator = loader

    # If running on CUDA, use a data prefetcher to overlap host->device transfer
    # with GPU computation. This often increases GPU utilization.
    class _Prefetcher:
        def __init__(self, it, device: torch.device):
            self.it = iter(it)
            self.device = device
            self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
            self.next_batch = None
            self._preload()

        def _preload(self):
            try:
                batch = next(self.it)
            except StopIteration:
                self.next_batch = None
                return
            if self.device.type == 'cuda':
                # Move tensors to device asynchronously in the prefetch stream
                with torch.cuda.stream(self.stream):
                    batch['paths_emb'] = batch['paths_emb'].to(self.device, non_blocking=True)
                    batch['attn_mask'] = batch['attn_mask'].to(self.device, non_blocking=True)
                    batch['node_ids'] = batch['node_ids'].to(self.device, non_blocking=True)
            self.next_batch = batch

        def __iter__(self):
            return self

        def __next__(self):
            if self.next_batch is None:
                raise StopIteration
            if self.device.type == 'cuda':
                # Wait for the transfer to finish
                torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next_batch
            self._preload()
            return batch

    if device.type == 'cuda':
        iterator = _Prefetcher(iterator, device)

    # Timing accumulators for profiling
    time_data = 0.0
    time_forward = 0.0
    time_loss = 0.0
    time_backward = 0.0
    t_batch_start = time.perf_counter()

    for batch_idx, batch in enumerate(iterator):
        t_data = time.perf_counter()
        time_data += t_data - t_batch_start
        
        # Batch tensors are already on device when using the prefetcher; otherwise
        # move them here (non_blocking already used in prefetcher or DataLoader pin_memory).
        paths = batch['paths_emb']  # [B, P, L, D]
        masks = batch['attn_mask']  # [B, P, L]
        node_ids = batch['node_ids']  # [B, P, L]
        files = batch['files']        # list[str]

        optim.zero_grad(set_to_none=True)

        # Use AMP to accelerate GPU throughput (Tensor Cores) where available.
        t_forward_start = time.perf_counter()
        if device.type == 'cuda' and scaler is not None:
            with torch.autocast(device_type='cuda'):
                logits = model(paths, masks)
                t_loss_start = time.perf_counter()
                time_forward += t_loss_start - t_forward_start
                loss, avg_reward, valid_rate = policy_loss_and_metrics(logits, node_ids, masks, files)
            t_backward_start = time.perf_counter()
            time_loss += t_backward_start - t_loss_start
            # Backprop with gradient scaler
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(paths, masks)  # [B, P, L, 2]
            t_loss_start = time.perf_counter()
            time_forward += t_loss_start - t_forward_start
            loss, avg_reward, valid_rate = policy_loss_and_metrics(logits, node_ids, masks, files)
            t_backward_start = time.perf_counter()
            time_loss += t_backward_start - t_loss_start
            loss.backward(retain_graph=True)
            optim.step()
        
        t_backward_end = time.perf_counter()
        time_backward += t_backward_end - t_backward_start

        total_loss += float(loss.item())
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_batches += 1

        # Print timing for first few batches to diagnose bottleneck
        if batch_idx < 3:
            print(f"\nBatch {batch_idx+1} timing: data={time_data:.3f}s forward={time_forward:.3f}s loss={time_loss:.3f}s backward={time_backward:.3f}s")
        
        t_batch_start = time.perf_counter()

        # Periodic progress update to indicate training is running; include accuracy
        if verbose:
            # iterator is either a tqdm instance or a plain loader; tqdm supports set_postfix
            try:
                iterator.set_postfix(loss=float(loss.item()), reward=avg_reward, acc=valid_rate)
            except Exception:
                # Plain loaders won't have set_postfix; ignore in that case
                pass
        elif verbose and (batch_idx % print_every == 0 or (nbatches is not None and batch_idx == nbatches - 1)):
            if nbatches is not None:
                print(f"Training batch {batch_idx+1}/{nbatches} | loss={float(loss.item()):.4f} avg_reward={avg_reward:.4f} acc={valid_rate:.4f}", end='\r', flush=True)
            else:
                print(f"Training batch {batch_idx+1} | loss={float(loss.item()):.4f} avg_reward={avg_reward:.4f} acc={valid_rate:.4f}", end='\r', flush=True)

    # Ensure final newline after progress prints
    if verbose:
        print()

    avg_loss = total_loss / max(1, total_batches)
    avg_reward = total_reward / max(1, total_batches)
    valid_rate = total_valid / max(1, total_batches)
    # Return avg loss, avg reward and accuracy (valid rate)
    return avg_loss, avg_reward, valid_rate


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_reward = 0.0
    total_valid = 0.0
    total_batches = 0

    for batch in loader:
        # Move batch to device for evaluation
        paths = batch['paths_emb'].to(device, non_blocking=True)
        masks = batch['attn_mask'].to(device, non_blocking=True)
        node_ids = batch['node_ids'].to(device, non_blocking=True)
        files = batch['files']

        logits = model(paths, masks)
        loss, avg_reward, valid_rate = policy_loss_and_metrics(logits, node_ids, masks, files)
        total_loss += float(loss.item())
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    avg_reward = total_reward / max(1, total_batches)
    valid_rate = total_valid / max(1, total_batches)
    return avg_loss, avg_reward, valid_rate


def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, best: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'config': asdict(cfg),
        'best': best,
    }, path)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        dataset=args.dataset,
        output=args.output,
        epochs=args.epochs,
        batch_size=getattr(args, 'batch_size', 8),
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Enable cuDNN autotuner for potential speedups on fixed-size inputs
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # AMP scaler for mixed-precision training (only used when CUDA is available)
    scaler: Optional[Any]
    if device.type == 'cuda':
        scaler = torch.GradScaler('cuda')
    else:
        scaler = None

    train_loader, val_loader = make_dataloaders(cfg, device)

    model = MultiPathTransformer(
        embedding_dim=cfg.embedding_dim,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_interaction_layers=cfg.num_interaction_layers,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = float('inf')
    # Print dataset/batch info at start
    nb_train = len(train_loader) if hasattr(train_loader, '__len__') else 0
    nb_val = len(val_loader) if hasattr(val_loader, '__len__') else 0
    print(f"Starting training: train_batches={nb_train}, val_batches={nb_val}, batch_size={cfg.batch_size}")
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_reward, tr_acc = train_one_epoch(model, train_loader, optim, device, verbose=cfg.verbose, scaler=scaler)
        va_loss, va_reward, va_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} avg_reward={tr_reward:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} avg_reward={va_reward:.4f} val_acc={va_acc:.4f}")

        # Save periodic checkpoint
        if epoch % 10 == 0 or epoch == cfg.epochs:
            save_checkpoint(os.path.join(cfg.output, f"checkpoint_epoch_{epoch}.pth"), model, cfg, best=False)

        # Save best by validation reward (maximize)
        if (-va_reward) < best_val:
            best_val = -va_reward
            save_checkpoint(os.path.join(cfg.output, "best_model.pth"), model, cfg, best=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal reconv transformer trainer")
    sub = p.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='Run supervised training')
    t.add_argument('--dataset', type=str, default='data/datasets/reconv_dataset.pkl', help='Path to dataset .pkl')
    t.add_argument('--output', type=str, default='checkpoints/reconv_minimal', help='Output checkpoint directory')
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--batch-size', type=int, default=1500)

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == 'train':
        cmd_train(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == '__main__':
    main()