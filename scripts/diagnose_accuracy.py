#!/usr/bin/env python3
"""
Diagnostic script to understand why accuracy plateaus at ~23% despite reward ~1.0.

Analyzes:
1. Edge error distribution: how many edges wrong per sample?
2. Gate type breakdown: which gate types fail most?
3. Reconvergence analysis: how often do path endpoints disagree?
4. Trivial prediction analysis: is the model predicting all-0 or all-1?
5. Position analysis: do errors cluster at certain path positions?
"""
import os, sys, torch, argparse
import numpy as np
from collections import Counter, defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.train_reconv import (
    TrainConfig, make_dataloaders, resolve_gate_types
)
from src.util.struct import GateType


GATE_NAMES = {
    GateType.INPT: 'INPUT', GateType.AND: 'AND', GateType.OR: 'OR',
    GateType.NOT: 'NOT', GateType.NAND: 'NAND', GateType.NOR: 'NOR',
    GateType.BUFF: 'BUFF', GateType.XOR: 'XOR', GateType.XNOR: 'XNOR',
}


def diagnose(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    from src.ml.reconv_lib import MultiPathTransformer
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt.get('config', {})
    
    # Get model dims from checkpoint
    model_dim = cfg_dict.get('model_dim', 512)
    nhead = cfg_dict.get('nhead', 4)
    enc_layers = cfg_dict.get('num_encoder_layers', cfg_dict.get('enc_layers', 3))
    int_layers = cfg_dict.get('num_interaction_layers', cfg_dict.get('int_layers', 3))
    ffn_dim = cfg_dict.get('dim_feedforward', cfg_dict.get('ffn_dim', 512))
    
    print(f"Model config: dim={model_dim}, nhead={nhead}, enc={enc_layers}, int={int_layers}, ffn={ffn_dim}")
    
    # Build config for data loading
    cfg = TrainConfig(
        dataset='',
        output='',
        processed_dir=args.data_dir,
        batch_size=args.batch_size,
        model_dim=model_dim,
        nhead=nhead,
        num_encoder_layers=enc_layers,
        num_interaction_layers=int_layers,
        dim_feedforward=ffn_dim,
        use_gate_type_embedding=cfg_dict.get('use_gate_type_embedding', True),
        verbose=False,
    )
    
    train_loader, val_loader = make_dataloaders(cfg, device)
    loader = val_loader if args.split == 'val' else train_loader
    
    # Get embedding dim from first batch
    sample_batch = next(iter(loader))
    emb_dim = sample_batch['paths_emb'].shape[-1]
    
    # Ensure model_dim divisible by nhead
    if model_dim % nhead != 0:
        model_dim = ((model_dim // nhead) + 1) * nhead
    
    model = MultiPathTransformer(
        input_dim=emb_dim,
        model_dim=model_dim,
        nhead=nhead,
        num_encoder_layers=enc_layers,
        num_interaction_layers=int_layers,
        dim_feedforward=ffn_dim,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    print(f"\nAnalyzing {args.max_batches} batches from {args.split} split...\n")
    
    # Accumulators
    total_samples = 0
    total_edges = 0
    total_wrong_edges = 0
    
    edge_errors_per_sample = []
    edges_checked_per_sample = []
    reconv_failures = 0
    trivial_count = 0
    fully_valid = 0
    sat_count = 0
    
    gate_type_errors = Counter()
    gate_type_total = Counter()
    
    position_errors = defaultdict(int)
    position_total = defaultdict(int)
    
    pred_value_counts = Counter()  # 0 vs 1 predictions
    
    path_len_errors = defaultdict(list)  # path_len -> [error_count, ...]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
                
            paths = batch['paths_emb'].to(device)
            masks = batch['attn_mask'].to(device)
            node_ids = batch['node_ids'].to(device)
            files = batch['files']
            
            gtypes = resolve_gate_types(node_ids, files, device)
            
            logits, solv_logits = model(paths, masks, gate_types=gtypes if cfg.use_gate_type_embedding else None)
            actions = logits.argmax(dim=-1)  # [B, P, L]
            
            B, P, L = actions.shape
            
            # --- Edge consistency ---
            valid_edges = masks[:, :, 1:] & masks[:, :, :-1]
            prev_vals = actions[:, :, :-1]
            cur_vals = actions[:, :, 1:]
            gt_cur = gtypes[:, :, 1:]
            
            edge_ok = torch.ones_like(prev_vals, dtype=torch.bool)
            
            m = gt_cur == GateType.NOT
            edge_ok[m] &= (cur_vals[m] == (1 - prev_vals[m]))
            m = gt_cur == GateType.BUFF
            edge_ok[m] &= (cur_vals[m] == prev_vals[m])
            m = gt_cur == GateType.AND
            edge_ok[m] &= (cur_vals[m] <= prev_vals[m])
            m = gt_cur == GateType.NAND
            edge_ok[m] &= (cur_vals[m] >= (1 - prev_vals[m]))
            m = gt_cur == GateType.OR
            edge_ok[m] &= (cur_vals[m] >= prev_vals[m])
            m = gt_cur == GateType.NOR
            edge_ok[m] &= (cur_vals[m] <= (1 - prev_vals[m]))
            
            wrong_edges = (~edge_ok) & valid_edges
            
            # Per-sample stats
            local_wrong = wrong_edges.sum(dim=(1, 2))
            checked = valid_edges.sum(dim=(1, 2))
            
            # --- Reconvergence ---
            path_len = masks.long().sum(dim=-1)
            last_idx = (path_len - 1).clamp(min=0)
            last_vals = actions.gather(2, last_idx.unsqueeze(-1)).squeeze(-1)
            path_valid = (path_len > 0)
            
            neg_inf = -999.0
            pos_inf = 999.0
            lv_f = last_vals.float()
            vm_f = path_valid.float()
            max_v = (lv_f * vm_f + neg_inf * (1 - vm_f)).max(dim=-1).values
            min_v = (lv_f * vm_f + pos_inf * (1 - vm_f)).min(dim=-1).values
            has_valid = (path_valid.sum(dim=-1) > 0)
            reconv_fail = (min_v < max_v) & has_valid
            
            # --- Solvability ---
            if solv_logits is not None:
                pred_solv = (solv_logits.squeeze(-1) > 0).long()
            
            # --- Per-sample analysis ---
            for b in range(B):
                n_wrong = local_wrong[b].item()
                n_checked = checked[b].item()
                is_reconv_fail = reconv_fail[b].item()
                
                edge_errors_per_sample.append(n_wrong)
                edges_checked_per_sample.append(n_checked)
                total_edges += n_checked
                total_wrong_edges += n_wrong
                total_samples += 1
                
                if is_reconv_fail:
                    reconv_failures += 1
                
                # Trivial check
                valid_actions = actions[b][masks[b]]
                n_zeros = (valid_actions == 0).sum().item()
                n_ones = (valid_actions == 1).sum().item()
                total_valid_nodes = n_zeros + n_ones
                if total_valid_nodes > 0:
                    pred_value_counts[0] += n_zeros
                    pred_value_counts[1] += n_ones
                    if n_zeros == total_valid_nodes or n_ones == total_valid_nodes:
                        trivial_count += 1
                
                is_sat = True  # Default
                is_valid = (n_wrong == 0) and (not is_reconv_fail) and (n_checked > 0) and is_sat
                if is_valid:
                    fully_valid += 1
                if is_sat:
                    sat_count += 1
                    
                # Path length vs errors
                avg_path_len = path_len[b][path_valid[b]].float().mean().item() if path_valid[b].any() else 0
                path_len_errors[int(avg_path_len)].append(n_wrong)
                
            # --- Gate type breakdown ---
            for gt_val in GATE_NAMES.keys():
                m = (gt_cur == gt_val) & valid_edges
                total_gt = m.sum().item()
                wrong_gt = (wrong_edges & m).sum().item()
                gate_type_total[gt_val] += total_gt
                gate_type_errors[gt_val] += wrong_gt
            
            # --- Position analysis ---
            for pos in range(wrong_edges.shape[-1]):
                pos_mask = valid_edges[:, :, pos]
                position_total[pos] += pos_mask.sum().item()
                pos_wrong = wrong_edges[:, :, pos]
                position_errors[pos] += pos_wrong.sum().item()
    
    # === REPORT ===
    print("=" * 70)
    print(f"DIAGNOSTIC REPORT — {total_samples} samples from {args.split}")
    print("=" * 70)
    
    print(f"\n--- Overall Metrics ---")
    print(f"Total samples: {total_samples}")
    print(f"Fully valid (acc): {fully_valid}/{total_samples} = {fully_valid/max(1,total_samples):.3f}")
    print(f"SAT samples: {sat_count}")
    print(f"Reconv failures: {reconv_failures}/{total_samples} = {reconv_failures/max(1,total_samples):.3f}")
    print(f"Trivial (all-0 or all-1): {trivial_count}/{total_samples} = {trivial_count/max(1,total_samples):.3f}")
    print(f"Edge accuracy: {(total_edges - total_wrong_edges)}/{total_edges} = {1 - total_wrong_edges/max(1,total_edges):.4f}")
    
    print(f"\n--- Prediction Value Distribution ---")
    total_preds = sum(pred_value_counts.values())
    for v in sorted(pred_value_counts.keys()):
        print(f"  Value {v}: {pred_value_counts[v]}/{total_preds} = {pred_value_counts[v]/max(1,total_preds):.3f}")
    
    print(f"\n--- Edge Error Distribution (per sample) ---")
    err_arr = np.array(edge_errors_per_sample)
    print(f"  Mean: {err_arr.mean():.2f}")
    print(f"  Median: {np.median(err_arr):.1f}")
    print(f"  Std: {err_arr.std():.2f}")
    print(f"  Min: {err_arr.min()}, Max: {err_arr.max()}")
    print(f"  Zero-error samples: {(err_arr == 0).sum()}/{total_samples} = {(err_arr==0).sum()/total_samples:.3f}")
    for threshold in [0, 1, 2, 3, 5, 10, 20]:
        count = (err_arr <= threshold).sum()
        print(f"  ≤{threshold} errors: {count}/{total_samples} = {count/total_samples:.3f}")
    
    print(f"\n--- Edges Checked per Sample ---")
    chk_arr = np.array(edges_checked_per_sample)
    print(f"  Mean: {chk_arr.mean():.1f}, Median: {np.median(chk_arr):.1f}, Max: {chk_arr.max()}")
    
    print(f"\n--- Gate Type Error Rates ---")
    for gt_val in sorted(GATE_NAMES.keys()):
        total = gate_type_total[gt_val]
        wrong = gate_type_errors[gt_val]
        if total > 0:
            rate = wrong / total
            print(f"  {GATE_NAMES[gt_val]:6s}: {wrong:7d}/{total:7d} = {rate:.4f} error rate")
    
    print(f"\n--- Position Error Rates (first 20) ---")
    for pos in range(min(20, max(position_total.keys()) + 1)):
        total = position_total.get(pos, 0)
        wrong = position_errors.get(pos, 0)
        if total > 0:
            rate = wrong / total
            print(f"  Pos {pos:3d}: {wrong:6d}/{total:6d} = {rate:.4f}")
    
    print(f"\n--- Path Length vs Error Count ---")
    for pl in sorted(path_len_errors.keys()):
        errs = path_len_errors[pl]
        if len(errs) >= 5:  # Only show bins with enough samples
            mean_err = np.mean(errs)
            zero_frac = sum(1 for e in errs if e == 0) / len(errs)
            print(f"  PathLen ~{pl:3d}: n={len(errs):5d}, mean_errors={mean_err:.2f}, zero_err_frac={zero_frac:.3f}")
    
    # --- Detailed sample-level breakdown for first few failures ---
    if args.verbose:
        print(f"\n--- Detailed Failure Analysis (first 10 failing samples) ---")
        fail_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= args.max_batches or fail_count >= 10:
                    break
                paths = batch['paths_emb'].to(device)
                masks = batch['attn_mask'].to(device)
                node_ids = batch['node_ids'].to(device)
                files = batch['files']
                gtypes = resolve_gate_types(node_ids, files, device)
                logits, solv_logits = model(paths, masks, gate_types=gtypes if cfg.use_gate_type_embedding else None)
                actions = logits.argmax(dim=-1)
                
                B, P, L = actions.shape
                valid_edges = masks[:, :, 1:] & masks[:, :, :-1]
                prev_vals = actions[:, :, :-1]
                cur_vals = actions[:, :, 1:]
                gt_cur = gtypes[:, :, 1:]
                
                edge_ok = torch.ones_like(prev_vals, dtype=torch.bool)
                m = gt_cur == GateType.NOT; edge_ok[m] &= (cur_vals[m] == (1 - prev_vals[m]))
                m = gt_cur == GateType.BUFF; edge_ok[m] &= (cur_vals[m] == prev_vals[m])
                m = gt_cur == GateType.AND; edge_ok[m] &= (cur_vals[m] <= prev_vals[m])
                m = gt_cur == GateType.NAND; edge_ok[m] &= (cur_vals[m] >= (1 - prev_vals[m]))
                m = gt_cur == GateType.OR; edge_ok[m] &= (cur_vals[m] >= prev_vals[m])
                m = gt_cur == GateType.NOR; edge_ok[m] &= (cur_vals[m] <= (1 - prev_vals[m]))
                
                wrong_edges = (~edge_ok) & valid_edges
                local_wrong = wrong_edges.sum(dim=(1, 2))
                
                for b in range(B):
                    if local_wrong[b].item() > 0 and fail_count < 10:
                        fail_count += 1
                        print(f"\n  Sample [{batch_idx}:{b}] — {local_wrong[b].item()} edge errors")
                        print(f"    File: {files[b]}")
                        for p in range(P):
                            pl = masks[b, p].sum().item()
                            if pl == 0: continue
                            path_errs = wrong_edges[b, p].sum().item()
                            if path_errs > 0:
                                print(f"    Path {p} (len={int(pl)}, errors={path_errs}):")
                                for pos in range(int(pl) - 1):
                                    if wrong_edges[b, p, pos].item():
                                        gt = gtypes[b, p, pos+1].item()
                                        gname = GATE_NAMES.get(gt, f'?{gt}')
                                        prev = prev_vals[b, p, pos].item()
                                        cur = cur_vals[b, p, pos].item()
                                        nid_prev = node_ids[b, p, pos].item()
                                        nid_cur = node_ids[b, p, pos+1].item()
                                        prob0 = torch.softmax(logits[b, p, pos+1], dim=-1)[0].item()
                                        prob1 = 1 - prob0
                                        print(f"      pos {pos}->{pos+1}: node {nid_prev}->{nid_cur} "
                                              f"gate={gname} prev={prev} cur={cur} "
                                              f"p(0)={prob0:.3f} p(1)={prob1:.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Diagnose accuracy plateau")
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    p.add_argument('--data-dir', default='data/datasets/shards_anchored', help='Processed data directory')
    p.add_argument('--split', default='val', choices=['train', 'val'])
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-batches', type=int, default=20, help='Max batches to analyze')
    p.add_argument('--verbose', action='store_true', help='Show detailed per-sample failures')
    args = p.parse_args()
    diagnose(args)
