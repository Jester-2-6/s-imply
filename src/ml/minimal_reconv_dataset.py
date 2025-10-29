"""
Minimal dataset wrapper for training the MultiPath reconvergent transformer.

This loader reads a pickled dataset of reconvergent path pairs and prepares
per-path, per-node embeddings compatible with `MultiPathTransformer`.

Assumptions (kept minimal by design):
- Each dataset entry is a dict with at least keys:
  - 'file': path to the source .bench circuit file (string)
  - 'info': { 'paths': [[node_ids...], [node_ids...], ...] } where node ids are
             integers or strings corresponding to gate ids in the original circuit
    - Dataset is paths-only; no justification labels are required or used.

- Embeddings are derived on-the-fly from the source circuit using the existing
  AIG conversion and embedding flow (`EmbeddingExtractor`). If DeepGate isn't
  available, dummy embeddings are produced (see src/ml/gcn.py), which is fine
  for smoke tests and minimal setups.

Outputs per item:
- paths_emb: Tensor [P, L, D] padded to the longest path length L in the sample
- attn_mask: Bool Tensor [P, L] (True for valid positions)
- node_ids: Long Tensor [P, L] with original gate ids (0 for padding)
- file: bench file path (string)
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from src.ml.embedding_extractor import EmbeddingExtractor


class ReconvergentPathsDataset(Dataset):
    """Minimal dataset for reconvergent path training.

    It lazily extracts embeddings per unique circuit file and maps original
    gate IDs to the AIG space used by the embedding backend.
    """

    def __init__(
        self,
        dataset_pickle: str,
        device: torch.device | None = None,
    prefer_value: int | None = None,
    ) -> None:
        """Create the dataset from a pickle file.

        Args:
            dataset_pickle: Path to the pickled list of samples.
            device: Optional device to place returned tensors on.
            prefer_value: Unused in paths-only setup; retained for compatibility.
        """
        super().__init__()
        self.dataset_path = dataset_pickle
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.prefer_value = 1 if prefer_value is None else int(prefer_value)

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected dataset pickle to contain a list of samples.")

        self.samples: List[Dict[str, Any]] = data
        self._embedder = EmbeddingExtractor(staging_dir="data/staging")

    def __len__(self) -> int:
        return len(self.samples)

    @lru_cache(maxsize=64)
    def _load_embeddings(self, bench_file: str) -> Tuple[torch.Tensor, Dict[str, str]]:
        """Get structural embeddings and original->AIG gate id mapping for a circuit file.

        Returns:
            struct_emb: Tensor [N, D]
            id_map: dict original_id (str) -> aig_id (str)
        """
        struct_emb, _func_emb, id_map, _orig_circuit = self._embedder.extract_embeddings(bench_file)
        # Ensure device consistency
        struct_emb = struct_emb.to(self.device)
        return struct_emb, id_map

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        bench_file_val = sample.get('file')
        if not isinstance(bench_file_val, str):
            raise ValueError("Malformed dataset sample: 'file' must be a string path.")
        bench_file: str = bench_file_val
        info: Dict[str, Any] = sample.get('info', {})
        paths: List[List[Any]] = info.get('paths', [])

        if not bench_file or not isinstance(paths, list) or len(paths) == 0:
            raise ValueError("Malformed dataset sample: missing 'file' or 'info.paths'.")

        struct_emb, id_map = self._load_embeddings(bench_file)

        # Build per-path embeddings and node id tensors
        path_vecs: List[torch.Tensor] = []
        path_node_ids: List[torch.Tensor] = []
        max_len = 0

        for path in paths:
            node_ids: List[str] = [str(n) for n in path]
            emb_list: List[torch.Tensor] = []
            ids_list: List[int] = []

            for orig_id in node_ids:
                aig_id_str = id_map.get(orig_id, orig_id)
                try:
                    idx_int = int(aig_id_str)
                except ValueError:
                    idx_int = -1

                if 0 <= idx_int < struct_emb.size(0):
                    emb_list.append(struct_emb[idx_int])
                else:
                    emb_list.append(torch.zeros(struct_emb.size(1), device=self.device))

                # Keep original node id for downstream LUT-based evaluation
                try:
                    ids_list.append(int(orig_id))
                except Exception:
                    ids_list.append(0)

            if len(emb_list) == 0:
                # Ensure at least a single padding token to avoid zero-length tensors
                emb_list.append(torch.zeros(struct_emb.size(1), device=self.device))
                ids_list.append(0)

            path_tensor = torch.stack(emb_list, dim=0)  # [L, D]
            ids_tensor = torch.tensor(ids_list, dtype=torch.long, device=self.device)  # [L]
            path_vecs.append(path_tensor)
            path_node_ids.append(ids_tensor)
            max_len = max(max_len, path_tensor.size(0))

        # Pad all paths to the same length for this sample
        P = len(path_vecs)
        D = path_vecs[0].size(1)
        padded_paths = torch.zeros(P, max_len, D, device=self.device)
        padded_ids = torch.zeros((P, max_len), dtype=torch.long, device=self.device)
        attn_mask = torch.zeros(P, max_len, dtype=torch.bool, device=self.device)

        for i, (pvec, pids) in enumerate(zip(path_vecs, path_node_ids)):
            L = pvec.size(0)
            padded_paths[i, :L] = pvec
            padded_ids[i, :L] = pids
            attn_mask[i, :L] = True

        return {
            'paths_emb': padded_paths,   # [P, L, D]
            'attn_mask': attn_mask,      # [P, L]
            'node_ids': padded_ids,      # [P, L]
            'file': bench_file,
        }


def reconv_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length per-sample tensors into a single batch.

    Pads across the sample dimension to the maximum path length in the batch.
    """
    device = batch[0]['paths_emb'].device
    P = max(item['paths_emb'].size(0) for item in batch)
    L = max(item['paths_emb'].size(1) for item in batch)
    D = batch[0]['paths_emb'].size(2)
    B = len(batch)

    paths = torch.zeros(B, P, L, D, device=device)
    masks = torch.zeros(B, P, L, dtype=torch.bool, device=device)
    node_ids = torch.zeros((B, P, L), dtype=torch.long, device=device)
    files: List[str] = []

    for b, item in enumerate(batch):
        p, L_i, d = item['paths_emb'].shape
        paths[b, :p, :L_i, :d] = item['paths_emb']
        masks[b, :p, :L_i] = item['attn_mask']
        node_ids[b, :p, :L_i] = item['node_ids']
        files.append(item['file'])

    return {
        'paths_emb': paths,
        'attn_mask': masks,
        'node_ids': node_ids,
        'files': files,
    }
