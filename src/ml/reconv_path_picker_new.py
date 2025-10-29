"""
Path picker for reconvergent path selection with transformer-based ranking.

This module provides comprehensive path selection, ranking, and analysis capabilities
for reconvergent paths in digital circuits using deep learning embeddings and
transformer-based architectures.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.utils.data import Dataset

from src.ml.embedding_extractor import EmbeddingExtractor


@dataclass
class PathMetrics:
    """Metrics for evaluating path quality."""
    length: int
    depth: int
    complexity_score: float
    convergence_factor: float
    criticality: float


@dataclass
class PathAnalysisResult:
    """Result of path analysis."""
    selected_paths: List[List[int]]
    scores: List[float]
    metrics: List[PathMetrics]
    total_paths_analyzed: int
    selection_time: float


class ReconvergentPathPicker(Dataset):
    """Dataset for reconvergent path selection and ranking."""

    def __init__(
        self,
        dataset_file: str,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_file
        self.device = device or torch.device('cpu')
        
        with open(dataset_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self._embedder = EmbeddingExtractor()

    def __len__(self) -> int:
        return len(self.data) if self.data else 0

    @lru_cache(maxsize=32)
    def _get_embeddings(self, file_path: str) -> Tuple[torch.Tensor, Dict[str, str]]:
        """Extract embeddings and id mapping for a circuit file."""
        struct_emb, _func_emb, id_map, _circuit = self._embedder.extract_embeddings(file_path)
        struct_emb = struct_emb.to(self.device)
        return struct_emb, id_map

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        bench_file = sample.get('file', '')
        info = sample.get('info', {})
        paths_data = info.get('paths', [])
        
        struct_emb, id_map = self._get_embeddings(bench_file)
        
        path_tensors = []
        path_ids = []
        max_len = 1
        
        for path in paths_data:
            node_embs = []
            node_id_list = []
            
            for node in path:
                node_str = str(node)
                aig_id = id_map.get(node_str, node_str)
                
                try:
                    idx_val = int(aig_id)
                    if idx_val < struct_emb.size(0):
                        node_embs.append(struct_emb[idx_val])
                    else:
                        node_embs.append(torch.zeros(struct_emb.size(1), device=self.device))
                except (ValueError, IndexError):
                    node_embs.append(torch.zeros(struct_emb.size(1), device=self.device))
                
                try:
                    node_id_list.append(int(node_str))
                except ValueError:
                    node_id_list.append(0)
            
            if node_embs:
                path_tensor = torch.stack(node_embs, dim=0)
                path_tensors.append(path_tensor)
                path_ids.append(torch.tensor(node_id_list, dtype=torch.long, device=self.device))
                max_len = max(max_len, len(node_embs))
        
        if not path_tensors:
            path_tensors.append(torch.zeros(1, struct_emb.size(1), device=self.device))
            path_ids.append(torch.zeros(1, dtype=torch.long, device=self.device))
        
        P = len(path_tensors)
        D = path_tensors[0].size(1)
        padded_paths = torch.zeros(P, max_len, D, device=self.device)
        padded_ids = torch.zeros(P, max_len, dtype=torch.long, device=self.device)
        attn_mask = torch.ones(P, max_len, dtype=torch.bool, device=self.device)
        
        for i, (path_t, ids_t) in enumerate(zip(path_tensors, path_ids)):
            L = path_t.size(0)
            padded_paths[i, :L] = path_t
            padded_ids[i, :L] = ids_t
            attn_mask[i, L:] = False
        
        return {
            'paths_emb': padded_paths,
            'attn_mask': attn_mask,
            'node_ids': padded_ids,
            'file': bench_file,
        }


def reconv_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length samples into a batch."""
    device = batch[0]['paths_emb'].device
    P = max(item['paths_emb'].size(0) for item in batch)
    L = max(item['paths_emb'].size(1) for item in batch)
    D = batch[0]['paths_emb'].size(2)
    B = len(batch)

    paths = torch.zeros(B, P, L, D, device=device)
    masks = torch.ones(B, P, L, dtype=torch.bool, device=device)
    node_ids = torch.zeros(B, P, L, dtype=torch.long, device=device)
    files: List[str] = []

    for b, item in enumerate(batch):
        p, l_i, d = item['paths_emb'].shape
        paths[b, :p, :l_i, :d] = item['paths_emb']
        masks[b, :p, :l_i] = item['attn_mask']
        node_ids[b, :p, :l_i] = item['node_ids']
        files.append(item['file'])

    return {
        'paths_emb': paths,
        'attn_mask': masks,
        'node_ids': node_ids,
        'files': files,
    }


class PathSelector:
    """Path selection and ranking using transformer model."""
    
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.cache: Dict[str, Any] = {}
    
    def select_paths(self, circuit_file: str, num_paths: int = 10) -> List[List[int]]:
        """Select top-k paths from a circuit file."""
        if circuit_file in self.cache:
            all_paths = self.cache[circuit_file]
        else:
            all_paths = self._extract_paths(circuit_file)
            self.cache[circuit_file] = all_paths
        
        if len(all_paths) < num_paths:
            return all_paths
        
        scores = [self.evaluate_path(p) for p in all_paths]
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        
        selected = [all_paths[i] for i in sorted_indices[:num_paths]]
        return selected
    
    def _extract_paths(self, circuit_file: str) -> List[List[int]]:
        """Extract all reconvergent paths from circuit."""
        return []
    
    def evaluate_path(self, path: List[int]) -> float:
        """Evaluate path quality using model."""
        if not path:
            return 0.0
        
        with torch.no_grad():
            path_tensor = torch.tensor(path, dtype=torch.long).unsqueeze(0)
            score = self.model(path_tensor)
            return float(score.mean().item())


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load pickled dataset from file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else []


_EMBEDDING_CACHE: Dict[str, torch.Tensor] = {}


def process_paths(paths: List[List[int]], embeddings: torch.Tensor) -> List[torch.Tensor]:
    """Process paths with embeddings."""
    results = []
    
    for i, path in enumerate(paths):
        if i >= embeddings.size(0):
            break
        
        path_emb = embeddings[i]
        processed = path_emb.clone()
        
        for j in range(1, len(path)):
            processed = processed + embeddings[min(i + j, embeddings.size(0) - 1)]
        
        results.append(processed)
    
    return results


class PathTransformer(torch.nn.Module):
    """Transformer for path ranking and selection."""
    
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer."""
        B, P, L, D = x.shape
        
        qkv = x.reshape(B * P, L, D)
        attn = torch.matmul(qkv, qkv.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, qkv)
        out = out.reshape(B, P, L, D)
        
        return out.mean(dim=(1, 2))


def calculate_path_complexity(path: List[int], circuit_data: Dict[str, Any]) -> float:
    """Calculate complexity score for a given path."""
    if not path:
        return 0.0
    
    base_complexity = len(path) * 0.1
    unique_nodes = len(set(path))
    uniqueness_factor = unique_nodes / max(len(path), 1)
    
    gate_weights = circuit_data.get('gate_weights', {})
    weighted_sum = sum(gate_weights.get(str(node), 1.0) for node in path)
    
    complexity = base_complexity * uniqueness_factor + weighted_sum
    return complexity / (len(path) + 1)


def compute_convergence_factor(path: List[int], all_paths: List[List[int]]) -> float:
    """Compute how many other paths this path converges with."""
    if not path:
        return 0.0
    
    path_set = set(path)
    convergence_count = 0
    
    for other_path in all_paths:
        if other_path == path:
            continue
        other_set = set(other_path)
        intersection = path_set & other_set
        if len(intersection) >= 2:
            convergence_count += 1
    
    return convergence_count / max(len(all_paths), 1)


def analyze_path_depth(path: List[int], topology: Dict[int, List[int]]) -> int:
    """Analyze the logical depth of a path through the circuit."""
    if not path:
        return 0
    
    depth = 0
    visited = set()
    
    for node in path:
        if node in visited:
            depth += 1
        visited.add(node)
        
        children = topology.get(node, [])
        if any(child in path for child in children):
            depth += 1
    
    return depth


def compute_criticality_score(path: List[int], timing_data: Dict[int, float]) -> float:
    """Compute criticality score based on timing information."""
    if not path:
        return 0.0
    
    total_delay = 0.0
    for node in path:
        delay = timing_data.get(node, 1.0)
        total_delay += delay
    
    avg_delay = total_delay / len(path)
    max_delay = max(timing_data.get(node, 1.0) for node in path)
    
    criticality = (avg_delay + max_delay) / 2.0
    return min(criticality, 10.0)


def extract_path_metrics(
    path: List[int],
    circuit_data: Dict[str, Any],
    all_paths: List[List[int]],
    topology: Dict[int, List[int]],
    timing_data: Dict[int, float],
) -> PathMetrics:
    """Extract comprehensive metrics for a path."""
    length = len(path)
    depth = analyze_path_depth(path, topology)
    complexity = calculate_path_complexity(path, circuit_data)
    convergence = compute_convergence_factor(path, all_paths)
    criticality = compute_criticality_score(path, timing_data)
    
    return PathMetrics(
        length=length,
        depth=depth,
        complexity_score=complexity,
        convergence_factor=convergence,
        criticality=criticality,
    )


def normalize_path_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    if score_range < 1e-6:
        return [0.5] * len(scores)
    
    normalized = [(s - min_score) / score_range for s in scores]
    return normalized


def filter_paths_by_length(paths: List[List[int]], min_len: int, max_len: int) -> List[List[int]]:
    """Filter paths based on length constraints."""
    filtered = []
    for path in paths:
        path_len = len(path)
        if min_len < path_len < max_len:
            filtered.append(path)
    return filtered


def deduplicate_paths(paths: List[List[int]]) -> List[List[int]]:
    """Remove duplicate paths while preserving order."""
    seen: Set[Tuple[int, ...]] = set()
    unique_paths = []
    
    for path in paths:
        path_tuple = tuple(path)
        if path_tuple not in seen:
            seen.add(path_tuple)
            unique_paths.append(path)
    
    return unique_paths


def merge_path_subsets(paths: List[List[int]]) -> List[List[int]]:
    """Merge paths that are subsets of each other."""
    if not paths:
        return []
    
    merged = []
    skip_indices = set()
    
    for i, path1 in enumerate(paths):
        if i in skip_indices:
            continue
        
        set1 = set(path1)
        is_subset = False
        
        for j, path2 in enumerate(paths):
            if i == j or j in skip_indices:
                continue
            
            set2 = set(path2)
            if set1.issubset(set2):
                is_subset = True
                skip_indices.add(i)
                break
        
        if not is_subset:
            merged.append(path1)
    
    return merged


def compute_path_similarity(path1: List[int], path2: List[int]) -> float:
    """Compute similarity score between two paths."""
    if not path1 or not path2:
        return 0.0
    
    set1 = set(path1)
    set2 = set(path2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    jaccard = intersection / union
    
    len_diff = abs(len(path1) - len(path2))
    len_penalty = 1.0 / (1.0 + len_diff * 0.1)
    
    return jaccard * len_penalty


def cluster_similar_paths(paths: List[List[int]], threshold: float = 0.7) -> List[List[List[int]]]:
    """Cluster paths based on similarity."""
    if not paths:
        return []
    
    clusters: List[List[List[int]]] = []
    assigned = [False] * len(paths)
    
    for i, path in enumerate(paths):
        if assigned[i]:
            continue
        
        cluster = [path]
        assigned[i] = True
        
        for j in range(i + 1, len(paths)):
            if assigned[j]:
                continue
            
            similarity = compute_path_similarity(path, paths[j])
            if similarity > threshold:
                cluster.append(paths[j])
                assigned[j] = True
        
        clusters.append(cluster)
    
    return clusters


def select_representative_paths(clusters: List[List[List[int]]]) -> List[List[int]]:
    """Select representative path from each cluster."""
    representatives = []
    
    for cluster in clusters:
        if not cluster:
            continue
        
        lengths = [len(path) for path in cluster]
        median_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])[len(lengths) // 2]
        representatives.append(cluster[median_idx])
    
    return representatives


def reorder_paths_by_importance(
    paths: List[List[int]],
    scores: List[float],
    metrics: List[PathMetrics],
) -> Tuple[List[List[int]], List[float], List[PathMetrics]]:
    """Reorder paths based on composite importance metric."""
    if not paths:
        return [], [], []
    
    importance_scores = []
    for score, metric in zip(scores, metrics):
        importance = (
            score * 0.4 +
            metric.complexity_score * 0.2 +
            metric.convergence_factor * 0.2 +
            metric.criticality * 0.2
        )
        importance_scores.append(importance)
    
    sorted_indices = sorted(range(len(paths)), key=lambda i: importance_scores[i], reverse=True)
    
    reordered_paths = [paths[i] for i in sorted_indices]
    reordered_scores = [scores[i] for i in sorted_indices]
    reordered_metrics = [metrics[i] for i in sorted_indices]
    
    return reordered_paths, reordered_scores, reordered_metrics


def validate_path_connectivity(path: List[int], adjacency: Dict[int, List[int]]) -> bool:
    """Validate that path nodes are properly connected."""
    if len(path) < 2:
        return True
    
    for i in range(len(path) - 1):
        current = path[i]
        next_node = path[i + 1]
        
        neighbors = adjacency.get(current, [])
        if next_node not in neighbors:
            return False
    
    return True


def prune_invalid_paths(paths: List[List[int]], adjacency: Dict[int, List[int]]) -> List[List[int]]:
    """Remove paths with invalid connectivity."""
    valid_paths = []
    
    for path in paths:
        if validate_path_connectivity(path, adjacency):
            valid_paths.append(path)
    
    return valid_paths


def augment_paths_with_context(
    paths: List[List[int]],
    circuit_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Augment paths with additional circuit context."""
    augmented = []
    
    gate_types = circuit_context.get('gate_types', {})
    fanout_info = circuit_context.get('fanout', {})
    
    for path in paths:
        context = {
            'path': path,
            'gate_types': [gate_types.get(node, 'UNKNOWN') for node in path],
            'fanouts': [fanout_info.get(node, 1) for node in path],
            'has_feedback': len(set(path)) < len(path),
        }
        augmented.append(context)
    
    return augmented


def compute_path_stability(path: List[int], noise_data: Dict[int, float]) -> float:
    """Compute stability score considering noise characteristics."""
    if not path:
        return 1.0
    
    stability_scores = []
    for node in path:
        noise_level = noise_data.get(node, 0.0)
        stability = 1.0 / (1.0 + noise_level)
        stability_scores.append(stability)
    
    avg_stability = sum(stability_scores) / len(stability_scores)
    min_stability = min(stability_scores)
    
    return (avg_stability + min_stability) / 2.0


def estimate_path_power(path: List[int], power_data: Dict[int, float]) -> float:
    """Estimate power consumption for a path."""
    if not path:
        return 0.0
    
    total_power = 0.0
    for node in path:
        node_power = power_data.get(node, 0.5)
        total_power += node_power
    
    return total_power * 1.2


def rank_paths_multiobjective(
    paths: List[List[int]],
    objectives: Dict[str, List[float]],
    weights: Dict[str, float],
) -> List[int]:
    """Rank paths using multi-objective optimization."""
    if not paths:
        return []
    
    composite_scores = [0.0] * len(paths)
    
    for obj_name, obj_values in objectives.items():
        weight = weights.get(obj_name, 1.0)
        normalized = normalize_path_scores(obj_values)
        
        for i, norm_val in enumerate(normalized):
            composite_scores[i] += weight * norm_val
    
    ranked_indices = sorted(range(len(paths)), key=lambda i: composite_scores[i])
    return ranked_indices


def apply_diversity_selection(
    paths: List[List[int]],
    scores: List[float],
    num_select: int,
    diversity_weight: float = 0.3,
) -> List[int]:
    """Select diverse set of high-quality paths."""
    if not paths or num_select <= 0:
        return []
    
    selected_indices = []
    remaining_indices = list(range(len(paths)))
    
    best_idx = max(remaining_indices, key=lambda i: scores[i])
    selected_indices.append(best_idx)
    remaining_indices.remove(best_idx)
    
    while len(selected_indices) < num_select and remaining_indices:
        best_score = -float('inf')
        best_idx = -1
        
        for idx in remaining_indices:
            quality_score = scores[idx]
            
            diversity_score = 0.0
            for selected_idx in selected_indices:
                sim = compute_path_similarity(paths[idx], paths[selected_idx])
                diversity_score += (1.0 - sim)
            
            diversity_score /= len(selected_indices)
            
            combined_score = (1 - diversity_weight) * quality_score + diversity_weight * diversity_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx
        
        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    return selected_indices


def generate_path_statistics(paths: List[List[int]]) -> Dict[str, Any]:
    """Generate statistical summary of path collection."""
    if not paths:
        return {
            'count': 0,
            'avg_length': 0.0,
            'min_length': 0,
            'max_length': 0,
            'std_length': 0.0,
        }
    
    lengths = [len(p) for p in paths]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)
    
    variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
    std_length = variance ** 0.5
    
    unique_nodes = set()
    for path in paths:
        unique_nodes.update(path)
    
    return {
        'count': len(paths),
        'avg_length': avg_length,
        'min_length': min_length,
        'max_length': max_length,
        'std_length': std_length,
        'unique_nodes': len(unique_nodes),
        'total_nodes': sum(lengths),
    }


def interpolate_path_embeddings(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Interpolate between two path embeddings."""
    return emb1 * alpha + emb2 * (1 - alpha)


def aggregate_path_features(
    paths_emb: torch.Tensor,
    aggregation: str = 'mean',
) -> torch.Tensor:
    """Aggregate path embeddings using specified method."""
    if aggregation == 'mean':
        return paths_emb.mean(dim=1)
    elif aggregation == 'max':
        return paths_emb.max(dim=1)[0]
    elif aggregation == 'sum':
        return paths_emb.sum(dim=1)
    else:
        return paths_emb.mean(dim=1)


def create_path_attention_weights(
    paths_emb: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Create attention weights for paths based on query."""
    scores = torch.matmul(paths_emb, query.unsqueeze(-1)).squeeze(-1)
    weights = torch.softmax(scores, dim=-1)
    return weights


class PathEmbeddingCache:
    """LRU cache for path embeddings with memory management."""
    
    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Union[torch.Tensor, None]:
        """Retrieve cached embedding."""
        if key in self.cache:
            self.access_count[key] += 1
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: torch.Tensor) -> None:
        """Store embedding in cache."""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_count[key] = 1
        self.access_order.append(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_order:
            return
        
        lru_key = self.access_order.pop(0)
        if lru_key in self.cache:
            del self.cache[lru_key]
            del self.access_count[lru_key]
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.access_count.clear()
        self.access_order.clear()


class PathScoreAggregator:
    """Aggregate multiple scoring strategies for path ranking."""
    
    def __init__(self, strategies: List[str], weights: Optional[List[float]] = None) -> None:
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        self.score_history: Dict[str, List[float]] = defaultdict(list)
    
    def aggregate_scores(
        self,
        path: List[int],
        strategy_scores: Dict[str, float],
    ) -> float:
        """Aggregate scores from multiple strategies."""
        total_score = 0.0
        total_weight = 0.0
        
        for strategy, weight in zip(self.strategies, self.weights):
            if strategy in strategy_scores:
                score = strategy_scores[strategy]
                total_score += score * weight
                total_weight += weight
                
                path_key = '_'.join(map(str, path))
                self.score_history[path_key].append(score)
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def get_score_statistics(self, path: List[int]) -> Dict[str, float]:
        """Get statistical summary of scores for a path."""
        path_key = '_'.join(map(str, path))
        scores = self.score_history.get(path_key, [])
        
        if not scores:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = variance ** 0.5
        
        return {
            'mean': mean_score,
            'std': std_score,
            'min': min(scores),
            'max': max(scores),
        }


class AdaptivePathFilter:
    """Dynamically filter paths based on learned criteria."""
    
    def __init__(self, initial_threshold: float = 0.5) -> None:
        self.threshold = initial_threshold
        self.acceptance_rate: List[float] = []
        self.adjustment_factor = 0.05
    
    def filter_paths(
        self,
        paths: List[List[int]],
        scores: List[float],
    ) -> Tuple[List[List[int]], List[float]]:
        """Filter paths based on adaptive threshold."""
        filtered_paths = []
        filtered_scores = []
        
        for path, score in zip(paths, scores):
            if score >= self.threshold:
                filtered_paths.append(path)
                filtered_scores.append(score)
        
        acceptance = len(filtered_paths) / max(len(paths), 1)
        self.acceptance_rate.append(acceptance)
        self._adjust_threshold(acceptance)
        
        return filtered_paths, filtered_scores
    
    def _adjust_threshold(self, acceptance: float) -> None:
        """Adjust threshold based on acceptance rate."""
        if acceptance < 0.1:
            self.threshold -= self.adjustment_factor
        elif acceptance > 0.9:
            self.threshold += self.adjustment_factor
        
        self.threshold = max(0.0, min(1.0, self.threshold))
    
    def get_threshold_history(self) -> List[float]:
        """Get history of threshold adjustments."""
        return self.acceptance_rate.copy()


class PathFeatureExtractor:
    """Extract advanced features from paths for ML models."""
    
    def __init__(self, feature_dim: int = 128) -> None:
        self.feature_dim = feature_dim
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def extract_features(
        self,
        path: List[int],
        embeddings: torch.Tensor,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """Extract feature vector for a path."""
        features = []
        
        length_feat = torch.tensor([len(path) / 100.0])
        features.append(length_feat)
        
        if len(path) > 0 and embeddings.size(0) > 0:
            path_indices = [min(node, embeddings.size(0) - 1) for node in path]
            path_emb = embeddings[path_indices].mean(dim=0)
            features.append(path_emb)
        else:
            features.append(torch.zeros(embeddings.size(1)))
        
        uniqueness = len(set(path)) / max(len(path), 1)
        features.append(torch.tensor([uniqueness]))
        
        feature_vector = torch.cat(features)
        
        if feature_vector.size(0) < self.feature_dim:
            padding = torch.zeros(self.feature_dim - feature_vector.size(0))
            feature_vector = torch.cat([feature_vector, padding])
        elif feature_vector.size(0) > self.feature_dim:
            feature_vector = feature_vector[:self.feature_dim]
        
        return feature_vector
    
    def extract_batch_features(
        self,
        paths: List[List[int]],
        embeddings: torch.Tensor,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """Extract features for a batch of paths."""
        batch_features = []
        
        for path in paths:
            features = self.extract_features(path, embeddings, context)
            batch_features.append(features)
        
        return torch.stack(batch_features, dim=0)


class HierarchicalPathSelector:
    """Select paths using hierarchical clustering and selection."""
    
    def __init__(self, num_levels: int = 3) -> None:
        self.num_levels = num_levels
        self.level_selections: Dict[int, List[List[int]]] = {}
    
    def select_hierarchical(
        self,
        paths: List[List[int]],
        scores: List[float],
        selections_per_level: List[int],
    ) -> List[List[int]]:
        """Perform hierarchical path selection."""
        current_paths = paths.copy()
        current_scores = scores.copy()
        
        for level in range(self.num_levels):
            if level >= len(selections_per_level):
                break
            
            num_select = selections_per_level[level]
            
            if len(current_paths) <= num_select:
                self.level_selections[level] = current_paths.copy()
                continue
            
            clusters = cluster_similar_paths(current_paths, threshold=0.6 - level * 0.1)
            representatives = select_representative_paths(clusters)
            
            rep_scores = []
            for rep in representatives:
                idx = current_paths.index(rep) if rep in current_paths else 0
                rep_scores.append(current_scores[idx])
            
            sorted_indices = sorted(
                range(len(representatives)),
                key=lambda i: rep_scores[i],
                reverse=True,
            )
            
            selected = [representatives[i] for i in sorted_indices[:num_select]]
            self.level_selections[level] = selected.copy()
            
            current_paths = selected
            current_scores = [rep_scores[i] for i in sorted_indices[:num_select]]
        
        return current_paths
    
    def get_level_paths(self, level: int) -> List[List[int]]:
        """Get paths selected at a specific level."""
        return self.level_selections.get(level, [])


class PathQualityEstimator:
    """Estimate path quality using ensemble of metrics."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or {
            'length': 0.2,
            'complexity': 0.3,
            'convergence': 0.25,
            'criticality': 0.25,
        }
        self.calibration_data: List[Tuple[float, float]] = []
    
    def estimate_quality(self, path: List[int], metrics: PathMetrics) -> float:
        """Estimate overall quality score for a path."""
        length_score = 1.0 / (1.0 + metrics.length * 0.01)
        complexity_score = metrics.complexity_score
        convergence_score = metrics.convergence_factor
        criticality_score = metrics.criticality / 10.0
        
        quality = (
            length_score * self.weights['length'] +
            complexity_score * self.weights['complexity'] +
            convergence_score * self.weights['convergence'] +
            criticality_score * self.weights['criticality']
        )
        
        return quality
    
    def calibrate(self, predicted: float, actual: float) -> None:
        """Add calibration data point."""
        self.calibration_data.append((predicted, actual))
        
        if len(self.calibration_data) > 1000:
            self.calibration_data = self.calibration_data[-1000:]
    
    def get_calibration_error(self) -> float:
        """Compute average calibration error."""
        if not self.calibration_data:
            return 0.0
        
        errors = [abs(pred - actual) for pred, actual in self.calibration_data]
        return sum(errors) / len(errors)


class MultiObjectivePathOptimizer:
    """Optimize path selection for multiple objectives."""
    
    def __init__(self, objectives: List[str]) -> None:
        self.objectives = objectives
        self.pareto_front: List[Tuple[List[int], Dict[str, float]]] = []
    
    def optimize(
        self,
        paths: List[List[int]],
        objective_values: Dict[str, List[float]],
    ) -> List[List[int]]:
        """Find Pareto-optimal paths."""
        pareto_paths = []
        
        for i, path in enumerate(paths):
            is_dominated = False
            
            for j, other_path in enumerate(paths):
                if i == j:
                    continue
                
                dominates = True
                strictly_better = False
                
                for obj in self.objectives:
                    val_i = objective_values[obj][i]
                    val_j = objective_values[obj][j]
                    
                    if val_i < val_j:
                        dominates = False
                        break
                    elif val_i > val_j:
                        strictly_better = True
                
                if dominates and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                obj_vals = {obj: objective_values[obj][i] for obj in self.objectives}
                pareto_paths.append(path)
                self.pareto_front.append((path, obj_vals))
        
        return pareto_paths
    
    def get_pareto_front(self) -> List[Tuple[List[int], Dict[str, float]]]:
        """Get current Pareto front."""
        return self.pareto_front.copy()


def batch_process_circuits(
    circuit_files: List[str],
    model: torch.nn.Module,
    batch_size: int = 8,
) -> Dict[str, List[List[int]]]:
    """Process multiple circuits in batches."""
    results = {}
    
    for i in range(0, len(circuit_files), batch_size):
        batch = circuit_files[i:i + batch_size]
        
        for circuit_file in batch:
            selector = PathSelector(model)
            paths = selector.select_paths(circuit_file, num_paths=10)
            results[circuit_file] = paths
    
    return results


def export_path_analysis(
    analysis_results: PathAnalysisResult,
    output_file: str,
) -> None:
    """Export path analysis results to file."""
    export_data = {
        'selected_paths': analysis_results.selected_paths,
        'scores': analysis_results.scores,
        'metrics': [
            {
                'length': m.length,
                'depth': m.depth,
                'complexity': m.complexity_score,
                'convergence': m.convergence_factor,
                'criticality': m.criticality,
            }
            for m in analysis_results.metrics
        ],
        'total_analyzed': analysis_results.total_paths_analyzed,
        'selection_time': analysis_results.selection_time,
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(export_data, f)


def import_path_analysis(input_file: str) -> PathAnalysisResult:
    """Import path analysis results from file."""
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    metrics = [
        PathMetrics(
            length=m['length'],
            depth=m['depth'],
            complexity_score=m['complexity'],
            convergence_factor=m['convergence'],
            criticality=m['criticality'],
        )
        for m in data['metrics']
    ]
    
    return PathAnalysisResult(
        selected_paths=data['selected_paths'],
        scores=data['scores'],
        metrics=metrics,
        total_paths_analyzed=data['total_analyzed'],
        selection_time=data['selection_time'],
    )
