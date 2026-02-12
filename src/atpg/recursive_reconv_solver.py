"""
Recursive Reconvergent Path Pair Solver.

This module implements a hierarchical justification flow that solves reconvergent
path pairs in a specific order (shortest to longest) to justify a target value.
It is designed to work with a predictive model (or oracle) that provides
candidate assignments for these pairs.
"""

from __future__ import annotations

import abc
import collections
from typing import List, Dict, Any, Optional, Set, Tuple
import heapq

from src.util.struct import LogicValue, Gate
from src.atpg.reconv_podem import PathConsistencySolver

class ReconvPairPredictor(abc.ABC):
    """Abstract base class for predicting solutions to reconvergent pairs."""
    
    @abc.abstractmethod
    def predict(
        self, 
        pair_info: Dict[str, Any], 
        constraints: Dict[int, LogicValue],
        seed: Optional[int] = None
    ) -> List[Dict[int, LogicValue]]:
        """
        Predict a list of valid assignments for the given pair, respecting constraints.
        
        Args:
            pair_info: Dictionary containing 'start', 'reconv', 'paths', etc.
            constraints: Dictionary of current node assignments {node_id: value}.
            
        Returns:
            A list of assignment dictionaries (partial solutions).
            The list is ordered by likelihood/preference.
        """
        pass


class HierarchicalReconvSolver:
    """
    Solves for a target value by recursively justifying reconvergent path pairs,
    ordered from shortest to longest.
    """

    def __init__(self, circuit: List[Gate], predictor: ReconvPairPredictor, recorder = None, verbose: bool = False):
        self.circuit = circuit
        self.predictor = predictor
        self.recorder = recorder
        self.verbose = verbose
        # Helper for consistency checks, reused from existing codebase
        self.consistency_checker = PathConsistencySolver(circuit)
        
        # Populate fanout lists from fanin (if not already present)
        self._populate_fanouts()

    def solve(self, target_node: int, target_val: LogicValue, constraints: Dict[int, LogicValue] = None, seed: Optional[int] = None) -> Optional[Dict[int, LogicValue]]:
        """
        Main entry point. Tries to justify target_node = target_val.
        
        1. Identify logic cone of target_node.
        2. Find and sort reconvergent pairs within the cone.
        3. Recursively solve pairs (backtracking).
        4. Return final assignment or None.
        """
        if self.verbose:
            print(f"[Solver] Solving for Gate {target_node} = {target_val} with {len(constraints) if constraints else 0} constraints")
            
        # 1. & 2. Find relevant pairs
        pairs = self._collect_and_sort_pairs(target_node)
        
        if self.verbose:
            print(f"[Solver] Found {len(pairs)} reconvergent pairs")
        
        # Initial constraints: target + provided constraints
        initial_constraints = {}
        if constraints:
            initial_constraints.update(constraints)
        
        # Set/Overwrite target requirement (critical!)
        initial_constraints[target_node] = target_val
        
        # 3. Recursive Solve
        final_assignment = self._solve_recursive(0, pairs, initial_constraints, seed)
        
        return final_assignment

    def _collect_and_sort_pairs(self, root_node: int) -> List[Dict[str, Any]]:
        """
        Identify reconvergent pairs in the transitive fanin of root_node,
        sorted by their proximity to the fault and path complexity.
        
        Priority: Pairs closer to the fault site (shorter distance from reconv to target)
        and with shorter total path lengths should be solved first.
        """
        # A. Transitive Fanin Cone
        cone_nodes = self._get_transitive_fanin(root_node)
        
        # B. Find Pairs within this cone
        pairs = self._find_pairs_in_set(cone_nodes)
        
        # C. Compute distance from reconvergence node to target for each pair
        # BFS from root_node backwards to find distance
        from collections import deque
        distances = {root_node: 0}
        queue = deque([root_node])
        
        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            gate = self.circuit[curr]
            fanins = getattr(gate, 'fin', []) or []
            
            for fin in fanins:
                if fin in cone_nodes and fin not in distances:
                    distances[fin] = curr_dist + 1
                    queue.append(fin)
        
        # D. Sort by priority:
        # Combined Score = Total Path Length + Distance to Target
        # This prioritizes compact loops near the fault over distant small loops.
        # Secondary sort by length ensures shortest paths are picked within similar distances.
        def pair_cost(p):
            reconv_node = p['reconv']
            dist_to_target = distances.get(reconv_node, 9999)
            total_path_len = len(p['paths'][0]) + len(p['paths'][1])
            # Return tuple: (Combined Score, Length)
            return (total_path_len + dist_to_target, total_path_len)
            
        pairs.sort(key=pair_cost)
        
        return pairs

    def _get_transitive_fanin(self, root: int) -> Set[int]:
        """BFS backwards to find all nodes feeding root."""
        seen = set()
        queue = collections.deque([root])
        seen.add(root)
        while queue:
            curr = queue.popleft()
            gate = self.circuit[curr]
            for fin in gate.fin:
                if fin not in seen:
                    seen.add(fin)
                    queue.append(fin)
        return seen

    def _populate_fanouts(self):
        """Build fanout lists from fanin relationships if not present."""
        # Initialize empty fanout lists
        for gate in self.circuit:
            if not hasattr(gate, 'fot') or gate.fot is None:
                gate.fot = []
        
        # Build fanouts from fanins
        for gate_id, gate in enumerate(self.circuit):
            for fin_id in gate.fin:
                if fin_id < len(self.circuit):
                    if gate_id not in self.circuit[fin_id].fot:
                        self.circuit[fin_id].fot.append(gate_id)

    def _find_pairs_in_set(self, allowed_nodes: Set[int]) -> List[Dict[str, Any]]:
        """
        Find reconvergent pairs where all path nodes are in allowed_nodes.
        Simplified BFS-based detection.
        """
        # Identify potential stems: nodes in allowed_nodes with multiple fanouts also in allowed_nodes
        stems = []
        for nid in allowed_nodes:
            gate = self.circuit[nid]
            valid_fot = [fo for fo in (getattr(gate, 'fot', []) or []) if fo in allowed_nodes]
            if len(valid_fot) >= 2:
                stems.append(nid)
                
        results = []
        
        # For each stem, launch BFS to find reconvergence points within allowed_nodes
        for s in stems:
            # We track which branch (index in valid_fot) reached a node
            start_gate = self.circuit[s]
            valid_fot = [fo for fo in (getattr(start_gate, 'fot', []) or []) if fo in allowed_nodes]
            
            # Track reported reconvergence nodes for this stem to avoid duplicates
            reported_reconvs = set()
            
            # reached[node] = {branch_idx: path_list}
            reached = {} 
            
            # Initial frontier
            queue = collections.deque()
            for i, fo in enumerate(valid_fot):
                path = [s, fo]
                reached.setdefault(fo, {})
                reached[fo][i] = path
                queue.append(fo)
                
            processed_nodes = {s} # Avoid cycles back to start
            
            # Limit search depth roughly
            while queue:
                curr = queue.popleft()
                
                # Check if this node is a reconvergence point for S
                # i.e., reached by >= 2 distinct branches
                if len(reached[curr]) >= 2 and curr != s:
                    # Found a pair!
                    # Extract pairs. If >2 branches, we can take all combinations or just first 2.
                    # Taking first 2 distinct branches for simplicity.
                    bs = list(reached[curr].keys())
                    b1, b2 = bs[0], bs[1]
                    p1 = reached[curr][b1]
                    p2 = reached[curr][b2]
                    
                    # Avoid duplicates? (s, curr)
                    # We add to results if not already reported for this stem.
                    if curr not in reported_reconvs:
                        reported_reconvs.add(curr)
                        results.append({
                            'start': s,
                            'reconv': curr,
                            'branches': [valid_fot[b1], valid_fot[b2]], # Branch specific nodes
                            'paths': [p1, p2]
                        })
                    
                    # Do we continue from here? Yes, might reach further reconvergence.
                    
                # Expand
                gate = self.circuit[curr]
                curr_branches = reached[curr].keys()
                
                valid_fot_curr = [fo for fo in (getattr(gate, 'fot', []) or []) if fo in allowed_nodes]
                
                for fo in valid_fot_curr:
                    if fo == s: continue 
                    
                    # Need to merge branch info
                    if fo not in reached:
                        reached[fo] = {}
                        new_visit = True
                    else:
                        new_visit = False
                        
                    changed = False
                    for b_idx in curr_branches:
                        if b_idx not in reached[fo]:
                            # Extend path
                            old_path = reached[curr][b_idx]
                            new_path = old_path + [fo]
                            reached[fo][b_idx] = new_path
                            changed = True
                            
                    if changed:
                        # If we added info, we must propagate, even if visited before (DAG)
                        # To avoid infinite loops in cyclic circuits we might need checks, but Bench circuits are usually DAGs.
                        # Simple optimization: only append if not already in queue? Set based queue?
                        # For now, just append.
                        queue.append(fo)

        # Post-processing: remove partial overlaps or duplicates if needed
        # For now, return all found.
        return results

    def _solve_recursive(
        self, 
        pair_idx: int, 
        pairs: List[Dict[str, Any]], 
        current_constraints: Dict[int, LogicValue],
        seed: Optional[int] = None
    ) -> Optional[Dict[int, LogicValue]]:
        """
        Backtracking solver.
        
        Args:
            pair_idx: Index of the pair we are currently solving.
            pairs: Sorted list of all pairs.
            current_constraints: Assignments made so far.
            
        Returns:
            Full assignment dictionary if solvable, None otherwise.
        """
        # Base Case: All pairs processed
        if pair_idx >= len(pairs):
            if self.verbose:
                print(f"[Solver] Base case reached. Returning partial solution.")
            return current_constraints
        
        pair = pairs[pair_idx]
        if self.verbose:
            indent = "  " * (pair_idx + 1)
            stem = pair.get('start', pair.get('stem'))
            print(f"[Solver]{indent} Processing Pair {pair_idx}: Stem {stem} -> Reconv {pair['reconv']}")
        
        # Get candidate solutions from Oracle
        # The predictor should return solutions that respect `current_constraints`.
        
        # UPDATE: Predictor now returns (candidates, inputs_snapshot)
        # Derive a unique seed for this step if a base seed is provided
        step_seed = None
        if seed is not None:
             # simple mixing: seed + pair_idx (could be more complex if needed)
             step_seed = (seed + pair_idx * 7919) % 2147483647
             
        prediction_result = self.predictor.predict(pair, current_constraints, seed=step_seed)
        
        # Handle backward compatibility if someone hasn't updated their predictor class
        inputs_snapshot = None
        if isinstance(prediction_result, tuple):
             candidates, inputs_snapshot = prediction_result
        else:
             candidates = prediction_result
        
        if not candidates:
            if self.verbose:
                indent = "  " * (pair_idx + 1)
                print(f"[Solver]{indent} No candidates from predictor. Backtracking.")
            # If the model cannot find any solution for this pair given constraints,
            # this path is dead. Backtrack.
            return None
            
        for i, assignment_part in enumerate(candidates):
            if self.verbose:
                indent = "  " * (pair_idx + 1)
                print(f"[Solver]{indent} Trying Candidate {i}: {assignment_part}")
                
            # Log this decision attempt if recording
            step_record = None
            if self.recorder and inputs_snapshot:
                 # Log the attempt. 
                 step_record = self.recorder.log_step(
                     node_ids=inputs_snapshot['node_ids'],
                     mask_valid=inputs_snapshot['mask_valid'],
                     gate_types=inputs_snapshot['gate_types'],
                     files=inputs_snapshot['files'],
                     pair_info=pair,
                     selected_assignment=assignment_part
                 )

            # 1. Merge assignment
            # (Logic consistency is assumed enforced by predictor, but we can double check)
            new_constraints = current_constraints.copy()
            conflict = False
            for k, v in assignment_part.items():
                if k in new_constraints:
                    if new_constraints[k] != v:
                        conflict = True
                        break
                else:
                    new_constraints[k] = v
            
            if conflict:
                if self.verbose:
                    print(f"[Solver]{indent} Conflict detected. Skipping candidate.")
                if step_record and self.recorder:
                     # Immediate conflict -> local failure
                     self.recorder.mark_backtrack(penalty=-0.5) 
                continue
                
            # 2. Recurse
            result = self._solve_recursive(pair_idx + 1, pairs, new_constraints, seed)
            
            if result is not None:
                if self.verbose:
                    print(f"[Solver]{indent} Candidate {i} successful.")
                # Found a valid complete assignment!
                return result
            
            # If we returned None, it means a conflict happened deeper in the recursion.
            # This choice (assignment_part) led to a failure.
            if self.verbose:
                print(f"[Solver]{indent} Candidate {i} failed in recursion. Backtracking.")
                
            if step_record and self.recorder:
                 self.recorder.mark_backtrack(penalty=-0.5)
                 
        # If no candidates lead to a solution, backtrack.
        if self.verbose:
            indent = "  " * (pair_idx + 1)
            print(f"[Solver]{indent} All candidates failed. Backtracking pair {pair_idx}.")
        return None

