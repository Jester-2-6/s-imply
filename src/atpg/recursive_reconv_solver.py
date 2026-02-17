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
from typing import Any, Dict, List, Optional, Set, Tuple

from src.atpg.reconv_podem import PathConsistencySolver
from src.util.struct import Gate, GateType, LogicValue


class ReconvPairPredictor(abc.ABC):
    """Abstract base class for predicting solutions to reconvergent pairs."""

    @abc.abstractmethod
    def predict(
        self,
        pair_info: Dict[str, Any],
        constraints: Dict[int, LogicValue],
        seed: Optional[int] = None,
    ) -> List[Dict[int, LogicValue]] | Tuple[List[Dict[int, LogicValue]], Any]:
        """
        Predict a list of valid assignments for the given pair, respecting constraints.

        Args:
            pair_info: Dictionary containing 'start', 'reconv', 'paths', etc.
            constraints: Dictionary of current node assignments {node_id: value}.
            seed: Optional random seed for deterministic sampling.

        Returns:
            A list of assignment dictionaries (partial solutions) OR
            (list of assignment dictionaries, snapshot information).
            The list is ordered by likelihood/preference.
        """
        pass


class HierarchicalReconvSolver:
    """
    Solves for a target value by recursively justifying reconvergent path pairs,
    ordered from shortest to longest.
    """

    def __init__(
        self,
        circuit: List[Gate],
        predictor: ReconvPairPredictor,
        recorder=None,
        verbose: bool = False,
    ):
        self.circuit = circuit
        self.predictor = predictor
        self.recorder = recorder
        self.verbose = verbose
        # Helper for consistency checks, reused from existing codebase
        self.consistency_checker = PathConsistencySolver(circuit)

        # Populate fanout lists from fanin (if not already present)
        self._populate_fanouts()
        self.pair_cache = {}  # Cache reconv pairs per root node

    def solve(
        self,
        target_node: int,
        target_val: LogicValue,
        constraints: Dict[int, LogicValue] = None,
        seed: Optional[int] = None,
    ) -> Optional[Dict[int, LogicValue]]:
        """
        Main entry point. Tries to justify target_node = target_val.
        """
        if self.verbose:
            print(
                f"[Solver] Solving for Gate {target_node} = {target_val} with "
                f"{len(constraints) if constraints else 0} constraints"
            )

        # 1. & 2. Find relevant pairs (use cache to avoid redundant BFS)
        if target_node not in self.pair_cache:
            self.pair_cache[target_node] = self._collect_and_sort_pairs(target_node)
        pairs = self.pair_cache[target_node]

        if self.verbose:
            print(f"[Solver] Found {len(pairs)} reconvergent pairs")

        # Initial constraints: target + provided constraints
        initial_constraints = {}
        if constraints:
            initial_constraints.update(constraints)

        # Set/Overwrite target requirement
        initial_constraints[target_node] = target_val

        # 3. Recursive Solve
        final_assignment = self._solve_recursive(0, pairs, initial_constraints, seed)

        return final_assignment

    def _collect_and_sort_pairs(self, root_node: int) -> List[Dict[str, Any]]:
        """Identify and sort reconvergent pairs in the transitive fanin of root_node."""
        cone_nodes = self._get_transitive_fanin(root_node)
        pairs = self._find_pairs_in_set(cone_nodes)

        from collections import deque

        distances = {root_node: 0}
        queue = deque([root_node])

        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            gate = self.circuit[curr]
            if gate is None:
                continue
            for fin in gate.fin:
                if fin in cone_nodes and fin not in distances:
                    distances[fin] = curr_dist + 1
                    queue.append(fin)

        def pair_cost(p):
            reconv_node = p["reconv"]
            dist_to_target = distances.get(reconv_node, 9999)
            total_path_len = len(p["paths"][0]) + len(p["paths"][1])
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
            if gate is None:
                continue
            for fin in gate.fin:
                if fin not in seen:
                    seen.add(fin)
                    queue.append(fin)
        return seen

    def _populate_fanouts(self):
        """Build fanout lists from fanin relationships if not present."""
        for gate in self.circuit:
            if gate is None:
                continue
            if not hasattr(gate, "fot") or gate.fot is None:
                gate.fot = []

        for gate_id, gate in enumerate(self.circuit):
            if gate is None:
                continue
            for fin_id in gate.fin:
                if fin_id < len(self.circuit):
                    target_gate = self.circuit[fin_id]
                    if target_gate is None:
                        continue
                    if not hasattr(target_gate, "fot") or target_gate.fot is None:
                        target_gate.fot = []
                    if gate_id not in target_gate.fot:
                        target_gate.fot.append(gate_id)

    def _find_pairs_in_set(self, allowed_nodes: Set[int]) -> List[Dict[str, Any]]:
        """Find reconvergent pairs within a set of allowed nodes."""
        stems = []
        for nid in allowed_nodes:
            gate = self.circuit[nid]
            if gate is None:
                continue
            valid_fot = [fo for fo in (getattr(gate, "fot", []) or []) if fo in allowed_nodes]
            if len(valid_fot) >= 2:
                stems.append(nid)

        results = []
        for s in stems:
            start_gate = self.circuit[s]
            valid_fot = [fo for fo in (getattr(start_gate, "fot", []) or []) if fo in allowed_nodes]
            reported_reconvs = set()
            reached = {}
            queue = collections.deque()
            for i, fo in enumerate(valid_fot):
                reached[fo] = {i: [s, fo]}
                queue.append(fo)

            while queue:
                curr = queue.popleft()
                if len(reached[curr]) >= 2 and curr != s:
                    if curr not in reported_reconvs:
                        reported_reconvs.add(curr)
                        bs = list(reached[curr].keys())
                        results.append(
                            {
                                "start": s,
                                "reconv": curr,
                                "branches": [valid_fot[bs[0]], valid_fot[bs[1]]],
                                "paths": [reached[curr][bs[0]], reached[curr][bs[1]]],
                            }
                        )

                gate = self.circuit[curr]
                curr_branches = reached[curr].keys()
                valid_fot_curr = [
                    fo for fo in (getattr(gate, "fot", []) or []) if fo in allowed_nodes
                ]

                for fo in valid_fot_curr:
                    if fo == s:
                        continue
                    if fo not in reached:
                        reached[fo] = {}
                    changed = False
                    for b_idx in curr_branches:
                        if b_idx not in reached[fo]:
                            reached[fo][b_idx] = reached[curr][b_idx] + [fo]
                            changed = True
                    if changed:
                        queue.append(fo)
        return results

    def _solve_recursive(
        self,
        pair_idx: int,
        pairs: List[Dict[str, Any]],
        current_constraints: Dict[int, LogicValue],
        seed: Optional[int] = None,
    ) -> Optional[Dict[int, LogicValue]]:
        """Backtracking solver with AI prediction support."""
        if pair_idx >= len(pairs):
            if self.verbose:
                print("[Solver] Base case: justifying remaining requirements...")
            return self._justify_all(current_constraints)

        pair = pairs[pair_idx]
        indent = "  " * (pair_idx + 1)
        if self.verbose:
            stem = pair.get("start", pair.get("stem"))
            print(f"[Solver]{indent} Solving Pair {pair_idx}: Stem {stem} -> {pair['reconv']}")

        step_seed = None
        if seed is not None:
            step_seed = (seed + pair_idx * 7919) % 2147483647

        prediction_result = self.predictor.predict(pair, current_constraints, seed=step_seed)
        inputs_snapshot = None
        if isinstance(prediction_result, tuple):
            candidates, inputs_snapshot = prediction_result
        else:
            candidates = prediction_result

        if not candidates:
            return None

        for i, assignment_part in enumerate(candidates):
            step_record = None
            if self.recorder and inputs_snapshot:
                step_record = self.recorder.log_step(
                    node_ids=inputs_snapshot["node_ids"],
                    mask_valid=inputs_snapshot["mask_valid"],
                    gate_types=inputs_snapshot["gate_types"],
                    files=inputs_snapshot["files"],
                    pair_info=pair,
                    selected_assignment=assignment_part,
                )

            new_constraints = current_constraints.copy()
            conflict = False
            for k, v in assignment_part.items():
                if not self._check_global_consistency(k, v, new_constraints):
                    conflict = True
                    break
                new_constraints[k] = v

            if conflict:
                if step_record and self.recorder:
                    self.recorder.mark_backtrack(penalty=-0.5)
                continue

            result = self._solve_recursive(pair_idx + 1, pairs, new_constraints, seed)
            if result is not None:
                return result

            if step_record and self.recorder:
                self.recorder.mark_backtrack(penalty=-0.5)

        return None

    def _justify_all(self, assignment: Dict[int, LogicValue]) -> Optional[Dict[int, LogicValue]]:
        """Justify all current assignments back to primary inputs."""
        full_assignment = assignment.copy()
        queue = [n for n in full_assignment if self.circuit[n].type != GateType.INPT]

        while queue:
            node = queue.pop(0)
            val = full_assignment[node]
            reqs = self._justify_gate(node, val, full_assignment)
            if reqs is None:
                return None

            for fin, fval in reqs.items():
                if fin not in full_assignment:
                    full_assignment[fin] = fval
                    if self.circuit[fin].type != GateType.INPT:
                        queue.append(fin)
                elif full_assignment[fin] != fval:
                    return None
        return full_assignment

    def _check_global_consistency(
        self, node: int, val: LogicValue, assignment: Dict[int, LogicValue]
    ) -> bool:
        if node in assignment:
            return assignment[node] == val
        gate = self.circuit[node]
        if gate.fin:
            input_vals = [assignment.get(fin, LogicValue.XD) for fin in gate.fin]
            if any(v != LogicValue.XD for v in input_vals):
                computed = self._compute_gate_robust(gate.type, input_vals)
                if computed != LogicValue.XD and computed != val:
                    return False
        for fout in getattr(gate, "fot", []) or []:
            if fout in assignment:
                fout_gate = self.circuit[fout]
                fout_val = assignment[fout]
                input_vals = [
                    val if fin == node else assignment.get(fin, LogicValue.XD)
                    for fin in fout_gate.fin
                ]
                computed = self._compute_gate_robust(fout_gate.type, input_vals)
                if computed != LogicValue.XD and computed != fout_val:
                    return False
        return True

    def _justify_gate(
        self, node: int, val: LogicValue, assignment: Dict[int, LogicValue]
    ) -> Optional[Dict[int, LogicValue]]:
        gate = self.circuit[node]
        unassigned = [fin for fin in gate.fin if fin not in assignment]
        if not unassigned:
            input_vals = [assignment[fin] for fin in gate.fin]
            res = self._compute_simple(gate.type, input_vals)
            return {} if res == val else None

        input_vals_partial = [assignment.get(fin, LogicValue.XD) for fin in gate.fin]
        computed = self._compute_gate_robust(gate.type, input_vals_partial)
        if computed != LogicValue.XD and computed != val:
            return None
        if computed == val:
            return {}

        reqs = {}
        if gate.type == GateType.AND:
            if val == LogicValue.ONE:
                for fin in unassigned:
                    reqs[fin] = LogicValue.ONE
            else:
                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ZERO, assignment):
                        reqs[fin] = LogicValue.ZERO
                        break
                else:
                    return None
        elif gate.type == GateType.NAND:
            if val == LogicValue.ZERO:
                for fin in unassigned:
                    reqs[fin] = LogicValue.ONE
            else:
                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ZERO, assignment):
                        reqs[fin] = LogicValue.ZERO
                        break
                else:
                    return None
        elif gate.type == GateType.OR:
            if val == LogicValue.ZERO:
                for fin in unassigned:
                    reqs[fin] = LogicValue.ZERO
            else:
                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ONE, assignment):
                        reqs[fin] = LogicValue.ONE
                        break
                else:
                    return None
        elif gate.type == GateType.NOR:
            if val == LogicValue.ONE:
                for fin in unassigned:
                    reqs[fin] = LogicValue.ZERO
            else:
                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ONE, assignment):
                        reqs[fin] = LogicValue.ONE
                        break
                else:
                    return None
        elif gate.type == GateType.NOT:
            reqs[gate.fin[0]] = LogicValue.ZERO if val == LogicValue.ONE else LogicValue.ONE
        elif gate.type == GateType.BUFF:
            reqs[gate.fin[0]] = val
        else:
            return None

        for r_node, r_val in reqs.items():
            if not self._check_global_consistency(r_node, r_val, assignment):
                return None
        return reqs

    def _compute_simple(self, gtype: int, inputs: List[LogicValue]) -> LogicValue:
        if gtype == GateType.AND:
            return LogicValue.ONE if all(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO
        if gtype == GateType.NAND:
            return LogicValue.ZERO if all(i == LogicValue.ONE for i in inputs) else LogicValue.ONE
        if gtype == GateType.OR:
            return LogicValue.ONE if any(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO
        if gtype == GateType.NOR:
            return LogicValue.ZERO if any(i == LogicValue.ONE for i in inputs) else LogicValue.ONE
        if gtype == GateType.NOT:
            return LogicValue.ZERO if inputs[0] == LogicValue.ONE else LogicValue.ONE
        if gtype == GateType.BUFF:
            return inputs[0]
        return LogicValue.XD

    def _compute_gate_robust(self, gtype: int, inputs: List[LogicValue]) -> LogicValue:
        if gtype == GateType.AND:
            if any(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ZERO
            if all(i == LogicValue.ONE for i in inputs):
                return LogicValue.ONE
            return LogicValue.XD
        elif gtype == GateType.NAND:
            if any(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ONE
            if all(i == LogicValue.ONE for i in inputs):
                return LogicValue.ZERO
            return LogicValue.XD
        elif gtype == GateType.OR:
            if any(i == LogicValue.ONE for i in inputs):
                return LogicValue.ONE
            if all(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ZERO
            return LogicValue.XD
        elif gtype == GateType.NOR:
            if any(i == LogicValue.ONE for i in inputs):
                return LogicValue.ZERO
            if all(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ONE
            return LogicValue.XD
        elif gtype == GateType.NOT:
            if inputs[0] == LogicValue.XD:
                return LogicValue.XD
            return LogicValue.ZERO if inputs[0] == LogicValue.ONE else LogicValue.ONE
        elif gtype == GateType.BUFF:
            return inputs[0]
        return LogicValue.XD
