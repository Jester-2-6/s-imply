import unittest

from src.atpg.reconv_podem import (
    PathConsistencySolver,
    get_lrr,
    identify_exit_lines,
)
from src.util.struct import Gate, GateType, LogicValue


class TestMaamariLogic(unittest.TestCase):
    def setUp(self):
        # Create a simple diamond circuit: S -> (B, C) -> R
        # S (1), B (2), C (3), R (4)
        # Add leakage: B -> Leak (5)
        self.circuit = [None] * 6

        # Helper to creating gate with fot populated
        def make_gate(gid, gtype, fins, fots):
            g = Gate(str(gid), gtype, len(fins), len(fots))  # Correct init args
            g.fin = fins  # Manually set fin list
            g.fot = fots  # Manually set fot list
            return g

        self.circuit[1] = make_gate(1, GateType.BUFF, [], [2, 3])  # S
        self.circuit[2] = make_gate(2, GateType.BUFF, [1], [4, 5])  # B
        self.circuit[3] = make_gate(3, GateType.BUFF, [1], [4])  # C
        self.circuit[4] = make_gate(4, GateType.AND, [2, 3], [])  # R
        self.circuit[5] = make_gate(5, GateType.BUFF, [2], [])  # Leak

    def test_get_lrr(self):
        lrr = get_lrr(self.circuit, 1, 4)
        # LRR should contain S, B, C, R.
        # Leak (5) should NOT be in LRR because it doesn't reach R.
        self.assertEqual(lrr, {1, 2, 3, 4})

    def test_identify_exit_lines(self):
        lrr = {1, 2, 3, 4}
        exit_lines = identify_exit_lines(self.circuit, lrr)
        # Exit line should be from B (2) to Leak (5)
        self.assertEqual(len(exit_lines), 1)
        self.assertEqual(exit_lines[0], (2, 5))

    def test_heuristic_scoring(self):
        # We can't easily unit test the internal loop of pick_reconv_pair directly
        # without mocking or refactoring, but we can verify the logic preference.
        # Let's create a circuit where one path is tight and one is leaky and see
        # if we pick the tight one?
        # Beam search picks "best".
        pass

    def test_solver_regional_consistency(self):
        # S=1 -> B=2, C=3
        # B -> R=4, B -> Leak=5
        # C -> R=4
        # R is AND. Target R=1. Requires B=1, C=1. Implies S=1.
        # If we constrain Leak=0, implies B=0 (if B->Leak is BUFF).
        # Then B=1 conflict.

        # Setup constraints
        constraints = {5: LogicValue.ZERO}  # Leak must be 0

        solver = PathConsistencySolver(self.circuit)
        pair_info = {"start": 1, "reconv": 4, "paths": [[1, 2, 4], [1, 3, 4]]}

        # Target R=1. Should fail because B needs to be 1, but Leak=0 forces B=0.
        res = solver.solve(pair_info, LogicValue.ONE, constraints)
        print(f"Solver result with conflict: {res}")
        self.assertIsNone(res, "Solver should fail due to Exit Line conflict")

        # Target R=0. Should succeed (e.g. C=0, S=0 -> B=0 -> Leak=0 ok)
        res_ok = solver.solve(pair_info, LogicValue.ZERO, constraints)
        print(f"Solver result ok: {res_ok}")
        self.assertIsNotNone(res_ok)
        self.assertEqual(res_ok[4], LogicValue.ZERO)


if __name__ == "__main__":
    unittest.main()
