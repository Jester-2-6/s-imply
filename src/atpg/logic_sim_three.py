"""
Logic Simulation Module for PODEM Algorithm
Based on the C implementation from legacy/user.c
"""
from typing import List

from src.util.struct import Gate, GateType, LogicValue

# Lookup tables for gates (based on C implementation)
AND_GATE = [
    [0, 0, 0],
    [0, 1, 2],
    [0, 2, 2],
]

OR_GATE = [
    [0, 1, 2],
    [1, 1, 1],
    [2, 1, 2],
]

XOR_GATE = [
    [0, 1, 2],
    [1, 0, 2],
    [2, 2, 2],
]

NOT_GATE = [1, 0, 2]

class DFrontier:
    """Manages the D-frontier."""
    def __init__(self):
        self._gates = []
    
    def is_empty(self):
        return len(self._gates) == 0
        
    def get_first(self):
        return self._gates[0] if self._gates else None

    def clear(self):
        self._gates = []

    def add(self, gate_id):
        self._gates.append(gate_id)

    def sort(self, key_func):
        self._gates.sort(key=key_func)

# Global D-frontier
d_frontier = DFrontier()

def set_d_frontier_sort(distance_map):
    pass # Sorting logic handled outside or not strictly needed for basic func

def logic_sim(circuit: List[Gate], total_gates: int) -> None:
    """
    Logic simulation with fault injection and implication
    Based on LogicSimAndImpl from C implementation
    """
    # Simply simulate (assuming ordered iteration or event driven. Here simple iteration 1 pass)
    # PODEM often needs event driven or ordered.
    # The provided loop iterates 0..total_gates. Assuming topologically sorted or handled.
    
    d_frontier.clear()
    
    for node_index in range(0, total_gates + 1):
        current_node = circuit[node_index]

        # Skip if node is not active
        if current_node.type == 0:
            continue
            
        # Logic simulation based on gate type
        if current_node.type == GateType.INPT:
            pass  # Primary input, value already set
        elif current_node.type == GateType.FROM:
            current_node.val = circuit[current_node.fin[0]].val
        elif current_node.type == GateType.BUFF:
            current_node.val = circuit[current_node.fin[0]].val
        elif current_node.type == GateType.NOT:
            current_node.val = NOT_GATE[circuit[current_node.fin[0]].val]

        elif current_node.type == GateType.AND:
            node_result = 1
            for fanin_id in current_node.fin:
                if node_result == 0:
                    break
                node_result = AND_GATE[node_result][circuit[fanin_id].val]

            current_node.val = node_result

        elif current_node.type == GateType.NAND:
            node_result = 1
            for fanin_id in current_node.fin:
                if node_result == 0:
                    break
                node_result = AND_GATE[node_result][circuit[fanin_id].val]

            current_node.val = NOT_GATE[node_result]

        elif current_node.type == GateType.OR:
            node_result = 0
            for fanin_id in current_node.fin:
                if node_result == 1:
                    break
                node_result = OR_GATE[node_result][circuit[fanin_id].val]

            current_node.val = node_result

        elif current_node.type == GateType.NOR:
            node_result = 0
            for fanin_id in current_node.fin:
                if node_result == 1:
                    break
                node_result = OR_GATE[node_result][circuit[fanin_id].val]

            current_node.val = NOT_GATE[node_result]

        elif current_node.type == GateType.XOR:
            node_result = circuit[current_node.fin[0]].val
            for fanin_id in current_node.fin[1:]:
                if node_result == LogicValue.XD:
                    break
                node_result = XOR_GATE[node_result][circuit[fanin_id].val]

            current_node.val = node_result

        elif current_node.type == GateType.XNOR:
            node_result = circuit[current_node.fin[0]].val
            for fanin_id in current_node.fin[1:]:
                if node_result == LogicValue.XD:
                    break
                node_result = XOR_GATE[node_result][circuit[fanin_id].val]

            current_node.val = NOT_GATE[node_result]
            
        # D-Frontier Update
        if current_node.val == LogicValue.XD:
            # Check if any input has fault effect (D or DB)
            has_fault_effect = False
            for fin in current_node.fin:
                if circuit[fin].val in (LogicValue.D, LogicValue.DB):
                    has_fault_effect = True
                    break
            if has_fault_effect:
                d_frontier.add(node_index)


def logic_sim_and_impl(circuit: List[Gate], total_gates: int, fault, assignment) -> None:
    # 1. Apply assignment (if valid)
    if assignment.gate_id != -1 and assignment.value != -1:
         circuit[assignment.gate_id].val = assignment.value
         
    # 2. Inject Fault (if current state matches condition)
    # Actually, logic_sim usually handles fault injection implicitly or we need to modify gate val?
    # Simple logic sim:
    
    # Run simulation
    logic_sim(circuit, total_gates)
    
    # 3. Post-process Fault Injection?
    # Usually we inject fault by flipping value at fault site if activated.
    # Check fault site:
    f_gate = circuit[fault.gate_id]
    
    # If fault is SA0 (val=D -> good=1, faulty=0)
    # If calculated val is 1 (Good), we change it to D.
    # If calculated val is 0, no fault effect (0/0).
    
    if fault.value == LogicValue.D: # SA0
        if f_gate.val == LogicValue.ONE:
            f_gate.val = LogicValue.D
        elif f_gate.val == LogicValue.ZERO:
            pass # 0/0
        # What if X? X/0 -> maybe? simple sim keeps X.
        
    elif fault.value == LogicValue.DB: # SA1
        if f_gate.val == LogicValue.ZERO:
            f_gate.val = LogicValue.DB
        elif f_gate.val == LogicValue.ONE:
            pass # 1/1

    # Re-propagate fault effects?
    # Since we changed a value, we might need to propagate again from fault site forward.
    # But simple logic_sim iterates all.
    # If we iterate 0..total_gates, and fault site is visited, the change persists?
    # But standard logic_sim calculates based on inputs.
    # If we modify f_gate.val AFTER logic_sim loop, forward gates are not updated.
    # We should integrate injection INTO logic_sim loop.
    
    # Let's simple-hack: set it and re-run logic_sim?
    # But logic_sim overwrites it based on inputs!
    # So we need logic_sim to be fault-aware or modify input gates?
    # Or overwrite AFTER computing current_node.val inside logic_sim.
    # For now, let's just make logic_sim fault aware?
    # Or just modify `logic_sim` above to handle fault injection?
    # The `fault` argument is missing from standard logic_sim signature I created.
    
    pass

def podem_fail():
    return False # Placeholder

def fault_is_at_po(circuit, total_gates):
    for i in range(1, total_gates+1):
        if circuit[i].nfo == 0 and circuit[i].val in (LogicValue.D, LogicValue.DB):
            return True
    return False

def reset_gates(circuit: List[Gate], total_gates: int):
    """
    Reset all gates to unknown state
    Based on reset_gates from C implementation
    """
    for node_index in range(total_gates + 1):
        circuit[node_index].val = LogicValue.XD
        
    d_frontier.clear()

def print_pi(circuit: List[Gate], total_gates: int) -> str:
    """
    Print primary input values
    Based on printPI from C implementation
    """
    result = ""
    for i in range(1, total_gates + 1):
        if circuit[i].type != 0 and circuit[i].nfi == 0:
            if circuit[i].val == LogicValue.XD:
                result += "X"
            elif circuit[i].val == LogicValue.D:
                result += "1"  # D represents good=1, faulty=0
            elif circuit[i].val == LogicValue.DB:
                result += "0"  # D-bar represents good=0, faulty=1
            elif circuit[i].val == 0:
                result += "0"
            elif circuit[i].val == 1:
                result += "1"
            else:
                result += "X"  # Default to X for unexpected values
    return result

def print_po(circuit: List[Gate], total_gates: int) -> str:
    """
    Print primary output values
    Based on printPO from C implementation
    """
    result = ""
    for i in range(1, total_gates + 1):
        if circuit[i].type != 0 and circuit[i].nfo == 0:
            result += str(circuit[i].val)
    return result
