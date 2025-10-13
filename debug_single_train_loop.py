#!/usr/bin/env python3
"""
Standalone script to debug a single training loop for ReverseCircuitTransformer.
Uses real embeddings and mirrors the trainer's supervised loop for one batch.
"""
import torch
import os
import sys
from src.ml.reverse_simulator import ReverseCircuitTransformer, ReverseSimulatorLoss
from src.ml.embedding_extractor import EmbeddingExtractor

# Settings
embedding_dim = 128
max_inputs = 100
learning_rate = 1e-3
device = 'cpu'

# Pick a single circuit
circuit_path = 'data/bench/arbitrary/single_and.bench'
if not os.path.exists(circuit_path):
    print(f'Circuit file not found: {circuit_path}')
    sys.exit(1)

# Extract embeddings
embedding_extractor = EmbeddingExtractor()
struct_emb, func_emb, gate_mapping, circuit_gates = embedding_extractor.extract_embeddings(circuit_path)
input_gates = [g for g in circuit_gates if g.type == 1 and g.nfi == 0]
output_gates = [g for g in circuit_gates if g.type != 0 and g.nfo == 0]
num_inputs = len(input_gates)
num_outputs = len(output_gates)
print(f"[INFO] Circuit: {circuit_path}")
print(f"[INFO] Num inputs: {num_inputs}, Num outputs: {num_outputs}")

# Prepare input embeddings (variable-length)
input_embeddings = func_emb[:num_inputs].unsqueeze(0)  # [1, num_inputs, embedding_dim]

# Select output gate
selected_output_idx = int(torch.randint(0, num_outputs, (1,)).item()) if num_outputs > 0 else None
if selected_output_idx is not None:
    output_start_idx = len(func_emb) - num_outputs
    output_embedding = func_emb[output_start_idx + selected_output_idx].unsqueeze(0)  # [1, embedding_dim]
else:
    output_embedding = torch.zeros(1, embedding_dim, device=device)

# Desired output
desired_output = torch.tensor([[torch.randint(0, 2, (1,)).item()]], dtype=torch.float32, device=device)
print(f"[INFO] Selected output gate idx: {selected_output_idx}")
print(f"[INFO] Desired output: {desired_output.item()}")

# Target pattern (random for debug)
target_pattern = torch.randint(0, 2, (1, num_inputs), dtype=torch.float32, device=device)
print(f"[DEBUG] Target pattern: {target_pattern}")

# Model
model = ReverseCircuitTransformer(embedding_dim=embedding_dim, max_inputs=max_inputs).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = ReverseSimulatorLoss()

# Forward pass
prediction = model(input_embeddings, output_embedding, desired_output)
print(f"[DEBUG] Raw model prediction: {prediction.detach().cpu().numpy()}")


# Binarize prediction for actual input pattern
binary_pattern = (prediction[:, :num_inputs] > 0.5).float()
input_pattern = binary_pattern[0].detach().cpu().numpy().astype(int)
print(f"[SIM] Predicted input pattern: {input_pattern}")

# Simulate circuit with predicted pattern
from src.atpg.logic_sim_three import reset_gates, logic_sim
from src.util.struct import LogicValue
reset_gates(circuit_gates, len(circuit_gates) - 1)
for i, gate in enumerate(input_gates):
    gate.val = int(input_pattern[i]) if i < len(input_pattern) else LogicValue.XD
logic_sim(circuit_gates, len(circuit_gates) - 1)

# Get actual output from simulator
if num_outputs > 0 and selected_output_idx is not None:
    actual_output = output_gates[selected_output_idx].val
    print(f"[SIM] Actual output: {actual_output}")
    print(f"[SIM] Desired output: {desired_output.item()}")
    simulation_success = (actual_output == desired_output.item())
    reward = 1.0 if simulation_success else -1.0
else:
    simulation_success = False
    reward = -1.0
print(f"[SIM] Simulation success: {simulation_success}")
print(f"[SIM] Reward: {reward}")

# Compute loss (only for actual inputs)
loss = loss_fn(prediction[:, :num_inputs], target_pattern)
print(f"[TRAIN] Loss: {loss.item()}")

# Backward and optimizer step
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
print("[TRAIN] Optimizer step complete.")

print("\n[SUMMARY] Single training loop debug complete.")
