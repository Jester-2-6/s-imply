import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper: Standard Positional Encoding ---
# Injects position information into the input embeddings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        pe_slice = self.pe[:x.size(1)].transpose(0, 1) # (1, Seq, Dim)
        x = x + pe_slice
        return self.dropout(x)

# --- The Main Transformer Architecture ---

class MultiPathTransformer(nn.Module):
    """
    A Transformer architecture to analyze and predict values for multiple,
    variable-length reconvergent paths.

    Parameters
    ----------
    input_dim : int
        Dimension of incoming node embeddings (from dataset/collate)
    model_dim : int
        Internal transformer model dimension (can be larger than input_dim)
    nhead : int
        Number of attention heads (must divide model_dim)
    num_encoder_layers : int
        Layers for the shared path encoder
    num_interaction_layers : int
        Layers for the path interaction encoder
    dim_feedforward : int
        Feedforward dimension inside Transformer layers
    """
    def __init__(self, input_dim: int, model_dim: int, nhead: int, num_encoder_layers: int, num_interaction_layers: int, dim_feedforward: int = 512):
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)

        # Optional input projection to expand/shrink to model_dim
        if self.input_dim != self.model_dim:
            self.input_proj = nn.Linear(self.input_dim, self.model_dim)
        else:
            self.input_proj = nn.Identity()

        # 1. Shared Path Encoder
        # This single encoder processes each path independently to learn the
        # general features of a logic path.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.shared_path_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 2. Path Interaction Layer
        # This layer allows the fully-encoded paths to "talk" to each other.
        # It's another Transformer encoder that treats each path as a single "token".
        interaction_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.path_interaction_layer = nn.TransformerEncoder(interaction_layer, num_layers=num_interaction_layers)
        
        # 3. Prediction Head
        # A simple linear layer to predict the final logic value (0 or 1) for each node.
        # We use embedding_dim as input and 2 as output for the two classes (0 and 1).
        self.prediction_head = nn.Linear(self.model_dim, 2)
        
        self.pos_encoder = PositionalEncoding(self.model_dim)

    def forward(self, path_list, attention_masks):
        """
        Args:
            path_list (Tensor): A padded tensor of path embeddings.
                                Shape: (batch_size, num_paths, seq_len, embedding_dim)
            attention_masks (Tensor): A boolean mask to ignore padded tokens.
                                     Shape: (batch_size, num_paths, seq_len)
        """
        batch_size, num_paths, seq_len, _ = path_list.shape

        # --- Step 1: Encode Each Path Independently ---
        # Reshape the input to process all paths in the batch at once.
        # (batch_size * num_paths, seq_len, input_dim)
        flat_paths = path_list.view(-1, seq_len, self.input_dim)
        # Project to model dimension if needed
        flat_paths = self.input_proj(flat_paths)
        
        # Apply Positional Encoding to learn sequential order (input -> output)
        flat_paths = self.pos_encoder(flat_paths)

        flat_masks = attention_masks.view(-1, seq_len)

        # Pass all paths through the same shared encoder.
        encoded_paths = self.shared_path_encoder(flat_paths, src_key_padding_mask=~flat_masks)

        # --- Step 2: Allow Paths to Interact ---
        # Use the first token as a summary representation for each path.
        path_representations = encoded_paths[:, 0, :]  # (batch_size * num_paths, model_dim)

        # Group by original batch item: (batch_size, num_paths, model_dim)
        path_representations = path_representations.view(batch_size, num_paths, self.model_dim)

        # Paths interact through a Transformer operating over the path axis.
        interaction_aware_reps = self.path_interaction_layer(path_representations)

        # --- Step 3: Combine and Predict ---
        # Broadcast interaction context back to each node position in each path.
        global_context = interaction_aware_reps.unsqueeze(2)  # (B, P, 1, model_dim)

        # Reshape encoded paths back to grouped form and add context.
        encoded_paths = encoded_paths.view(batch_size, num_paths, seq_len, self.model_dim)
        final_representations = encoded_paths + global_context

        # Per-node logits
        predictions = self.prediction_head(final_representations)  # (B, P, L, 2)
        return predictions

# --- Training Logic Placeholder ---
def custom_loss_function(predictions, targets, original_lengths):
    """
    Placeholder for the full training logic.
    """
    # 1. Main Prediction Loss (e.g., Cross-Entropy)
    # This would compare predictions to the ground truth, ignoring padded values.
    main_loss = F.cross_entropy(predictions.permute(0, 3, 1, 2), targets, ignore_index=-1) # Assuming -1 for padding
    
    # 2. Consistency Loss for Shared Nodes
    # Enforces that the first and last nodes of all paths in a set are the same.
    # Get predictions for the first node of each path (for all items in batch)
    first_node_preds = predictions[:, :, 0, :] # Shape: (batch_size, num_paths, 2)
    # Get predictions for the last node of each path (using original_lengths)
    # (More complex indexing needed here based on original_lengths)
    
    # Calculate variance or MSE between predictions for path 0, path 1, etc.
    consistency_loss = torch.var(first_node_preds, dim=1).mean() # Simplified example
    
    # 3. Combine Losses
    total_loss = main_loss + (0.5 * consistency_loss) # Weighting factor of 0.5
    
    return total_loss