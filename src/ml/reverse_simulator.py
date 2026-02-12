"""
Reverse Circuit Simulator using Transformer Architecture

This module implements a transformer-based model that takes functional embeddings
of circuit inputs/outputs and predicts input patterns that would produce the desired output.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :] # type: ignore


class EmbeddingProjection(nn.Module):
    """Projects functional embeddings to transformer dimension."""
    
    def __init__(self, embedding_dim: int = 128, d_model: int = 256):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, embedding_dim] or [seq_len, batch_size, embedding_dim]
        """
        projected = self.projection(x)
        return self.layer_norm(projected)


class ReverseCircuitTransformer(nn.Module):
    """
    Transformer model for reverse circuit simulation.
    
    Takes functional embeddings of inputs and desired output, predicts input pattern.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_inputs: int = 100
    ):
        super().__init__()
        self.d_model = d_model
        self.max_inputs = max_inputs
        self.input_embedding_proj = EmbeddingProjection(embedding_dim, d_model)
        self.output_embedding_proj = EmbeddingProjection(embedding_dim, d_model)
        self.output_token = nn.Parameter(torch.randn(d_model))
        self.separator_token = nn.Parameter(torch.randn(d_model))
        self.desired_zero_token = nn.Parameter(torch.randn(d_model))
        self.desired_one_token = nn.Parameter(torch.randn(d_model))
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_inputs + 200)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        # Use new nested tensor API to avoid warnings
        encoder_layer.enable_nested_tensor = True
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        output_embedding: torch.Tensor,
        desired_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the reverse circuit simulator.
        Accepts variable-length input sequences (2-100).
        input_embeddings: [batch_size, num_inputs, embedding_dim]
        output_embedding: [batch_size, embedding_dim]
        desired_output: [batch_size, 1]
        Returns: [batch_size, num_inputs] predictions
        """
        batch_size, num_inputs, _ = input_embeddings.shape
        num_inputs = max(2, min(num_inputs, self.max_inputs))
        input_embeddings = input_embeddings[:, :num_inputs, :]
        input_proj = self.input_embedding_proj(input_embeddings)
        output_proj = self.output_embedding_proj(output_embedding)
        seq_len = 4 + num_inputs
        sequence = torch.zeros(seq_len, batch_size, self.d_model, device=input_embeddings.device)
        sequence[0] = self.output_token.unsqueeze(0).expand(batch_size, -1)
        sequence[1] = output_proj
        desired_output_tokens = torch.where(
            desired_output > 0.5,
            self.desired_one_token.unsqueeze(0).expand(batch_size, -1),
            self.desired_zero_token.unsqueeze(0).expand(batch_size, -1)
        )
        sequence[2] = desired_output_tokens
        sequence[3] = self.separator_token.unsqueeze(0).expand(batch_size, -1)
        sequence[4:4+num_inputs] = input_proj.transpose(0, 1)
        sequence = self.pos_encoding(sequence)
        transformer_output = self.transformer(sequence)
        desired_output_token = transformer_output[2:3]
        input_embeddings_for_attention = transformer_output[4:4+num_inputs]
        attended_inputs, _ = self.cross_attention(
            query=input_embeddings_for_attention,
            key=desired_output_token,
            value=desired_output_token
        )
        input_predictions = input_embeddings_for_attention + attended_inputs
        input_predictions = input_predictions.transpose(0, 1)
        predicted_inputs = self.output_head(input_predictions).squeeze(-1)
        return predicted_inputs

    def predict_with_confidence(
        self,
        input_embeddings: torch.Tensor,
        output_embedding: torch.Tensor,
        desired_output: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict input pattern with confidence scores.
        Returns:
            binary_predictions: Binary predictions [batch_size, num_inputs]
            confidence_scores: Confidence scores [batch_size, num_inputs]
        """
        with torch.no_grad():
            probabilities = self.forward(input_embeddings, output_embedding, desired_output)
            binary_predictions = (probabilities > threshold).float()
            confidence_scores = torch.abs(probabilities - 0.5) * 2  # Convert to 0-1 confidence
        return binary_predictions, confidence_scores


class ReverseSimulatorLoss(nn.Module):
    """
    Custom loss function for reverse circuit simulation.
    Combines binary cross-entropy with circuit simulation feedback.
    """
    
    def __init__(self, simulation_weight: float = 1.0, bce_weight: float = 0.5, 
                 consistency_weight: float = 0.3):
        super().__init__()
        self.simulation_weight = simulation_weight
        self.bce_weight = bce_weight
        self.consistency_weight = consistency_weight
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        simulation_rewards: Optional[torch.Tensor] = None,
        desired_outputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss combining BCE and simulation feedback.
        
        Args:
            predictions: Model predictions [batch_size, num_inputs]
            targets: Ground truth targets [batch_size, num_inputs] 
            simulation_rewards: Rewards from circuit simulation [batch_size]
            desired_outputs: Desired output values [batch_size, 1]
        
        Returns:
            total_loss: Combined loss value
        """
        # Ensure predictions and targets are in valid range for BCE loss
        predictions = torch.clamp(predictions, min=1e-7, max=1-1e-7)
        targets = torch.clamp(targets, min=0.0, max=1.0)
        
        # Binary cross-entropy loss
        bce_loss = self.bce_loss(predictions, targets)
        
        total_loss = self.bce_weight * bce_loss
        
        if simulation_rewards is not None:
            # Convert rewards to loss (negative rewards become positive loss)
            simulation_loss = -simulation_rewards.mean()
            total_loss += self.simulation_weight * simulation_loss
        
        # Add consistency loss to encourage predictions that align with desired output
        if desired_outputs is not None:
            # Encourage more confident predictions when desired output is 1
            # and less confident when desired output is 0
            confidence = torch.abs(predictions - 0.5) * 2  # Convert to 0-1 confidence
            target_confidence = desired_outputs.squeeze()  # [batch_size]
            # Ensure same shape for MSE loss
            if confidence.mean(dim=1).shape != target_confidence.shape:
                target_confidence = target_confidence.unsqueeze(0).expand_as(confidence.mean(dim=1))
            consistency_loss = self.mse_loss(confidence.mean(dim=1), target_confidence)
            total_loss += self.consistency_weight * consistency_loss
            
        return total_loss


def create_model(
    embedding_dim: int = 128,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ReverseCircuitTransformer:
    """Factory function to create a reverse circuit transformer model."""
    
    model = ReverseCircuitTransformer(
        embedding_dim=embedding_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    return model.to(device)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=str(device))
    
    # Create dummy data
    batch_size = 4
    num_inputs = 8
    embedding_dim = 128
    
    input_embeddings = torch.randn(batch_size, num_inputs, embedding_dim).to(device)
    output_embedding = torch.randn(batch_size, embedding_dim).to(device)
    desired_output = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    
    # Forward pass
    predictions = model(input_embeddings, output_embedding, desired_output)
    print(f"Input embeddings shape: {input_embeddings.shape}")
    print(f"Output embedding shape: {output_embedding.shape}")
    print(f"Desired output shape: {desired_output.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # Test with confidence
    binary_pred, confidence = model.predict_with_confidence(
        input_embeddings, output_embedding, desired_output
    )
    print(f"Binary predictions: {binary_pred}")
    print(f"Confidence scores: {confidence}")
