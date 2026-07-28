"""
trajectory/encoder.py

Cross-turn trajectory encoder. Takes the sequence of activation snapshots
from ConversationBuffer and produces a single trajectory embedding that
represents the safety-relevant direction of the conversation.

Architecture:
    - Positional encoding over turns (not tokens)
    - Lightweight transformer encoder (2–4 layers) over turn sequence
    - Projection head that maps to a low-dimensional safety subspace
    - Contrastively trained: safe convos cluster away from unsafe ones

Why not just mean-pool activations across turns?
    Mean-pooling loses the directional information (drift). Two conversations
    can have similar mean activations but very different trajectories —
    one stays flat, one escalates. The transformer encoder captures that
    ordering-sensitive drift.

Key design decision: frozen base model
    We freeze Llama-3.1-8B entirely. The encoder only trains a small
    projection head on top of extracted activations. This keeps Week 3
    compute tractable and avoids catastrophic forgetting concerns.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TurnPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding over turns (not tokens).
    Max 64 turns — more than enough for any realistic conversation.
    """

    def __init__(self, d_model: int, max_turns: int = 64, dropout: float = 0.1):
        """
        Args:
            d_model: embedding dimension the positional encoding is added to
                (the encoder's working dim, not the raw LLM hidden dim)
            max_turns: number of turn positions to precompute encodings for
            dropout: dropout applied after adding the positional encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_turns, d_model)
        position = torch.arange(0, max_turns, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_turns, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_turns, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TrajectoryEncoder(nn.Module):
    """
    Cross-turn trajectory encoder.

    Input:  (batch, n_turns, d_model)  — turn activation sequence
    Output: (batch, proj_dim)          — trajectory embedding in safety subspace

    The safety subspace dimension (proj_dim) is intentionally small (128–256)
    to force the projection to learn a compact, interpretable representation
    of safety-relevant trajectory features.

    Training objective: contrastive (see subspace.py)
    """

    def __init__(
        self,
        d_model: int = 4096,         # Llama-3.1-8B hidden dim
        proj_dim: int = 256,          # safety subspace dimension
        n_heads: int = 8,
        n_layers: int = 2,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_turns: int = 32,
    ):
        """
        Args:
            d_model: hidden dim of the frozen base LLM's activations (input size)
            proj_dim: dimension of the output safety subspace embedding
            n_heads: attention heads in the turn-level transformer
            n_layers: number of transformer encoder layers
            feedforward_dim: hidden size of each transformer layer's FFN
            dropout: dropout used in positional encoding and transformer layers
            max_turns: max conversation length (in turns) supported by the
                positional encoding
        """
        super().__init__()

        self.d_model = d_model
        self.proj_dim = proj_dim

        # Input projection: reduce d_model to a working dimension
        # Llama's 4096-dim is too large for efficient attention over turns
        self.working_dim = 512
        self.input_proj = nn.Linear(d_model, self.working_dim, bias=False)

        # Positional encoding over turns
        self.pos_enc = TurnPositionalEncoding(self.working_dim, max_turns, dropout)

        # Transformer encoder over turn sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.working_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,      # (batch, seq, dim)
            norm_first=True,       # pre-norm: more stable
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False
        )

        # CLS-style aggregation token (learned)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.working_dim))

        # Final projection to safety subspace
        self.proj_head = nn.Sequential(
            nn.LayerNorm(self.working_dim),
            nn.Linear(self.working_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Truncated-normal init for all Linear layers (std=0.02, zero bias)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        turn_activations: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            turn_activations: (batch, n_turns, d_model)
            padding_mask: (batch, n_turns) bool mask, True = padding turn

        Returns:
            trajectory_embedding: (batch, proj_dim)
                L2-normalized embedding in safety subspace
        """
        batch_size = turn_activations.size(0)

        # Project down from LLM hidden dim to working dim
        x = self.input_proj(turn_activations)       # (B, T, working_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, working_dim)
        x = torch.cat([cls, x], dim=1)              # (B, T+1, working_dim)

        # Add positional encoding
        x = self.pos_enc(x)

        # Extend padding mask for CLS token (CLS is never masked)
        if padding_mask is not None:
            cls_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=padding_mask.device
            )
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Transformer over turn sequence
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Extract CLS token representation
        cls_out = x[:, 0, :]                        # (B, working_dim)

        # Project to safety subspace and L2-normalize
        embedding = self.proj_head(cls_out)         # (B, proj_dim)
        embedding = F.normalize(embedding, dim=-1)

        return embedding

    def encode_single(self, turn_activations: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for single conversation (no batch dim).

        Args:
            turn_activations: (n_turns, d_model)
        Returns:
            embedding: (proj_dim,)
        """
        return self.forward(turn_activations.unsqueeze(0)).squeeze(0)

    def encode_buffer(self, buffer) -> torch.Tensor:
        """
        Encode directly from a ConversationBuffer.

        Args:
            buffer: a `trajectory.buffer.ConversationBuffer` instance
        Returns:
            (proj_dim,) trajectory embedding, or zeros if the buffer has
            no turns (see `ConversationBuffer.get_trajectory`)
        """
        trajectory = buffer.get_trajectory()
        if trajectory is None:
            return torch.zeros(self.proj_dim)
        return self.encode_single(trajectory)


class DeltaTrajectoryEncoder(TrajectoryEncoder):
    """
    Variant that explicitly encodes turn-to-turn *deltas* rather than
    raw activations. More sensitive to sudden escalation changes.

    Useful as an ablation baseline — swap with TrajectoryEncoder to
    test whether drift signal or absolute position drives detection.
    """

    def forward(
        self,
        turn_activations: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            turn_activations: (batch, n_turns, d_model) raw per-turn activations
            padding_mask: (batch, n_turns) bool mask, True = padding turn

        Returns:
            trajectory_embedding: (batch, proj_dim), same as TrajectoryEncoder.forward
        """
        # Compute consecutive differences
        if turn_activations.size(1) > 1:
            deltas = turn_activations[:, 1:, :] - turn_activations[:, :-1, :]
            # Prepend zero delta for first turn
            zeros = torch.zeros_like(turn_activations[:, :1, :])
            turn_activations = torch.cat([zeros, deltas], dim=1)

        return super().forward(turn_activations, padding_mask)