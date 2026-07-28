"""
trajectory/buffer.py

Per-conversation activation buffer. Stores hidden states across turns
so the trajectory encoder can operate over the full conversation history.

Design notes:
- One buffer instance per active conversation
- Stores both user and assistant turn activations separately
- Supports windowing for very long conversations (>128k token range)
- Serializable so you can checkpoint mid-conversation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import numpy as np


@dataclass
class TurnActivation:
    """Activation snapshot for a single turn."""
    turn_index: int
    role: str                          # "user" or "assistant"
    text: str                          # raw text of this turn
    hidden_state: torch.Tensor         # shape: (d_model,) — mean-pooled last layer
    layer_states: Optional[torch.Tensor] = None  # shape: (n_layers, d_model) — all layers
    token_count: int = 0

    def to_numpy(self) -> np.ndarray:
        """Return hidden_state as a (d_model,) float32 numpy array.

        Cast to float32 first since numpy has no native bf16/fp16 support
        and the hidden state may come off the model in a reduced-precision dtype.
        """
        return self.hidden_state.float().cpu().numpy()


class ConversationBuffer:
    """
    Maintains the full activation history for one conversation.

    Usage:
        buffer = ConversationBuffer(max_turns=20)
        buffer.add(TurnActivation(...))
        trajectory = buffer.get_trajectory()   # (n_turns, d_model)
        recent = buffer.get_window(k=5)        # last 5 turns
    """

    def __init__(self, max_turns: int = 20, d_model: int = 4096):
        """
        Args:
            max_turns: maximum number of turns retained; oldest turns are
                evicted once this is exceeded (see `add`)
            d_model: hidden dimension of the activations being stored
        """
        self.max_turns = max_turns
        self.d_model = d_model
        self.turns: List[TurnActivation] = []
        self._conversation_id: Optional[str] = None

    def add(self, turn: TurnActivation) -> None:
        """Add a turn activation. Evicts oldest if over max_turns."""
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_trajectory(
        self,
        roles: Optional[List[str]] = None,
        as_tensor: bool = True
    ) -> torch.Tensor:
        """
        Return stacked activations across all turns.

        Args:
            roles: if set, filter to ["user"], ["assistant"], or both
            as_tensor: return torch.Tensor if True, else np.ndarray

        Returns:
            shape: (n_turns, d_model)
        """
        turns = self.turns
        if roles is not None:
            turns = [t for t in turns if t.role in roles]

        if not turns:
            return None

        stacked = torch.stack([t.hidden_state.float() for t in turns], dim=0)
        return stacked if as_tensor else stacked.numpy()

    def get_window(self, k: int = 5) -> torch.Tensor:
        """Return the last k turns as a trajectory tensor.

        Returns:
            shape: (min(k, n_turns), d_model), or (1, d_model) of zeros
            if the buffer is empty
        """
        recent = self.turns[-k:] if len(self.turns) >= k else self.turns
        if not recent:
            return torch.zeros(1, self.d_model)
        return torch.stack([t.hidden_state.float() for t in recent], dim=0)

    def get_drift_vector(self) -> Optional[torch.Tensor]:
        """
        Compute the direction of activation drift across the conversation.
        Returns (d_model,) vector representing the net drift direction,
        or None if fewer than 2 turns available.

        This is a cheap proxy for trajectory direction before the full
        encoder runs — useful for early-exit flagging.
        """
        if len(self.turns) < 2:
            return None
        first = self.turns[0].hidden_state.float()
        last = self.turns[-1].hidden_state.float()
        drift = last - first
        norm = drift.norm()
        # Guard against divide-by-zero when first/last activations are
        # (near-)identical, e.g. a conversation that never moves semantically.
        if norm < 1e-8:
            return torch.zeros_like(drift)
        return drift / norm

    def get_pairwise_cosine_trajectory(self) -> List[float]:
        """
        Return cosine similarities between consecutive turn activations.
        A sharp drop signals a semantic shift — useful as a cheap
        anomaly signal before running the full encoder.
        """
        if len(self.turns) < 2:
            return []
        sims = []
        for i in range(len(self.turns) - 1):
            a = self.turns[i].hidden_state.float()
            b = self.turns[i + 1].hidden_state.float()
            sim = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()
            sims.append(sim)
        return sims

    def n_turns(self) -> int:
        """Number of turns currently stored (<= max_turns)."""
        return len(self.turns)

    def is_empty(self) -> bool:
        """True if no turns have been added (or all were reset)."""
        return len(self.turns) == 0

    def reset(self) -> None:
        """Clear all stored turns, e.g. when starting a new conversation."""
        self.turns.clear()

    def state_dict(self) -> Dict:
        """Serialize for checkpointing."""
        return {
            "max_turns": self.max_turns,
            "d_model": self.d_model,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "role": t.role,
                    "text": t.text,
                    "hidden_state": t.hidden_state.cpu(),
                    "token_count": t.token_count,
                }
                for t in self.turns
            ]
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "ConversationBuffer":
        """Reconstruct a ConversationBuffer from `state_dict()` output.

        Note: `layer_states` is not part of the serialized state (see
        `state_dict`), so restored TurnActivations only carry the
        mean-pooled `hidden_state`.
        """
        buf = cls(max_turns=state["max_turns"], d_model=state["d_model"])
        for t in state["turns"]:
            buf.add(TurnActivation(
                turn_index=t["turn_index"],
                role=t["role"],
                text=t["text"],
                hidden_state=t["hidden_state"],
                token_count=t["token_count"]
            ))
        return buf