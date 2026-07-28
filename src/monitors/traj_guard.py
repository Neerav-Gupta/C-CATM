"""
TrajGuard baseline monitor: single-turn activation-direction probing.

Fits a linear "safety direction" in hidden-state space from labeled
safe/unsafe examples (mean-difference direction), then flags a turn if
its hidden state projects far enough onto that direction. Each assistant
turn is scored independently — there's no memory of prior turns, which
is the cross-turn gap that C-CATM's trajectory tracking is designed to close.
"""
import numpy as np
import torch
from typing import Dict, Any, List
from .base import BaseMonitor
from ..utils.data_structures import AttackTrace


class TrajGuardMonitor(BaseMonitor):
    """
    TrajGuard baseline: monitors hidden-state drift within
    a single response. Does NOT track across turns —
    that's the gap C-CATM fills.
    Requires fit() to be called before score_trace().
    """

    def __init__(self, threshold: float = 0.0):
        """
        Args:
            threshold: projection score above which a turn is flagged
                as unsafe (see `_project`).
        """
        self.threshold = threshold
        self.safety_direction = None  # fitted from safe/unsafe pairs

    def fit(
        self,
        safe_hidden_states: List[np.ndarray],
        unsafe_hidden_states: List[np.ndarray]
    ) -> None:
        """
        Learn safety direction via mean difference in activation space.
        safe/unsafe_hidden_states: lists of (d_model,) numpy arrays.
        """
        safe_mean = np.mean(safe_hidden_states, axis=0)
        unsafe_mean = np.mean(unsafe_hidden_states, axis=0)
        direction = unsafe_mean - safe_mean
        self.safety_direction = direction / np.linalg.norm(direction)
        print(f"✓ TrajGuard fitted — direction norm: {np.linalg.norm(self.safety_direction):.4f}")

    def _project(self, hidden_state: torch.Tensor) -> float:
        """Project a hidden state onto the unsafe direction."""
        if self.safety_direction is None:
            raise RuntimeError("Call fit() before scoring")
        hs = hidden_state.numpy() if isinstance(hidden_state, torch.Tensor) else hidden_state
        return float(np.dot(hs, self.safety_direction))

    def score_trace(self, trace: AttackTrace) -> Dict[str, Any]:
        """
        Score each assistant turn independently.
        No cross-turn memory — single-turn limitation preserved.
        """
        asst_turns = trace.assistant_turns()
        per_turn_scores = []
        detection_turn = None

        for i, turn in enumerate(asst_turns):
            if turn.hidden_state is not None:
                score = self._project(turn.hidden_state)
                per_turn_scores.append(score)
                if score > self.threshold and detection_turn is None:
                    detection_turn = i
            else:
                per_turn_scores.append(None)

        flagged = detection_turn is not None

        return self._make_result(
            flagged=flagged,
            detection_turn=detection_turn,
            per_turn_scores=per_turn_scores
        )