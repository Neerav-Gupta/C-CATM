"""
C-CATM monitor wrapper: this project's cross-turn trajectory monitor.

Thin adapter around the standalone `c_catm` package's `CCATMMonitor`,
translating this codebase's AttackTrace/Turn representation into the
turn-by-turn activation buffer C-CATM expects. Unlike the other
baselines here (LlamaGuard, TrajGuard, constitutional), C-CATM
accumulates state across the whole conversation, which is what lets it
catch attacks that only become apparent when turns are considered together.
"""
import sys
import torch
from typing import Dict, Any
from .base import BaseMonitor
from ..utils.data_structures import AttackTrace

# The C-CATM monitor implementation lives in a separate package outside
# this repo; add it to the path so `from monitor import CCATMMonitor`
# (below) resolves without requiring it to be pip-installed.
sys.path.insert(0, "/workspace/c_catm")


class CCATMMonitorWrapper(BaseMonitor):
    """Adapter exposing the C-CATM trajectory monitor via the BaseMonitor interface."""

    def __init__(
        self,
        encoder_checkpoint: str = "/workspace/c_catm/checkpoints/encoder_best.pt",
        concept_checkpoint: str = "/workspace/c_catm/checkpoints/concepts.pt",
        gate_checkpoint: str = "/workspace/c_catm/checkpoints/gate_trained.pt",
        gate_threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Load the C-CATM monitor (encoder + concept bank) and then load
        the separately-trained gate weights on top of it.

        Args:
            encoder_checkpoint: path to the trained trajectory encoder.
            concept_checkpoint: path to the trained concept bank.
            gate_checkpoint: path to the trained gate classifier weights,
                applied on top of the pretrained monitor.
            gate_threshold: decision threshold for the gate's flagged output.
            device: torch device to load the monitor onto.
        """
        # Imported here (rather than at module level) because `monitor`
        # only resolves once sys.path has been patched above, and to
        # avoid requiring the c_catm package for callers that only use
        # the other (non-C-CATM) monitors in this module.
        from monitor import CCATMMonitor
        self.monitor = CCATMMonitor.from_pretrained(
            encoder_checkpoint=encoder_checkpoint,
            concept_checkpoint=concept_checkpoint,
            gate_threshold=gate_threshold,
            device=device,
        )

        # Load trained gate weights on top
        import torch
        gate_data = torch.load(gate_checkpoint, map_location=device)
        self.monitor.gate.load_state_dict(gate_data["gate_state_dict"])
        self.monitor.gate.eval()
        print(f"✓ C-CATM monitor loaded with TRAINED gate (threshold={gate_threshold})")

    def score_trace(self, trace: AttackTrace) -> Dict[str, Any]:
        """
        Feed every turn's hidden state through the C-CATM trajectory
        buffer (as normalized inter-turn deltas, matching how the
        monitor was trained) and return its flagged/gate verdict.

        Args:
            trace: the AttackTrace to score.

        Returns:
            Result dict with flagged, detection_turn (always None here —
            C-CATM's gate verdict applies to the trajectory as a whole
            rather than a single turn), and details including
            gate_probability and triggered rules/categories.
        """
        from trajectory.buffer import TurnActivation

        self.monitor.reset()
        expected_dim = self.monitor.buffer.d_model

        # Collect and normalize all hidden states first
        processed_hs = []
        valid_turns = []

        for turn in trace.turns:
            if turn.hidden_state is None:
                continue
            hs = turn.hidden_state.float()
            # Resize to expected dim — hidden states may come from a
            # different-width target model than the monitor was trained on.
            if hs.shape[0] > expected_dim:
                hs = hs[:expected_dim]
            elif hs.shape[0] < expected_dim:
                pad = torch.zeros(expected_dim - hs.shape[0])
                hs = torch.cat([hs, pad])
            # Normalize
            norm = hs.norm()
            if norm > 1e-8:
                hs = hs / norm
            processed_hs.append(hs)
            valid_turns.append(turn)

        if not processed_hs:
            return self._make_result(flagged=False, detection_turn=None)

        # Compute deltas — same as training preprocessing. The monitor
        # was trained on turn-to-turn activation deltas (not raw hidden
        # states) since it's the direction/magnitude of drift between
        # turns that signals escalation, not the absolute activation.
        delta_hs = [processed_hs[0]]  # first turn: raw
        for i in range(1, len(processed_hs)):
            delta = processed_hs[i] - processed_hs[i-1]
            norm = delta.norm()
            if norm > 1e-8:
                delta = delta / norm
            delta_hs.append(delta)

        # Add delta features to buffer
        for i, (turn, hs) in enumerate(zip(valid_turns, delta_hs)):
            self.monitor.add_turn(TurnActivation(
                turn_index=turn.turn_index,
                role=turn.role,
                text=turn.content,
                hidden_state=hs,
            ))

        result = self.monitor.score_current()

        return self._make_result(
            flagged=result.flagged,
            detection_turn=None,
            gate_probability=result.gate_probability,
            triggered_rules=result.triggered_rules,
            triggered_categories=result.triggered_categories,
        )