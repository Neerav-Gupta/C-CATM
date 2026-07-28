"""
Base class shared by all defense/detection monitors.

Defines the common interface (`score_trace`) and result shape used by
every monitor baseline in this project — no-defense, LlamaGuard,
TrajGuard, the constitutional filter, and the C-CATM monitor — so they
can be swapped interchangeably when evaluating attack traces.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..utils.data_structures import AttackTrace


class BaseMonitor(ABC):
    """
    Base class for all monitors.
    All monitors take an AttackTrace and return a result dict with:
    - flagged: bool
    - detection_turn: int or None
    - details: dict of monitor-specific scores
    """

    @abstractmethod
    def score_trace(self, trace: AttackTrace) -> Dict[str, Any]:
        """
        Score a full attack trace and decide whether to flag it.

        Args:
            trace: the AttackTrace (all turns + hidden states) to evaluate.

        Returns:
            Result dict as produced by `_make_result` — flagged,
            detection_turn, and monitor-specific details.
        """
        raise NotImplementedError

    def _make_result(
        self,
        flagged: bool,
        detection_turn: int = None,
        **details
    ) -> Dict[str, Any]:
        """
        Build the standard monitor result dict.

        Args:
            flagged: whether the monitor flagged the trace as unsafe.
            detection_turn: index of the turn where detection first
                triggered, or None if never flagged.
            **details: any monitor-specific scores/metadata, collected
                into the "details" key.

        Returns:
            Dict with keys "flagged", "detection_turn", "details".
        """
        return {
            "flagged": flagged,
            "detection_turn": detection_turn,
            "details": details
        }