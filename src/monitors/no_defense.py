"""
No-defense baseline monitor: never flags anything.

Used to measure the unmitigated attack success rate (ASR) that every
other monitor (LlamaGuard, TrajGuard, constitutional, C-CATM) is
compared against.
"""
from typing import Dict, Any
from .base import BaseMonitor
from ..utils.data_structures import AttackTrace


class NoDefenseMonitor(BaseMonitor):
    """
    No defense — always passes everything through.
    Gives the upper bound on ASR: how often attacks succeed
    with zero mitigation. Every other monitor is compared against this.
    """

    def score_trace(self, trace: AttackTrace) -> Dict[str, Any]:
        """Always returns unflagged, regardless of trace content."""
        return self._make_result(
            flagged=False,
            detection_turn=None
        )