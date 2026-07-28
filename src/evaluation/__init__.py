"""
Public API re-exports for the evaluation package.

Downstream code imports scorers and the evaluation harness directly from
`src.evaluation` (e.g. `from src.evaluation import EvaluationHarness`), so
the names below are intentionally re-exported here even though this file
does not use them itself — do not remove as "unused imports".
"""

from .scorer import HarmBenchScorer, StrongRejectScorer
from .eval_loop import EvaluationHarness
