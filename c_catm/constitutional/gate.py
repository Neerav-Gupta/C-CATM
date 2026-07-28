"""
constitutional/gate.py

The constitutional gating layer. This is the component that fuses
the text-side constitutional rules with the activation-space trajectory.

How it works:
    1. The trajectory encoder produces an embedding in the safety subspace: (proj_dim,)
    2. The concept matrix maps constitutional rules to activation space: (n_rules, d_model)
    3. The gate projects the trajectory embedding against each constitutional
       concept vector and computes a per-rule alignment score
    4. Scores above threshold on any rule → flag the conversation
    5. A learned combination layer blends all rule scores into a final
       gate probability

The key innovation vs. TrajGuard:
    TrajGuard monitors drift toward a generic "unsafe" direction.
    The constitutional gate checks drift toward SPECIFIC harmful concepts
    defined in natural language — meaning the gate is interpretable
    (you can say "flagged due to weapons synthesis rule") and can be
    updated by editing the constitution without retraining.

The key innovation vs. Constitutional Classifiers:
    CC checks per-turn text. The gate checks the trajectory embedding —
    which is the cross-turn cumulative drift signal, not any single turn.
    Decomposition attacks that split harm across turns are invisible to CC
    but detectable by the gate.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GateResult:
    """Output of the constitutional gate for one conversation."""
    flagged: bool
    gate_probability: float              # 0–1, probability of unsafe
    per_rule_scores: List[float]         # alignment score per rule
    triggered_rules: List[str]           # short_names of triggered rules
    triggered_categories: List[str]      # categories of triggered rules
    trajectory_norm: float               # L2 norm of trajectory embedding
    max_rule_score: float                # highest per-rule score


class ConstitutionalGate(nn.Module):
    """
    Constitutional gating layer.

    Takes:
        - trajectory_embedding: (batch, proj_dim) from TrajectoryEncoder
        - concept_matrix: (n_rules, d_model) from SAEConceptExtractor

    Returns:
        - gate_probability: (batch,) probability of unsafe
        - per_rule_scores: (batch, n_rules)

    The concept matrix is registered as a fixed buffer (not trained).
    Only the combination layer is trained.
    """

    def __init__(
        self,
        proj_dim: int = 256,             # trajectory embedding dimension
        d_model: int = 4096,             # LLM hidden dimension
        n_rules: int = 32,               # number of constitutional rules
        rule_weights: Optional[List[float]] = None,
        gate_threshold: float = 0.5,     # flag if gate_probability > threshold
        rule_threshold: float = 0.3,     # individual rule trigger threshold
    ):
        super().__init__()

        self.proj_dim = proj_dim
        self.d_model = d_model
        self.n_rules = n_rules
        self.gate_threshold = gate_threshold
        self.rule_threshold = rule_threshold

        # Concept matrix: (n_rules, d_model) — set via set_concept_matrix()
        # Registered as buffer so it moves with .to(device) but isn't trained
        self.register_buffer(
            "concept_matrix",
            torch.zeros(n_rules, d_model)
        )

        # Rule weights: upweight high-severity rules
        if rule_weights is None:
            rule_weights = [1.0] * n_rules
        self.register_buffer(
            "rule_weights",
            torch.tensor(rule_weights, dtype=torch.float32)
        )

        # Bridge: project trajectory embedding into d_model space
        # so it's comparable to constitutional concept vectors
        self.bridge = nn.Sequential(
            nn.Linear(proj_dim, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Combination layer: (n_rules,) → scalar gate probability
        # This is trained to combine rule scores optimally
        self.combiner = nn.Sequential(
            nn.Linear(n_rules, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._concept_matrix_set = False

    def set_concept_matrix(
        self,
        concept_matrix: torch.Tensor,
        rule_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Set the constitutional concept matrix.

        Args:
            concept_matrix: (n_rules, d_model) — one concept vector per rule
            rule_weights: optional per-rule weights
        """
        assert concept_matrix.shape[0] == self.n_rules, (
            f"Expected {self.n_rules} rules, got {concept_matrix.shape[0]}"
        )
        self.concept_matrix.data = F.normalize(
            concept_matrix.float(), dim=-1
        ).to(self.concept_matrix.device)

        if rule_weights is not None:
            self.rule_weights.data = torch.tensor(
                rule_weights, dtype=torch.float32
            ).to(self.rule_weights.device)

        self._concept_matrix_set = True

    def forward(
        self,
        trajectory_embedding: torch.Tensor,   # (batch, proj_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory_embedding: L2-normalized (batch, proj_dim)

        Returns:
            gate_probability: (batch,)
            per_rule_scores: (batch, n_rules)
        """
        # Project trajectory embedding to d_model space
        traj_in_model_space = self.bridge(trajectory_embedding)  # (B, d_model)
        traj_in_model_space = F.normalize(traj_in_model_space, dim=-1)

        # Cosine similarity against each constitutional concept vector
        # concept_matrix: (n_rules, d_model)
        # result: (B, n_rules)
        per_rule_scores = torch.matmul(
            traj_in_model_space, self.concept_matrix.T
        )                                                          # (B, n_rules)

        # Apply rule weights
        weighted_scores = per_rule_scores * self.rule_weights.unsqueeze(0)

        # Combine into final gate probability
        gate_probability = self.combiner(weighted_scores).squeeze(-1)  # (B,)

        return gate_probability, per_rule_scores

    def score_conversation(
        self,
        trajectory_embedding: torch.Tensor,   # (proj_dim,) single conversation
        rule_short_names: List[str],
        rule_categories: List[str],
    ) -> GateResult:
        """
        Score a single conversation and return an interpretable GateResult.
        """
        with torch.no_grad():
            gate_prob, rule_scores = self.forward(
                trajectory_embedding.unsqueeze(0)
            )

        gate_prob_val = gate_prob.item()
        rule_scores_list = rule_scores.squeeze(0).tolist()

        # Identify triggered rules
        triggered_rules = [
            name for name, score in zip(rule_short_names, rule_scores_list)
            if score > self.rule_threshold
        ]
        triggered_categories = list(set(
            cat for cat, score in zip(rule_categories, rule_scores_list)
            if score > self.rule_threshold
        ))

        return GateResult(
            flagged=gate_prob_val > self.gate_threshold,
            gate_probability=gate_prob_val,
            per_rule_scores=rule_scores_list,
            triggered_rules=triggered_rules,
            triggered_categories=triggered_categories,
            trajectory_norm=trajectory_embedding.norm().item(),
            max_rule_score=max(rule_scores_list) if rule_scores_list else 0.0,
        )

    def get_top_triggered_rules(
        self,
        rule_scores: List[float],
        rule_names: List[str],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return top-k triggered rules with their scores. For logging."""
        pairs = sorted(zip(rule_names, rule_scores), key=lambda x: -x[1])
        return pairs[:top_k]


class AdaptiveConstitutionalGate(ConstitutionalGate):
    """
    Extension for Week 4: adaptive gate that adjusts thresholds based
    on conversation length and recent trajectory velocity.

    Longer conversations with fast activation drift should trigger at
    lower thresholds — the attacker has had more turns to decompose the request.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Learned threshold adjustment based on conversation features
        self.threshold_adapter = nn.Sequential(
            nn.Linear(3, 16),   # inputs: n_turns, drift_norm, max_cosine_drop
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def get_adaptive_threshold(
        self,
        n_turns: int,
        drift_norm: float,
        max_cosine_drop: float,
    ) -> float:
        """
        Compute adaptive gate threshold based on conversation dynamics.

        Intuition:
            - More turns + high drift + sharp cosine drops → lower threshold
              (more suspicious, flag earlier)
            - Few turns + low drift → higher threshold (less suspicious)
        """
        features = torch.tensor(
            [[n_turns / 20.0, drift_norm, max_cosine_drop]],
            dtype=torch.float32
        ).to(next(self.parameters()).device)

        with torch.no_grad():
            adjustment = self.threshold_adapter(features).item()

        # Adjustment: [0, 1] → [-0.2, +0.1] shift on base threshold
        return self.gate_threshold + (adjustment - 0.67) * 0.3