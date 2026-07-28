"""
monitor.py

Unified C-CATM monitor. This is the single entry point your evaluation
harness calls. It wires together:

    ConversationBuffer  →  TrajectoryEncoder  →  ConstitutionalGate
                                                      ↑
                                           Constitutional concept vectors
                                           (from SAEConceptExtractor)

Usage in Week 2 harness:
    monitor = CCATMMonitor.from_pretrained("checkpoints/encoder_best.pt",
                                           "checkpoints/concepts.pt")
    result = monitor.score_turn(buffer, turn_activation)
    # or
    result = monitor.score_trace(attack_trace)

The monitor scores after EVERY assistant turn — not just the final one.
This gives you:
    1. Detection turn index (when did C-CATM first flag it?)
    2. Trajectory of gate probabilities across turns
    3. Which constitutional rules were triggered and when
"""

from __future__ import annotations
import torch
from typing import Optional, List, Dict
from dataclasses import dataclass

from trajectory.buffer import ConversationBuffer, TurnActivation
from trajectory.encoder import TrajectoryEncoder
from constitutional.gate import ConstitutionalGate, GateResult
from constitutional.constitution import CONSTITUTION, get_rule_weights


@dataclass
class MonitorResult:
    """Full monitoring result for a conversation."""
    flagged: bool
    detection_turn: Optional[int]        # first turn where flagged=True
    final_gate_probability: float
    gate_probabilities: List[float]      # gate prob after each turn
    gate_results: List[GateResult]       # full result per turn
    n_turns_evaluated: int
    triggered_rules: List[str]           # union of all triggered rules
    triggered_categories: List[str]
    early_exit: bool = False             # flagged before final turn


class CCATMMonitor:
    """
    C-CATM: Cross-Turn Constitutional Activation Trajectory Monitor.

    This is the defense your paper proposes. Instantiate once, reuse
    across conversations by calling reset() between them.
    """

    def __init__(
        self,
        encoder: TrajectoryEncoder,
        gate: ConstitutionalGate,
        rule_short_names: List[str],
        rule_categories: List[str],
        d_model: int = 4096,
        max_turns: int = 32,
        device: str = "cuda",
        early_exit: bool = True,         # flag and stop as soon as threshold exceeded
        early_exit_threshold: float = 0.8,  # higher threshold for early exit
    ):
        self.encoder = encoder.to(device).eval()
        self.gate = gate.to(device).eval()
        self.rule_short_names = rule_short_names
        self.rule_categories = rule_categories
        self.device = device
        self.early_exit = early_exit
        self.early_exit_threshold = early_exit_threshold

        self.buffer = ConversationBuffer(max_turns=max_turns, d_model=d_model)

    def add_turn(self, turn: TurnActivation) -> None:
        """Add a turn activation to the conversation buffer."""
        self.buffer.add(turn)

    def score_current(self) -> GateResult:
        """
        Score the current conversation state (all turns in buffer so far).
        Call this after each assistant turn.
        """
        with torch.no_grad():
            # Get full trajectory from buffer
            trajectory = self.buffer.get_trajectory()       # (n_turns, d_model)
            trajectory = trajectory.to(self.device)

            # Encode trajectory to safety subspace embedding
            traj_embedding = self.encoder.encode_single(trajectory)  # (proj_dim,)

            # Score against constitutional gate
            result = self.gate.score_conversation(
                traj_embedding,
                self.rule_short_names,
                self.rule_categories,
            )

        return result

    def score_trace(self, trace) -> MonitorResult:
        """
        Score a complete AttackTrace from your Week 2 harness.
        Processes turns sequentially, scoring after each assistant response.

        Args:
            trace: AttackTrace object from Week 2 attack harness

        Returns:
            MonitorResult with full trajectory of gate probabilities
        """
        self.reset()
        gate_probabilities = []
        gate_results = []
        detection_turn = None
        all_triggered_rules = set()
        all_triggered_categories = set()

        # Process turns in order
        for turn in trace.turns:
            if turn.hidden_state is None:
                continue

            # Add to buffer
            self.add_turn(turn)

            # Only score after assistant turns (we want to catch the response)
            if turn.role == "assistant":
                result = self.score_current()
                gate_probabilities.append(result.gate_probability)
                gate_results.append(result)
                all_triggered_rules.update(result.triggered_rules)
                all_triggered_categories.update(result.triggered_categories)

                if result.flagged and detection_turn is None:
                    detection_turn = turn.turn_index

                # Early exit: if very high confidence, stop evaluating
                if self.early_exit and result.gate_probability > self.early_exit_threshold:
                    break

        flagged = detection_turn is not None
        final_prob = gate_probabilities[-1] if gate_probabilities else 0.0

        return MonitorResult(
            flagged=flagged,
            detection_turn=detection_turn,
            final_gate_probability=final_prob,
            gate_probabilities=gate_probabilities,
            gate_results=gate_results,
            n_turns_evaluated=len(gate_probabilities),
            triggered_rules=list(all_triggered_rules),
            triggered_categories=list(all_triggered_categories),
            early_exit=self.early_exit and (
                gate_probabilities[-1] > self.early_exit_threshold
                if gate_probabilities else False
            ),
        )

    def reset(self) -> None:
        """Reset buffer for a new conversation."""
        self.buffer.reset()

    # ── Factory methods ────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, encoder_checkpoint, concept_checkpoint,
                        device="cuda", gate_threshold=0.5, **kwargs):
        """
        Build a ready-to-use CCATMMonitor from a trained encoder checkpoint
        and a saved constitutional concept matrix (both produced by
        train.py / extract_concepts.py). This is the normal way to
        construct a monitor for evaluation.

        Args:
            encoder_checkpoint: path to a checkpoint saved by SubspaceTrainer
            concept_checkpoint: path to a concept matrix saved by
                SAEConceptExtractor.save() (constitutional/extract_concepts.py)
            gate_threshold: probability above which a conversation is flagged
            **kwargs: forwarded to CCATMMonitor.__init__ (e.g. early_exit)
        """
        from trajectory.subspace import SubspaceTrainer
        from constitutional.extractor import SAEConceptExtractor

        encoder = SubspaceTrainer.load_encoder(encoder_checkpoint, device)
        
        # Get d_model from encoder
        d_model = encoder.d_model
        print(f"  Encoder d_model: {d_model}")

        concept_matrix, rules = SAEConceptExtractor.load(concept_checkpoint)
        rule_short_names = [r.short_name for r in CONSTITUTION]
        rule_categories = [r.category for r in CONSTITUTION]
        rule_weights = get_rule_weights()

        n_rules = concept_matrix.shape[0]

        gate = ConstitutionalGate(
            proj_dim=encoder.proj_dim,
            d_model=concept_matrix.shape[1],
            n_rules=n_rules,
            rule_weights=rule_weights,
            gate_threshold=gate_threshold,
        )
        gate.set_concept_matrix(concept_matrix, rule_weights)

        return cls(
            encoder=encoder,
            gate=gate,
            rule_short_names=rule_short_names,
            rule_categories=rule_categories,
            d_model=d_model,  # use encoder's actual d_model
            device=device,
            **kwargs,
        )

    @classmethod
    def from_scratch(
        cls,
        model,
        tokenizer,
        sae=None,
        device: str = "cuda",
        encoder_kwargs: Dict = None,
        gate_kwargs: Dict = None,
    ) -> "CCATMMonitor":
        """
        Initialize C-CATM from scratch (before training).
        Used at the start of Week 3 to set up the training pipeline.
        """
        from constitutional.extractor import SAEConceptExtractor
        from constitutional.constitution import CONSTITUTION, get_all_rule_texts, get_rule_weights

        encoder_kwargs = encoder_kwargs or {}
        gate_kwargs = gate_kwargs or {}

        encoder = TrajectoryEncoder(**encoder_kwargs)
        rules = get_all_rule_texts()
        rule_weights = get_rule_weights()
        n_rules = len(rules)

        gate = ConstitutionalGate(
            proj_dim=encoder.proj_dim,
            d_model=encoder_kwargs.get("d_model", 4096),
            n_rules=n_rules,
            rule_weights=rule_weights,
            **gate_kwargs,
        )

        # Extract concept vectors
        if sae is not None:
            print("Extracting SAE concept vectors...")
            extractor = SAEConceptExtractor(model, tokenizer, sae, device=device)
            concept_matrix = extractor.extract(rules)
        else:
            print("SAE not provided — initializing with random concept vectors.")
            print("Call gate.set_concept_matrix() with real vectors before training.")
            concept_matrix = torch.randn(n_rules, encoder_kwargs.get("d_model", 4096))

        gate.set_concept_matrix(concept_matrix, rule_weights)

        rule_short_names = [r.short_name for r in CONSTITUTION]
        rule_categories = [r.category for r in CONSTITUTION]

        return cls(
            encoder=encoder,
            gate=gate,
            rule_short_names=rule_short_names,
            rule_categories=rule_categories,
            device=device,
        )


# ─── Sanity check ──────────────────────────────────────────────────────────────

def run_sanity_check(monitor: CCATMMonitor, d_model: int = 3584) -> None:
    """
    Quick sanity check: feed random safe and unsafe trajectories,
    confirm the monitor runs without error.

    Run this after Week 3 setup before starting any real evaluation.
    """
    print("Running C-CATM sanity check...")

    from trajectory.buffer import TurnActivation

    # Simulate a safe 3-turn conversation
    monitor.reset()
    for i in range(6):  # 3 user + 3 assistant
        role = "user" if i % 2 == 0 else "assistant"
        fake_hs = torch.randn(d_model) * 0.1  # small norm = safe-ish
        monitor.add_turn(TurnActivation(
            turn_index=i // 2, role=role,
            text=f"safe turn {i}", hidden_state=fake_hs
        ))

    safe_result = monitor.score_current()
    print(f"  Safe conversation: gate_prob={safe_result.gate_probability:.3f} flagged={safe_result.flagged}")

    # Simulate an unsafe trajectory (larger activations drifting toward unsafe)
    monitor.reset()
    base = torch.randn(d_model)
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        # Escalating activations
        fake_hs = base * (1 + i * 0.3) + torch.randn(d_model) * 0.05
        monitor.add_turn(TurnActivation(
            turn_index=i // 2, role=role,
            text=f"escalating turn {i}", hidden_state=fake_hs
        ))

    unsafe_result = monitor.score_current()
    print(f"  Escalating conversation: gate_prob={unsafe_result.gate_probability:.3f} flagged={unsafe_result.flagged}")

    print("Sanity check complete. ✓ (scores may not be meaningful until trained)")