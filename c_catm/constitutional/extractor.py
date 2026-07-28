"""
constitutional/extractor.py

Extracts constitutional concept vectors from a pre-trained SAE.

The core idea (from LLMSymGuard, arXiv:2508.16325):
    - A Sparse Autoencoder (SAE) decomposes LLM activations into a
      large dictionary of interpretable features (~16k–65k features)
    - Each feature corresponds to a human-interpretable concept
    - We identify which features correlate with safety-relevant concepts
      described in a safety constitution
    - Those feature directions become our "constitutional vectors" in
      activation space

We use SAELens (github.com/jbloomAus/SAELens) for pre-trained SAEs on
Llama-3.1-8B. No SAE training needed — just feature identification.

Pipeline:
    1. Load pre-trained SAE from SAELens
    2. For each constitutional rule, encode it as text and run through
       the model + SAE to get feature activations
    3. Identify the top-k features that activate strongly for that rule
    4. Average their decoder directions → constitutional concept vector
    5. Store as a (n_rules, d_model) matrix for the gate

Fallback: if SAELens is unavailable, use linear probes trained on
labeled safe/unsafe examples — see LinearProbeExtractor below.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class SAEConceptExtractor:
    """
    Extracts concept vectors using pre-trained SAE features.

    Usage:
        extractor = SAEConceptExtractor(model, tokenizer, sae)
        vectors = extractor.extract(constitution_rules)
        # vectors: (n_rules, d_model)
    """

    def __init__(self, model, tokenizer, sae, layer_idx=None,
             top_k_features=20, device="cuda"):
        """
        Args:
            model, tokenizer: the base LLM the SAE was trained on
            sae: pre-trained SAELens SAE for `layer_idx` of `model`
            layer_idx: transformer layer to extract activations from;
                auto-scaled from model depth if not given (see below)
            top_k_features: number of top-activating SAE features averaged
                into each rule's concept vector
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        # Auto-scale layer based on model depth if not specified
        if layer_idx is None:
            n_layers = len(model.model.layers)
            layer_idx = int(n_layers * 0.625)  # ~62.5% depth = layer 20 for Llama
        self.layer_idx = layer_idx
        self.top_k = top_k_features
        self.device = device
        self._cache = {}   # rule_text -> concept vector, avoids recomputing

    def _get_layer_activation(self, text: str) -> torch.Tensor:
        """Mean-pooled hidden state of `text` at self.layer_idx, (d_model,)."""
        is_gemma = hasattr(self.tokenizer, 'tokenizer')
        if is_gemma:
            inputs = self.tokenizer(text=text, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)
        else:
            inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        layer_act = outputs.hidden_states[self.layer_idx + 1]
        return layer_act.mean(dim=1).squeeze(0)

    def _get_sae_features(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Run activation through SAE encoder.
        Returns feature activations: (n_features,)
        """
        with torch.no_grad():
            # SAELens interface: sae.encode() returns sparse feature acts
            feature_acts = self.sae.encode(activation.unsqueeze(0))  # (1, n_features)
        return feature_acts.squeeze(0)                                # (n_features,)

    def _features_to_concept_vector(
        self, feature_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert sparse feature activations to a dense concept vector
        in the original activation space.

        Takes the top-k most active features and averages their
        decoder directions (weighted by activation strength).

        Returns: (d_model,) concept vector
        """
        # Get top-k feature indices
        topk_vals, topk_indices = torch.topk(feature_activations, self.top_k)

        # Weight feature decoder directions by activation strength
        # SAELens: sae.W_dec has shape (n_features, d_model)
        decoder_directions = self.sae.W_dec[topk_indices]       # (k, d_model)
        weights = F.softmax(topk_vals, dim=0).unsqueeze(1)       # (k, 1)
        concept_vector = (weights * decoder_directions).sum(dim=0)  # (d_model,)

        return F.normalize(concept_vector, dim=0)

    def extract_single(self, rule_text: str) -> torch.Tensor:
        """
        Extract concept vector for a single constitutional rule.
        Returns (d_model,) vector.
        """
        if rule_text in self._cache:
            return self._cache[rule_text]

        activation = self._get_layer_activation(rule_text)
        feature_acts = self._get_sae_features(activation)
        concept_vec = self._features_to_concept_vector(feature_acts)

        self._cache[rule_text] = concept_vec
        return concept_vec

    def extract(self, rules: List[str]) -> torch.Tensor:
        """
        Extract concept vectors for a list of constitutional rules.

        Args:
            rules: list of natural language safety rules

        Returns:
            concept_matrix: (n_rules, d_model) — one vector per rule
        """
        vectors = []
        for i, rule in enumerate(rules):
            print(f"  Extracting concept vector {i+1}/{len(rules)}: {rule[:60]}...")
            vec = self.extract_single(rule)
            vectors.append(vec)

        return torch.stack(vectors, dim=0)   # (n_rules, d_model)

    def save(self, path: str, rules: List[str]) -> None:
        """Save concept matrix and rules for reuse."""
        concept_matrix = self.extract(rules)
        torch.save({
            "concept_matrix": concept_matrix.cpu(),
            "rules": rules,
            "layer_idx": self.layer_idx,
            "top_k_features": self.top_k,
        }, path)
        print(f"Saved {len(rules)} concept vectors to {path}")

    @classmethod
    def load(cls, path: str) -> Tuple[torch.Tensor, List[str]]:
        """Load saved concept matrix and rules."""
        data = torch.load(path, map_location="cpu")
        return data["concept_matrix"], data["rules"]


class LinearProbeExtractor:
    """
    Fallback concept vector extractor using linear probes.

    If SAELens is not available or SAE features are too noisy,
    train a linear probe on labeled safe/unsafe examples and use
    the probe weight vector as the concept direction.

    This is less interpretable than SAE features but more robust
    and doesn't require a trained SAE.
    """

    def __init__(
        self,
        model,
        tokenizer,
        layer_idx: int = 20,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.probe_weights: Optional[torch.Tensor] = None

    def _get_activation(self, text: str) -> torch.Tensor:
        """Mean-pooled hidden state of `text` at self.layer_idx, (d_model,)."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[self.layer_idx + 1].mean(dim=1).squeeze(0)

    def fit(
        self,
        safe_texts: List[str],
        unsafe_texts: List[str],
        n_epochs: int = 10,
    ) -> torch.Tensor:
        """
        Train a linear probe and return its weight vector as the
        safety concept direction.

        Returns: (d_model,) normalized concept vector
        """
        print("Extracting activations for linear probe training...")
        safe_acts = torch.stack([self._get_activation(t) for t in safe_texts])
        unsafe_acts = torch.stack([self._get_activation(t) for t in unsafe_texts])

        X = torch.cat([safe_acts, unsafe_acts], dim=0).float()
        y = torch.cat([
            torch.zeros(len(safe_acts)),
            torch.ones(len(unsafe_acts))
        ]).long().to(self.device)
        X = X.to(self.device)

        # Train logistic regression via SGD
        probe = torch.nn.Linear(X.shape[1], 2, bias=True).to(self.device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

        for epoch in range(n_epochs):
            opt.zero_grad()
            logits = probe(X)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            if (epoch + 1) % 5 == 0:
                acc = (logits.argmax(dim=1) == y).float().mean()
                print(f"  Probe epoch {epoch+1}: loss={loss.item():.4f} acc={acc.item():.3f}")

        # The weight difference (unsafe class - safe class) is the concept direction
        w = probe.weight.detach()                           # (2, d_model)
        concept_dir = w[1] - w[0]                           # (d_model,)
        self.probe_weights = F.normalize(concept_dir, dim=0)

        return self.probe_weights

    def get_concept_vectors(self, rules: List[str]) -> torch.Tensor:
        """
        For linear probe, all rules share the same unsafe direction.
        Returns (n_rules, d_model) where each row is the same vector.
        This is a simplification — use SAEExtractor for rule-specific vectors.
        """
        if self.probe_weights is None:
            raise RuntimeError("Call fit() first")
        return self.probe_weights.unsqueeze(0).expand(len(rules), -1)