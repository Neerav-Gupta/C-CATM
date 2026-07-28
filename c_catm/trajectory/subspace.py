"""
trajectory/subspace.py

Contrastive training for the safety subspace.

Loss function: NT-Xent (normalized temperature-scaled cross-entropy)
    — same as SimCLR, adapted for our setting where positives are
    (safe_conv_A, safe_conv_B) pairs and negatives are (safe, unsafe) pairs.

Why contrastive and not just binary classification?
    Binary classification would push safe/unsafe into two point clusters.
    Contrastive training creates a rich geometry in the subspace: safe
    conversations form a compact cluster, unsafe conversations form their
    own cluster, and crucially the *direction* from safe→unsafe becomes
    a meaningful axis you can project constitutional concept vectors onto.
    That directional structure is what the constitutional gate needs.

Training data construction:
    - Safe positives: benign multi-turn conversations (Ultrachat, ShareGPT safe subset)
    - Unsafe examples: HarmBench attack traces from Week 2 with success=True
    - Augmentation: random turn dropout (drop 1 turn), role swap noise

This module handles:
    1. Dataset class
    2. Contrastive loss
    3. Training loop
    4. Evaluation: separation metric (inter-class distance / intra-class distance)
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .encoder import TrajectoryEncoder


# ─── Dataset ───────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Loads pre-extracted activation trajectories from JSONL.

    Each line in the JSONL has:
        {
          "label": 0 or 1,          # 0=safe, 1=unsafe
          "turns": [
            {"role": "user", "hidden_state": [...], "text": "..."},
            ...
          ]
        }

    You produce this file by running your Week 2 attack traces through
    extract_activations.py (see bottom of this file).
    """

    def __init__(
        self,
        jsonl_path: str,
        max_turns: int = 20,
        d_model: int = 4096,
        augment: bool = True,
        augment_prob: float = 0.3,
    ):
        """
        Args:
            jsonl_path: path to a JSONL file of pre-extracted trajectories
                (see class docstring for the record schema)
            max_turns: truncate/pad trajectories to this many turns
            d_model: expected hidden dim of stored activations; shorter
                vectors are zero-padded, longer ones truncated (see
                `_load_trajectory`)
            augment: whether to apply random turn-dropout augmentation
            augment_prob: per-sample probability of applying the dropout
                augmentation when `augment=True`
        """
        self.max_turns = max_turns
        self.d_model = d_model
        self.augment = augment
        self.augment_prob = augment_prob
        self.samples: List[Dict] = []

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        self.safe_indices = [i for i, s in enumerate(self.samples) if s["label"] == 0]
        self.unsafe_indices = [i for i, s in enumerate(self.samples) if s["label"] == 1]

        print(f"Loaded {len(self.safe_indices)} safe, {len(self.unsafe_indices)} unsafe trajectories")

    def __len__(self) -> int:
        """Number of trajectory samples in the dataset."""
        return len(self.samples)

    def _load_trajectory(self, sample: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a fixed-length (max_turns, d_model) trajectory tensor from a
        raw sample's variable-length list of turns, applying optional
        dropout augmentation and converting absolute activations into
        turn-to-turn deltas.

        Args:
            sample: one decoded JSONL record, i.e. {"label": ..., "turns": [...]}

        Returns:
            trajectory: (max_turns, d_model) float tensor. Unused trailing
                rows (beyond the number of turns present) are left as zero.
            mask: (max_turns,) bool padding mask, True = padding row
                (consumed by the transformer as `src_key_padding_mask`)
        """
        turns = sample["turns"][:self.max_turns]

        # Turn-dropout augmentation: randomly remove one turn so the encoder
        # doesn't overfit to exact turn positions/counts.
        if self.augment and len(turns) > 2 and random.random() < self.augment_prob:
            drop_idx = random.randint(0, len(turns) - 1)
            turns = [t for i, t in enumerate(turns) if i != drop_idx]

        # Extract raw hidden states first
        raw = []
        for turn in turns:
            hs = torch.tensor(turn["hidden_state"], dtype=torch.float32)
            # Reconcile stored hidden-state size against the configured
            # d_model (activations may have been extracted with a
            # different model/layer than d_model expects).
            if hs.shape[0] > self.d_model:
                hs = hs[:self.d_model]
            elif hs.shape[0] < self.d_model:
                pad = torch.zeros(self.d_model - hs.shape[0])
                hs = torch.cat([hs, pad])
            # L2-normalize so the encoder sees direction, not magnitude —
            # raw activation norms can vary a lot across turns/conversations
            # and aren't themselves safety-relevant.
            norm = hs.norm()
            if norm > 1e-8:
                hs = hs / norm
            raw.append(hs)

        trajectory = torch.zeros(self.max_turns, self.d_model)
        mask = torch.ones(self.max_turns, dtype=torch.bool)

        if len(raw) == 0:
            return trajectory, mask

        # Turn 0: use raw hidden state
        trajectory[0] = raw[0]
        mask[0] = False

        # Turns 1+: use delta from previous turn
        # This captures direction of change, not absolute position
        for i in range(1, len(raw)):
            delta = raw[i] - raw[i-1]
            norm = delta.norm()
            if norm > 1e-8:
                delta = delta / norm
            trajectory[i] = delta
            mask[i] = False

        return trajectory, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            trajectory: (max_turns, d_model) float tensor
            padding_mask: (max_turns,) bool tensor, True = padding row
            label: 0 (safe) or 1 (unsafe)
        """
        sample = self.samples[idx]
        traj, mask = self._load_trajectory(sample)
        label = sample["label"]
        return traj, mask, label

    def get_contrastive_pair(self, idx: int) -> Tuple[
        torch.Tensor, torch.Tensor,  # anchor traj, mask
        torch.Tensor, torch.Tensor,  # positive traj, mask
        int                          # anchor label
    ]:
        """
        Returns an anchor and a positive from the same class.
        Used for contrastive training.
        """
        sample = self.samples[idx]
        label = sample["label"]

        # Pick a positive from same class
        if label == 0:
            pos_idx = random.choice(self.safe_indices)
        else:
            pos_idx = random.choice(self.unsafe_indices)

        anchor_traj, anchor_mask = self._load_trajectory(sample)
        pos_traj, pos_mask = self._load_trajectory(self.samples[pos_idx])

        return anchor_traj, anchor_mask, pos_traj, pos_mask, label


# ─── Loss ──────────────────────────────────────────────────────────────────────

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised NT-Xent / SupCon loss (see module docstring): for each
    embedding, all other same-label embeddings in the batch are positives
    and all differently-labeled embeddings are negatives.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch, proj_dim) trajectory embeddings (need not be
                pre-normalized; this re-normalizes internally)
            labels: (batch,) int labels, 0 (safe) or 1 (unsafe)

        Returns:
            scalar loss tensor. Falls back to a zero loss with
            `requires_grad=True` (rather than NaN) if the batch is
            degenerate — either all one class, or no sample has a positive
            partner — so callers can skip the backward step cleanly.
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Degenerate batch: nothing to contrast against.
        if labels.unique().numel() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Force normalize
        embeddings = F.normalize(embeddings.float(), dim=-1)

        # Cosine similarity scaled by temperature
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Self mask
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Positive mask
        labels_col = labels.unsqueeze(1)
        pos_mask = (labels_col == labels_col.T) & ~self_mask
        neg_mask = (labels_col != labels_col.T)

        # Numerically stable: subtract max per row before exp
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim_stable = sim - sim_max.detach()

        # Zero out self
        exp_sim = torch.exp(sim_stable) * (~self_mask).float()

        # Sum of negatives per row
        neg_sum = (exp_sim * neg_mask.float()).sum(dim=1)
        
        # For each positive pair, compute loss
        losses = []
        for i in range(batch_size):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue
            for j in pos_indices:
                pos_sim = sim_stable[i, j]
                # log(exp(pos) / (exp(pos) + sum_neg))
                log_prob = pos_sim - torch.log(
                    torch.exp(pos_sim) + neg_sum[i] + 1e-8
                )
                losses.append(-log_prob)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()

# ─── Training loop ─────────────────────────────────────────────────────────────

class SubspaceTrainer:
    """
    Trains the TrajectoryEncoder on contrastive safety subspace objective.

    Typical Week 3 usage:
        trainer = SubspaceTrainer(encoder, train_dataset, val_dataset)
        trainer.train(epochs=20)
        trainer.save("checkpoints/encoder_epoch20.pt")
        metrics = trainer.evaluate_separation()
    """

    def __init__(
        self,
        encoder: TrajectoryEncoder,
        train_dataset: TrajectoryDataset,
        val_dataset: Optional[TrajectoryDataset] = None,
        batch_size: int = 32,
        lr: float = 1e-4,
        temperature: float = 0.07,
        device: str = "cuda",
    ):
        """
        Args:
            encoder: the TrajectoryEncoder (or subclass) to train
            train_dataset: labeled trajectories used for the contrastive objective
            val_dataset: held-out trajectories used by `evaluate_separation`
            batch_size: batch size used for `val_loader` only — the
                training batch size is fixed by BalancedBatchSampler below
                (batch_size=32, n_unsafe_per_batch=8), independent of this arg
            lr: (unused as of this trainer version — `self.optimizer` below
                is constructed with a fixed learning rate instead)
            temperature: (unused as of this trainer version — `self.loss_fn`
                below is constructed with a fixed temperature instead)
            device: torch device string the encoder and data are moved to
        """
        self.encoder = encoder.to(device)
        self.device = device
        self.loss_fn = SupervisedContrastiveLoss(temperature=0.2)

        safe_indices = [i for i, s in enumerate(train_dataset.samples) if s["label"] == 0]
        unsafe_indices = [i for i, s in enumerate(train_dataset.samples) if s["label"] == 1]

        print(f"Training with {len(safe_indices)} safe, {len(unsafe_indices)} unsafe")

        class BalancedBatchSampler:
            """
            Batch sampler that guarantees every batch contains a fixed
            number of unsafe examples.

            Unsafe (attack) trajectories are typically far rarer than safe
            ones, so uniform random sampling would make many batches
            entirely safe — with no negatives to contrast against, those
            batches contribute zero loss (see SupervisedContrastiveLoss).
            Forcing `n_unsafe_per_batch` unsafe examples into every batch
            ensures the contrastive loss always has both classes to work with.
            """

            def __init__(self, safe_idx, unsafe_idx, batch_size, n_unsafe_per_batch=4):
                self.safe_idx = safe_idx
                self.unsafe_idx = unsafe_idx
                self.batch_size = batch_size
                self.n_unsafe = min(n_unsafe_per_batch, len(unsafe_idx))
                self.n_safe = batch_size - self.n_unsafe

            def __iter__(self):
                safe_perm = torch.randperm(len(self.safe_idx)).tolist()
                unsafe_perm = torch.randperm(len(self.unsafe_idx)).tolist()

                safe_pool = [self.safe_idx[i] for i in safe_perm]
                unsafe_pool = [self.unsafe_idx[i] for i in unsafe_perm]

                # Unsafe examples are scarcer than safe ones, so the unsafe
                # pool is replenished (with reshuffled repeats) until there
                # are enough to cover every batch drawn from the safe pool.
                while len(unsafe_pool) < self.n_unsafe * (len(safe_pool) // self.n_safe + 1):
                    unsafe_pool += [self.unsafe_idx[i]
                                for i in torch.randperm(len(self.unsafe_idx)).tolist()]

                batches = []
                safe_pos = 0
                unsafe_pos = 0

                while safe_pos + self.n_safe <= len(safe_pool):
                    batch = (safe_pool[safe_pos:safe_pos + self.n_safe] +
                            unsafe_pool[unsafe_pos:unsafe_pos + self.n_unsafe])
                    random.shuffle(batch)
                    batches.append(batch)
                    safe_pos += self.n_safe
                    unsafe_pos += self.n_unsafe

                return iter(batches)

            def __len__(self):
                return len(self.safe_idx) // (self.batch_size - self.n_unsafe)

        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=BalancedBatchSampler(
                safe_indices, unsafe_indices,
                batch_size=32,
                n_unsafe_per_batch=8
            ),
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        ) if val_dataset else None

        # Only train the encoder — base LLM is already frozen
        self.optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=3e-5,
            weight_decay=1e-4,
            eps=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )

        self.best_separation = 0.0
        self.history = {"train_loss": [], "val_separation": []}

    def train_epoch(self) -> float:
        """
        Run one training epoch over `train_loader`.

        Returns:
            mean training loss over batches that contributed a gradient
            step (see below — NaN/Inf and exactly-zero-loss batches are
            skipped and excluded from the average)
        """
        self.encoder.train()
        total_loss = 0.0
        n_batches = 0

        for trajectories, masks, labels in self.train_loader:
            trajectories = trajectories.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            embeddings = self.encoder(trajectories, masks)
            loss = self.loss_fn(embeddings, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # SupervisedContrastiveLoss returns exactly 0.0 (not just near-zero)
            # for degenerate batches with no valid positive pair — skip those
            # rather than taking a no-op optimizer step.
            if loss.item() == 0.0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / n_batches if n_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate_separation(self) -> Dict[str, float]:
        """
        Separation metric: ratio of inter-class distance to intra-class distance.
        Higher = better separation = better safety subspace.
        Target: > 2.0 before moving on.
        """
        if self.val_loader is None:
            return {}

        self.encoder.eval()
        safe_embeds, unsafe_embeds = [], []

        for trajectories, masks, labels in self.val_loader:
            trajectories = trajectories.to(self.device)
            masks = masks.to(self.device)
            embeds = self.encoder(trajectories, masks).cpu()

            for emb, lbl in zip(embeds, labels):
                if lbl == 0:
                    safe_embeds.append(emb)
                else:
                    unsafe_embeds.append(emb)

        safe_embeds = torch.stack(safe_embeds) if safe_embeds else torch.zeros(1, self.encoder.proj_dim)
        unsafe_embeds = torch.stack(unsafe_embeds) if unsafe_embeds else torch.zeros(1, self.encoder.proj_dim)

        safe_centroid = safe_embeds.mean(dim=0)
        unsafe_centroid = unsafe_embeds.mean(dim=0)

        # Inter-class distance
        inter = F.pairwise_distance(
            safe_centroid.unsqueeze(0), unsafe_centroid.unsqueeze(0)
        ).item()

        # Intra-class distances
        safe_intra = torch.pdist(safe_embeds).mean().item() if len(safe_embeds) > 1 else 1.0
        unsafe_intra = torch.pdist(unsafe_embeds).mean().item() if len(unsafe_embeds) > 1 else 1.0
        avg_intra = (safe_intra + unsafe_intra) / 2

        separation = inter / (avg_intra + 1e-8)

        return {
            "separation_ratio": separation,
            "inter_class_dist": inter,
            "intra_safe": safe_intra,
            "intra_unsafe": unsafe_intra,
        }

    def train(self, epochs: int = 20, save_dir: str = "checkpoints") -> None:
        """
        Run the full training loop: `epochs` calls to `train_epoch`, with
        separation evaluated each epoch and the best-separation checkpoint
        saved to `{save_dir}/encoder_best.pt`.
        """
        Path(save_dir).mkdir(exist_ok=True)
        print(f"Training trajectory encoder for {epochs} epochs...")

        for epoch in range(epochs):
            loss = self.train_epoch()
            self.history["train_loss"].append(loss)

            metrics = self.evaluate_separation()
            sep = metrics.get("separation_ratio", 0)
            self.history["val_separation"].append(sep)

            print(f"Epoch {epoch+1:3d}/{epochs} | loss={loss:.4f} | separation={sep:.3f}")

            # Save best checkpoint
            if sep > self.best_separation:
                self.best_separation = sep
                self.save(f"{save_dir}/encoder_best.pt")

        print(f"Training complete. Best separation ratio: {self.best_separation:.3f}")
        print("Target was >2.0 — ", "✓ PASS" if self.best_separation > 2.0 else "✗ FAIL — check data quality")

    def save(self, path: str) -> None:
        """Checkpoint encoder weights, training history, and enough config
        (d_model, proj_dim) for `load_encoder` to reconstruct the module."""
        torch.save({
            "encoder_state_dict": self.encoder.state_dict(),
            "best_separation": self.best_separation,
            "history": self.history,
            "encoder_config": {
                "d_model": self.encoder.d_model,
                "proj_dim": self.encoder.proj_dim,
            }
        }, path)
        print(f"Saved encoder to {path}")

    @classmethod
    def load_encoder(cls, path: str, device: str = "cuda") -> TrajectoryEncoder:
        """
        Load a checkpoint written by `save` and return an eval-mode encoder.

        Note: this always reconstructs a plain `TrajectoryEncoder`, even if
        the checkpoint came from a `DeltaTrajectoryEncoder` — the saved
        `encoder_config` doesn't record the subclass. Callers that trained a
        `DeltaTrajectoryEncoder` should reconstruct that class manually and
        load `state_dict()` into it directly.
        """
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint["encoder_config"]
        encoder = TrajectoryEncoder(d_model=cfg["d_model"], proj_dim=cfg["proj_dim"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        return encoder.to(device).eval()