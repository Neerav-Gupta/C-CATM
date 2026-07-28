"""
train.py

Week 3 training script. Two stages:

Stage 1: Extract activations from Week 2 attack traces + safe conversations
    → produces data/trajectories_train.jsonl and data/trajectories_val.jsonl

Stage 2: Train the TrajectoryEncoder on contrastive objective
    → produces checkpoints/encoder_best.pt

Run as:
    python train.py --stage extract --traces_path results/week2.jsonl
    python train.py --stage train --epochs 20
    python train.py --stage both  (runs both sequentially)

Expected runtime on A100:
    Stage 1: ~12 hours (2,000 conversations × 7 turns × activation extraction)
    Stage 2: ~2 hours (contrastive training, small head on frozen activations)
"""

from __future__ import annotations
import argparse
import json
from typing import List
import torch

from trajectory.subspace import TrajectoryDataset, SubspaceTrainer


# ─── Stage 1: Activation extraction ───────────────────────────────────────────

def extract_activations_from_traces(
    traces_dir, model, tokenizer, model_id,
    output_path, layer_idx=20, device="cuda"
):
    """
    Stage 1 driver: walk every trace JSONL under `traces_dir`, ensure each
    turn has a hidden-state vector (extracting one via `model`/`tokenizer`
    if missing), and write the flattened records to `output_path` as a
    single JSONL file consumable by TrajectoryDataset.

    `model`/`tokenizer` may be None — traces that already carry a
    "hidden_state" field never touch the model, so callers who know all
    traces are pre-extracted can skip loading the LLM entirely (see
    `main()`, which does exactly this).
    """
    import jsonlines as jl
    from pathlib import Path

    trace_files = list(Path(traces_dir).rglob("*.jsonl"))
    print(f"Found {len(trace_files)} trace files")
    Path(output_path).parent.mkdir(exist_ok=True)

    is_gemma = "gemma" in model_id.lower() if model_id else False

    def get_hidden_state(text: str) -> list:
        """Run `text` through the model and return its mean-pooled hidden
        state at `layer_idx` as a plain Python list (for JSON storage)."""
        # Only called if hidden_state missing — requires model
        if model is None:
            raise RuntimeError("Model needed but not loaded")
        if is_gemma:
            # Gemma's tokenizer needs the `text=` kwarg rather than positional
            inputs = tokenizer(text=text, return_tensors="pt",
                             truncation=True, max_length=512).to(device)
        else:
            inputs = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=512).to(device)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # hidden_states[0] is the embedding layer output, so index layer_idx+1
        # is the output of transformer block layer_idx
        hs = outputs.hidden_states[layer_idx + 1]
        return hs.mean(dim=1).squeeze(0).cpu().float().tolist()

    written = 0
    with open(output_path, "w") as f_out:
        for trace_file in trace_files:
            with jl.open(trace_file) as reader:
                for record in reader:
                    turns = record.get("turns", [])
                    if not turns:
                        continue

                    turns_with_hs = []
                    for turn in turns:
                        if "hidden_state" in turn:
                            hs = turn["hidden_state"]
                        else:
                            hs = get_hidden_state(turn.get("content", ""))
                        turns_with_hs.append({
                            "role": turn["role"],
                            "text": turn.get("content", turn.get("text", "")),
                            "hidden_state": hs,
                        })

                    out_record = {
                        "label": record["label"],
                        "turns": turns_with_hs,
                        "attack": record.get("attack", "unknown"),
                        "behavior": record.get("behavior", ""),
                        "model_id": record.get("model_id", ""),
                    }
                    f_out.write(json.dumps(out_record) + "\n")
                    written += 1

                    if written % 100 == 0:
                        print(f"  Processed {written} traces...")

    print(f"Extraction complete. {written} traces written to {output_path}")


def add_safe_conversations(
    output_path: str,
    model,
    tokenizer,
    n_safe: int = 1000,
    layer_idx: int = 20,
    device: str = "cuda",
) -> None:
    """
    Add safe conversation trajectories to the training data.

    Safe conversations sourced from:
    - Ultrachat 200k (benign instructional conversations)
    - ShareGPT safe subset (filtered for no harmful content)

    For Week 3, you can bootstrap with a small set of hand-crafted
    safe multi-turn conversations if the full datasets aren't available.
    """
    from datasets import load_dataset

    print(f"Adding {n_safe} safe conversations...")

    def get_hidden_state(text: str) -> List[float]:
        """Mean-pool the model's hidden state at `layer_idx` for `text`."""
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[layer_idx + 1]
        return hs.mean(dim=1).squeeze(0).cpu().tolist()

    # Load Ultrachat
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    written = 0
    with open(output_path, "a") as f_out:
        for sample in dataset:
            if written >= n_safe:
                break

            messages = sample.get("messages", [])
            if len(messages) < 2:
                continue

            turns_with_hs = []
            for msg in messages[:10]:  # cap at 10 turns
                hs = get_hidden_state(msg["content"])
                turns_with_hs.append({
                    "role": msg["role"],
                    "text": msg["content"],
                    "hidden_state": hs,
                })

            record = {
                "label": 0,
                "turns": turns_with_hs,
                "source": "ultrachat",
            }
            f_out.write(json.dumps(record) + "\n")
            written += 1

            if written % 100 == 0:
                print(f"  Added {written} safe conversations...")

    print(f"Added {written} safe conversations to {output_path}")


# ─── Stage 2: Train encoder ────────────────────────────────────────────────────

from trajectory.encoder import DeltaTrajectoryEncoder

def train_encoder(
    data_path: str,
    val_path: str,
    output_dir: str = "checkpoints",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    d_model: int = 4096,
    proj_dim: int = 256,
    device: str = "cuda",
) -> None:
    """
    Stage 2 driver: train a DeltaTrajectoryEncoder with SubspaceTrainer's
    contrastive objective on the extracted trajectory datasets, save
    checkpoints to `output_dir`, and print final safe/unsafe separation
    metrics (see SubspaceTrainer.evaluate_separation).
    """
    print(f"Training trajectory encoder...")
    print(f"  data: {data_path}")
    print(f"  epochs: {epochs}, batch_size: {batch_size}")

    train_dataset = TrajectoryDataset(data_path, d_model=d_model, augment=True)
    val_dataset = TrajectoryDataset(val_path, d_model=d_model, augment=False)

    encoder = DeltaTrajectoryEncoder(d_model=d_model, proj_dim=proj_dim)

    trainer = SubspaceTrainer(
        encoder=encoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    trainer.train(epochs=epochs, save_dir=output_dir)

    # Final evaluation
    metrics = trainer.evaluate_separation()
    print("\nFinal separation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if metrics.get("separation_ratio", 0) < 2.0:
        print("\nWARNING: Separation ratio below target (2.0).")
        print("Consider: more training data, more epochs, or smaller proj_dim.")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point: parse args and run Stage 1 (extract), Stage 2
    (train), or both in sequence, per `--stage`."""
    parser = argparse.ArgumentParser(description="C-CATM Week 3 training pipeline")
    parser.add_argument("--stage", choices=["extract", "train", "both"], default="both")
    parser.add_argument("--traces_path", default="results/week2.jsonl")
    parser.add_argument("--data_path", default="data/trajectories_train.jsonl")
    parser.add_argument("--val_path", default="data/trajectories_val.jsonl")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--traces_dir",
                        default="/workspace/results/traces",
                        help="Directory containing full trace JSONL files")
    parser.add_argument("--d_model", type=int, default=3584)
    args = parser.parse_args()

    if args.stage in ("extract", "both"):
        # All traces already have hidden states saved — no model needed
        print("All traces have pre-saved hidden states — skipping model loading...")
        
        extract_activations_from_traces(
            traces_dir=args.traces_dir,
            model=None,      # not needed
            tokenizer=None,  # not needed
            model_id=args.model_id,
            output_path=args.data_path,
            layer_idx=args.layer_idx,
            device=args.device,
        )

        # Simple 80/20 train/val split
        import random
        with open(args.data_path) as f:
            lines = f.readlines()
        random.shuffle(lines)
        split = int(0.8 * len(lines))
        with open(args.data_path, "w") as f:
            f.writelines(lines[:split])
        with open(args.val_path, "w") as f:
            f.writelines(lines[split:])
        print(f"Train: {split} samples, Val: {len(lines)-split} samples")

    if args.stage in ("train", "both"):
        train_encoder(
            data_path=args.data_path,
            val_path=args.val_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_model=args.d_model,
            device=args.device,
        )

    print("\nWeek 3 training pipeline complete.")
    print("Next: extract constitutional concept vectors, then wire monitor.py into harness.")


if __name__ == "__main__":
    main()