"""
check_training_data.py

Pre-flight sanity check for the raw attack-trace data that train.py's
Stage 1 (extract_activations_from_traces) consumes. Run this before
train.py to catch data problems early — e.g. too few unsafe traces,
an imbalanced/missing model or attack type — rather than discovering
them after a long extraction run.

Scans every JSONL trace file under `trace_dir` and prints:
    - label balance (safe vs. unsafe trace counts)
    - per-model and per-attack-type counts
    - turn-length distribution
    - a readiness verdict against minimum dataset-size heuristics

This is a standalone script (module-level code, no importable functions);
run directly with `python check_training_data.py`.
"""
from pathlib import Path
import jsonlines
from collections import defaultdict

trace_dir = Path("/workspace/results/traces")
label_counts = defaultdict(int)
model_counts = defaultdict(int)
attack_counts = defaultdict(int)
turn_lengths = []

for f in trace_dir.rglob("*.jsonl"):
    with jsonlines.open(f) as r:
        for row in r:
            label = row.get("label")
            label_counts[label] += 1
            model_counts[row.get("model_id", "?").split("/")[-1]] += 1
            attack_counts[row.get("attack", "?")] += 1
            turn_lengths.append(len(row.get("turns", [])))

total = sum(label_counts.values())
print(f"Total traces: {total}")
print(f"\nLabel distribution:")
print(f"  Safe   (0): {label_counts[0]} ({label_counts[0]/total*100:.1f}%)")
print(f"  Unsafe (1): {label_counts[1]} ({label_counts[1]/total*100:.1f}%)")

print(f"\nPer model:")
for k, v in sorted(model_counts.items()):
    print(f"  {k}: {v}")

print(f"\nPer attack:")
for k, v in sorted(attack_counts.items()):
    print(f"  {k}: {v}")

print(f"\nTurn length stats:")
if turn_lengths:
    import statistics
    print(f"  Mean: {statistics.mean(turn_lengths):.1f}")
    print(f"  Min:  {min(turn_lengths)}")
    print(f"  Max:  {max(turn_lengths)}")

print(f"\nTraining readiness:")
# Minimum dataset-size heuristics for a viable contrastive training run;
# unsafe traces are the scarcer, harder-to-collect class so the bar is lower.
print(f"  Unsafe traces >= 100: {'✓' if label_counts[1] >= 100 else '✗ need more'} ({label_counts[1]})")
print(f"  Safe traces >= 500:   {'✓' if label_counts[0] >= 500 else '✗ need more'} ({label_counts[0]})")
print(f"  Ready to train:       {'✓ YES' if label_counts[1] >= 100 and label_counts[0] >= 500 else '✗ NOT YET'}")