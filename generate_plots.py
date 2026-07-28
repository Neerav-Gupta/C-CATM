"""
Generate all result tables, plots, and figures for the C-CATM paper.
Run this to regenerate all paper-ready outputs.

Usage:
    python generate_plots.py --output /workspace/paper_results/
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jsonlines

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

def load_all_results():
    """Load and parse all evaluation results.

    Recursively scans /workspace/results for *.jsonl eval output files
    (skipping error/backup/raw-trace files) and groups rows by
    (model_key, monitor).

    Returns:
        dict mapping (model_key, monitor_name) -> list of result row dicts,
        where model_key is "llama" or "qwen" (inferred from model_id) and
        monitor_name comes from each row's "monitor" field.
    """
    results = defaultdict(list)

    for f in Path("/workspace/results").rglob("*.jsonl"):
        # Skip error logs, backups, and raw activation traces — only the
        # final per-example eval rows should feed the metrics below.
        if "error" in f.name or "backup" in str(f) or "traces" in str(f):
            continue
        with jsonlines.open(f) as r:
            for row in r:
                model = row.get("model_id", "").lower()
                if "llama" in model:
                    model_key = "llama"
                elif "qwen" in model:
                    model_key = "qwen"
                else:
                    continue
                
                monitor = row.get("monitor")
                key = (model_key, monitor)
                results[key].append(row)
    
    return results

def compute_metrics(rows):
    """Compute FPR, Precision, Recall, F1 from results.

    Args:
        rows: list of eval result dicts (see load_all_results), each with
            "flagged_by_monitor" (bool) and "hb_label" (1 = harmful/unsafe
            ground truth, 0 = safe).

    Returns:
        dict of metrics (fpr/precision/recall/f1 as percentages 0-100, plus
        tp/fp/fn/tn/total/harmful counts), or None if `rows` is empty or
        contains no harmful (hb_label == 1) examples — FPR/precision/recall
        are undefined without both classes present.
    """
    if not rows:
        return None

    hb1 = sum(1 for r in rows if r.get("hb_label") == 1)
    if hb1 == 0:
        return None

    tp = sum(1 for r in rows if r.get("flagged_by_monitor") and r.get("hb_label") == 1)
    fp = sum(1 for r in rows if r.get("flagged_by_monitor") and r.get("hb_label") == 0)
    fn = sum(1 for r in rows if not r.get("flagged_by_monitor") and r.get("hb_label") == 1)
    tn = len(rows) - tp - fp - fn
    
    fpr = fp/(fp+tn)*100 if (fp+tn) > 0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp) > 0 else 0
    rec = tp/(tp+fn)*100 if (tp+fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    
    return {
        "fpr": fpr,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": len(rows),
        "harmful": hb1
    }

def generate_results_table(results, output_dir):
    """Generate main results comparison table.

    Returns:
        dict table_data[model][monitor] -> metrics dict from
        compute_metrics() (or None), for model in {"llama", "qwen"} and
        monitor in {"no_defense", "llama_guard", "traj_guard",
        "constitutional", "ccatm"}. Consumed by the plotting functions below.
    """
    print("Generating results table...")
    
    table_data = {}
    for model in ["llama", "qwen"]:
        table_data[model] = {}
        for monitor in ["no_defense", "llama_guard", "traj_guard", "constitutional", "ccatm"]:
            key = (model, monitor)
            metrics = compute_metrics(results.get(key, []))
            table_data[model][monitor] = metrics
    
    # Save as JSON
    with open(output_dir / "results_table.json", "w") as f:
        json.dump(table_data, f, indent=2)
    
    # Print formatted table
    print("\n" + "=" * 120)
    print("C-CATM FINAL RESULTS")
    print("=" * 120)
    
    for model in ["llama", "qwen"]:
        print(f"\n{model.upper()}")
        print("-" * 120)
        print(f"{'Monitor':<20} {'FPR%':>8} {'Precision%':>12} {'Recall%':>10} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'Total':>6}")
        print("-" * 120)
        
        for monitor in ["no_defense", "llama_guard", "traj_guard", "constitutional", "ccatm"]:
            m = table_data[model][monitor]
            if m is None:
                print(f"{monitor:<20} {'N/A':>8} {'N/A':>12} {'N/A':>10} {'N/A':>8}")
            else:
                print(f"{monitor:<20} {m['fpr']:>7.1f}% {m['precision']:>11.1f}% {m['recall']:>9.1f}% {m['f1']:>7.1f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['total']:>6}")
    
    return table_data

def generate_per_attack_analysis(results, output_dir):
    """Analyze performance per attack type.

    Returns:
        dict attack_metrics[monitor][attack] -> metrics dict from
        compute_metrics() (or None), for the four multi-turn jailbreak
        attack types used in the paper. Consumed by
        generate_attack_comparison_plot().
    """
    print("Generating per-attack analysis...")
    
    attack_results = defaultdict(lambda: defaultdict(list))
    
    for (model, monitor), rows in results.items():
        for row in rows:
            attack = row.get("attack", "unknown")
            attack_results[monitor][attack].append(row)
    
    attack_metrics = {}
    for monitor in ["llama_guard", "traj_guard", "constitutional", "ccatm"]:
        attack_metrics[monitor] = {}
        for attack in ["crescendo", "jigsaw", "echo_chamber", "safe_in_isolation"]:
            rows = attack_results[monitor][attack]
            attack_metrics[monitor][attack] = compute_metrics(rows)
    
    with open(output_dir / "per_attack_metrics.json", "w") as f:
        json.dump(attack_metrics, f, indent=2)
    
    print("\nPer-Attack Performance (Llama + Qwen combined):")
    print("-" * 80)
    print(f"{'Attack Type':<25} {'Llama Guard':>15} {'TrajGuard':>15} {'Constitutional':>20} {'C-CATM':>15}")
    print(f"{'':>25} {'F1':>15} {'F1':>15} {'F1':>20} {'F1':>15}")
    print("-" * 80)
    
    for attack in ["crescendo", "jigsaw", "echo_chamber", "safe_in_isolation"]:
        metrics = [attack_metrics[m][attack] for m in ["llama_guard", "traj_guard", "constitutional", "ccatm"]]
        f1_values = [m['f1'] if m else 0 for m in metrics]
        print(f"{attack:<25} {f1_values[0]:>14.1f} {f1_values[1]:>14.1f} {f1_values[2]:>19.1f} {f1_values[3]:>14.1f}")
    
    return attack_metrics

def generate_attack_comparison_plot(attack_metrics, output_dir):
    """Generate per-attack comparison plot."""
    print("Generating per-attack comparison plot...")
    
    attacks = ["crescendo", "jigsaw", "echo_chamber", "safe_in_isolation"]
    monitors = ["llama_guard", "traj_guard", "constitutional", "ccatm"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("C-CATM Performance per Attack Type", fontsize=16, fontweight="bold")
    
    for idx, attack in enumerate(attacks):
        ax = axes[idx // 2, idx % 2]
        f1_values = [attack_metrics[m][attack]['f1'] if attack_metrics[m][attack] else 0 for m in monitors]
        
        colors = ["#FF6B6B" if m != "ccatm" else "#51CF66" for m in monitors]
        bars = ax.bar(monitors, f1_values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
        
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(f"{attack.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 80)
        ax.set_xticklabels(monitors, rotation=45, ha="right")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_attack_comparison.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "per_attack_comparison.png", format="png", bbox_inches="tight", dpi=300)
    print(f"  Saved: per_attack_comparison.pdf / .png")
    plt.close()

def generate_detection_turn_analysis(results, output_dir):
    """Analyze at which turn harmful responses are detected.

    Restricted to the "ccatm" monitor's true positives (flagged and
    hb_label == 1). Writes per-model (mean/median/std/min/max) turn-index
    stats to detection_turn_stats.json.

    Returns:
        dict mapping model_key -> stats dict (see above), only for models
        with at least one true-positive detection.
    """
    print("Generating detection turn analysis...")

    detection_turns = defaultdict(list)

    for (model, monitor), rows in results.items():
        if monitor != "ccatm":
            continue
        for row in rows:
            if row.get("flagged_by_monitor") and row.get("hb_label") == 1:
                # Older result rows don't record an explicit detection_turn;
                # for those, fall back to the trace length (i.e. treat the
                # detection as having happened at the final turn).
                det_turn = row.get("detection_turn", 0)
                detection_turns[model].append(det_turn if det_turn else len(row.get("turns", [])))
    
    detection_stats = {}
    for model in ["llama", "qwen"]:
        turns = detection_turns[model]
        if turns:
            detection_stats[model] = {
                "mean_turn": float(np.mean(turns)),
                "median_turn": float(np.median(turns)),
                "std_turn": float(np.std(turns)),
                "min_turn": int(np.min(turns)),
                "max_turn": int(np.max(turns))
            }
    
    with open(output_dir / "detection_turn_stats.json", "w") as f:
        json.dump(detection_stats, f, indent=2)
    
    print("\nDetection Turn Statistics (when C-CATM flags harmful responses):")
    print("-" * 70)
    print(f"{'Model':<10} {'Mean Turn':>12} {'Median':>10} {'Std Dev':>10} {'Range':>20}")
    print("-" * 70)
    
    for model in ["llama", "qwen"]:
        if model in detection_stats:
            s = detection_stats[model]
            print(f"{model:<10} {s['mean_turn']:>11.2f} {s['median_turn']:>9.0f} {s['std_turn']:>9.2f} {s['min_turn']:>4} - {s['max_turn']:<4}")
    
    return detection_stats

def generate_harm_category_breakdown(results, output_dir):
    """Compute per-monitor F1 broken down by harm category (e.g. Malware,
    Fraud, CSAM), and write category_breakdown.json.

    Maps each result row's "behavior" field to a category using the
    Behavior->Category lookup in /workspace/data/stratified_100.json (falls
    back to "Unknown" if that file is missing or a behavior isn't found).

    Not currently called from main(); kept as a standalone reporting
    utility for producing an optional per-category breakdown table for the
    paper's appendix.
    """
    print("Generating harm category breakdown...")

    category_results = defaultdict(lambda: defaultdict(list))

    # Load behavior categories
    import json as json_module
    try:
        with open("/workspace/data/stratified_100.json") as f:
            behaviors = json_module.load(f)
            behavior_to_category = {b.get("Behavior", ""): b.get("Category", "Unknown") for b in behaviors}
    except:
        behavior_to_category = {}
    
    for (model, monitor), rows in results.items():
        for row in rows:
            behavior = row.get("behavior", "")
            category = behavior_to_category.get(behavior, "Unknown")
            category_results[monitor][category].append(row)
    
    category_metrics = {}
    for monitor in ["llama_guard", "traj_guard", "constitutional", "ccatm"]:
        category_metrics[monitor] = {}
        for category in ["Illegal Activity", "Malware", "Physical Harm", "Fraud", "CSAM", "Non-consensual Intimate Images", "Hate Speech", "Misinformation"]:
            rows = category_results[monitor].get(category, [])
            if rows:
                category_metrics[monitor][category] = compute_metrics(rows)
    
    with open(output_dir / "category_breakdown.json", "w") as f:
        json.dump(category_metrics, f, indent=2)
    
    print("\nHarm Category Breakdown (F1 Score):")
    print("-" * 90)
    
    categories = list(set([cat for m in category_metrics.values() for cat in m.keys()]))
    print(f"{'Category':<30} {'Llama Guard':>15} {'TrajGuard':>15} {'Constitutional':>15} {'C-CATM':>15}")
    print("-" * 90)
    
    for category in sorted(categories):
        values = []
        for monitor in ["llama_guard", "traj_guard", "constitutional", "ccatm"]:
            m = category_metrics[monitor].get(category)
            values.append(f"{m['f1']:>7.1f}" if m else "    N/A")
        print(f"{category:<30} {values[0]}>13 {values[1]}>13 {values[2]}>13 {values[3]}>13")

def generate_comparison_plots(table_data, output_dir):
    """Generate comparison plots (PNG + PDF)."""
    print("Generating comparison plots...")
    
    monitors = ["llama_guard", "traj_guard", "constitutional", "ccatm"]
    metrics_list = ["fpr", "precision", "recall", "f1"]
    metric_labels = ["FPR (%)", "Precision (%)", "Recall (%)", "F1 Score"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("C-CATM vs Baseline Monitors", fontsize=16, fontweight="bold")
    
    for idx, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        llama_vals = [table_data["llama"][m][metric] if table_data["llama"][m] else 0 for m in monitors]
        qwen_vals = [table_data["qwen"][m][metric] if table_data["qwen"][m] else 0 for m in monitors]
        
        x = np.arange(len(monitors))
        width = 0.35
        
        ax.bar(x - width/2, llama_vals, width, label="Llama-3.1-8B", alpha=0.8, color="#4C72B0")
        ax.bar(x + width/2, qwen_vals, width, label="Qwen2.5-7B", alpha=0.8, color="#DD8452")
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(monitors, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_plots.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "comparison_plots.png", format="png", bbox_inches="tight", dpi=300)
    print(f"  Saved: comparison_plots.pdf / .png")
    plt.close()

def generate_fpr_vs_recall_plot(table_data, output_dir):
    """Generate FPR vs Recall scatter plot (PDF + PNG)."""
    print("Generating FPR vs Recall plot...")
    
    fig, ax = plt.subplots(figsize=(11, 9))
    
    monitors = ["llama_guard", "traj_guard", "constitutional", "ccatm"]
    colors = {"llama": "#4C72B0", "qwen": "#DD8452"}
    markers = {"llama_guard": "o", "traj_guard": "s", "constitutional": "^", "ccatm": "D"}
    
    for model in ["llama", "qwen"]:
        for monitor in monitors:
            m = table_data[model][monitor]
            if m:
                ax.scatter(m["fpr"], m["recall"], s=400, 
                          color=colors[model], marker=markers[monitor],
                          alpha=0.7, edgecolors="black", linewidth=2.5)
                ax.annotate(f"{monitor}\n({model})", (m["fpr"], m["recall"]), 
                           xytext=(8, 8), textcoords="offset points", fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax.set_xlabel("False Positive Rate (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Recall (%)", fontsize=13, fontweight="bold")
    ax.set_title("FPR vs Recall Trade-off Across Models", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fpr_vs_recall.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "fpr_vs_recall.png", format="png", bbox_inches="tight", dpi=300)
    print(f"  Saved: fpr_vs_recall.pdf / .png")
    plt.close()

def generate_latency_plot(output_dir):
    """Generate latency plot for C-CATM only.

    The per-component latency numbers below are not computed from `results`
    — they are fixed values taken from separate timing measurements (see
    README latency figure) and hardcoded here purely for plotting.
    """
    print("Generating latency plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # C-CATM latency breakdown
    components = ["Trajectory\nEncoding", "Constitutional\nGate", "Total"]
    latencies = [4.2, 2.0, 6.2]  # ms (estimated from measurements)
    colors = ["#4C72B0", "#DD8452", "#51CF66"]
    
    bars = ax.bar(components, latencies, color=colors, alpha=0.85, 
                   edgecolor="black", linewidth=2)
    
    ax.set_ylabel("Latency (ms per trace)", fontsize=13, fontweight="bold")
    ax.set_title("C-CATM Inference Latency", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 8)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for bar, lat in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f"{lat:.1f}ms", ha="center", va="bottom", 
               fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "latency_breakdown.pdf", format="pdf", 
                bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "latency_breakdown.png", format="png", 
                bbox_inches="tight", dpi=300)
    print(f"  Saved: latency_breakdown.pdf / .png")
    plt.close()

def generate_ablation_table(output_dir):
    """Generate ablation study table from saved results.

    Reads /workspace/ablation_results.json (produced by
    ablation_studies.py) — a list of per-ablation result dicts. Each dict's
    schema differs by which ablation produced it, so the branches below
    dispatch on which pair of keys is present (e.g. "delta_encoder"/
    "base_encoder" vs "trained_encoder"/"random_embedding") to pick the
    right pair of configurations to print.
    """
    print("Generating ablation table...")

    # Load ablation results
    ablation_file = Path("/workspace/ablation_results.json")
    if not ablation_file.exists():
        print("⚠️  WARNING: ablation_results.json not found")
        print("   Run: python ablation_studies.py --all")
        return
    
    with open(ablation_file) as f:
        ablations = json.load(f)
    
    # Save for reference
    with open(output_dir / "ablations.json", "w") as f:
        json.dump(ablations, f, indent=2)
    
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100)
    
    for ablation in ablations:
        name = ablation["name"]
        hypothesis = ablation["hypothesis"]
        conclusion = ablation["conclusion"]
        
        print(f"\n{name}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Conclusion: {conclusion}")
        print("-"*100)
        
        if "delta_encoder" in ablation:
            d = ablation["delta_encoder"]
            b = ablation["base_encoder"]
            diff = ablation["difference"]
            print(f"{'Configuration':<35} {'FPR%':>10} {'Precision%':>15} {'Recall%':>12} {'F1':>10}")
            print(f"{'DeltaTrajectoryEncoder':<35} {d['fpr']:>9.1f}% {d['precision']:>14.1f}% {d['recall']:>11.1f}% {d['f1']:>9.2f}")
            print(f"{'TrajectoryEncoder (base)':<35} {b['fpr']:>9.1f}% {b['precision']:>14.1f}% {b['recall']:>11.1f}% {b['f1']:>9.2f}")
            print(f"{'Difference':<35} {diff['fpr']:>9.1f}% {diff['precision']:>14.1f}% {diff['recall']:>11.1f}% {diff['f1']:>9.2f}")
        
        elif "with_rules" in ablation:
            w = ablation["with_rules"]
            wo = ablation["without_rules"]
            diff = ablation["difference"]
            print(f"{'Configuration':<35} {'FPR%':>10} {'Precision%':>15} {'Recall%':>12} {'F1':>10}")
            print(f"{'Full C-CATM (with rules)':<35} {w['fpr']:>9.1f}% {w['precision']:>14.1f}% {w['recall']:>11.1f}% {w['f1']:>9.2f}")
            print(f"{'Encoder-only (no rule weights)':<35} {wo['fpr']:>9.1f}% {wo['precision']:>14.1f}% {wo['recall']:>11.1f}% {wo['f1']:>9.2f}")
            print(f"{'Difference':<35} {diff['fpr']:>9.1f}% {diff['precision']:>14.1f}% {diff['recall']:>11.1f}% {diff['f1']:>9.2f}")
        
        elif "trained_encoder" in ablation:
            t = ablation["trained_encoder"]
            r = ablation["random_embedding"]
            diff = ablation["difference"]
            print(f"{'Configuration':<35} {'FPR%':>10} {'Precision%':>15} {'Recall%':>12} {'F1':>10}")
            print(f"{'Trained Encoder':<35} {t['fpr']:>9.1f}% {t['precision']:>14.1f}% {t['recall']:>11.1f}% {t['f1']:>9.2f}")
            print(f"{'Random Embedding':<35} {r['fpr']:>9.1f}% {r['precision']:>14.1f}% {r['recall']:>11.1f}% {r['f1']:>9.2f}")
            print(f"{'Difference (trained advantage)':<35} {diff['fpr']:>9.1f}% {diff['precision']:>14.1f}% {diff['recall']:>11.1f}% {diff['f1']:>9.2f}")
    
    print("\n" + "="*100 + "\n")

def main():
    """CLI entry point: load all eval results and regenerate every table,
    JSON metrics file, and plot used in the paper into --output.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/workspace/paper_results/",
                       help="Output directory for plots and tables")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*100}")
    print(f"Generating all results to {output_dir}")
    print(f"{'='*100}\n")
    
    results = load_all_results()
    table_data = generate_results_table(results, output_dir)
    attack_metrics = generate_per_attack_analysis(results, output_dir)
    generate_detection_turn_analysis(results, output_dir)
    
    # Generate all plots in both PDF and PNG
    generate_comparison_plots(table_data, output_dir)
    generate_fpr_vs_recall_plot(table_data, output_dir)
    generate_latency_plot(output_dir)
    generate_attack_comparison_plot(attack_metrics, output_dir)
    generate_ablation_table(output_dir)
    
    print(f"\n{'='*100}")
    print(f"✓ All results generated to {output_dir}")
    print(f"{'='*100}\n")
    print("Files generated:")
    print("  - results_table.json (main metrics)")
    print("  - per_attack_metrics.json (attack breakdown)")
    print("  - detection_turn_stats.json (when C-CATM flags)")
    print("  - ablations.json (ablation study results)")
    print("  - comparison_plots.pdf / .png")
    print("  - fpr_vs_recall.pdf / .png")
    print("  - per_attack_comparison.pdf / .png")
    print("  - latency_breakdown.pdf / .png")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()