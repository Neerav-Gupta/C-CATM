"""
Live terminal dashboard for monitoring an in-progress evaluation run.

Meant to be run in a separate terminal/process alongside `run_experiment.py`
or `run_secondary.py` (see `run_dashboard`'s docstring for usage). It polls
the same JSONL result files those scripts write to and renders a
continuously-refreshing `rich` layout with ASR, monitor flagging rate, and
recent-cell summaries. Purely observational — does not mutate any results.
"""

import time
import jsonlines
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def load_results():
    """
    Read all completed result rows from every JSONL file under
    RESULTS_DIR (recursively, since results are split into per-model
    subfolders), skipping error logs. Best-effort: a file that fails to
    parse (e.g. being written to concurrently) is silently skipped rather
    than crashing the dashboard's refresh loop.
    """
    results = []
    for f in RESULTS_DIR.rglob("*.jsonl"):  # rglob instead of glob
        if "error" in f.name:
            continue
        try:
            with jsonlines.open(f) as reader:
                for row in reader:
                    results.append(row)
        except Exception:
            pass
    return results


def build_dashboard(results, total_expected, model_name, start_time):
    """
    Build a `rich.layout.Layout` summarizing current run progress: header
    with ETA, ASR-by-attack, monitor flagging rate, HarmBench label
    distribution, StrongREJECT scores by attack, ASR by harm category, and
    the last 8 completed cells. Called once per refresh tick by
    `run_dashboard`.
    """
    completed = len(results)
    progress_pct = (completed / total_expected * 100) if total_expected > 0 else 0
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = (total_expected - completed) / rate if rate > 0 else 0

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=12),
        Layout(name="middle", size=10),
        Layout(name="bottom", size=10),
        Layout(name="footer", size=6),
    )
    layout["top"].split_row(
        Layout(name="asr"),
        Layout(name="monitor"),
    )
    layout["middle"].split_row(
        Layout(name="hb"),
        Layout(name="sr"),
    )

    # ── Header ────────────────────────────────────────────────────
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
    eta = datetime.fromtimestamp(time.time() + remaining).strftime("%H:%M:%S")

    header_text = Text(justify="center")
    header_text.append(f"⚡ Red-Team Evaluation — ", style="bold white")
    header_text.append(f"{model_name}", style="bold cyan")
    header_text.append(f"  |  ", style="dim")
    header_text.append(f"{completed}/{total_expected} cells ", style="bold yellow")
    header_text.append(f"({progress_pct:.1f}%)", style="yellow")
    header_text.append(f"  |  Elapsed: ", style="dim")
    header_text.append(f"{elapsed_str}", style="green")
    header_text.append(f"  |  Remaining: ", style="dim")
    header_text.append(f"{remaining_str}", style="red")
    header_text.append(f"  |  ETA: ", style="dim")
    header_text.append(f"{eta}", style="bold green")
    layout["header"].update(Panel(header_text))

    # ── ASR by Attack ─────────────────────────────────────────────
    asr_table = Table(
        title="ASR by Attack (no_defense)",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta"
    )
    asr_table.add_column("Attack", style="cyan", width=22)
    asr_table.add_column("ASR %", justify="right", width=8)
    asr_table.add_column("Success", justify="right", width=8)
    asr_table.add_column("Total", justify="right", width=8)
    asr_table.add_column("Bar", width=20)

    attack_stats = defaultdict(lambda: {"success": 0, "total": 0})
    for r in results:
        if r.get("monitor") == "no_defense":
            a = r["attack"]
            attack_stats[a]["total"] += 1
            if r.get("attack_success"):
                attack_stats[a]["success"] += 1

    for attack in ["crescendo", "jigsaw", "echo_chamber", "safe_in_isolation"]:
        s = attack_stats[attack]
        if s["total"] > 0:
            asr = s["success"] / s["total"] * 100
            bar_filled = int(asr / 5)  # 20 chars = 100%
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            color = "green" if asr > 10 else "yellow" if asr > 3 else "red"
            asr_table.add_row(
                attack,
                f"{asr:.1f}%",
                str(s["success"]),
                str(s["total"]),
                f"[{color}]{bar}[/{color}]"
            )
        else:
            asr_table.add_row(attack, "-", "0", "0", "░" * 20)

    layout["asr"].update(Panel(asr_table))

    # ── Monitor Flagging Rate ─────────────────────────────────────
    mon_table = Table(
        title="Monitor Flagging Rate",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta"
    )
    mon_table.add_column("Monitor", style="cyan", width=18)
    mon_table.add_column("Flag %", justify="right", width=8)
    mon_table.add_column("Flagged", justify="right", width=8)
    mon_table.add_column("Total", justify="right", width=8)
    mon_table.add_column("FPR?", width=8)

    monitor_stats = defaultdict(lambda: {"flagged": 0, "total": 0})
    for r in results:
        m = r.get("monitor", "unknown")
        monitor_stats[m]["total"] += 1
        if r.get("flagged_by_monitor"):
            monitor_stats[m]["flagged"] += 1

    for monitor in ["no_defense", "llama_guard", "traj_guard", "constitutional"]:
        s = monitor_stats[monitor]
        if s["total"] > 0:
            rate = s["flagged"] / s["total"] * 100
            fpr_note = ""
            if monitor == "traj_guard" and rate > 50:
                fpr_note = "[red]HIGH[/red]"
            elif monitor == "llama_guard" and rate < 20:
                fpr_note = "[green]OK[/green]"
            elif monitor == "constitutional" and rate < 2:
                fpr_note = "[yellow]LOW[/yellow]"
            mon_table.add_row(
                monitor,
                f"{rate:.1f}%",
                str(s["flagged"]),
                str(s["total"]),
                fpr_note
            )
        else:
            mon_table.add_row(monitor, "-", "0", "0", "")

    layout["monitor"].update(Panel(mon_table))

    # ── HarmBench Distribution ────────────────────────────────────
    hb_table = Table(
        title="HarmBench Labels",
        box=box.SIMPLE,
        header_style="bold magenta"
    )
    hb_table.add_column("Label", style="cyan", width=12)
    hb_table.add_column("Count", justify="right", width=8)
    hb_table.add_column("Pct", justify="right", width=8)

    hb_counts = defaultdict(int)
    for r in results:
        hb_counts[r.get("hb_label", "?")] += 1

    total = len(results) or 1
    for label, count in sorted(hb_counts.items()):
        color = "green" if label == 0 else "red"
        hb_table.add_row(
            f"[{color}]HB={label}[/{color}]",
            str(count),
            f"{count/total*100:.1f}%"
        )

    layout["hb"].update(Panel(hb_table))

    # ── SR Scores by Attack ───────────────────────────────────────
    sr_table = Table(
        title="Avg SR Score by Attack",
        box=box.SIMPLE,
        header_style="bold magenta"
    )
    sr_table.add_column("Attack", style="cyan", width=22)
    sr_table.add_column("Avg SR", justify="right", width=8)
    sr_table.add_column("Bar", width=20)

    sr_scores = defaultdict(list)
    for r in results:
        if r.get("monitor") == "no_defense":
            sr_scores[r["attack"]].append(r.get("sr_score", 0))

    for attack in ["crescendo", "jigsaw", "echo_chamber", "safe_in_isolation"]:
        scores = sr_scores[attack]
        if scores:
            avg = sum(scores) / len(scores)
            bar_filled = int(avg * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            color = "red" if avg > 0.5 else "yellow" if avg > 0.2 else "green"
            sr_table.add_row(
                attack,
                f"{avg:.3f}",
                f"[{color}]{bar}[/{color}]"
            )
        else:
            sr_table.add_row(attack, "-", "░" * 20)

    layout["sr"].update(Panel(sr_table))

    # ── Category Breakdown ────────────────────────────────────────
    cat_table = Table(
        title="ASR by Harm Category (no_defense)",
        box=box.SIMPLE,
        header_style="bold magenta"
    )
    cat_table.add_column("Category", style="cyan", width=35)
    cat_table.add_column("ASR %", justify="right", width=8)
    cat_table.add_column("S/T", justify="right", width=10)

    cat_stats = defaultdict(lambda: {"success": 0, "total": 0})
    for r in results:
        if r.get("monitor") == "no_defense":
            cat = r.get("category", "unknown")
            cat_stats[cat]["total"] += 1
            if r.get("attack_success"):
                cat_stats[cat]["success"] += 1

    for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["total"]):
        if s["total"] > 0:
            asr = s["success"] / s["total"] * 100
            color = "red" if asr > 10 else "yellow" if asr > 3 else "white"
            cat_table.add_row(
                cat,
                f"[{color}]{asr:.1f}%[/{color}]",
                f"{s['success']}/{s['total']}"
            )

    layout["bottom"].update(Panel(cat_table))

    # ── Recent Cells ──────────────────────────────────────────────
    recent_table = Table(
        title="Last 8 Completed Cells",
        box=box.SIMPLE,
        header_style="bold magenta"
    )
    recent_table.add_column("", width=2)
    recent_table.add_column("Attack", style="cyan", width=20)
    recent_table.add_column("Monitor", style="blue", width=16)
    recent_table.add_column("SR", justify="right", width=6)
    recent_table.add_column("HB", justify="right", width=4)
    recent_table.add_column("Flag", justify="right", width=5)
    recent_table.add_column("Behavior", width=45)

    recent = sorted(results, key=lambda x: x.get("timestamp", ""))[-8:]
    for r in reversed(recent):
        status = "✓" if r.get("attack_success") else "✗"
        color = "green" if r.get("attack_success") else "red"
        flag = "🚩" if r.get("flagged_by_monitor") else "  "
        recent_table.add_row(
            f"[{color}]{status}[/{color}]",
            r.get("attack", ""),
            r.get("monitor", ""),
            f"{r.get('sr_score', 0):.2f}",
            str(r.get("hb_label", 0)),
            flag,
            r.get("behavior", "")[:45] + "..."
        )

    layout["footer"].update(Panel(recent_table))

    return layout


def run_dashboard(total_expected: int, model_name: str, refresh_every: int = 10):
    """
    Run live dashboard. Call this in a separate terminal while
    the experiment is running.
    Usage: python -c "
        from src.evaluation.live_dashboard import run_dashboard
        run_dashboard(3200, 'llama31')
    "
    """
    console = Console()
    start_time = time.time()

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            try:
                results = load_results()
                dashboard = build_dashboard(
                    results, total_expected, model_name, start_time
                )
                live.update(dashboard)

                # Stop if complete
                if len(results) >= total_expected:
                    time.sleep(5)
                    break

                time.sleep(refresh_every)

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"Dashboard error: {e}")
                time.sleep(refresh_every)