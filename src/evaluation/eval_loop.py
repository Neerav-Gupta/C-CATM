"""
Core evaluation loop: runs the attack x monitor x behavior matrix and
persists results.

`EvaluationHarness` is the main entry point used by the run_*.py scripts.
It orchestrates running each attack against the target model, scoring the
resulting transcript with HarmBench and StrongREJECT, passing the trace
through each monitor, and writing results to disk (with resume/backup
support for long unattended runs).
"""

from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console

from ..utils.results_logger import save_trace

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[2]


class EvaluationHarness:
    """
    Main evaluation loop.
    Runs the full attack × monitor × behavior matrix and logs results.
    """

    def __init__(
        self,
        attackers,
        monitors,
        harmbench_scorer,
        strongreject_scorer,
        model_id: str = "unknown"
    ):
        self.attackers = attackers
        self.monitors = monitors
        self.hb_scorer = harmbench_scorer
        self.sr_scorer = strongreject_scorer
        self.model_id = model_id
        self.results = []

    def run_cell(
        self,
        behavior: Dict[str, str],
        attack_name: str,
        monitor_name: str,
        trace=None  # accept pre-computed trace to avoid re-running attack
    ) -> Dict[str, Any]:
        """
        Score and log a single (behavior, attack, monitor) cell.

        If `trace` is not supplied, the attack is run fresh against the
        target model; otherwise a trace already produced for this
        behavior/attack pair is reused so the same attack transcript can be
        scored against multiple monitors without re-running the LLM calls
        (see `run_matrix`, which relies on this for its main optimization).

        Args:
            behavior: HarmBench-style row with at least "Behavior" and
                optionally "BehaviorID" / "SemanticCategory".
            attack_name: key into `self.attackers`.
            monitor_name: key into `self.monitors`.
            trace: optional pre-computed `AttackTrace` to reuse.

        Returns:
            The result dict written by `save_trace` (timestamp, scores,
            success/flag flags, truncated final response, etc.).
        """
        behavior_text = behavior["Behavior"]
        behavior_id = behavior.get("BehaviorID", "unknown")

        monitor = self.monitors[monitor_name]

        # Run attack only if trace not provided
        if trace is None:
            attacker = self.attackers[attack_name]
            trace = attacker.run(behavior_text)

        # Score final response
        final_response = trace.final_response()
        trace.harmbench_label = self.hb_scorer.score(behavior_text, final_response)
        trace.strongreject_score = self.sr_scorer.score(behavior_text, final_response)

        # Run monitor
        monitor_result = monitor.score_trace(trace)
        trace.flagged_by_monitor = monitor_result["flagged"]
        trace.detection_turn = monitor_result.get("detection_turn")
        trace.success = trace.harmbench_label == 1 and not trace.flagged_by_monitor

        # Save to disk
        result = save_trace(
            trace,
            monitor_name,
            behavior_id=behavior_id,
            category=behavior.get("SemanticCategory", "unknown")
        )
        if monitor_name == list(self.monitors.keys())[0]:
            # Only persist the full per-turn trace (with hidden states, used
            # for C-CATM training) once per attack — not once per monitor —
            # by gating on the first monitor in iteration order.
            from ..utils.results_logger import save_full_trace
            save_full_trace(
                trace,
                behavior_id=behavior_id,
                category=behavior.get("SemanticCategory", "unknown")
            )
        self.results.append(result)

        return result

    def run_matrix(
        self,
        behaviors: List[Dict],
        attack_names: List[str] = None,
        monitor_names: List[str] = None,
        resume: bool = True,
        backup_every: int = 50
    ) -> List[Dict]:
        """
        Run the full behaviors x attacks x monitors matrix.

        Key optimization: for a given (behavior, attack) pair, the attack is
        only executed once — the resulting trace is then scored against
        every monitor in `monitor_names`, since attack generation (LLM
        calls) is far more expensive than monitor scoring.

        Args:
            behaviors: list of HarmBench-style behavior rows.
            attack_names: subset of `self.attackers` keys to run (default:
                all configured attackers).
            monitor_names: subset of `self.monitors` keys to run (default:
                all configured monitors).
            resume: if True, skip cells already present in prior result
                files (keyed by model_id/behavior_id/attack/monitor) so an
                interrupted run can be restarted without redoing work.
            backup_every: snapshot all result files to a timestamped backup
                folder after this many newly-completed cells, as insurance
                against corruption during long unattended GPU runs.

        Returns:
            `self.results` — the list of result dicts produced by every
            `run_cell` call made during this invocation (not including
            cells that were skipped because they were already done).
        """
        attack_names = attack_names or list(self.attackers.keys())
        monitor_names = monitor_names or list(self.monitors.keys())

        total = len(behaviors) * len(attack_names) * len(monitor_names)
        completed = 0
        skipped = 0
        cells_since_backup = 0

        console.print(f"\n[bold]Starting evaluation matrix[/bold]")
        console.print(f"  Behaviors:  {len(behaviors)}")
        console.print(f"  Attacks:    {len(attack_names)}")
        console.print(f"  Monitors:   {len(monitor_names)}")
        console.print(f"  Total cells: {total}")
        console.print(f"  Backup every: {backup_every} cells\n")

        completed_keys = set()
        if resume:
            completed_keys = self._load_completed_keys()
            if completed_keys:
                console.print(f"[yellow]Resuming — {len(completed_keys)} cells already done[/yellow]\n")

        # ── Heartbeat thread ──────────────────────────────────────────
        # Writes a small status file every 60s so an external process (or a
        # human) can tell a long-running unattended job is still alive vs.
        # hung, without needing to parse the result files.
        import threading
        import time
        from datetime import datetime
        from pathlib import Path

        self._done = False

        def _heartbeat():
            while not self._done:
                try:
                    (REPO_ROOT / "logs").mkdir(exist_ok=True)
                    with open(REPO_ROOT / "logs" / "heartbeat.txt", "w") as f:
                        f.write(
                            f"alive at {datetime.now().isoformat()}\n"
                            f"completed: {completed}\n"
                            f"skipped: {skipped}\n"
                            f"total: {total}\n"
                        )
                except Exception:
                    pass
                time.sleep(60)

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
        # ─────────────────────────────────────────────────────────────

        for behavior in behaviors:
            for attack_name in attack_names:

                # ── KEY OPTIMIZATION: run attack ONCE, reuse across all monitors ──
                # Check if all monitor cells for this attack+behavior are done
                all_done = all(
                    f"{behavior.get('BehaviorID')}_{attack_name}_{m}" in completed_keys
                    for m in monitor_names
                )
                if all_done:
                    skipped += len(monitor_names)
                    continue

                # Check if any monitor cells need this attack
                needs_attack = any(
                    f"{behavior.get('BehaviorID')}_{attack_name}_{m}" not in completed_keys
                    for m in monitor_names
                )

                # Run attack once
                trace = None
                if needs_attack:
                    try:
                        attacker = self.attackers[attack_name]
                        trace = attacker.run(behavior["Behavior"])
                        # Score once — same for all monitors
                        trace.harmbench_label = self.hb_scorer.score(
                            behavior["Behavior"], trace.final_response()
                        )
                        trace.strongreject_score = self.sr_scorer.score(
                            behavior["Behavior"], trace.final_response()
                        )
                    except Exception as e:
                        console.print(f"[red]ATTACK ERROR[/red] {attack_name} × "
                                    f"{behavior.get('BehaviorID')}: {e}")
                        self._log_error(behavior, attack_name, "attack", str(e))
                        continue

                # Pass same trace to all monitors
                for monitor_name in monitor_names:
                    cell_key = f"{self.model_id}_{behavior.get('BehaviorID')}_{attack_name}_{monitor_name}"

                    if cell_key in completed_keys:
                        skipped += 1
                        continue

                    try:
                        result = self.run_cell(
                            behavior, attack_name, monitor_name,
                            trace=trace  # reuse trace — no re-running attack
                        )
                        completed += 1
                        cells_since_backup += 1

                        status = "✓" if result["attack_success"] else "✗"
                        color = "green" if result["attack_success"] else "red"
                        flag_icon = "🚩" if result["flagged_by_monitor"] else "  "
                        console.print(
                            f"[{color}]{status}[/{color}] "
                            f"[dim][{completed}/{total-skipped}][/dim] "
                            f"[cyan]{attack_name:20s}[/cyan] × "
                            f"[blue]{monitor_name:15s}[/blue] {flag_icon} "
                            f"[yellow]SR={result['sr_score']:.2f}[/yellow] "
                            f"[{'red' if result['hb_label'] else 'green'}]"
                            f"HB={result['hb_label']}"
                            f"[/{'red' if result['hb_label'] else 'green'}] "
                            f"[dim]{behavior['Behavior'][:40]}...[/dim]"
                        )

                        if cells_since_backup >= backup_every:
                            self._backup_results()
                            cells_since_backup = 0

                    except Exception as e:
                        console.print(f"[red]ERROR[/red] {attack_name} × {monitor_name} × "
                                    f"{behavior.get('BehaviorID')}: {e}")
                        self._log_error(behavior, attack_name, monitor_name, str(e))

        self._backup_results()
        self._done = True

        console.print(f"\n[bold green]Matrix complete[/bold green] — "
                    f"{completed} cells run, {skipped} skipped (already done)")
        return self.results
    
    def _backup_results(self) -> None:
        """
        Copy all result files to a timestamped backup folder.
        Keeps last 5 backups to avoid filling disk.
        """
        import shutil
        from datetime import datetime

        results_dir = REPO_ROOT / "results"
        backup_base = REPO_ROOT / "results" / "backups"
        backup_base.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_base / timestamp
        backup_dir.mkdir(exist_ok=True)

        # Copy all JSONL files except errors and previous backups
        backed_up = 0
        for f in results_dir.glob("*.jsonl"):
            shutil.copy2(f, backup_dir / f.name)
            backed_up += 1

        console.print(f"[blue]💾 Backup: {backed_up} files → {backup_dir}[/blue]")

        # Keep only last 5 backups to save disk space
        backups = sorted(backup_base.iterdir())
        if len(backups) > 5:
            for old in backups[:-5]:
                shutil.rmtree(old)
            console.print(f"[blue]   Pruned old backups, keeping last 5[/blue]")

    def _load_completed_keys(self) -> set:
        """
        Scan existing result files under the repo's results/ directory and
        build the set of "{model_id}_{behavior_id}_{attack}_{monitor}" keys
        that have already been completed, so `run_matrix` can resume
        without re-running expensive attack/monitor cells.
        """
        keys = set()
        results_dir = REPO_ROOT / "results"
        if not results_dir.exists():
            return keys

        import jsonlines
        for f in results_dir.rglob("*.jsonl"):
            # Skip errors, backups, archives
            if any(x in str(f) for x in ["error", "backup", "archive", "traces"]):
                continue
            try:
                with jsonlines.open(f) as reader:
                    for row in reader:
                        if row.get("monitor") == "attack":
                            continue
                        # sr_score is None only for partially-written/failed
                        # rows — treat those as not completed.
                        if row.get("sr_score") is None:
                            continue
                        # Only count rows for the model currently being
                        # evaluated — result files can contain rows for
                        # multiple models sharing the same directory.
                        if self.model_id and self.model_id not in row.get("model_id", ""):
                            continue
                        key = (f"{row.get('model_id', '')}_{row.get('behavior_id')}"
                            f"_{row.get('attack')}_{row.get('monitor')}")
                        keys.add(key)
            except Exception:
                pass
        return keys

    def _log_error(
        self,
        behavior: Dict,
        attack_name: str,
        monitor_name: str,
        error: str
    ) -> None:
        """Log failed cells so you know what to re-run."""
        import jsonlines
        error_path = REPO_ROOT / "results" / "errors.jsonl"
        with jsonlines.open(error_path, mode="a") as writer:
            writer.write({
                "behavior_id": behavior.get("BehaviorID"),
                "attack": attack_name,
                "monitor": monitor_name,
                "error": error
            })