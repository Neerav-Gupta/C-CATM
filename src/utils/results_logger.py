"""
Writes evaluation results to disk as JSONL: a flat summary row per
(trace, monitor) cell for analysis, plus optional full per-turn traces
(with hidden states) for C-CATM training.
"""

import jsonlines
from pathlib import Path
from datetime import datetime
from typing import Dict
from .data_structures import AttackTrace


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def save_trace(trace: AttackTrace, monitor_name: str,
               behavior_id: str = None, category: str = None) -> Dict:
    """
    Append a flat summary row for one (attack trace, monitor) cell to
    results/{model_name}/{attack}_{monitor}.jsonl under the repo root.

    Args:
        trace: the scored AttackTrace (harmbench_label, strongreject_score,
            flagged_by_monitor, detection_turn must already be set).
        monitor_name: which monitor produced this cell's flagging result.
        behavior_id: HarmBench BehaviorID, or "unknown" if not provided.
        category: HarmBench SemanticCategory, or "unknown" if not provided.

    Returns:
        The result dict that was written (also appended to
        EvaluationHarness.results by the caller). Fields include the
        scores/labels above plus "attack_success" (harmbench_label == 1
        and not flagged_by_monitor) and a "final_response" truncated to
        500 chars to keep result files small.
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "model_id": trace.model_id,
        "attack": trace.attack_name,
        "monitor": monitor_name,
        "behavior": trace.target_behavior,
        "behavior_id": behavior_id or "unknown",
        "category": category or "unknown",
        "n_turns": len(trace.assistant_turns()),
        "sr_score": trace.strongreject_score,
        "hb_label": trace.harmbench_label,
        "flagged_by_monitor": trace.flagged_by_monitor,
        "detection_turn": trace.detection_turn,
        "attack_success": trace.harmbench_label == 1 and not trace.flagged_by_monitor,
        "final_response": trace.final_response()[:500],
    }

    # Create model-specific subfolder automatically
    model_name = trace.model_id.split('/')[-1]
    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / f"{trace.attack_name}_{monitor_name}.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    with jsonlines.open(out_path, mode="a") as writer:
        writer.write(result)

    return result


def load_results(path: str) -> list:
    """Load a results JSONL file into a list of dicts."""
    with jsonlines.open(path) as f:
        return list(f)
    
TRACES_DIR = REPO_ROOT / "results" / "traces"

def save_full_trace(trace: AttackTrace, behavior_id: str = None,
                    category: str = None) -> None:
    """
    Save full attack trace with per-turn hidden states for C-CATM training.
    Saved separately from flat results (`save_trace`) — these are large
    files due to the per-turn hidden states.

    NOTE: as currently implemented this saves every trace it is called
    with, unconditionally (labeled by harmbench_label as 0/1) — there is
    no success-based filtering or safe-trace subsampling here. Any class
    balancing for C-CATM training must happen downstream when consuming
    these files. (Flagged for author review: an earlier version of this
    docstring described filtering/sampling that does not appear in the
    code below.)
    """
    TRACES_DIR.mkdir(exist_ok=True)

    model_name = trace.model_id.split("/")[-1]
    out_dir = TRACES_DIR / model_name
    out_dir.mkdir(exist_ok=True)

    # Serialize turns including hidden states
    turns_data = []
    for turn in trace.turns:
        turn_dict = {
            "role": turn.role,
            "content": turn.content,
            "turn_index": turn.turn_index,
        }
        # Save hidden state as list (JSON serializable)
        if turn.hidden_state is not None:
            turn_dict["hidden_state"] = turn.hidden_state.float().tolist()
        turns_data.append(turn_dict)

    record = {
        "label": 1 if (trace.harmbench_label == 1) else 0,
        "model_id": trace.model_id,
        "attack": trace.attack_name,
        "behavior": trace.target_behavior,
        "behavior_id": behavior_id or "unknown",
        "category": category or "unknown",
        "turns": turns_data,
        "n_turns": len(turns_data),
        "sr_score": trace.strongreject_score,
        "hb_label": trace.harmbench_label,
    }

    # Append to model+attack specific file
    out_path = out_dir / f"{trace.attack_name}.jsonl"
    with jsonlines.open(out_path, mode="a") as writer:
        writer.write(record)