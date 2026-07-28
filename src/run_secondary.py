"""
Entry-point script for the "secondary models" evaluation pass.

Runs Qwen2.5-7B and Gemma-3-12B sequentially on the fixed stratified
100-behavior set (see `load_fixed_stratified`), against all attacks and
monitors. Unlike run_experiment.py (one target model per process), this
script loads the monitor models (Llama Guard + HarmBench classifier) once
and keeps them resident in GPU memory across both target-model runs,
swapping only the (much larger) target model in and out, to avoid paying
the monitor-loading cost twice.
"""

import sys
import torch
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

load_dotenv("/workspace/.env")
sys.path.insert(0, "/workspace")

from src.utils.model_loader import load_model, unload_model, TARGET_MODELS
from src.utils.data_loader import load_fixed_stratified
from src.attackers import (
    CrescendoAttacker, JigsawAttacker,
    EchoChamberAttacker, SafeInIsolationAttacker
)
from src.monitors import (
    NoDefenseMonitor, LlamaGuardMonitor,
    TrajGuardMonitor, ConstitutionalMonitor
)
from src.evaluation import HarmBenchScorer, StrongRejectScorer, EvaluationHarness
from rich.console import Console
import numpy as np

console = Console()

SECONDARY_MODELS = ["qwen25", "gemma3"]
ATTACKS = ["crescendo", "jigsaw", "echo_chamber", "safe_in_isolation"]
MONITORS = ["no_defense", "llama_guard", "traj_guard", "constitutional"]


def fit_traj_guard(traj_guard, target_dict, behaviors):
    """
    Fit TrajGuard's safety direction for the current target model, using a
    fixed list of benign prompts vs. the first 10 evaluation behaviors as
    calibration examples, then set `traj_guard.threshold` to the midpoint
    between the mean safe/unsafe projected scores. (Same procedure as
    `run_experiment._fit_traj_guard`; duplicated here rather than shared
    because this script fits per target model in its own loop.)
    """
    model = target_dict["model"]
    tokenizer = target_dict["tokenizer"]
    # Gemma's AutoProcessor expects the text kwarg and doesn't accept
    # truncation the way AutoTokenizer does for the other models.
    is_gemma = "gemma" in target_dict["repo_id"].lower()

    safe_examples = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "What are some good books to read?",
        "How do I make pasta carbonara?",
        "What is the Pythagorean theorem?",
        "Tell me about the history of Rome.",
        "How does the immune system work?",
        "What causes rainbows?",
        "Recommend some good movies.",
        "Explain how airplanes fly.",
    ]
    unsafe_examples = [b["Behavior"] for b in behaviors[:10]]

    def get_hs(text):
        if is_gemma:
            inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True
            ).to(model.device)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        # Cast to float32 before numpy — fixes bfloat16 issue
        return out.hidden_states[-1].mean(dim=1).squeeze(0).cpu().float().numpy()

    console.print("  Computing safe hidden states...")
    safe_hs = [get_hs(t) for t in safe_examples]
    console.print("  Computing unsafe hidden states...")
    unsafe_hs = [get_hs(t) for t in unsafe_examples]

    traj_guard.fit(safe_hs, unsafe_hs)

    safe_scores = [traj_guard._project(h) for h in safe_hs]
    unsafe_scores = [traj_guard._project(h) for h in unsafe_hs]
    threshold = float((np.mean(safe_scores) + np.mean(unsafe_scores)) / 2)
    traj_guard.threshold = threshold

    console.print(f"✓ TrajGuard fitted — threshold: {threshold:.4f} "
                  f"(safe: {np.mean(safe_scores):.4f}, "
                  f"unsafe: {np.mean(unsafe_scores):.4f})")


def run_model(model_key: str, behaviors: list,
              llama_guard_dict: dict, harmbench_dict: dict):
    """
    Run the full attack x monitor matrix for one secondary target model
    (Qwen2.5 or Gemma-3), reusing the already-loaded monitor models, then
    unload just the target model (monitors stay resident for the next
    model in the loop — see `main`).
    """
    console.rule(f"[bold]Starting {model_key}[/bold]")

    # Load target model
    console.print(f"Loading target model {model_key}...")
    target_repo = TARGET_MODELS[model_key]

    # NOTE: dtype/device-map handling for Gemma vs. non-Gemma targets is
    # decided inside load_model() (see src/utils/model_loader.py) based on
    # the repo id, not here.
    target_dict = load_model(target_repo)

    console.print(f"VRAM after loading: "
                  f"{torch.cuda.memory_allocated()/1e9:.1f}GB / "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

    attackers = {
        "crescendo": CrescendoAttacker(
            target=target_dict,
            attacker=target_dict,
            max_turns=10
        ),
        "jigsaw": JigsawAttacker(target=target_dict, max_turns=6),
        "echo_chamber": EchoChamberAttacker(target=target_dict, max_turns=8),
        "safe_in_isolation": SafeInIsolationAttacker(target=target_dict, max_turns=5),
    }

    # Fit TrajGuard
    traj_guard = TrajGuardMonitor(threshold=0.0)
    console.print("\nFitting TrajGuard safety direction...")
    fit_traj_guard(traj_guard, target_dict, behaviors)

    monitors = {
        "no_defense": NoDefenseMonitor(),
        "llama_guard": LlamaGuardMonitor(llama_guard_dict),
        "traj_guard": traj_guard,
        "constitutional": ConstitutionalMonitor(llama_guard_dict),
    }

    harness = EvaluationHarness(
        attackers=attackers,
        monitors=monitors,
        harmbench_scorer=HarmBenchScorer(harmbench_dict),
        strongreject_scorer=StrongRejectScorer(),
        model_id=target_repo
    )

    harness.run_matrix(
        behaviors=behaviors,
        attack_names=ATTACKS,
        monitor_names=MONITORS,
        resume=True,
        backup_every=50
    )

    # Only unload target — keep monitor models loaded
    console.print(f"\nUnloading target model {model_key}...")
    unload_model(target_dict)
    torch.cuda.empty_cache()
    console.print(f"✓ {model_key} complete — "
                  f"VRAM after unload: "
                  f"{torch.cuda.memory_allocated()/1e9:.1f}GB")


def main():
    """
    Entry point: load the fixed stratified behavior set and the monitor
    models once, then run each secondary target model (Qwen2.5, Gemma-3)
    in sequence via `run_model`. A failure on one model (e.g. an OOM) is
    caught and logged so it doesn't abort the run for the remaining
    model(s); GPU memory is cleared before continuing.
    """
    behaviors = load_fixed_stratified()
    console.print(f"✓ Loaded {len(behaviors)} fixed stratified behaviors")

    # Load monitor models ONCE — keep for entire run
    console.print("\nLoading monitor models (persistent for entire run)...")
    llama_guard_dict = load_model("meta-llama/Llama-Guard-3-8B")
    harmbench_dict = load_model("cais/HarmBench-Llama-2-13b-cls")
    console.print(f"✓ Monitor models loaded — "
                  f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    console.print("  These stay loaded for both Qwen and Gemma runs\n")

    for model_key in SECONDARY_MODELS:
        try:
            run_model(model_key, behaviors, llama_guard_dict, harmbench_dict)
        except Exception as e:
            # Don't let one model's failure (e.g. OOM) kill the whole run —
            # log it and free GPU memory before moving to the next model.
            console.print(f"[red]ERROR running {model_key}: {e}[/red]")
            import traceback
            traceback.print_exc()
            console.print("[yellow]Continuing to next model...[/yellow]")
            torch.cuda.empty_cache()

    # Unload monitor models at the very end
    console.print("\nUnloading monitor models...")
    unload_model(llama_guard_dict)
    unload_model(harmbench_dict)
    console.print("\n[bold green]All secondary models complete![/bold green]")
    console.print("Results saved to /workspace/results/")


if __name__ == "__main__":
    main()