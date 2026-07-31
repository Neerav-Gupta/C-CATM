# C-CATM: Cross-Turn Contrastive Activation Trajectory Monitor

A lightweight multi-turn LLM jailbreak monitor that operates on a target model's **hidden-state trajectory** across conversation turns, rather than on its output text. C-CATM combines (1) a trajectory encoder trained with a supervised contrastive objective on turn-to-turn activation deltas, and (2) a constitutional gating layer that maps trajectory embeddings to a calibrated safety probability.

**Paper:** [`paper/main.pdf`](paper/main.pdf) — *C-CATM: Cross-Turn Contrastive Activation Trajectory Monitor for Multi-Turn LLM Jailbreak Detection*, Neerav Gupta, 2026 (independent research, preprint).
**Trained models:** [ccatm-safety-monitor on Hugging Face](https://huggingface.co/Neerav-Gupta/ccatm-safety-monitor)

## Results

Evaluated against four multi-turn attacks (Crescendo, Jigsaw, Echo Chamber, Safe-in-Isolation) and three baseline monitors (Llama Guard, TrajGuard, Constitutional), across 1,192 evaluated conversation traces.

**Llama-3.1-8B-Instruct** (200 behaviors × 4 attacks):

| Monitor | FPR | Precision | Recall | F1 |
|---|---|---|---|---|
| Llama Guard | 8.5% | 13.6% | 58.3% | 22.1 |
| TrajGuard | 88.3% | 2.3% | 88.9% | 4.4 |
| Constitutional | 6.1% | 15.0% | 47.2% | 22.8 |
| **C-CATM** | **2.3%** | **43.8%** | **50.0%** | **46.7** |

**Qwen2.5-7B-Instruct** (98 stratified behaviors × 4 attacks, unseen at training time):

| Monitor | FPR | Precision | Recall | F1 |
|---|---|---|---|---|
| Llama Guard | 47.3% | 23.8% | 77.4% | 36.4 |
| TrajGuard | 100.0% | 16.0% | 100.0% | 27.6 |
| Constitutional | 47.3% | 23.8% | 77.4% | 36.4 |
| **C-CATM** | **19.1%** | **46.6%** | **88.7%** | **61.1** |

**Latency:** 6.2ms per trace, monitor-only inference (trajectory encoding + gate scoring; see [paper §4.4](paper/main.pdf) for what this excludes).

An ablation shows the learned trajectory encoder — not the gate architecture alone — is responsible for the bulk of this performance (+22.0 F1 over a randomly initialized encoder of identical architecture). We also report, transparently, that the trained gate does not currently make use of its constitutional rule-scoring pathway (see **Limitations** below). Full tables, per-attack breakdowns, and figures are in [`paper_results/`](paper_results/) and Sections 5–6 of the paper.

## Repository Structure

```
c_catm/                  Standalone C-CATM monitor package (encoder, gate, constitution)
  trajectory/            Delta trajectory encoder + contrastive training (encoder.py, subspace.py)
  constitutional/        32-rule constitution, concept extraction, gating (gate.py, extractor.py)
  train.py               Trains the trajectory encoder + gate from collected traces
  monitor.py             CCATMMonitor — loads trained checkpoints, scores a trace
  checkpoints/           Trained weights go here (not in git; download from Hugging Face)

src/                     Evaluation harness used to produce the paper's results
  attackers/              Crescendo, Jigsaw, Echo Chamber, Safe-in-Isolation
  monitors/               Baseline monitors + the CCATMMonitorWrapper adapter
  evaluation/             HarmBench/StrongReject scorers, evaluation loop
  run_experiment.py       Primary-model evaluation sweep (Llama-3.1-8B)
  run_secondary.py        Secondary-model sweep (Qwen2.5-7B, Gemma-3-12B)

data/                    Stratified HarmBench behavior subsets used for evaluation
paper/                   Paper source (main.tex) and compiled PDF
paper_results/           Result tables, ablations, and figures referenced by the paper
ablation_studies.py      Reproduces the encoder ablation in paper §6
generate_plots.py        Regenerates all paper tables/figures from raw eval output
```

## Installation

Requires Python ≥3.10 and a CUDA GPU (results in the paper were produced on a single A100-80GB or L40-48GB).

```bash
pip install torch==2.8.0          # install torch first — flash-attn's build needs it present
pip install -r requirements.txt   # everything else, including flash-attn
```

`flash-attn` compiles a native extension against your installed torch/CUDA version; if the build fails, consult the [flash-attention installation notes](https://github.com/Dao-AILab/flash-attention#installation) for a prebuilt wheel matching your environment.

## Trained Models

Download the trajectory encoder, gate, and constitutional concept checkpoints from [Hugging Face](https://huggingface.co/Neerav-Gupta/ccatm-safety-monitor) and place them under `c_catm/checkpoints/`.

## Usage

```python
from src.monitors.ccatm import CCATMMonitorWrapper

monitor = CCATMMonitorWrapper(
    encoder_checkpoint="c_catm/checkpoints/encoder_best.pt",
    concept_checkpoint="c_catm/checkpoints/concepts.pt",
    gate_checkpoint="c_catm/checkpoints/gate_trained.pt",
    gate_threshold=0.5,
    device="cuda"
)

result = monitor.score_trace(trace)
print(f"Flagged: {result['flagged']}")
```

## Reproducing the Paper's Results

All scripts resolve their paths relative to the repo (via `Path(__file__)`), so they work from a clone anywhere on disk — no fixed mount point required. `run_experiment.py`/`run_secondary.py` load API keys from a `.env` file at the repo root if present. `src/utils/data_loader.py` expects [HarmBench](https://github.com/centerforaisafety/HarmBench) and SafeDialBench-Dataset checked out as sibling directories next to this repo by default; point it elsewhere via the `HARMBENCH_PATH` / `SAFEDIAL_EN_PATH` environment variables if you keep them somewhere else.

1. **Regenerate paper tables/figures from the released raw results** (fastest path — no GPU needed):
   ```bash
   python generate_plots.py --output ./paper_results/
   ```
2. **Reproduce the encoder ablation** (needs the trained checkpoints from Hugging Face, see above):
   ```bash
   python ablation_studies.py --all
   ```
3. **Reproduce the full evaluation matrix from scratch** (needs a GPU and HF access to Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct / Llama-Guard-3-8B / the HarmBench classifier):
   ```bash
   python src/run_experiment.py --model llama31 --behaviors 200
   python src/run_secondary.py
   ```
4. **Retrain the encoder + gate from scratch:** `c_catm/train.py` trains on hidden-state trajectory traces collected during step 3 above (`/workspace/results/traces` by default). These raw traces are large (the training/validation trajectory files used for this paper total ~2.2GB) and are not included in this repository; regenerate them via step 3, then run:
   ```bash
   python c_catm/train.py
   ```

## Limitations

See paper §7 for the full discussion. Briefly: the trained gate does not currently use its constitutional rule-scoring pathway (the trajectory embedding alone drives its decisions); only two target models were evaluated end-to-end (a third, Gemma-3-12B, was excluded for compute-budget reasons); threshold selection was not done on a fully disjoint calibration set; and the attack implementations are our own re-implementations of published strategies, not the original authors' code.

## Citation

```bibtex
@misc{gupta2026ccatm,
  title  = {C-CATM: Cross-Turn Contrastive Activation Trajectory Monitor for Multi-Turn LLM Jailbreak Detection},
  author = {Gupta, Neerav},
  year   = {2026},
  note   = {Independent research, preprint},
  url    = {https://github.com/Neerav-Gupta/C-CATM}
}
```

## License

MIT License
