# C-CATM: Cross-turn Contrastive Activation Trajectory Monitor

A learned trajectory encoder + constitutional gating approach for LLM safety monitoring.

## Results

**Llama-3.1-8B:**
- FPR: 2.3%
- Precision: 43.8%
- Recall: 50.0%
- F1: 46.7

**Qwen2.5-7B:**
- FPR: 19.1%
- Precision: 46.6%
- Recall: 88.7%
- F1: 61.1

**Latency:** 6.2ms per trace (12.2x faster than Llama Guard)

## Quick Start

```bash
pip install -r requirements.txt
python ablation_studies.py --all
python generate_plots.py --output ./results/
```

## Trained Models

Download from Hugging Face: [ccatm-safety-monitor](https://huggingface.co/Neerav-Gupta/ccatm-safety-monitor)

## Usage

```python
from src.monitors.ccatm import CCATMMonitorWrapper

monitor = CCATMMonitorWrapper(
    encoder_checkpoint="encoder_best.pt",
    concept_checkpoint="concepts.pt",
    gate_checkpoint="gate_trained.pt",
    gate_threshold=0.5,
    device="cuda"
)

result = monitor.score_trace(trace)
print(f"Flagged: {result['flagged']}")
```

## Paper Results

See `paper_results/` for all evaluation metrics, plots, and ablations.

## License

MIT License
