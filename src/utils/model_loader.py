"""
Loads target and monitor HuggingFace models with the dtype/attention/
tokenizer settings each family needs, and frees GPU memory afterward.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from typing import Dict, Any


TOKEN = os.environ.get("HF_TOKEN")

# Target models evaluated as attack victims (selectable via --model on the
# run_*.py scripts).
TARGET_MODELS = {
    "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen25":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma3":  "google/gemma-3-12b-it",
}

# Reference of monitor/scorer model repo IDs used elsewhere in the
# pipeline (LlamaGuard/ConstitutionalMonitor, HarmBenchScorer, etc.).
MONITOR_MODELS = {
    "llama_guard":   "meta-llama/Llama-Guard-3-8B",
    "harmbench_cls": "cais/HarmBench-Llama-2-13b-cls",
    "strongreject":  "google/gemma-2b",
}


def load_model(repo_id: str, output_hidden_states: bool = False) -> Dict[str, Any]:
    """
    Load a HuggingFace causal LM (or Gemma-3 conditional-generation model)
    and its tokenizer/processor onto GPU, ready for generation.

    Args:
        repo_id: HuggingFace repo id, e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct".
        output_hidden_states: currently unused by this function (hidden
            states are requested per-call via `model(..., output_hidden_states=True)`
            where needed, e.g. TrajGuard fitting) — kept as a documented,
            currently-inert parameter rather than removed, since callers
            may pass it.

    Returns:
        Dict with "model", "tokenizer" (or processor, for Gemma-3), and
        "repo_id".
    """
    print(f"Loading {repo_id}...")

    # Gemma-3 is a vision-language model in transformers (loaded via
    # Gemma3ForConditionalGeneration + AutoProcessor) even though it's only
    # used here for text, so it needs a different loading path than the
    # plain causal LMs.
    is_gemma3 = "gemma-3" in repo_id.lower()

    # flash-attn is an optional dependency (requires a matching CUDA/GPU
    # build); fall back to the slower "eager" attention implementation
    # when it isn't installed rather than failing to load the model.
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    if is_gemma3:
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,  # bfloat16: Gemma-3 was trained/released in bf16
            device_map="auto",
            attn_implementation=attn_impl,
            token=TOKEN
        )
        tokenizer = AutoProcessor.from_pretrained(repo_id, token=TOKEN, use_fast=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,  # float16 for the other (non-Gemma) target/monitor models
            device_map="auto",
            attn_implementation=attn_impl,
            token=TOKEN
        )
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=TOKEN)

    model.eval()
    print(f"✓ Loaded {repo_id} [{attn_impl}] — VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "repo_id": repo_id
    }


def unload_model(model_dict: Dict[str, Any]) -> None:
    """Free VRAM from a loaded model."""
    repo_id = model_dict.get("repo_id", "unknown")
    del model_dict["model"]
    del model_dict["tokenizer"]
    torch.cuda.empty_cache()
    print(f"✓ Unloaded {repo_id}")