"""
Dataset loading utilities for the evaluation pipeline.

Loads behavior/prompt sets from HarmBench, SafeDialBench, and StrongREJECT
into a common record shape: dicts with at least "Behavior" (the prompt/
target behavior text), "BehaviorID", and "SemanticCategory" keys, matching
the HarmBench CSV schema, so downstream code (attackers, scorers, the
evaluation harness) can treat all datasets uniformly.
"""

import pandas as pd
import jsonlines
from typing import List, Dict


HARMBENCH_PATH = "/workspace/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
SAFEDIAL_EN_PATH = "/workspace/SafeDialBench-Dataset/data/complete/datasets_en.jsonl"
STRONGREJECT_SMALL_PATH = None  # loaded via strong_reject package directly


def load_harmbench(n: int = None, categories: List[str] = None) -> List[Dict]:
    """
    Load HarmBench text-only behaviors (excludes behaviors that require a
    ContextString, e.g. multimodal/contextual attacks).

    Args:
        n: if given, truncate to the first `n` rows after filtering.
        categories: if given, restrict to these SemanticCategory values.

    Returns:
        List of dicts with "BehaviorID", "Behavior", "SemanticCategory".
    """
    df = pd.read_csv(HARMBENCH_PATH)

    # Filter to text-only behaviors
    if "ContextString" in df.columns:
        df = df[df["ContextString"].isna() | (df["ContextString"] == "")]

    # Filter out null or empty Behavior field
    df = df[df["Behavior"].notna() & (df["Behavior"].str.strip() != "")]

    if categories:
        df = df[df["SemanticCategory"].isin(categories)]

    if n:
        df = df.head(n)

    return df[["BehaviorID", "Behavior", "SemanticCategory"]].to_dict("records")


def load_harmbench_stratified(n: int = 50) -> List[Dict]:
    """
    Load up to `n` HarmBench behaviors, sampled evenly across
    SemanticCategory so rarer categories aren't crowded out (unlike
    `load_harmbench`, which just takes the first `n` rows in file order).

    Args:
        n: target total number of behaviors; actual count may be slightly
            less than `n` due to integer division across categories.

    Returns:
        List of dicts with "BehaviorID", "Behavior", "SemanticCategory".
    """
    df = pd.read_csv(HARMBENCH_PATH)

    if "ContextString" in df.columns:
        df = df[df["ContextString"].isna() | (df["ContextString"] == "")]

    categories = df["SemanticCategory"].unique()
    per_category = max(1, n // len(categories))

    sampled = []
    for cat in categories:
        subset = df[df["SemanticCategory"] == cat].head(per_category)
        sampled.append(subset)

    result = pd.concat(sampled).head(n)
    return result[["BehaviorID", "Behavior", "SemanticCategory"]].to_dict("records")


def load_safedial(n: int = None) -> List[Dict]:
    """
    Load SafeDialBench English dataset.

    Note: not currently called anywhere else in this repo (the paper's
    reported results use HarmBench). Kept as a dataset-loading utility for
    reproducibility / potential future use — flagged here for author
    review before publication in case it should be removed or documented.
    """
    records = []
    with jsonlines.open(SAFEDIAL_EN_PATH) as f:
        for item in f:
            records.append(item)
            if n and len(records) >= n:
                break
    return records


def load_strongreject_behaviors() -> List[Dict]:
    """
    Load StrongREJECT forbidden prompts via the `strong_reject` package,
    remapped into the same {Behavior, BehaviorID, SemanticCategory} shape
    used by the HarmBench loaders above.

    Note: not currently called anywhere else in this repo. Kept as a
    dataset-loading utility for reproducibility / potential future use —
    flagged here for author review before publication in case it should
    be removed or documented.
    """
    from strong_reject.load_datasets import load_strongreject_small
    ds = load_strongreject_small()
    return [{"Behavior": row["forbidden_prompt"],
             "BehaviorID": f"sr_{i}",
             "SemanticCategory": row.get("category", "unknown")}
            for i, row in enumerate(ds)]

def load_fixed_stratified() -> List[Dict]:
    """
    Load the fixed, pre-sampled 100-behavior stratified set used for the
    secondary-model runs (see run_secondary.py), so all secondary models
    are evaluated on the exact same behavior sample for a fair comparison.
    """
    import json
    path = "/workspace/data/stratified_100.json"
    with open(path) as f:
        return json.load(f)