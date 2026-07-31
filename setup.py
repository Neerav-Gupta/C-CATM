"""
Packaging config for the C-CATM library. Install with `pip install -e .`
(editable) or `pip install .` to make the `ccatm` package and its runtime
dependencies (torch, transformers, flash-attn, etc.) available for the
scripts in this repo.
"""

from setuptools import setup, find_packages

setup(
    name="ccatm",
    version="1.0.0",
    description="Cross-turn Contrastive Activation Trajectory Monitor for LLM Safety",
    author="Neerav Gupta",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.8.0",
        "transformers>=4.40.0",
        "flash-attn>=2.8.3",
        "numpy>=1.24.0",
        "jsonlines>=4.0.0",
    ],
)
