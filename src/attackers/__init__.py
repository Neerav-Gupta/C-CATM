# /workspace/src/attackers/__init__.py
"""
Public API for the `attackers` package.

Re-exports the base class and each multi-turn jailbreak attack
implementation so callers can do `from src.attackers import CrescendoAttacker`
etc. without reaching into individual submodules. These imports are
intentionally "unused" from pyflakes' perspective — that's expected for
a package __init__.py acting as a re-export surface.
"""
from .base import MultiTurnAttacker
from .crescendo import CrescendoAttacker
from .jigsaw import JigsawAttacker
from .echo_chamber import EchoChamberAttacker
from .safe_in_isolation import SafeInIsolationAttacker