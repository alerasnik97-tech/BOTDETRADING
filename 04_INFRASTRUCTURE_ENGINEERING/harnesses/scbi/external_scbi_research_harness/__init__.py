"""
Harness externo de research para SCBI_M5_GLOBAL.
No modifica ningun archivo del laboratorio sellado.
"""

from .config import HarnessPaths, TruthModelConfig, default_paths
from .matrix import build_variants
from .orchestrator import load_research_inputs
from .strategy import run_truth_model

__all__ = [
    "HarnessPaths",
    "TruthModelConfig",
    "build_variants",
    "default_paths",
    "load_research_inputs",
    "run_truth_model",
]
