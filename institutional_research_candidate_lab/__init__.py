"""
Lab aislado de research institucional para candidatos SCBI.
RESEARCH_ONLY / NO_PRODUCTION.
"""

from .config import CandidateConfig, LabPaths, default_paths
from .candidate_matrix import build_baseline_config, build_candidate_matrix
from .orchestrator import execute_baseline, execute_matrix, resolve_paths

__all__ = [
    "CandidateConfig",
    "LabPaths",
    "build_baseline_config",
    "build_candidate_matrix",
    "default_paths",
    "execute_baseline",
    "execute_matrix",
    "resolve_paths",
]
