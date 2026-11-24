"""Locations of data, intermediate files and figures. Override with environment variables."""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
DATA = os.environ.get("HCBON_DATA", str(_ROOT / "data"))      # DiffIQA/, SRIQA-Bench/, FGRestore/, label CSVs
WORK = os.environ.get("HCBON_WORK", str(_ROOT / "work"))      # tables, scores, predictions, JSON results
FIGS = os.environ.get("HCBON_FIGS", str(_ROOT / "figs"))      # paper figures
