"""
test_wage_comparison.py

Sandbox for testing plot_wage_scarring_by_duration.
Loads cached N10 results and calls the function directly — no full pipeline run needed.
"""

import os
import sys
import pickle

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import plot_funs as pf
from model_colors import MODEL_COLORS

# ── Constants ─────────────────────────────────────────────────────────────────
CALIB_DATE = ["2001-01-01", "2024-05-01"]
DELAY      = 25

PKL_PATH = os.path.join(
    PROJECT_ROOT,
    "output/calib_final/full_omn/exog_vac/figures/covid_oos/cached_results/model_results_N10.pkl"
)

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {PKL_PATH} ...")
with open(PKL_PATH, 'rb') as f:
    cached_data = pickle.load(f)

_DESIRED_ORDER = list(MODEL_COLORS.keys())
_raw = cached_data['filtered_transition_dfs']
filtered_transition_dfs = {k: _raw[k] for k in _DESIRED_ORDER if k in _raw}
print(f"Models loaded: {list(filtered_transition_dfs.keys())}")

# ── Plot ──────────────────────────────────────────────────────────────────────
pf.plot_wage_scarring_by_duration(
    transition_dfs_dict = filtered_transition_dfs,
    calib_date          = CALIB_DATE,
    delay               = DELAY,
    colors              = MODEL_COLORS,
    latex_ready         = True,
    save                = False,
    path                = None,
)
