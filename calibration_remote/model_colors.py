"""
model_colors.py

Single source of truth for model color assignments.
Import in any script that produces plots:

    from model_colors import MODEL_COLORS, MODEL_COLORS_BY_ID

MODEL_COLORS        — keyed by display name (used in plot_funs, diagnose_* scripts)
MODEL_COLORS_BY_ID  — keyed by model identifier (used in auto_calibration_summary)

Color pairing logic (sns.color_palette("Paired", 12)):
  Blues   (#a6cee3, #1f78b4) — non-behavioural variants
  Purples (#cab2d6, #6a3d9a) — behavioural w.o. reservation wages
  Oranges (#fdbf6f, #ff7f00) — behavioural w. reservation wages
  Greens  (#b2df8a, #33a02c) — additional model variants (future)
"""

import seaborn as sns

_palette = sns.color_palette("Paired", 12)
_blues   = _palette[0:2]
_greens  = _palette[2:4]
_oranges = _palette[6:8]
_purples = _palette[8:10]
_reds = _palette[4:6]

# ── keyed by display name ─────────────────────────────────────────────────────
MODEL_COLORS = {
    "Non-behavioural":                   _blues[0],
    "Non-behavioural w. OTJ":            _blues[1],
    "Behavioural w.o. Cyc. OTJ w.o RW": _purples[1],
    "Behavioural w. Cyc. OTJ w.o RW": _purples[0],
    "Behavioural w.o. Cyc. OTJ w. RW":  _oranges[1],
    # future models:
   
    "Behavioural w. Cyc. OTJ w. RW":  _oranges[0],
    # "Additional model A":              _greens[0],
    # "Additional model B":              _greens[0],
}

# ── keyed by model identifier ─────────────────────────────────────────────────
MODEL_COLORS_BY_ID = {
    "nonbehav":                    _blues[0],
    "otj_nonbehav":                _blues[1],
    "otj_disc_no_rw":              _purples[1],
    "otj_cyclical_e_disc_no_rw": _purples[0],
    "otj_disc":                    _oranges[1],
    # future models:
    "otj_cyclical_e_disc":       _oranges[0],
    # additional model A:          _greens[0],
    # additional model B:          _greens[0],
}

