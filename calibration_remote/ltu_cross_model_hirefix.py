"""
ltu_cross_model_hirefix.py

Re-runs only what fig_cross_model_combined (entry_policy_exercise.py, Section 5B:
LTU bias x2.0, three non-cyclical models) needs after the hire() fix in abm_funs.py,
where biased hiring had skipped already-hired applicants instead of using the same
sample-one-and-fail rule as unbiased hiring.

Stage 1: 25 ltu_hire_bias = 1.0 sims per non-cyclical model (5 + 20 top-up, same
         sim_ids as the original caches), plus 25 baseline (no bias) sims per model,
         one worker process per model. The baseline is re-run rather than taken
         from model_results_N25.pkl because that cache predates the Failed_Hires
         instrumentation; using the new baseline for every panel keeps all panels
         on the same runs.
Stage 2: rebuilds fig_cross_model_combined by executing entry_policy_exercise.py up
         to the end of Section 5B, with the three models' LTU-bias runs read from
         the new caches. Everything else that script loads (stale bias caches for
         other sections) is only used for figures that are not saved here.

All output goes to entry_policy_hirefix/; nothing under entry_policy/ is touched.
Params, network and run_policy_scenario come from entry_policy_exercise.py's own
source, so the runs use exactly the same settings as the original caches.
"""

import os
import sys
import pickle
import multiprocessing as mp

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

EPE_PATH = os.path.join(PROJECT_ROOT, "entry_policy_exercise.py")
HIREFIX_DIR = os.path.join(
    PROJECT_ROOT, "output/calib_final/full_omn/exog_vac/figures/covid_oos/entry_policy_hirefix/"
)
HIREFIX_MODELS = ['otj_nonbehav', 'otj_disc', 'otj_disc_no_rw']   # Section 5B's _CM_SPECS_NO_CYC
N_WORKERS = 3

with open(EPE_PATH) as _f:
    EPE_SRC = _f.read()


def _between(src, start, end):
    """The source from `start` (inclusive) up to `end` (exclusive); each must occur exactly once."""
    assert src.count(start) == 1 and src.count(end) == 1, (start, end)
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


def _ltu_pkl(mc):
    return os.path.join(HIREFIX_DIR, f"cross_model_{mc}_ltu_bias.pkl")


def _topup_pkl(mc):
    return os.path.join(HIREFIX_DIR, f"cross_model_{mc}_topup.pkl")


def _baseline_pkl(mc):
    return os.path.join(HIREFIX_DIR, f"cross_model_{mc}_baseline.pkl")


def _all_pkls(mc):
    return (_ltu_pkl(mc), _topup_pkl(mc), _baseline_pkl(mc))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: LTU-bias runs
# ══════════════════════════════════════════════════════════════════════════════
_SETUP_SRC = "\n".join([
    # imports, settings, network build (stops before the N25 cache load)
    _between(EPE_SRC, '"""\nentry_policy_exercise.py', "# ── Load cached simulation results"),
    # REF_BASE_PARAMS + run_policy_scenario
    _between(EPE_SRC, "from copy import deepcopy\nfrom scipy.stats import gaussian_kde as _gkde",
             "# ── Baseline economy-wide time series"),
    # _CM_SPECS + _build_model_params
    _between(EPE_SRC, "_CM_SPECS = {", "if os.path.exists(CROSS_MODEL_PKL) and not FORCE_CROSS_RERUN:"),
])
_setup_ns = {'__file__': EPE_PATH, '__name__': 'entry_policy_exercise_setup'}


def _run_model(mc):
    """Worker (forked, so it inherits _setup_ns): both LTU-bias batches and the baseline for one model."""
    _bp = _setup_ns['_build_model_params'](mc)
    for _path, _key, _label, _overrides, _n, _offset in [
        (_ltu_pkl(mc),      f"{mc}_ltu_bias", f"{mc}_ltu_bias",       {'ltu_hire_bias': 1.0},
         _setup_ns['N_POLICY_SIMS'],    0),
        (_topup_pkl(mc),    f"{mc}_ltu_bias", f"{mc}_ltu_bias_topup", {'ltu_hire_bias': 1.0},
         _setup_ns['N_TOPUP_SIMS_DEF'], _setup_ns['N_POLICY_SIMS']),
        (_baseline_pkl(mc), f"{mc}_baseline", f"{mc}_baseline",       {},
         _setup_ns['N_TOTAL_SIMS'],     0),
    ]:
        if os.path.exists(_path):
            print(f"  [{mc}] {_path} exists, skipping.", flush=True)
            continue
        _res = _setup_ns['run_policy_scenario'](_label, _overrides, _bp,
                                                _setup_ns['CALIB_DATE'], n_sims=_n, sim_id_offset=_offset)
        with open(_path, 'wb') as _f:
            pickle.dump({_key: _res}, _f)
        print(f"  [{mc}] saved {_path}", flush=True)
        del _res
    return mc


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: fig_cross_model_combined from the new runs
#
# Builds only what Section 5B needs from entry_policy_exercise.py's source: the
# setup above, the helpers 5B calls, and 5B itself, with _load_cm_model reading
# the three models' new caches. The rest of that script (and the large pre-fix
# caches it loads) is skipped.
# ══════════════════════════════════════════════════════════════════════════════
_LOADER_SRC = """
from model_colors import MODEL_COLORS_BY_ID

def _concat_results(a, b):
    return tuple(pd.concat([x, y], ignore_index=True) for x, y in zip(a, b))

def _load_cm_model(mc):
    with open(_baseline_pkl(mc), 'rb') as _f:
        _bl = pickle.load(_f)[f'{mc}_baseline']
    with open(_ltu_pkl(mc), 'rb') as _f:
        _po = pickle.load(_f)[f'{mc}_ltu_bias']
    with open(_topup_pkl(mc), 'rb') as _f:
        _po = _concat_results(_po, pickle.load(_f)[f'{mc}_ltu_bias'])
    return _bl, _po
"""


def _figure_source():
    _cut = EPE_SRC.index("# SECTION 5E: Cross-model combined figure")
    section_5b = EPE_SRC[EPE_SRC.index('print("\\n=== Section 5B'):EPE_SRC.rindex("\n# ═", 0, _cut) + 1]
    src = "\n".join([
        _SETUP_SRC.replace("SAVE_FIG5_TO_DROPBOX = True", "SAVE_FIG5_TO_DROPBOX = False"),
        _between(EPE_SRC, "occ_wage_info = STATIC_OCC_DF[['Occupation', 'wage_mu']].copy()",
                 "MAX_DUR    = 36"),
        _between(EPE_SRC, "def _ltuer_mean(occ_df):", "_ltuer_base5 = _ltuer_mean"),
        _between(EPE_SRC, "def _ltu_stock_and_rate(", "fig_ld, (ax_ld1"),
        _LOADER_SRC,
        section_5b,
    ])
    _old = "save_plot_standard(fig_cb, os.path.join(OUTPUT_PATH, _fname_cb"
    assert src.count(_old) == 2, _old
    return src.replace(_old, "save_plot_standard(fig_cb, os.path.join(HIREFIX_DIR, _fname_cb")


if __name__ == "__main__":
    os.makedirs(HIREFIX_DIR, exist_ok=True)

    print("=== Stage 1: LTU-bias runs (hire() fix) ===", flush=True)
    exec(compile(_SETUP_SRC, EPE_PATH, 'exec'), _setup_ns)
    _todo = [mc for mc in HIREFIX_MODELS if not all(os.path.exists(p) for p in _all_pkls(mc))]
    if _todo:
        with mp.get_context('fork').Pool(min(N_WORKERS, len(_todo))) as _pool:
            for _mc in _pool.imap_unordered(_run_model, _todo):
                print(f"  ✓ {_mc} done", flush=True)
    del _setup_ns

    missing = [p for mc in HIREFIX_MODELS for p in _all_pkls(mc) if not os.path.exists(p)]
    assert not missing, f"Stage 1 incomplete, missing: {missing}"

    print("\n=== Stage 2: fig_cross_model_combined ===", flush=True)
    _fig_ns = {'__file__': EPE_PATH, '__name__': '__main__',
               'HIREFIX_DIR': HIREFIX_DIR, 'HIREFIX_MODELS': HIREFIX_MODELS,
               '_ltu_pkl': _ltu_pkl, '_topup_pkl': _topup_pkl, '_baseline_pkl': _baseline_pkl}
    exec(compile(_figure_source(), EPE_PATH, 'exec'), _fig_ns)
    print(f"\n✓ Done. Output written to {HIREFIX_DIR}")
