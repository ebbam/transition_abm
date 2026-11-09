"""
exposure_entry_policy_exercise.py

Exposure-shock policy exercise for the labour market transformation paper.

Unlike entry_policy_exercise.py (which runs the calibrated model over the
historical 2001-2024 window), this script asks a forward-looking question:
what happens to entry-level workers if AI depresses labour demand in specific
occupations, and does preferential hiring for entry-level workers help?

Design:
  1. Warm start: run the calibrated reference model to the end of the
     calibration window (2024-05) and capture its live network state (worker-
     level employment/unemployment, open vacancies) rather than re-initialising
     from scratch. The occupational value-added shock series our calibration
     is built on ends at that date, so there is no real "future" demand signal
     to feed in — every occupation's target_demand multiplier (occ_shocks_data)
     is held at 1.0 (its calibrated, no-shock level) going forward, EXCEPT for
     the imposed AI shock below. target_demand itself (the absolute headcount
     anchor used in the separation/vacancy-creation formulas) is left at its
     calibrated value throughout, not reset to a literal 1 — see the design
     discussion this script's commit message links to for why that literal
     reading would collapse the whole economy rather than isolate the shock.
  2. The real-world vacancy rate is held at its last observed calibration-
     window level for the whole shock horizon.
  3. Occupation-specific AI exposure shock, built from EXACTLY ONE of two
     available exposure scores -- Anthropic Economic Index or Eloundou et al.
     (2024, Science) (see data/ai_exposure/README.md and the EXPOSURE_SOURCE
     setting below; the two are not blended together) -- applied as a
     declining S-curve multiplier on target demand. Stanford Digital Economy
     Lab's "Canaries in the Coal Mine" doesn't publish its own continuous
     exposure score (it reuses AIOE/AEI against ADP payroll data), so it
     isn't a third exposure option here; its ~19% entry-level employment gap
     finding instead anchors the shock's maximum magnitude
     (MAX_DEMAND_REDUCTION).
  4. Two shock-phase runs per warm-started state: with and without entry-level
     preferential hiring (entry_hire_bias), so the comparison is paired on the
     same starting conditions.

Output layout: everything that depends on EXPOSURE_SOURCE (shock results,
figures, summaries) is written to a source-named subfolder
(OUTPUT_PATH/anthropic/ or OUTPUT_PATH/eloundou/) so both sources' results
can coexist. Warm-start states don't depend on the exposure source and stay
in OUTPUT_PATH directly, shared by both.

Topping up sims: N_WARMSTART_TOPUP_SIMS adds more simulations on top of
N_WARMSTART_SIMS (for both the reference model and each cross-model variant)
without re-running the original batch -- the original and topup batches are
cached separately and concatenated in memory. Set N_WARMSTART_TOPUP_SIMS = 0
to skip topping up.

Run interactively in VS Code "Run in Interactive Window".
"""

import os
import sys
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from network_input_builder import network_input_builder
from model_fun import run_single_local
from collate_macro_vars import macro_observations
from abm_funs import duration_to_prob_dict
from plot_style import TEXTWIDTH, ROW_HEIGHT, save_plot_standard, add_shared_legend, add_suptitle_spacing
from matplotlib.lines import Line2D

# ── Settings ──────────────────────────────────────────────────────────────────
NW              = "full_omn"
CALIB_DATE      = ["2001-01-01", "2024-05-01"]
DELAY           = 25
VAC_SETTING     = "exog_vac"
REFERENCE_MODEL = "otj_cyclical_e_disc"   # Behavioural w. Cyc. OTJ w. RW

SHOCK_HORIZON_MONTHS = 240     # total months simulated after the shock begins (20 years). Decoupled
                                # from the S-curve's own ramp-in speed (RAMP_HORIZON_MONTHS below): the
                                # shock still phases in over the same ~5 years as before, then holds
                                # flat at its fully-ramped level for the remaining ~15 years. Chosen so
                                # that, together with COHORT_WINDOW_MONTHS below, every cohort member
                                # (even the last to arrive, at the end of the cohort window) still gets
                                # a full 15-year uniform follow-up for the lifetime-earnings comparison
                                # (Section 13) -- see MAX_FOLLOWUP_YEARS. Extending this constant does
                                # NOT slow the ramp itself down, it just simulates further past it.
NEAR_TERM_HORIZON_MONTHS = 60  # Sections 6-8/12 (the original near-term shock-response figures/summary)
                                # are truncated to this window, so they keep showing the same interpretable
                                # ~5-year story as before rather than being swamped by 20 years of a flat
                                # post-ramp plateau. Section 13 (lifetime earnings) uses the full horizon.
SHOCK_DELAY          = 0       # no burn-in needed: we're warm-starting from a real state
N_WARMSTART_SIMS      = 5
N_WARMSTART_TOPUP_SIMS = 5     # additional sims on top of N_WARMSTART_SIMS, cached separately
N_TOTAL_WARMSTART_SIMS = N_WARMSTART_SIMS + N_WARMSTART_TOPUP_SIMS

EXPOSURE_SOURCE = "eloundou_v2"   # "anthropic" or "eloundou" -- which exposure score drives the
                                # shock. These are NOT blended together -- the shock uses exactly
                                # one source at a time. Every downstream output that depends on
                                # this choice is written to OUTPUT_PATH/<EXPOSURE_SOURCE>/, so
                                # switching sources never overwrites or silently reuses the other
                                # source's cache/figures.
                                # "_v2" sources ("eloundou_v2", "anthropic_v2") read the hierarchical SOC
                                # matching in data/occ_macro_vars/eloundou_exposure/ (all 528 occupations
                                # matched, vs 361 for Eloundou in the original matching below); the
                                # original sources are kept so earlier results can be reproduced.
assert EXPOSURE_SOURCE in ("anthropic", "eloundou", "anthropic_v2", "eloundou_v2")

EXERCISE = os.environ.get("EXPOSURE_EXERCISE", "shock")
                                # "shock":  scenario analysis -- the exposure shock vs no shock, no hiring bias
                                #           in either, for every CM_SPECS model.
                                # "policy": entry-level preferential hiring vs none, both under the shock, for
                                #           every CM_SPECS model. Its no_policy arm is the "shock" scenario of
                                #           the shock-mode runs (same warm starts and sim_ids), so run "shock"
                                #           mode first. Set EXPOSURE_EXERCISE=policy to choose it from the shell.
assert EXERCISE in ("shock", "policy")
if EXERCISE == "shock":
    BASE_SCEN, TREAT_SCEN = 'no_shock', 'shock'
    BASE_LABEL, TREAT_LABEL = 'No shock', 'Shock'
else:
    BASE_SCEN, TREAT_SCEN = 'no_policy', 'policy'
    BASE_LABEL, TREAT_LABEL = 'No policy', 'Policy'
DELTA_LABEL = f"{TREAT_LABEL} − {BASE_LABEL.lower()}"

ONLY_MODEL = None
                                # None: all cross-model variants.
                                # A CM_SPECS key (e.g. "otj_disc"): run only that model -- the reference
                                # model (Sections 4-6, 8, 13) and the multi-model figures (Sections 12,
                                # 14) are skipped; Section 15's single-model summary figures cover it.
RUN_REFERENCE_MODEL = False     # the reference (cyclical-OTJ) model's cached runs predate the hire() fix and the
                                # 7% OTJ search share, so Sections 4-6, 8 and 13 are off in both modes
N_CM_WORKERS = 3                # Section 11's shock-phase runs, one process per model

MAX_DEMAND_REDUCTION = 0.19    # max fractional cut to target-demand multiplier at
                                # full exposure & full ramp; anchored to the Canaries
                                # paper's ~19% entry-level employment gap in AI-exposed
                                # occupations (a magnitude calibration, not a direct
                                # replication -- the entry-level employment effect
                                # here is a model output, not an input)
RAMP_HORIZON_MONTHS  = 60      # how long the S-curve itself takes to fully ramp in -- independent of
                                # SHOCK_HORIZON_MONTHS (the total simulated length)
RAMP_MIDPOINT_FRAC   = 0.5     # S-curve midpoint as a fraction of the RAMP horizon
RAMP_STEEPNESS       = 8.0 / RAMP_HORIZON_MONTHS
ELOUNDOU_EXPOSURE_COL = 'dv_rating_beta'   # E2: LLM + software exposure

DATA_DIR               = os.path.join(PROJECT_ROOT, "data", "ai_exposure")
ANTHROPIC_EXPOSURE_CSV = os.path.join(DATA_DIR, "anthropic_economic_index_job_exposure.csv")
ELOUNDOU_EXPOSURE_CSV  = os.path.join(DATA_DIR, "eloundou_2024_occ_level.csv")
EXPOSURE_V2_CSV        = os.path.join(PROJECT_ROOT, "..", "data", "occ_macro_vars", "eloundou_exposure",
                                      "ai_exposure_by_acs_full_omn.csv")
OCC_NAMES_CSV  = "/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/data/occ_names_employment_asec_occ.csv"
OCC_SOC_CW_CSV = "/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/ONET/occ_names_employment_asec_occ_2010_soc_cw.csv"

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    f"output/calib_final/{NW}/{VAC_SETTING}/figures/covid_oos/exposure_entry_policy/"
)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Source-specific outputs (shock results, figures, summaries) live here.
# Warm-start states are exposure-independent and stay directly in OUTPUT_PATH.
# "_hirefix": results after the hire() fix (biased hiring now uses the same
# sample-one-and-fail rule as unbiased hiring, instead of skipping already-hired
# applicants). The pre-fix outputs in PRE_FIX_SOURCE_OUTPUT_PATH are left as they
# were; only their no_policy runs (bias 0, unaffected by the fix) are reused.
# "_shock_scenarios": EXERCISE = "shock" (all of its runs are new; none are reused).
# "_policy_7pct": EXERCISE = "policy" after the 7% OTJ search share was restored (the "_hirefix" runs used
# 3%). Only the policy arm is simulated; no_policy is the "shock" scenario read from the shock-mode caches.
PRE_FIX_SOURCE_OUTPUT_PATH = os.path.join(OUTPUT_PATH, f"{EXPOSURE_SOURCE}_shock_scenarios")
NO_POLICY_SOURCE_KEY = 'shock'   # scenario key of the no_policy arm inside PRE_FIX_SOURCE_OUTPUT_PATH's caches
SOURCE_OUTPUT_PATH = os.path.join(
    OUTPUT_PATH, f"{EXPOSURE_SOURCE}_shock_scenarios" if EXERCISE == "shock" else f"{EXPOSURE_SOURCE}_policy_7pct")
os.makedirs(SOURCE_OUTPUT_PATH, exist_ok=True)

WARMSTART_PKL       = os.path.join(OUTPUT_PATH, "warmstart_states.pkl")
WARMSTART_TOPUP_PKL = os.path.join(OUTPUT_PATH, "warmstart_states_topup.pkl")
SHOCK_PKL           = os.path.join(SOURCE_OUTPUT_PATH, "shock_results.pkl")
SHOCK_TOPUP_PKL     = os.path.join(SOURCE_OUTPUT_PATH, "shock_results_topup.pkl")
FORCE_WARMSTART_RERUN       = False
FORCE_WARMSTART_TOPUP_RERUN = False
FORCE_SHOCK_RERUN           = False
FORCE_SHOCK_TOPUP_RERUN     = False
FORCE_CM_WARMSTART_RERUN       = False   # cross-model warm-start (Section 10) -- separate flags:
FORCE_CM_WARMSTART_TOPUP_RERUN = False   # each variant's warm-start is its own expensive 24yr run
FORCE_CM_SHOCK_RERUN           = False
FORCE_CM_SHOCK_TOPUP_RERUN     = False

NO_POLICY_COLOR = '#888888'
POLICY_COLOR    = '#7b5ea7'   # matches the "entry-level" purple used elsewhere in this project
LW              = 1.5         # main line width -- slightly thinner than a first pass at 2,
                               # matching entry_policy_exercise.py's convention for 2-4-series
                               # panels (its ax_C/ax4A/etc use 2, but its busier per-model
                               # comparison panels drop to 1.0-1.8; see LW_CM below)
LW_CM           = 1.0         # cross-model line width -- matches entry_policy_exercise.py's
                               # own cross-model Δ time-series panels (ax_C/ax_HR), which use
                               # linewidth=1.0 for the same reason: several overlapping model lines


def _load_or_build(path, force, build_fn, label):
    """Cache-or-compute helper: load `path` if it exists (and not `force`),
    otherwise call build_fn(), pickle the result to `path`, and return it."""
    if os.path.exists(path) and not force:
        print(f"  [{label}] loading cached result from {path} ...")
        with open(path, 'rb') as _f:
            return pickle.load(_f)
    print(f"  [{label}] cache miss or forced rerun -- computing ...")
    result = build_fn()
    with open(path, 'wb') as _f:
        pickle.dump(result, _f)
    print(f"  [{label}] saved to {path}")
    return result


def _run_warmstart_batch(params, n_sims, label):
    states = []
    for _i in range(n_sims):
        _result = run_single_local(**deepcopy(params))
        states.append({'net': _result[2], 'vacs': _result[8]})
        print(f"    [{label}] warm-start sim {_i + 1}/{n_sims} complete")
    return states


def _run_shock_batch(base_params, scenarios, warmstart_states, sim_id_offset, label):
    """Run every scenario in `scenarios` (name -> param overrides) against
    each given warm-started state, tagging sim_id starting at sim_id_offset
    (so a topup batch's sim_ids continue on from the base batch's, rather
    than colliding with them). Returns (results, wage_age_results): `results`
    keeps its original {scenario: (trans_df, occ_df)} shape so every existing
    caller/cache is unaffected; wage_age_results is a new, separate
    {scenario: wage_age_df} dict (annual per-worker wage/age snapshots,
    including Entry_Time_Step for entrants -- used for the lifetime-earnings
    comparison, Section 13)."""
    results = {}
    wage_age_results = {}
    for _scen_name, _overrides in scenarios.items():
        _trans_runs, _occ_runs, _wage_age_runs = [], [], []
        for _i, _ws in enumerate(warmstart_states):
            _params = deepcopy(base_params)
            _params.update(_overrides)
            _params['net_temp'] = _ws['net']
            _params['vacs']     = _ws['vacs']
            _result = run_single_local(**_params)
            _sim_id = sim_id_offset + _i
            _record_df = _result[0].copy();      _record_df['sim_id']      = _sim_id
            _wage_age_df = _result[9].copy();     _wage_age_df['sim_id']    = _sim_id
            _transition_df = _result[11].copy(); _transition_df['sim_id'] = _sim_id
            _trans_runs.append(_transition_df)
            _occ_runs.append(_record_df)
            _wage_age_runs.append(_wage_age_df)
            print(f"  [{label}][{_scen_name}] shock-phase run {_i + 1}/{len(warmstart_states)} complete")
        results[_scen_name] = (pd.concat(_trans_runs, ignore_index=True),
                                pd.concat(_occ_runs,   ignore_index=True))
        wage_age_results[_scen_name] = pd.concat(_wage_age_runs, ignore_index=True)
    return results, wage_age_results


def _concat_shock_results(a, b):
    """Concatenate two {scenario: (trans_df, occ_df)} dicts (base + topup)."""
    return {_scen: (pd.concat([a[_scen][0], b[_scen][0]], ignore_index=True),
                    pd.concat([a[_scen][1], b[_scen][1]], ignore_index=True))
            for _scen in a}


def _concat_wage_age_results(a, b):
    """Concatenate two {scenario: wage_age_df} dicts (base + topup)."""
    return {_scen: pd.concat([a[_scen], b[_scen]], ignore_index=True) for _scen in a}


def _assert_shock_batch_format(loaded, pkl_path):
    """_run_shock_batch's return shape changed (now (results, wage_age_results)
    instead of just results) when Entry_Time_Step/wage_age_df capture was
    added. A stale pre-change cache would otherwise unpack silently wrong
    (SHOCK_SCENARIOS has exactly 2 keys, so `a, b = old_dict` "succeeds" by
    assigning the two scenario names to a and b) -- fail loudly instead."""
    ok = (isinstance(loaded, tuple) and len(loaded) == 2
          and isinstance(loaded[0], dict) and isinstance(loaded[1], dict)
          and all(isinstance(v, tuple) and len(v) == 2 for v in loaded[0].values()))
    assert ok, (
        f"{pkl_path} looks like it's in the old cache format (from before wage_age_df "
        f"capture was added). Delete it (and its topup counterpart) or set the matching "
        f"FORCE_*_RERUN flag to True, then rerun."
    )


def _run_policy_only(base_params, warmstart_states, sim_id_offset, label):
    """_run_shock_batch for the policy scenario only. no_policy has entry_hire_bias = 0,
    which never reached the fixed hire() branch, so its pre-fix runs are still valid
    (and, since the model is unseeded, rerunning them would not reproduce them anyway):
    they are read from the pre-fix cache by _with_pre_fix_no_policy instead of being
    re-simulated or stored a second time. Same warm-start states and sim_ids as the
    pre-fix batch, so the pairing by sim_id is unchanged."""
    return _run_shock_batch(base_params, {TREAT_SCEN: SHOCK_SCENARIOS[TREAT_SCEN]},
                            warmstart_states, sim_id_offset, label)


def _with_pre_fix_no_policy(policy_raw, pre_fix_pkl):
    """Combine a policy-only (results, wage_age_results) batch with the no_policy
    scenario from the matching pre-fix cache."""
    with open(pre_fix_pkl, 'rb') as _f:
        _pre_fix = pickle.load(_f)
    _assert_shock_batch_format(_pre_fix, pre_fix_pkl)
    _pre_res, _pre_wage_age = _pre_fix
    _res, _wage_age = policy_raw
    assert set(_pre_res[NO_POLICY_SOURCE_KEY][1]['sim_id'].unique()) == set(_res[TREAT_SCEN][1]['sim_id'].unique()), \
        f"{pre_fix_pkl}: cached no_policy sim_ids do not match the re-run policy sim_ids"
    return ({BASE_SCEN: _pre_res[NO_POLICY_SOURCE_KEY], TREAT_SCEN: _res[TREAT_SCEN]},
            {BASE_SCEN: _pre_wage_age[NO_POLICY_SOURCE_KEY], TREAT_SCEN: _wage_age[TREAT_SCEN]})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: network + calibrated reference parameters
# ══════════════════════════════════════════════════════════════════════════════
print("=== Section 1: Building network and reference parameters ===")

_mod_data, _net_temp, _vacs, _, _occ_shocks_dat = network_input_builder(NW, False, CALIB_DATE, delay=DELAY)
N_OCC = len(_net_temp)
ENTRY_IDS = set(o.occupation_id for o in _net_temp if o.entry_level)
NON_ENTRY_IDS = set(range(N_OCC)) - ENTRY_IDS
print(f"  {N_OCC} occupations, {len(ENTRY_IDS)} entry-level.")

OCC_WAGE_INFO = pd.DataFrame({
    'Occupation':       [o.occupation_id for o in _net_temp],
    'occ_median_wage':  [float(np.exp(o.wage_mu)) for o in _net_temp],
})

CALIB_CSV = os.path.join(
    PROJECT_ROOT,
    "output/calib_final/full_omn/exog_vac/"
    "calibrated_params_all_theta_signal_exog_wage_update_True.csv"
)
_param_df = pd.read_csv(CALIB_CSV)
_ref = (_param_df[_param_df['model_cat'] == REFERENCE_MODEL]
        .sort_values('Timestamp', ascending=False))

_obs = macro_observations.loc[
    (macro_observations['DATE'] >= CALIB_DATE[0]) &
    (macro_observations['DATE'] <= CALIB_DATE[1])
].dropna(subset=['VACRATE'])
_vac_dat_hist = _obs['VACRATE'].to_numpy()
LAST_VACRATE  = float(_obs['VACRATE'].iloc[-1])
print(f"  Last observed VACRATE ({_obs['DATE'].iloc[-1].date()}): {LAST_VACRATE:.4f} — held flat for the shock horizon.")

REF_BASE_PARAMS = {
    'mod_data':           _mod_data,
    'net_temp':           _net_temp,
    'vacs':               _vacs,
    'time_steps':         int(_occ_shocks_dat.shape[1]),
    'occ_shocks_data':    _occ_shocks_dat,
    'app_effort_dat':     duration_to_prob_dict,
    'gdp_data':           None,   # unused inside run_single_local
    'bus_confidence_dat': None,   # unused inside run_single_local
    'vac_data':           _vac_dat_hist,
    'delay':              DELAY,
    'simple_res':         False,
    'dumb_hire':          False,
    'mistake_rate':       0.1,
    'vac_method':         'exog',
    'vac_wage_update':    True,
    'steady_state':       False,
    'otj':                True,
    'cyc_otj':             True,
    'cyc_ue':              False,
    'disc':                True,
    'wage_prefs':          True,
    'emp_apps':            1,
    'strict_rw':           False,
    'd_v':                 None,
    'gamma_v':              None,
    'gamma_u': float(_ref.loc[_ref['Parameter'] == 'gamma_u', 'Value'].iloc[0]),
    'd_u':     float(_ref.loc[_ref['Parameter'] == 'd_u',     'Value'].iloc[0]),
    'theta':   float(_ref.loc[_ref['Parameter'] == 'theta',   'Value'].iloc[0]),
}
print(f"  Reference model params: gamma_u={REF_BASE_PARAMS['gamma_u']:.4f}, "
      f"d_u={REF_BASE_PARAMS['d_u']:.5f}, theta={REF_BASE_PARAMS['theta']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: occupational AI-exposure vector
#
# full_omn occupations are Census/ACS 2010 occupation codes (~528 detailed
# occupations, one row per network occupation_id). We crosswalk these to 2010
# SOC detailed codes, then attach exposure scores from EXACTLY ONE of:
#   - Anthropic Economic Index (labor_market_impacts/job_exposure.csv):
#     observed_exposure, keyed by 2018 SOC detailed code.
#   - Eloundou et al. (2024): dv_rating_beta (E2), keyed by O*NET-SOC, averaged
#     up to the 6-digit SOC detailed code (O*NET-SOC splits some SOC codes
#     into finer sub-occupations).
# The selected source's score is min-max normalised to [0,1]; occupations
# with no match in that source (mostly small/niche 2010 ACS codes not carried
# into the 2018 exposure work) get exposure = 0, i.e. treated as unshocked.
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 2: Building occupational AI-exposure vector ===")

_occ_names = pd.read_csv(OCC_NAMES_CSV)[['Code', 'Label']]
_occ_names['occupation_id'] = _occ_names.index.values
assert list(_occ_names['occupation_id']) == list(range(N_OCC)), \
    "occ_names_employment_asec_occ.csv row order must match network occupation_id"

def _norm01(s):
    return (s - s.min()) / (s.max() - s.min())


if EXPOSURE_SOURCE.endswith('_v2'):
    # Hierarchical SOC matching (data/occ_macro_vars/eloundou_exposure/ai_exposure_matching_full_omn.R):
    # one row per network occupation, in network order, every occupation matched.
    _base = EXPOSURE_SOURCE[:-len('_v2')]
    _v2 = pd.read_csv(EXPOSURE_V2_CSV)
    assert list(_v2['acs_occ_code']) == list(_occ_names['Code']), \
        "ai_exposure_by_acs_full_omn.csv must be in network occupation order"
    _exp = _occ_names.copy()
    _exp[f'{EXPOSURE_SOURCE}_exposure'] = _v2[f'{_base}_exposure'].to_numpy()
    _exp['match_level'] = _v2[f'{_base}_match_level'].to_numpy()
    _exp['selected_exposure'] = _norm01(_exp[f'{EXPOSURE_SOURCE}_exposure'])
else:
    _cw = pd.read_csv(OCC_SOC_CW_CSV)[['acs_2010_code', 'soc_2010_code']]

    _anthropic = pd.read_csv(ANTHROPIC_EXPOSURE_CSV)
    _anthropic_by_soc = _anthropic.set_index('occ_code')['observed_exposure'].rename('anthropic_exposure')

    _eloundou = pd.read_csv(ELOUNDOU_EXPOSURE_CSV)
    _eloundou['soc6'] = _eloundou['O*NET-SOC Code'].str.split('.').str[0]
    _eloundou_by_soc = _eloundou.groupby('soc6')[ELOUNDOU_EXPOSURE_COL].mean().rename('eloundou_exposure')

    _exp = _occ_names.merge(_cw, left_on='Code', right_on='acs_2010_code', how='left')
    _exp = _exp.merge(_anthropic_by_soc, left_on='soc_2010_code', right_index=True, how='left')
    _exp = _exp.merge(_eloundou_by_soc, left_on='soc_2010_code', right_index=True, how='left')

    _exp['anthropic_norm'] = _norm01(_exp['anthropic_exposure'])
    _exp['eloundou_norm']  = _norm01(_exp['eloundou_exposure'])
    # EXPOSURE_SOURCE selects exactly one source -- no blending across sources.
    _exp['selected_exposure'] = _exp[f'{EXPOSURE_SOURCE}_norm']

# A single ACS code can map to >1 SOC code (crosswalk fan-out) -- collapse
# back to one exposure value per occupation_id before reindexing.
EXPOSURE_BY_OCCID = (
    _exp.groupby('occupation_id')['selected_exposure'].mean()
        .reindex(range(N_OCC))
        .fillna(0.0)
)
EXPOSURE_VEC = EXPOSURE_BY_OCCID.to_numpy()
# Top-quartile exposure vs the rest -- the same split as Section 7's shock-implementation figure.
HIGH_EXPOSURE_IDS = set(np.flatnonzero(EXPOSURE_VEC >= np.quantile(EXPOSURE_VEC, 0.75)))
LOW_EXPOSURE_IDS  = set(range(N_OCC)) - HIGH_EXPOSURE_IDS

_n_matched = int(_exp.groupby('occupation_id')[f'{EXPOSURE_SOURCE}_exposure'].first().notna().sum())
print(f"  Exposure source: {EXPOSURE_SOURCE}")
print(f"  {_n_matched}/{N_OCC} occupations matched to {EXPOSURE_SOURCE}.")
print(f"  Exposure summary: mean={EXPOSURE_VEC.mean():.3f}, "
      f"p90={np.quantile(EXPOSURE_VEC, 0.9):.3f}, max={EXPOSURE_VEC.max():.3f}")

_top10 = (_exp.groupby('occupation_id')
              .agg(Label=('Label', 'first'), selected_exposure=('selected_exposure', 'mean'))
              .reindex(range(N_OCC)).fillna(0.0)
              .sort_values('selected_exposure', ascending=False).head(10))
print("  Top-10 most exposed occupations:")
for _occid, _row in _top10.iterrows():
    _tag = " [entry-level]" if _occid in ENTRY_IDS else ""
    print(f"    occ {_occid:>3} {_row['Label']:<45} exposure={_row['selected_exposure']:.3f}{_tag}")

pd.DataFrame({
    'occupation_id': range(N_OCC),
    'label':         _occ_names.set_index('occupation_id')['Label'].reindex(range(N_OCC)).values,
    'entry_level':   [oid in ENTRY_IDS for oid in range(N_OCC)],
    'exposure':      EXPOSURE_VEC,
    **({'match_level': _exp.groupby('occupation_id')['match_level'].first().reindex(range(N_OCC)).values}
       if 'match_level' in _exp else {}),
}).to_csv(os.path.join(SOURCE_OUTPUT_PATH, "exposure_by_occupation.csv"), index=False)
print("  Saved: exposure_by_occupation.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: S-curve shock construction
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 3: Building S-curve shock ===")

H = SHOCK_HORIZON_MONTHS + 1   # +1 accounts for model_fun's `range(time_steps-1)` loop
_t = np.arange(H)

# The S-curve ramps in over RAMP_HORIZON_MONTHS (unchanged shape/speed regardless
# of the total horizon H), then holds flat at 1.0 (fully ramped) for the rest.
_ramp_h = RAMP_HORIZON_MONTHS + 1
_ramp_t = np.arange(_ramp_h)
_raw_ramp = 1.0 / (1.0 + np.exp(-RAMP_STEEPNESS * (_ramp_t - RAMP_MIDPOINT_FRAC * (_ramp_h - 1))))
_ramp_normalised = (_raw_ramp - _raw_ramp.min()) / (_raw_ramp.max() - _raw_ramp.min())   # exactly 0 -> 1
RAMP = np.concatenate([_ramp_normalised, np.ones(max(0, H - _ramp_h))])[:H]

# occ_shocks_data needs 24 pre-fill columns (unused here: vac_method='exog' means
# the endogenous-vacancy deque they feed is never read) followed by H shock columns.
SHOCK_OCC_ARRAY = np.ones((N_OCC, 24 + H))
for _occid in range(N_OCC):
    SHOCK_OCC_ARRAY[_occid, 24:] = 1.0 - MAX_DEMAND_REDUCTION * EXPOSURE_VEC[_occid] * RAMP

SHOCK_VAC_DATA = np.full(H, LAST_VACRATE)

if EXERCISE == "shock":
    # no_shock: the same 20 years with the target-demand multiplier held at 1 throughout
    # (same constant vacancy rate, same warm starts) -- the shock is the only difference.
    SHOCK_SCENARIOS = {
        BASE_SCEN:  {'entry_hire_bias': 0.0, 'occ_shocks_data': np.ones_like(SHOCK_OCC_ARRAY)},
        TREAT_SCEN: {'entry_hire_bias': 0.0},
    }
else:
    SHOCK_SCENARIOS = {
        BASE_SCEN:  {'entry_hire_bias': 0.0},
        TREAT_SCEN: {'entry_hire_bias': 1.0},
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: warm start — run the reference model to the end of the calibration
# window and capture its live network + open-vacancy state. Topped up with
# N_WARMSTART_TOPUP_SIMS additional sims, cached separately from the original
# N_WARMSTART_SIMS batch so the original never needs re-running.
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 4: Warm-start runs ===")

if RUN_REFERENCE_MODEL:
    WARMSTART_STATES = _load_or_build(
        WARMSTART_PKL, FORCE_WARMSTART_RERUN,
        lambda: _run_warmstart_batch(REF_BASE_PARAMS, N_WARMSTART_SIMS, "ref"),
        label="ref warm-start",
    )
    if N_WARMSTART_TOPUP_SIMS > 0:
        _ws_topup = _load_or_build(
            WARMSTART_TOPUP_PKL, FORCE_WARMSTART_TOPUP_RERUN,
            lambda: _run_warmstart_batch(REF_BASE_PARAMS, N_WARMSTART_TOPUP_SIMS, "ref topup"),
            label="ref warm-start topup",
        )
        WARMSTART_STATES = WARMSTART_STATES + _ws_topup
    print(f"  {len(WARMSTART_STATES)} total warm-start states ready.")

    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 5: shock-phase runs — with vs without entry-level preferential hiring
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n=== Section 5: Shock-phase runs ===")

    SHOCK_BASE_PARAMS = deepcopy(REF_BASE_PARAMS)
    SHOCK_BASE_PARAMS.update({
        'time_steps':      int(SHOCK_OCC_ARRAY.shape[1]),
        'occ_shocks_data': SHOCK_OCC_ARRAY,
        'vac_data':        SHOCK_VAC_DATA,
        'delay':           SHOCK_DELAY,
    })

    _shock_base_raw = _load_or_build(
        SHOCK_PKL, FORCE_SHOCK_RERUN,
        lambda: _run_policy_only(SHOCK_BASE_PARAMS, WARMSTART_STATES[:N_WARMSTART_SIMS],
                                  sim_id_offset=0, label="ref"),
        label="ref shock-phase",
    )
    _assert_shock_batch_format(_shock_base_raw, SHOCK_PKL)
    SHOCK_RESULTS, SHOCK_WAGE_AGE = _with_pre_fix_no_policy(
        _shock_base_raw, os.path.join(PRE_FIX_SOURCE_OUTPUT_PATH, "shock_results.pkl"))
    if N_WARMSTART_TOPUP_SIMS > 0:
        _shock_topup_raw = _load_or_build(
            SHOCK_TOPUP_PKL, FORCE_SHOCK_TOPUP_RERUN,
            lambda: _run_policy_only(SHOCK_BASE_PARAMS, WARMSTART_STATES[N_WARMSTART_SIMS:],
                                      sim_id_offset=N_WARMSTART_SIMS, label="ref topup"),
            label="ref shock-phase topup",
        )
        _assert_shock_batch_format(_shock_topup_raw, SHOCK_TOPUP_PKL)
        _shock_topup, _wage_age_topup = _with_pre_fix_no_policy(
            _shock_topup_raw, os.path.join(PRE_FIX_SOURCE_OUTPUT_PATH, "shock_results_topup.pkl"))
        SHOCK_RESULTS = _concat_shock_results(SHOCK_RESULTS, _shock_topup)
        SHOCK_WAGE_AGE = _concat_wage_age_results(SHOCK_WAGE_AGE, _wage_age_topup)
    print(f"  {SHOCK_RESULTS[BASE_SCEN][1]['sim_id'].nunique()} total sims per scenario after topup.")


    def _truncate_near_term(shock_results, months=NEAR_TERM_HORIZON_MONTHS):
        """Restrict a {scenario: (trans_df, occ_df)} dict to the first `months`
        months -- Sections 6-8/12 describe the shock's near-term response and
        stay on the original ~5-year window regardless of how long
        SHOCK_HORIZON_MONTHS simulates for; only Section 13 (lifetime earnings)
        uses the full horizon."""
        _cutoff = months + 1   # Time Step is 1-indexed (t+1), delay=0
        return {_scen: (_trans[_trans['Time Step'] <= _cutoff], _occ[_occ['Time Step'] <= _cutoff])
                for _scen, (_trans, _occ) in shock_results.items()}


    SHOCK_RESULTS_NEAR_TERM = _truncate_near_term(SHOCK_RESULTS)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: outcome metrics — declined jobs, EE transitions, wage growth
# (near-term window -- see _truncate_near_term above)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 6: Outcome summary ===")


def _shock_declined_jobs(occ_df):
    if 'Failed_Hires' not in occ_df.columns:
        return np.nan, np.nan, np.nan, np.nan
    econ = occ_df.groupby('sim_id')['Failed_Hires'].sum()
    entry = (occ_df[occ_df['Occupation'].isin(ENTRY_IDS)]
             .groupby('sim_id')['Failed_Hires'].sum()
             .reindex(econ.index, fill_value=0))
    n = len(econ)
    econ_se  = econ.std(ddof=1)  / np.sqrt(n) if n > 1 else np.nan
    entry_se = entry.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return econ.mean(), econ_se, entry.mean(), entry_se


def _shock_ee_transitions(trans_df):
    ee = trans_df[trans_df['TransitionType'] == 'EE']
    all_sims = trans_df['sim_id'].unique()
    counts = ee.groupby('sim_id').size().reindex(all_sims, fill_value=0).to_numpy(dtype=float)
    n = len(counts)
    se = counts.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return counts.mean(), se


def _shock_wage_growth(occ_df):
    econ = (occ_df.groupby(['Time Step', 'sim_id'])[['Total_Wages', 'Employment']]
                  .sum().reset_index())
    econ['wage'] = econ['Total_Wages'] / econ['Employment']
    growths = []
    for _, g in econ.groupby('sim_id'):
        g = g.sort_values('Time Step')
        w = g['wage'].dropna()
        if len(w) < 2:
            continue
        n_years = (g['Time Step'].max() - g['Time Step'].min()) / 12.0
        growths.append(((w.iloc[-1] / w.iloc[0]) ** (1.0 / max(n_years, 1e-6)) - 1) * 100)
    growths = np.array(growths)
    n = len(growths)
    se = growths.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return growths.mean(), se


if RUN_REFERENCE_MODEL:
    _summary_rows = []
    for _scen in (BASE_SCEN, TREAT_SCEN):
        _trans_df, _occ_df = SHOCK_RESULTS_NEAR_TERM[_scen]
        _econ_fh, _econ_fh_se, _entry_fh, _entry_fh_se = _shock_declined_jobs(_occ_df)
        _ee_mean, _ee_se = _shock_ee_transitions(_trans_df)
        _wg_mean, _wg_se = _shock_wage_growth(_occ_df)
        _summary_rows.append({
            'scenario': _scen,
            'declined_jobs_economy_mean': _econ_fh, 'declined_jobs_economy_se': _econ_fh_se,
            'declined_jobs_entry_level_mean': _entry_fh, 'declined_jobs_entry_level_se': _entry_fh_se,
            'ee_transitions_mean': _ee_mean, 'ee_transitions_se': _ee_se,
            'annualised_wage_growth_pct': _wg_mean, 'annualised_wage_growth_pct_se': _wg_se,
        })
    shock_summary = pd.DataFrame(_summary_rows)
    print("\n" + shock_summary.to_string(index=False))
    shock_summary.to_csv(os.path.join(SOURCE_OUTPUT_PATH, "shock_outcomes_summary.csv"), index=False)
    print("Saved: shock_outcomes_summary.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: figure — AI-exposure shock implementation (standalone)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 7: Building shock-implementation figure ===")

fig_shock, ax_shock = plt.subplots(figsize=(8, 5))
_q75 = np.quantile(EXPOSURE_VEC, 0.75)
_hi, _lo = EXPOSURE_VEC >= _q75, EXPOSURE_VEC < _q75
ax_shock.plot(_t, SHOCK_OCC_ARRAY[_hi, 24:].mean(axis=0), color='#e05c5c', linewidth=LW,
              label=f'Top-quartile exposure (n={int(_hi.sum())})')
ax_shock.plot(_t, SHOCK_OCC_ARRAY[_lo, 24:].mean(axis=0), color=NO_POLICY_COLOR, linewidth=LW,
              label=f'Rest (n={int(_lo.sum())})')
ax_shock.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
ax_shock.set_xlabel("Months since shock onset", fontsize=10)
ax_shock.set_ylabel("Target-demand multiplier", fontsize=10)
ax_shock.set_title("Imposed AI-exposure shock (S-curve)", fontsize=12, fontweight='bold')
# Both curves start flat near 1.0 and only diverge downward as the horizon
# progresses, so the low-x/low-y corner is the one region neither line ever
# passes through -- anchor the legend there instead of letting 'best' guess
# (which kept placing it on top of the curves' early, still-close-to-1.0 segment).
ax_shock.legend(fontsize=9, loc='lower left'); ax_shock.grid(alpha=0.3)
# The ramp itself only takes RAMP_HORIZON_MONTHS -- after that both lines are
# flat plateaus for the remaining ~15 years of SHOCK_HORIZON_MONTHS. Show the
# ramp plus a bit of the plateau, not 20 years of flat line.
ax_shock.set_xlim(0, RAMP_HORIZON_MONTHS + 12)
plt.tight_layout()
_shock_fname = "exposure_shock_implementation.png"
save_plot_standard(fig_shock, os.path.join(SOURCE_OUTPUT_PATH, _shock_fname))
save_plot_standard(fig_shock, os.path.join(SOURCE_OUTPUT_PATH, _shock_fname.replace('.png', '.pdf')))
plt.show()
print(f"Saved: {_shock_fname} (PNG + PDF)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: figure — effect of entry-level preferential hiring under the shock
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 8: Building figure ===")


def _uer_ts(trans_df, occ_df, occ_ids):
    e = occ_df[occ_df['Occupation'].isin(occ_ids)]
    agg = e.groupby(['Time Step', 'sim_id'])[['Unemployment', 'Workers']].sum().reset_index()
    agg['uer'] = agg['Unemployment'] / agg['Workers'].clip(lower=1)
    return agg.groupby('Time Step')['uer'].agg(['mean', 'sem']).reset_index()


def _failed_hire_rate_ts(trans_df, occ_df, occ_ids):
    """Share of hiring attempts (vacancies with >=1 applicant this period)
    that failed to result in a hire -- Failed_Hires / (Hires + Failed_Hires).
    A vacancy "fails" when its sampled/available applicant(s) were already
    hired by a competing vacancy earlier in that period's shuffle; see the
    Failed_Hires instrumentation in model_fun.py.
    """
    e = occ_df[occ_df['Occupation'].isin(occ_ids)]
    agg = e.groupby(['Time Step', 'sim_id'])[['Hires', 'Failed_Hires']].sum().reset_index()
    agg['rate'] = agg['Failed_Hires'] / (agg['Hires'] + agg['Failed_Hires']).clip(lower=1)
    return agg.groupby('Time Step')['rate'].agg(['mean', 'sem']).reset_index()


def _wage_ts(trans_df, occ_df, occ_ids, bin_width=3):
    ue = trans_df[(trans_df['TransitionType'] == 'UE') &
                  (trans_df['DestinationOccupation'].isin(occ_ids))].copy()
    ue = ue.merge(OCC_WAGE_INFO, left_on='DestinationOccupation', right_on='Occupation', how='left')
    ue['wr']  = ue['AcceptedWage'] / ue['occ_median_wage']
    ue['bin'] = (ue['Time Step'] // bin_width) * bin_width
    per_sim = ue.groupby(['bin', 'sim_id'])['wr'].mean().reset_index()
    return (per_sim.groupby('bin')['wr'].agg(['mean', 'sem']).reset_index()
                    .rename(columns={'bin': 'Time Step'}))


def _economy_wage_ratio_ts(trans_df, occ_df, occ_ids):
    """Wage LEVEL (relative to occ. median) across every currently-employed
    worker in `occ_ids`, as opposed to _wage_ts's re-employment wage: _wage_ts
    only reflects the wage of workers who were JUST hired via UE that period
    (a small, selected subset, and silent whenever no one in occ_ids was
    hired that period). Concretely, for each occupation this is that
    occupation's Total_Wages/occ_median_wage (an "effective headcount at
    median wage"), summed across occupations in `occ_ids` and divided by
    their combined Employment -- equivalently, the EMPLOYMENT-WEIGHTED
    AVERAGE, across those occupations, of (current average wage / that
    occupation's own calibrated median wage). A value of 1.0 means the
    workforce is earning exactly its calibrated reference level on average;
    above/below 1 means aggregate pay is running above/below that baseline.
    `occ_ids` sets the scope -- callers pass ALL_OCC_IDS for a genuine
    economy-wide check (Section 12's Panel C) or ENTRY_IDS to match the
    other entry-level-only panels; this function itself has no opinion on
    which.
    """
    e = occ_df[occ_df['Occupation'].isin(occ_ids)].merge(OCC_WAGE_INFO, on='Occupation', how='left')
    e['wage_bill_ratio'] = e['Total_Wages'] / e['occ_median_wage']
    agg = e.groupby(['Time Step', 'sim_id'])[['wage_bill_ratio', 'Employment']].sum().reset_index()
    agg['wr'] = agg['wage_bill_ratio'] / agg['Employment'].clip(lower=1)
    return agg.groupby('Time Step')['wr'].agg(['mean', 'sem']).reset_index()


def _economy_wage_level_ts(trans_df, occ_df, occ_ids):
    """Raw dollar economy-wide average wage level across every currently-
    employed worker in `occ_ids`: sum(Total_Wages) / sum(Employment), in
    dollars -- no normalisation by occupation median wage. Complements
    _economy_wage_ratio_ts: that ratio isolates WITHIN-occupation pay changes
    from compositional shifts (more/fewer people working in occupations that
    simply pay differently); this raw level bundles both together, which is
    arguably the more direct answer to "did the policy make the workforce
    worse off in pay terms" -- a compositional shift toward lower-paying
    occupations is itself a real adverse wage effect, not noise to strip out.
    """
    e = occ_df[occ_df['Occupation'].isin(occ_ids)]
    agg = e.groupby(['Time Step', 'sim_id'])[['Total_Wages', 'Employment']].sum().reset_index()
    agg['wage_level'] = agg['Total_Wages'] / agg['Employment'].clip(lower=1)
    return agg.groupby('Time Step')['wage_level'].agg(['mean', 'sem']).reset_index()


def _transition_rate_ts(occ_df, col):
    """Economy-wide transition rate (UE_Transitions or EE_Transitions as a
    share of total Workers) -- no entry-level/rest-of-economy split, unlike
    the other panels: this compares transition *type* (UE vs EE) instead.
    """
    agg = occ_df.groupby(['Time Step', 'sim_id'])[[col, 'Workers']].sum().reset_index()
    agg['rate'] = agg[col] / agg['Workers'].clip(lower=1)
    return agg.groupby('Time Step')['rate'].agg(['mean', 'sem']).reset_index()


_SCENARIOS = [(BASE_SCEN, NO_POLICY_COLOR, BASE_LABEL),
              (TREAT_SCEN, POLICY_COLOR, TREAT_LABEL)]

if RUN_REFERENCE_MODEL:
    fig, ((ax_A, ax_B), (ax_C, ax_D)) = plt.subplots(2, 2, figsize=(TEXTWIDTH, ROW_HEIGHT * 2))


    def _plot_entry_vs_rest(ax, ts_fn, val_fn, err_fn, ylabel, title, marker=None):
        """Entry-level (solid) vs rest-of-economy (dotted), one pair of lines per
        scenario, colour = scenario, linestyle = occupation group. No per-panel
        legend -- A/B/C share one legend below the figure (see add_shared_legend
        call below), since all three plot the exact same four series."""
        for _scen, _color, _ in _SCENARIOS:
            _trans_df, _occ_df = SHOCK_RESULTS_NEAR_TERM[_scen]
            for _ids, _ls in [(ENTRY_IDS, '-'), (NON_ENTRY_IDS, ':')]:
                _ts = ts_fn(_trans_df, _occ_df, _ids)
                _v, _e = val_fn(_ts), err_fn(_ts)
                ax.plot(_ts['Time Step'], _v, color=_color, linewidth=LW, linestyle=_ls, marker=marker, markersize=3)
                ax.fill_between(_ts['Time Step'], _v - 1.96 * _e, _v + 1.96 * _e, alpha=0.12, color=_color)
        ax.set_xlabel("Months since shock onset", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)


    _plot_entry_vs_rest(ax_A, _uer_ts, lambda ts: ts['mean'] * 100, lambda ts: ts['sem'] * 100,
                         "UER (%)", "A.  Unemployment rate")
    _plot_entry_vs_rest(ax_B, _wage_ts, lambda ts: ts['mean'], lambda ts: ts['sem'],
                         "First-job wage / occ. median", "B.  Re-employment wage", marker='o')
    ax_B.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.7)
    _plot_entry_vs_rest(ax_C, _failed_hire_rate_ts, lambda ts: ts['mean'] * 100, lambda ts: ts['sem'] * 100,
                         "Failed-hire rate (%)", "C.  Failed hires")

    # Panel D: economy-wide UE & EE transition rates -- own inline legend, since
    # its linestyle dimension (transition type) means something different from
    # A/B/C's (occupation group), and reusing solid/dotted for both meanings in
    # one shared legend would be ambiguous.
    for _scen, _color, _lbl in _SCENARIOS:
        _, _occ_df = SHOCK_RESULTS_NEAR_TERM[_scen]
        for _col, _ls, _type_lbl in [('UE_Transitions', '-', 'UE'), ('EE_Transitions', ':', 'EE')]:
            _ts = _transition_rate_ts(_occ_df, _col)
            _v, _e = _ts['mean'] * 100, _ts['sem'] * 100
            ax_D.plot(_ts['Time Step'], _v, color=_color, linewidth=LW, linestyle=_ls,
                      label=f'{_lbl} — {_type_lbl}')
            ax_D.fill_between(_ts['Time Step'], _v - 1.96 * _e, _v + 1.96 * _e, alpha=0.12, color=_color)
    ax_D.set_xlabel("Months since shock onset", fontsize=10)
    ax_D.set_ylabel("Transition rate (%)", fontsize=10)
    ax_D.set_title("D.  UE & EE transitions", fontsize=11, fontweight='bold')
    ax_D.legend(fontsize=8); ax_D.grid(alpha=0.3)

    plt.tight_layout()

    # One shared legend for A/B/C (identical 4 series across all three): colour =
    # scenario (solid entries), linestyle = occupation group (dotted extras),
    # plus the wage panel's median-wage reference line. Panel D keeps its own
    # inline legend (see above) since its linestyle means something different.
    _legend_colors = {f'{_lbl} — entry-level': _color for _, _color, _lbl in _SCENARIOS}
    _legend_extras = [Line2D([0], [0], color=_color, linewidth=LW, linestyle=':', label=f'{_lbl} — rest of economy')
                       for _, _color, _lbl in _SCENARIOS]
    _legend_extras.append(Line2D([0], [0], color='black', linewidth=1, linestyle='--', label='Occupation median wage'))
    add_shared_legend(fig, _legend_colors, extra_handles=_legend_extras, ncol=2)
    # No caption text on the figure itself -- captions live in exposure_shock_writeup.tex.

    _fname = "fig_exposure_shock_entry_policy.png"
    save_plot_standard(fig, os.path.join(SOURCE_OUTPUT_PATH, _fname), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    save_plot_standard(fig, os.path.join(SOURCE_OUTPUT_PATH, _fname.replace('.png', '.pdf')), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    plt.show()
    print(f"Saved: {_fname} (PNG + PDF)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: cross-model comparison — setup
#
# Mirrors entry_policy_exercise.py's Section 5 cross-model machinery: the same
# 3 non-cyclical model variants it actually compares (its _CM_SPECS_NO_CYC —
# cyclical variants are excluded there too), each with its own calibrated
# gamma_u/d_u/theta.
#
# Each variant gets its OWN warm-start below, not the single-model REFERENCE_
# MODEL's: 24 years of different search/hiring mechanics produce different
# equilibrium wage/unemployment states by 2024, so reusing one model's end-
# state under a different model's rules would be internally inconsistent.
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 9: Cross-model setup ===")

from model_colors import MODEL_COLORS_BY_ID

CM_SPECS = {
    'otj_nonbehav':   {'otj': True, 'cyc_otj': False, 'cyc_ue': False, 'disc': False, 'wage_prefs': False, 'strict_rw': True},
    'otj_disc':       {'otj': True, 'cyc_otj': False, 'cyc_ue': False, 'disc': True,  'wage_prefs': True,  'strict_rw': False},
    'otj_disc_no_rw': {'otj': True, 'cyc_otj': False, 'cyc_ue': False, 'disc': True,  'wage_prefs': False, 'strict_rw': True},
}
CM_DISPLAY = {
    'otj_nonbehav':   'Non-behav.\nw. OTJ',
    'otj_disc':       'Behav. w.o.\nCyc. OTJ\nw. RW',
    'otj_disc_no_rw': 'Behav. w.o.\nCyc. OTJ\nw.o RW',
}

if ONLY_MODEL is not None:
    assert ONLY_MODEL in CM_SPECS, f"ONLY_MODEL must be one of {list(CM_SPECS)}"
    CM_SPECS = {ONLY_MODEL: CM_SPECS[ONLY_MODEL]}
    print(f"  ONLY_MODEL = {ONLY_MODEL}: reference model and other variants skipped.")

_calib_base_csv = os.path.join(
    PROJECT_ROOT,
    "output/calib_final/full_omn/exog_vac/calibrated_params_all_exog_wage_update_True.csv"
)
_cm_param_df = pd.concat(
    [pd.read_csv(_calib_base_csv), pd.read_csv(CALIB_CSV)],
    ignore_index=True
).sort_values('Timestamp', ascending=False)


def _build_cm_params(model_cat):
    """REF_BASE_PARAMS with model_cat's own calibrated gamma_u/d_u/theta and
    behavioral flags substituted in -- same logic as entry_policy_exercise.py's
    _build_model_params. None of CM_SPECS' 3 variants are cyclical, so theta
    stays None for all of them, but the lookup is kept general in case a
    cyclical variant is ever added to CM_SPECS.
    """
    _pf = _cm_param_df
    _has_theta = 'cyclical_e' in model_cat
    _calib = {
        'gamma_u': float(_pf.loc[(_pf['model_cat'] == model_cat) & (_pf['Parameter'] == 'gamma_u'), 'Value'].iloc[0]),
        'd_u':     float(_pf.loc[(_pf['model_cat'] == model_cat) & (_pf['Parameter'] == 'd_u'),     'Value'].iloc[0]),
        'theta':   (float(_pf.loc[(_pf['model_cat'] == model_cat) & (_pf['Parameter'] == 'theta'), 'Value'].iloc[0])
                    if _has_theta else None),
    }
    params = deepcopy(REF_BASE_PARAMS)
    params.update(_calib)
    params.update(CM_SPECS[model_cat])
    return params


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: cross-model warm-start (+ topup)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 10: Cross-model warm-start ===")

CM_WARMSTART_STATES = {}
for _mc in CM_SPECS:
    _cm_bp = _build_cm_params(_mc)
    _cm_ws_pkl = os.path.join(OUTPUT_PATH, f"warmstart_states_{_mc}.pkl")
    _cm_states = _load_or_build(
        _cm_ws_pkl, FORCE_CM_WARMSTART_RERUN,
        lambda bp=_cm_bp, mc=_mc: _run_warmstart_batch(bp, N_WARMSTART_SIMS, mc),
        label=f"{_mc} warm-start",
    )
    if N_WARMSTART_TOPUP_SIMS > 0:
        _cm_ws_topup_pkl = os.path.join(OUTPUT_PATH, f"warmstart_states_{_mc}_topup.pkl")
        _cm_ws_topup = _load_or_build(
            _cm_ws_topup_pkl, FORCE_CM_WARMSTART_TOPUP_RERUN,
            lambda bp=_cm_bp, mc=_mc: _run_warmstart_batch(bp, N_WARMSTART_TOPUP_SIMS, f"{mc} topup"),
            label=f"{_mc} warm-start topup",
        )
        _cm_states = _cm_states + _cm_ws_topup
    CM_WARMSTART_STATES[_mc] = _cm_states
    print(f"  [{_mc}] {len(_cm_states)} total warm-start states ready.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: cross-model shock-phase runs (+ topup)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 11: Cross-model shock-phase runs ===")


def _cm_shock_params(mc):
    _bp = _build_cm_params(mc)
    _bp.update({
        'time_steps':      int(SHOCK_OCC_ARRAY.shape[1]),
        'occ_shocks_data': SHOCK_OCC_ARRAY,
        'vac_data':        SHOCK_VAC_DATA,
        'delay':           SHOCK_DELAY,
    })
    return _bp


def _cm_shock_pkls(mc):
    return (os.path.join(SOURCE_OUTPUT_PATH, f"shock_results_{mc}.pkl"),
            os.path.join(SOURCE_OUTPUT_PATH, f"shock_results_{mc}_topup.pkl"))


def _build_cm_shock_caches(mc):
    """The base and top-up warm-start batches for one model (shock mode: both scenarios; policy mode:
    the policy arm only), each pickled to its own cache. Run in a forked worker (one per model) by
    the block below."""
    _bp = _cm_shock_params(mc)
    _batches = [(0, CM_WARMSTART_STATES[mc][:N_WARMSTART_SIMS], mc),
                (N_WARMSTART_SIMS, CM_WARMSTART_STATES[mc][N_WARMSTART_SIMS:], f"{mc} topup")]
    for _pkl, (_offset, _states, _label) in zip(_cm_shock_pkls(mc), _batches):
        if _states:
            _load_or_build(_pkl, FORCE_CM_SHOCK_RERUN,
                           lambda: (_run_shock_batch(_bp, SHOCK_SCENARIOS, _states, sim_id_offset=_offset, label=_label)
                                    if EXERCISE == "shock" else
                                    _run_policy_only(_bp, _states, sim_id_offset=_offset, label=_label)),
                           label=f"{_label} shock-phase")
    return mc


_todo = [mc for mc in CM_SPECS if FORCE_CM_SHOCK_RERUN
         or not all(os.path.exists(p) for p in _cm_shock_pkls(mc)[:1 + (N_WARMSTART_TOPUP_SIMS > 0)])]
if _todo:
    import multiprocessing as _mp
    with _mp.get_context('fork').Pool(min(N_CM_WORKERS, len(_todo))) as _pool:
        for _done in _pool.imap_unordered(_build_cm_shock_caches, _todo):
            print(f"  ✓ [{_done}] shock-phase caches built", flush=True)

CM_SHOCK_RESULTS = {}
CM_SHOCK_WAGE_AGE = {}
for _mc in CM_SPECS:
    _cm_shock_bp = _cm_shock_params(_mc)
    _cm_shock_pkl = os.path.join(SOURCE_OUTPUT_PATH, f"shock_results_{_mc}.pkl")
    _cm_base_states = CM_WARMSTART_STATES[_mc][:N_WARMSTART_SIMS]
    _cm_res_raw = _load_or_build(
        _cm_shock_pkl, FORCE_CM_SHOCK_RERUN,
        lambda bp=_cm_shock_bp, st=_cm_base_states, mc=_mc:
            _run_policy_only(bp, st, sim_id_offset=0, label=mc),   # not reached: the workers above built every cache
        label=f"{_mc} shock-phase",
    )
    _assert_shock_batch_format(_cm_res_raw, _cm_shock_pkl)
    _cm_res, _cm_wage_age = (_cm_res_raw if EXERCISE == "shock" else _with_pre_fix_no_policy(
        _cm_res_raw, os.path.join(PRE_FIX_SOURCE_OUTPUT_PATH, f"shock_results_{_mc}.pkl")))
    if N_WARMSTART_TOPUP_SIMS > 0:
        _cm_topup_states = CM_WARMSTART_STATES[_mc][N_WARMSTART_SIMS:]
        _cm_shock_topup_pkl = os.path.join(SOURCE_OUTPUT_PATH, f"shock_results_{_mc}_topup.pkl")
        _cm_res_topup_raw = _load_or_build(
            _cm_shock_topup_pkl, FORCE_CM_SHOCK_TOPUP_RERUN,
            lambda bp=_cm_shock_bp, st=_cm_topup_states, mc=_mc:
                _run_policy_only(bp, st, sim_id_offset=N_WARMSTART_SIMS, label=f"{mc} topup"),
            label=f"{_mc} shock-phase topup",
        )
        _assert_shock_batch_format(_cm_res_topup_raw, _cm_shock_topup_pkl)
        _cm_res_topup, _cm_wage_age_topup = (_cm_res_topup_raw if EXERCISE == "shock" else _with_pre_fix_no_policy(
            _cm_res_topup_raw, os.path.join(PRE_FIX_SOURCE_OUTPUT_PATH, f"shock_results_{_mc}_topup.pkl")))
        _cm_res = _concat_shock_results(_cm_res, _cm_res_topup)
        _cm_wage_age = _concat_wage_age_results(_cm_wage_age, _cm_wage_age_topup)
    CM_SHOCK_RESULTS[_mc] = _cm_res
    CM_SHOCK_WAGE_AGE[_mc] = _cm_wage_age   # used by Section 14's cross-model lifetime-earnings figure
    print(f"  [{_mc}] {_cm_res[BASE_SCEN][1]['sim_id'].nunique()} total sims per scenario after topup.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: cross-model comparison figure (full SHOCK_HORIZON_MONTHS window
# -- unlike Figure~fig:main (Section 8), which stays truncated to the near-
# term window via _truncate_near_term/SHOCK_RESULTS_NEAR_TERM, this figure
# uses the full, untruncated CM_SHOCK_RESULTS: all of it was already
# simulated and cached for the lifetime-earnings comparison (Sections 13-14),
# so showing the full horizon here costs nothing extra and lets the reader
# see the whole trajectory, not just the first 5 years.
#
# Time series of the policy effect (Δ = policy - no_policy) over the shock
# horizon, one line per model variant -- shows *when* the effect emerges,
# which a single bar (mean over the horizon) collapses away. Kept to 3 lines
# per panel (one per model) by collapsing the policy dimension into the paired
# Δ and dropping the entry/rest split (entry-level occupations only, matching
# the primary policy question) -- otherwise 3 models x entry/rest x
# policy/no_policy would need 12 lines per panel. Paired by warm-started sim
# (same starting state underlies both scenarios for a given sim), which
# cancels warm-start variance rather than compounding it.
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 12: Cross-model comparison figure ===")


def _cm_delta_ts(mc, ts_fn, occ_ids=ENTRY_IDS):
    """Time series of paired Δ (policy - no_policy): mean/SE across sims at
    each Time Step, rather than collapsed to one number over the horizon."""
    _res = CM_SHOCK_RESULTS[mc]
    _per_sim = []
    _sim_ids = sorted(_res[BASE_SCEN][1]['sim_id'].unique())
    for _sim in _sim_ids:
        _vals = {}
        for _scen in (BASE_SCEN, TREAT_SCEN):
            _trans_df, _occ_df = _res[_scen]
            _trans_s = _trans_df[_trans_df['sim_id'] == _sim]
            _occ_s = _occ_df[_occ_df['sim_id'] == _sim]
            _vals[_scen] = ts_fn(_trans_s, _occ_s, occ_ids).set_index('Time Step')['mean']
        _per_sim.append(_vals[TREAT_SCEN] - _vals[BASE_SCEN])
    _wide = pd.concat(_per_sim, axis=1)
    _n = _wide.shape[1]
    _sem = (_wide.std(axis=1, ddof=1) / np.sqrt(_n)) if _n > 1 else pd.Series(np.nan, index=_wide.index)
    return pd.DataFrame({'Time Step': _wide.index, 'mean': _wide.mean(axis=1), 'sem': _sem}).reset_index(drop=True)


ALL_OCC_IDS = set(range(N_OCC))   # every occupation, entry-level or not -- used for Panel C below so it's
                                    # a genuine economy-wide check, unlike Panels A/B which deliberately
                                    # stay restricted to entry-level occupations (the policy's direct
                                    # target). A policy that only ever helps/hurts entry-level workers
                                    # would show nothing in Panels A/B if it also has spillover effects
                                    # (crowding out, competition for vacancies) on the rest of the
                                    # economy -- Panel C is the one place that could reveal that.


def _cm_line(ax, ts_fn, val_fn, err_fn, ylabel, title, occ_ids=ENTRY_IDS):
    for _mc in CM_SPECS:
        _color = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
        _ts = _cm_delta_ts(_mc, ts_fn, occ_ids=occ_ids)
        _v, _e = val_fn(_ts), err_fn(_ts)
        _years = _ts['Time Step'] / 12   # this figure now spans 20 years -- months would be a dense,
                                          # hard-to-read axis, so convert to years for display only
                                          # (the underlying Δ-per-sim computation stays in months).
        ax.plot(_years, _v, color=_color, linewidth=LW_CM)
        ax.fill_between(_years, _v - 1.96 * _e, _v + 1.96 * _e, alpha=0.15, color=_color)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Years since shock onset", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)


if ONLY_MODEL is None:
    fig_cm, ((ax_cm_A, ax_cm_B), (ax_cm_C, ax_cm_D)) = plt.subplots(2, 2, figsize=(TEXTWIDTH, ROW_HEIGHT * 2))

    _cm_line(ax_cm_A, _uer_ts, lambda ts: ts['mean'] * 100, lambda ts: ts['sem'] * 100,
             "Δ UER (pp)", "A.  Unemployment rate")
    _cm_line(ax_cm_B, _failed_hire_rate_ts, lambda ts: ts['mean'] * 100, lambda ts: ts['sem'] * 100,
             "Δ Failed-hire rate (pp)", "B.  Failed hires")
    _cm_line(ax_cm_C, _economy_wage_ratio_ts, lambda ts: ts['mean'], lambda ts: ts['sem'],
             "Δ Economy-wide wage ratio", "C.  Economy-wide wage (ratio)", occ_ids=ALL_OCC_IDS)
    _cm_line(ax_cm_D, _economy_wage_level_ts, lambda ts: ts['mean'], lambda ts: ts['sem'],
             "Δ Economy-wide wage ($)", "D.  Economy-wide wage (level)", occ_ids=ALL_OCC_IDS)

    plt.tight_layout()

    _cm_legend_colors = {CM_DISPLAY[_mc].replace('\n', ' '): MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa') for _mc in CM_SPECS}
    add_shared_legend(fig_cm, _cm_legend_colors, ncol=3)
    # No caption text on the figure itself -- captions live in exposure_shock_writeup.tex.

    _cm_fname = "fig_exposure_shock_cross_model.png"
    save_plot_standard(fig_cm, os.path.join(SOURCE_OUTPUT_PATH, _cm_fname), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    save_plot_standard(fig_cm, os.path.join(SOURCE_OUTPUT_PATH, _cm_fname.replace('.png', '.pdf')), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    plt.show()
    print(f"Saved: {_cm_fname} (PNG + PDF)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13: lifetime earnings of new entrants — policy vs no-policy
#
# Cohort: entrants arriving in the first COHORT_WINDOW_MONTHS months of the
# shock, so every cohort member has at least MAX_FOLLOWUP_YEARS of follow-up
# within the horizon -- otherwise late entrants would mechanically show near-
# zero cumulative earnings just from lack of follow-up time, not from the
# policy. Tracked via wage_age_df's ~annual per-worker snapshots, linked to
# each worker's Entry_Time_Step (set once, at creation, in abm_funs.py's
# entry() -- see model_fun.py's occ.entry(...) calls).
#
# A year absent from a tracked worker's snapshot (they were unemployed at
# that annual checkpoint) counts as $0 earned that year, not excluded --
# excluding it would mean dropping exactly the workers the policy is most
# likely to affect (those struggling to find/keep work).
#
# TWO earnings metrics are tracked in parallel:
#   - 'ratio':   Wage / the worker's ENTRY occupation's median annual wage,
#     fixed ONCE at the worker's first observed (employed) snapshot and held
#     constant for the rest of their follow-up. An earlier version instead
#     re-looked-up occ_median_wage for whichever occupation the worker
#     currently held each year -- but ~90% of cohort workers visit more than
#     one occupation over the follow-up window, and the occupations they
#     move between differ in median wage by a median ratio of ~2x, so that
#     version silently re-normalised the ratio back toward ~1.0 on every
#     occupation change, masking real earnings differences instead of
#     revealing them (job-to-job pay growth/decline was invisible). Anchoring
#     to the entry occupation fixes this while still controlling for
#     cross-occupation pay-level differences when averaging across a cohort
#     that spans many occupations.
#   - 'dollars': raw annual Wage, no normalisation at all -- the most literal
#     reading of "lifetime earnings," at the cost of cohort-level averages
#     being somewhat sensitive to which (differently-paid) occupations
#     entrants happen to land in, independent of the policy.
#
# IMPORTANT CAVEAT: this is paired at the warm-started-SIM level, not at the
# individual-entrant level. Entrant Worker_IDs are only lazily assigned the
# first time a worker is observed employed at an annual snapshot (see
# model_fun.py's wage_age_df block), in an order driven by each run's own
# hiring history. Since entry_hire_bias changes how many random draws hire()
# consumes each period, the shared RNG stream -- and therefore which specific
# occupation each period's (deterministic-in-total) entrant count actually
# lands in -- diverges between policy/no-policy runs, even though the total
# count of entrants per period is identical (verified earlier: Entry_Level_
# Hires is bit-identical across scenarios). So this compares the same cohort
# *definition*, drawn from the same starting economies (same warm-started sim
# underlies both scenarios), not literally the same individuals.
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 13: Lifetime earnings of new entrants ===")

COHORT_WINDOW_MONTHS = 60   # entrants arriving in months [1, COHORT_WINDOW_MONTHS] -- matches the
                             # full ramp-in period (RAMP_HORIZON_MONTHS), so the shock is imposed
                             # during the same window the cohort is drawn from. MAX_FOLLOWUP_YEARS
                             # (below) is set by the LAST cohort member to arrive (month 60): they
                             # have the least simulated time left before SHOCK_HORIZON_MONTHS ends, so
                             # every cohort member is uniformly tracked for that same (shorter) number
                             # of years -- an earlier entrant (month 1) has ~5 more years of data
                             # available but it's truncated to match, otherwise later years of the
                             # figure would silently be drawn from a shrinking, earlier-biased sample.
# Annual snapshots are taken at Time Steps 1, 13, 25, ... while model_fun's loop runs t = 0..H-2, so the
# last snapshot is at Time Step 12*floor((H-2)/12) + 1 (= 229 for a 240-month horizon), not at month 240.
# The last entrant (month COHORT_WINDOW_MONTHS) therefore has floor((229 - 60)/12) = 14 full years of
# follow-up; assuming 15 left entrants after month 49 without a year-15 snapshot (filled as $0).
LAST_SNAPSHOT_TIME_STEP = 12 * ((SHOCK_HORIZON_MONTHS + 1 - 2) // 12) + 1
MAX_FOLLOWUP_YEARS = (LAST_SNAPSHOT_TIME_STEP - COHORT_WINDOW_MONTHS) // 12   # 14


def _entrant_cumulative_earnings_by_sim(wage_age_df, occ_wage_info, entry_occ_ids=None):
    """Per-sim cumulative earnings across the cohort, by years since entry,
    for the COHORT_WINDOW_MONTHS entry cohort, under both the 'ratio' (entry-
    occupation-anchored) and 'dollars' (raw) metrics described in the
    Section 13 header. Computes BOTH the per-sim MEDIAN and MEAN across the
    cohort (columns 'cum_ratio_median'/'cum_ratio_mean'/'cum_dollars_median'/
    'cum_dollars_mean') so callers can compare the two side by side (see
    Section 14's 2x2 figure) -- median is the primary/default statistic (see
    the comment below the cumsum for why), mean is kept as an explicit
    robustness comparison. See the Section 13 header for the sim-level (not
    individual-level) pairing caveat. entry_occ_ids: if given, only entrants
    whose entry occupation (first observed employed occupation, as below) is
    in this set -- note the shock itself shifts where entrants first find work,
    so groups defined this way are selected differently across scenarios.
    """
    df = wage_age_df.dropna(subset=['Entry_Time_Step']).copy()
    df = df[df['Entry_Time_Step'] <= COHORT_WINDOW_MONTHS]
    # floor, not round: annual snapshots fall at Time Steps 1, 13, 25, ..., so for an entrant arriving in
    # month E the snapshot offsets are k - (E-1)/12. Rounding sent offsets of exactly .5 (E = 7, 19, 31, ...)
    # to the nearest EVEN integer (two snapshots in even years, none in odd years -- filled as $0 below) and
    # dropped year 0 for anyone whose first snapshot came more than half a year after entry. With floor, the
    # snapshot taken during the k-th year after entry is year k: exactly one per worker per year.
    df['years_since_entry'] = np.floor((df['Time Step'] - df['Entry_Time_Step']) / 12).astype(int)
    df = df[(df['years_since_entry'] >= 0) & (df['years_since_entry'] <= MAX_FOLLOWUP_YEARS)]
    df = df.merge(occ_wage_info, on='Occupation', how='left')
    df = df.sort_values(['sim_id', 'Worker_ID', 'years_since_entry'])

    # Entry occupation's median wage: occ_median_wage at each worker's FIRST
    # observed (employed) snapshot, fixed for their whole trajectory.
    _entry_occ_wage = (df.groupby(['sim_id', 'Worker_ID'])['occ_median_wage']
                          .first().rename('entry_occ_median_wage'))
    df = df.merge(_entry_occ_wage, on=['sim_id', 'Worker_ID'], how='left')
    if entry_occ_ids is not None:
        _entry_occ = df.groupby(['sim_id', 'Worker_ID'])['Occupation'].transform('first')
        df = df[_entry_occ.isin(entry_occ_ids)]
    df['ratio']   = df['Wage'] / df['entry_occ_median_wage']
    df['dollars'] = df['Wage']

    # A worker shouldn't appear twice in the same (sim, year) -- annual
    # snapshots are disjoint by construction -- but average defensively.
    df = (df.groupby(['sim_id', 'Worker_ID', 'years_since_entry'])[['ratio', 'dollars']]
            .mean().reset_index())

    # Full (sim, worker) x year grid so a missing year fills as $0 earned
    # rather than silently vanishing from the cumulative sum.
    cohort_workers = df[['sim_id', 'Worker_ID']].drop_duplicates()
    full_years = pd.DataFrame({'years_since_entry': range(0, MAX_FOLLOWUP_YEARS + 1)})
    full = cohort_workers.merge(full_years, how='cross').merge(
        df, on=['sim_id', 'Worker_ID', 'years_since_entry'], how='left')
    full[['ratio', 'dollars']] = full[['ratio', 'dollars']].fillna(0.0)
    full = full.sort_values(['sim_id', 'Worker_ID', 'years_since_entry'])
    full['cum_ratio']   = full.groupby(['sim_id', 'Worker_ID'])['ratio'].cumsum()
    full['cum_dollars'] = full.groupby(['sim_id', 'Worker_ID'])['dollars'].cumsum()

    # MEDIAN across the cohort, per sim, per year, is the primary statistic --
    # deliberately median, not mean: a cohort can include a small number of
    # workers with unusually large cumulative earnings (e.g. long unbroken
    # employment spells in a high-paying occupation) who would otherwise pull
    # the per-sim summary away from what a "typical" cohort member
    # experienced. MEAN is also computed, purely as a robustness comparison
    # (Section 14's bottom row). Either way, this per-sim summary is what
    # gets paired across scenarios (same sim = same starting economy); the
    # across-sim step (see _lifetime_earnings_delta_ts /
    # _cm_lifetime_earnings_delta_ts) still uses the MEAN (with SEM) across
    # the ten per-sim summaries regardless -- a median of only 10 values
    # would be a much noisier, less standard summary than a mean with
    # confidence intervals.
    _agg = (full.groupby(['sim_id', 'years_since_entry'])[['cum_ratio', 'cum_dollars']]
                .agg(['median', 'mean']))
    _agg.columns = [f'{_col}_{_stat}' for _col, _stat in _agg.columns]
    return _agg.reset_index()


def _lifetime_earnings_ts(scen, metric, stat='median'):
    """Mean (across sims) of each sim's cohort `stat` ('median' or 'mean'),
    with SEM across sims."""
    _col = f'{metric}_{stat}'
    per_sim = _entrant_cumulative_earnings_by_sim(SHOCK_WAGE_AGE[scen], OCC_WAGE_INFO)
    return (per_sim.groupby('years_since_entry')[_col]
                    .agg(['mean', 'sem']).reset_index()
                    .rename(columns={'mean': f'mean_of_{stat}s'}))


def _lifetime_earnings_delta_ts(metric, stat='median'):
    """Paired Δ (policy - no_policy) time series by years_since_entry: mean/SEM
    across sims at each year, matching the cross-model Δ pattern in Section 12
    (_cm_delta_ts) -- same warm-started sim underlies both scenarios, so
    differencing per sim first cancels warm-start variance rather than
    compounding it. `stat` selects whether the per-sim cohort summary being
    differenced is the 'median' (default) or 'mean' -- see
    _entrant_cumulative_earnings_by_sim."""
    _col = f'{metric}_{stat}'
    _per_sim = {_scen: _entrant_cumulative_earnings_by_sim(SHOCK_WAGE_AGE[_scen], OCC_WAGE_INFO)
                       .set_index(['sim_id', 'years_since_entry'])[_col]
                for _scen in (BASE_SCEN, TREAT_SCEN)}
    _delta = (_per_sim[TREAT_SCEN] - _per_sim[BASE_SCEN]).reset_index(name=_col)
    _wide = _delta.pivot(index='years_since_entry', columns='sim_id', values=_col)
    _n = _wide.shape[1]
    _sem = (_wide.std(axis=1, ddof=1) / np.sqrt(_n)) if _n > 1 else pd.Series(np.nan, index=_wide.index)
    return pd.DataFrame({'years_since_entry': _wide.index, 'mean': _wide.mean(axis=1).values,
                          'sem': _sem.values})


LE_DISPLAY_MIN_YEAR = 1   # year 0 is dropped from all lifetime-earnings figures: at that first annual
                          # snapshot (0-6 months post-entry) roughly half the cohort hasn't found its
                          # first job yet, so the cross-sectional distribution is zero-inflated right at
                          # the ~50th percentile -- the MEDIAN is discontinuous there, and which side of
                          # that near-50/50 split a given simulation happens to land on (a coin-flip-
                          # level difference) swings it between ~0 and a real positive value. By year 1
                          # (~97%+ of the cohort employed at least once) this instability is gone. The
                          # underlying cumsum still starts at year 0 internally -- only the DISPLAYED
                          # range is truncated, so year-1-onward values are unaffected.

_le_tick_step = max(1, round(MAX_FOLLOWUP_YEARS / 10))   # ~10 ticks regardless of how long the horizon is

if RUN_REFERENCE_MODEL:
    fig_le, (ax_le_ratio, ax_le_dollars) = plt.subplots(1, 2, figsize=(TEXTWIDTH, ROW_HEIGHT))
    for _ax, _metric, _ylabel, _title in [
        (ax_le_ratio,   'cum_ratio',
         "Δ Cumulative earnings ratio", "A.  Relative to entry occupation"),
        (ax_le_dollars, 'cum_dollars',
         "Δ Cumulative earnings ($)", "B.  Raw dollar earnings"),
    ]:
        _ts = _lifetime_earnings_delta_ts(_metric)
        _ts = _ts[_ts['years_since_entry'] >= LE_DISPLAY_MIN_YEAR]
        _v, _e = _ts['mean'], _ts['sem']
        _ax.plot(_ts['years_since_entry'], _v, color=POLICY_COLOR, linewidth=LW, marker='o', markersize=4)
        _ax.fill_between(_ts['years_since_entry'], _v - 1.96 * _e, _v + 1.96 * _e, alpha=0.15, color=POLICY_COLOR)
        _ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        _ax.set_xlabel("Years since labour-market entry", fontsize=10)
        _ax.set_ylabel(_ylabel, fontsize=10)
        _ax.set_title(_title, fontsize=11, fontweight='bold')
        _ax.set_xticks(range(LE_DISPLAY_MIN_YEAR, MAX_FOLLOWUP_YEARS + 1, _le_tick_step))
        _ax.grid(alpha=0.3)
    plt.tight_layout()
    # No caption text on the figure itself -- captions live in exposure_shock_writeup.tex.

    _le_fname = "fig_exposure_shock_lifetime_earnings.png"
    save_plot_standard(fig_le, os.path.join(SOURCE_OUTPUT_PATH, _le_fname), aspect_ratio=ROW_HEIGHT / TEXTWIDTH)
    save_plot_standard(fig_le, os.path.join(SOURCE_OUTPUT_PATH, _le_fname.replace('.png', '.pdf')), aspect_ratio=ROW_HEIGHT / TEXTWIDTH)
    plt.show()
    print(f"Saved: {_le_fname} (PNG + PDF)")

    # Console readout at the final tracked year: scenario levels plus the paired Δ.
    print(f"\n  Lifetime-earnings summary at year {MAX_FOLLOWUP_YEARS} (mean ± SEM across {N_TOTAL_WARMSTART_SIMS} sims):")
    for _metric, _mlabel in [('cum_ratio', 'Cumulative earnings / entry occ. median wage'),
                              ('cum_dollars', 'Cumulative earnings ($)')]:
        print(f"    {_mlabel}:")
        for _scen, _, _lbl in _SCENARIOS:
            _row = _lifetime_earnings_ts(_scen, _metric).iloc[-1]
            print(f"      {_lbl:10s}: {_row['mean_of_medians']:.4f} ± {_row['sem']:.4f}")
        _drow = _lifetime_earnings_delta_ts(_metric).iloc[-1]
        print(f"      {'Δ (policy - no policy)':10s}: {_drow['mean']:.4f} ± {_drow['sem']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14: cross-model lifetime-earnings comparison figure
#
# Same idea as Section 13's Δ(policy - no_policy) lifetime-earnings figure,
# but repeated for each of the 3 cross-model variants (CM_SPECS) and drawn as
# one line per model, styled like Figure 3 (Section 12's fig:crossmodel) --
# same color mapping (MODEL_COLORS_BY_ID), same shared legend, same paired-Δ-
# per-sim logic. Uses CM_SHOCK_WAGE_AGE (captured in Section 11 alongside
# CM_SHOCK_RESULTS) rather than SHOCK_WAGE_AGE, which is reference-model only.
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 14: Cross-model lifetime-earnings figure ===")


def _cm_lifetime_earnings_delta_ts(mc, metric, stat='median', entry_occ_ids=None):
    """Same pattern as _lifetime_earnings_delta_ts (Section 13), for
    cross-model variant `mc`. `stat` selects 'median' (default) or 'mean';
    entry_occ_ids restricts the cohort by entry occupation."""
    _col = f'{metric}_{stat}'
    _per_sim = {_scen: _entrant_cumulative_earnings_by_sim(CM_SHOCK_WAGE_AGE[mc][_scen], OCC_WAGE_INFO,
                                                            entry_occ_ids=entry_occ_ids)
                       .set_index(['sim_id', 'years_since_entry'])[_col]
                for _scen in (BASE_SCEN, TREAT_SCEN)}
    _delta = (_per_sim[TREAT_SCEN] - _per_sim[BASE_SCEN]).reset_index(name=_col)
    _wide = _delta.pivot(index='years_since_entry', columns='sim_id', values=_col)
    _n = _wide.shape[1]
    _sem = (_wide.std(axis=1, ddof=1) / np.sqrt(_n)) if _n > 1 else pd.Series(np.nan, index=_wide.index)
    return pd.DataFrame({'years_since_entry': _wide.index, 'mean': _wide.mean(axis=1).values,
                          'sem': _sem.values})


if ONLY_MODEL is None:
    # 2x2 layout: top row uses the per-sim cohort MEDIAN (the primary statistic
    # used everywhere else, e.g. Figure~fig:lifetime), bottom row uses the per-sim
    # cohort MEAN -- shown side by side purely as a robustness comparison, per
    # Ebba's request, so a reader can see how sensitive the cross-model result is
    # to that choice.
    fig_cm_le, ((ax_cm_le_A, ax_cm_le_B), (ax_cm_le_C, ax_cm_le_D)) = plt.subplots(
        2, 2, figsize=(TEXTWIDTH, ROW_HEIGHT * 2))
    for _ax, _metric, _stat, _ylabel, _title in [
        (ax_cm_le_A, 'cum_ratio',   'median', "Δ Cumulative earnings ratio",
         "A.  Relative to entry occupation (median)"),
        (ax_cm_le_B, 'cum_dollars', 'median', "Δ Cumulative earnings ($)",
         "B.  Raw dollar earnings (median)"),
        (ax_cm_le_C, 'cum_ratio',   'mean',   "Δ Cumulative earnings ratio",
         "C.  Relative to entry occupation (mean)"),
        (ax_cm_le_D, 'cum_dollars', 'mean',   "Δ Cumulative earnings ($)",
         "D.  Raw dollar earnings (mean)"),
    ]:
        for _mc in CM_SPECS:
            _color = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
            _ts = _cm_lifetime_earnings_delta_ts(_mc, _metric, stat=_stat)
            _ts = _ts[_ts['years_since_entry'] >= LE_DISPLAY_MIN_YEAR]   # see Section 13's note on year 0
            _v, _e = _ts['mean'], _ts['sem']
            _ax.plot(_ts['years_since_entry'], _v, color=_color, linewidth=LW_CM)
            _ax.fill_between(_ts['years_since_entry'], _v - 1.96 * _e, _v + 1.96 * _e, alpha=0.15, color=_color)
        _ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        _ax.set_xlabel("Years since labour-market entry", fontsize=10)
        _ax.set_ylabel(_ylabel, fontsize=10)
        _ax.set_title(_title, fontsize=11, fontweight='bold')
        _ax.set_xticks(range(LE_DISPLAY_MIN_YEAR, MAX_FOLLOWUP_YEARS + 1, _le_tick_step))
        _ax.grid(alpha=0.3)
    plt.tight_layout()

    _cm_le_legend_colors = {CM_DISPLAY[_mc].replace('\n', ' '): MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa') for _mc in CM_SPECS}
    add_shared_legend(fig_cm_le, _cm_le_legend_colors, ncol=3)
    # No caption text on the figure itself -- captions live in exposure_shock_writeup.tex.

    _cm_le_fname = "fig_exposure_shock_cross_model_lifetime_earnings.png"
    save_plot_standard(fig_cm_le, os.path.join(SOURCE_OUTPUT_PATH, _cm_le_fname), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    save_plot_standard(fig_cm_le, os.path.join(SOURCE_OUTPUT_PATH, _cm_le_fname.replace('.png', '.pdf')), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    plt.show()
    print(f"Saved: {_cm_le_fname} (PNG + PDF)")

    print(f"\n  Cross-model lifetime-earnings Δ at year {MAX_FOLLOWUP_YEARS} (mean ± SEM across {N_TOTAL_WARMSTART_SIMS} sims):")
    for _metric, _mlabel in [('cum_ratio', 'Cumulative earnings / entry occ. median wage'),
                              ('cum_dollars', 'Cumulative earnings ($)')]:
        for _stat in ('median', 'mean'):
            print(f"    {_mlabel} ({_stat}):")
            for _mc in CM_SPECS:
                _row = _cm_lifetime_earnings_delta_ts(_mc, _metric, stat=_stat).iloc[-1]
                print(f"      {_mc:16s}: {_row['mean']:.4f} ± {_row['sem']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15: six-panel summary figures
#
# An alternative to the 3-model figures (Sections 12 and 14), which are kept
# unchanged: six outcomes for one model, otj_disc ("Behav. w.o. Cyc. OTJ w. RW",
# the orange line in those figures), in two versions, plus the same six panels
# for all cross-model variants (_cross_model_summary_delta, when ONLY_MODEL is None):
#   - _delta:  paired policy - no_policy difference, mean and 95% band across sims
#   - _levels: policy and no_policy each as its own line, mean and 95% band
# Panel A is entry-level occupations (the policy's target, solid) with all
# occupations (dashed) for comparison, B-D are economy-wide, E-F are the entry
# cohort's cumulative earnings (per-sim cohort
# median, as in Sections 13-14). Panel B is the economy-wide failed-hire rate.
# Uses the cached CM_SHOCK_RESULTS /
# CM_SHOCK_WAGE_AGE only -- no new simulations.
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 15: summary figures ===")

SUMMARY_MODEL = ONLY_MODEL or 'otj_disc'
SUMMARY_POLICY_COLOR = '#33A02C' if EXERCISE == "policy" else '#006D2C'   # green (policy) / dark green (shock)
_SUMMARY_SCENARIOS = [(BASE_SCEN, NO_POLICY_COLOR, BASE_LABEL), (TREAT_SCEN, SUMMARY_POLICY_COLOR, TREAT_LABEL)]


def _cm_level_ts(mc, ts_fn, scen, occ_ids=ENTRY_IDS):
    """Per-scenario level time series for model mc: mean/SEM across sims of each
    sim's own series (same per-sim construction as _cm_delta_ts, without differencing)."""
    _trans_df, _occ_df = CM_SHOCK_RESULTS[mc][scen]
    _per_sim = []
    for _sim in sorted(_occ_df['sim_id'].unique()):
        _per_sim.append(ts_fn(_trans_df[_trans_df['sim_id'] == _sim], _occ_df[_occ_df['sim_id'] == _sim],
                              occ_ids).set_index('Time Step')['mean'])
    _wide = pd.concat(_per_sim, axis=1)
    _n = _wide.shape[1]
    _sem = (_wide.std(axis=1, ddof=1) / np.sqrt(_n)) if _n > 1 else pd.Series(np.nan, index=_wide.index)
    return pd.DataFrame({'Time Step': _wide.index, 'mean': _wide.mean(axis=1), 'sem': _sem}).reset_index(drop=True)


def _cm_lifetime_earnings_level_ts(mc, metric, scen, stat='median'):
    """Per-scenario cumulative earnings for model mc: mean/SEM across sims of each
    sim's cohort `stat` (same per-sim summary as _cm_lifetime_earnings_delta_ts)."""
    _col = f'{metric}_{stat}'
    per_sim = _entrant_cumulative_earnings_by_sim(CM_SHOCK_WAGE_AGE[mc][scen], OCC_WAGE_INFO)
    return (per_sim.groupby('years_since_entry')[_col].agg(['mean', 'sem']).reset_index())


# (ts_fn, scale, [(occ_ids, linestyle), ...], ylabel stem, title); scale converts rates to pp.
# Panel A: entry-level occupations (solid) with all occupations (dashed) for comparison.
# Panel B is economy-wide: with 427 of 528 occupations entry-level, the entry-level rate is nearly identical.
_SUMMARY_TS_PANELS = [
    (_uer_ts,                100, [(ENTRY_IDS, '-'), (ALL_OCC_IDS, '--')], "UER (pp)",              "A.  UER"),
    (_failed_hire_rate_ts,   100, [(ALL_OCC_IDS, '-')],                    "Failed-hire rate (pp)", "B.  Failed hires (all)"),
    (_economy_wage_ratio_ts, 1,   [(ALL_OCC_IDS, '-')],                    "Wage ratio",            "C.  Wage ratio (all)"),
    (_economy_wage_level_ts, 1,   [(ALL_OCC_IDS, '-')],                    "Wage ($)",              "D.  Wage level (all)"),
]
_SUMMARY_LE_PANELS = [
    ('cum_ratio',   "Cumulative earnings ratio", "E.  Cum. earnings (ratio)"),
    ('cum_dollars', "Cumulative earnings ($)",   "F.  Cum. earnings ($)"),
]
_PANEL_A_HANDLES = [Line2D([0], [0], color='black', linewidth=LW, linestyle='-',
                           label='Panel A: entry-level occupations'),
                    Line2D([0], [0], color='black', linewidth=LW, linestyle='--',
                           label='Panel A: all occupations')]
if EXERCISE == "shock":
    # The shock is occupation-specific, so Panel A splits UER by exposure instead of by entry level.
    _SUMMARY_TS_PANELS[0] = (_uer_ts, 100, [(HIGH_EXPOSURE_IDS, '-'), (LOW_EXPOSURE_IDS, '--')],
                             "UER (pp)", "A.  UER by exposure")
    _PANEL_A_HANDLES = [Line2D([0], [0], color='black', linewidth=LW, linestyle='-',
                               label='Panel A: top-quartile exposure'),
                        Line2D([0], [0], color='black', linewidth=LW, linestyle='--',
                               label='Panel A: rest')]


SUMMARY_LW, SUMMARY_LW_MULTI, SUMMARY_MS = 0.9, 0.6, 2   # thinner lines/markers than the rest of the script


def _band(ax, x, v, e, color, lw=SUMMARY_LW, **kw):
    ax.plot(x, v, color=color, linewidth=lw, **kw)
    ax.fill_between(x, v - 1.96 * e, v + 1.96 * e, alpha=0.15, color=color)


def _summary_figure(version, models):
    """version: 'delta' (policy - no_policy) or 'levels' (both scenarios as lines; one model only).
    models: one model (green, as in the single-model figures) or several (one colour per model,
    MODEL_COLORS_BY_ID, as in the cross-model figures)."""
    multi = len(models) > 1
    assert not (multi and version == 'levels'), "levels version is single-model only"
    colors = {mc: (MODEL_COLORS_BY_ID.get(mc, '#aaaaaa') if multi else SUMMARY_POLICY_COLOR) for mc in models}
    lw = SUMMARY_LW_MULTI if multi else SUMMARY_LW
    ts_panels = list(_SUMMARY_TS_PANELS)
    if multi and EXERCISE == "policy":   # cross-model version: panel A shows economy-wide unemployment only
        ts_panels[0] = (_uer_ts, 100, [(ALL_OCC_IDS, '-')], "UER (pp)", "A.  UER (all)")
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, ROW_HEIGHT * 2))
    for ax, (ts_fn, scale, scopes, ylab, title) in zip(axes.flat[:4], ts_panels):
        for occ_ids, ls in scopes:
            if version == 'delta':
                for mc in models:
                    ts = _cm_delta_ts(mc, ts_fn, occ_ids=occ_ids)
                    _band(ax, ts['Time Step'] / 12, ts['mean'] * scale, ts['sem'] * scale, colors[mc],
                          lw=lw, linestyle=ls)
            else:
                for scen, color, _ in _SUMMARY_SCENARIOS:
                    ts = _cm_level_ts(models[0], ts_fn, scen, occ_ids=occ_ids)
                    _band(ax, ts['Time Step'] / 12, ts['mean'] * scale, ts['sem'] * scale, color, lw=lw, linestyle=ls)
        if version == 'delta':
            ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
            ax.set_ylabel(f"Δ {ylab}", fontsize=9)
        else:
            ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlabel("Years since shock onset", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
    for ax, (metric, ylab, title) in zip(axes.flat[4:], _SUMMARY_LE_PANELS):
        if version == 'delta':
            for mc in models:
                ts = _cm_lifetime_earnings_delta_ts(mc, metric)
                ts = ts[ts['years_since_entry'] >= LE_DISPLAY_MIN_YEAR]   # see Section 13's note on year 0
                _band(ax, ts['years_since_entry'], ts['mean'], ts['sem'], colors[mc], lw=lw,
                      marker='o', markersize=SUMMARY_MS)
            ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
            ax.set_ylabel(f"Δ {ylab}", fontsize=9)
        else:
            for scen, color, _ in _SUMMARY_SCENARIOS:
                ts = _cm_lifetime_earnings_level_ts(models[0], metric, scen)
                ts = ts[ts['years_since_entry'] >= LE_DISPLAY_MIN_YEAR]
                _band(ax, ts['years_since_entry'], ts['mean'], ts['sem'], color, lw=lw, marker='o', markersize=SUMMARY_MS)
            ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlabel("Years since entry", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks(range(LE_DISPLAY_MIN_YEAR, MAX_FOLLOWUP_YEARS + 1, _le_tick_step))
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if multi:
        _legend = {f"{CM_DISPLAY[mc].replace(chr(10), ' ')} ({DELTA_LABEL.lower()})": colors[mc] for mc in models}
        add_shared_legend(fig, _legend, ncol=2,
                          **({'extra_handles': _PANEL_A_HANDLES} if EXERCISE == "shock" else {}))
    elif version == 'delta':
        add_shared_legend(fig, {DELTA_LABEL: SUMMARY_POLICY_COLOR}, extra_handles=_PANEL_A_HANDLES, ncol=3)
    else:
        add_shared_legend(fig, {lbl: color for _, color, lbl in _SUMMARY_SCENARIOS},
                          extra_handles=_PANEL_A_HANDLES, ncol=2)
    # No caption text on the figure itself -- captions live in exposure_shock_writeup.tex.
    stem = "cross_model" if multi else models[0]
    fname = f"fig_exposure_shock_{stem}_summary_{version}.png"
    save_plot_standard(fig, os.path.join(SOURCE_OUTPUT_PATH, fname), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    save_plot_standard(fig, os.path.join(SOURCE_OUTPUT_PATH, fname.replace('.png', '.pdf')),
                       aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    plt.show()
    print(f"Saved: {fname} (PNG + PDF)")


EXPOSURE_REST_COLOR = '#888888'   # shock-mode delta figure: "rest" of the occupations, next to the top quartile
ENTRY_COHORT_COLOR  = '#B15928'   # shock-mode delta figure: all entrants (ColorBrewer Paired brown -- not a model colour)


def _shock_exposure_delta_figure(mc):
    """Shock mode, one model: shock - no_shock for UER, wage ratio and wage level, each split into
    top-quartile-exposure occupations (SUMMARY_POLICY_COLOR, solid) and the rest (grey, dashed),
    plus the whole entry cohort's cumulative dollar earnings. Four panels in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, ROW_HEIGHT * 2))
    ts_axes, ax_le = list(axes.flat[:3]), axes.flat[3]
    scopes = [(HIGH_EXPOSURE_IDS, '-', SUMMARY_POLICY_COLOR), (LOW_EXPOSURE_IDS, '--', EXPOSURE_REST_COLOR)]
    ts_panels = [
        (_uer_ts,                100, "UER (pp)",   "A.  UER"),
        (_economy_wage_ratio_ts, 1,   "Wage ratio", "B.  Wage ratio"),
        (_economy_wage_level_ts, 1,   "Wage ($)",   "C.  Wage level"),
    ]
    for ax, (ts_fn, scale, ylab, title) in zip(ts_axes, ts_panels):
        for occ_ids, ls, color in scopes:
            ts = _cm_delta_ts(mc, ts_fn, occ_ids=occ_ids)
            _band(ax, ts['Time Step'] / 12, ts['mean'] * scale, ts['sem'] * scale, color, linestyle=ls)
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.set_ylabel(f"Δ {ylab}", fontsize=9)
        ax.set_xlabel("Years since shock onset", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
    # D: the whole entry cohort, not split by exposure -- the only occupation recorded for an
    # entrant is their first job's, which the shock itself shifts (see entry_occ_ids). Dollars
    # only: the entry-occupation-anchored ratio follows the same path here.
    ts = _cm_lifetime_earnings_delta_ts(mc, 'cum_dollars')
    ts = ts[ts['years_since_entry'] >= LE_DISPLAY_MIN_YEAR]   # see Section 13's note on year 0
    _band(ax_le, ts['years_since_entry'], ts['mean'], ts['sem'], ENTRY_COHORT_COLOR,
          marker='o', markersize=SUMMARY_MS)
    ax_le.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax_le.set_ylabel("Δ Cumulative earnings ($)", fontsize=9)
    ax_le.set_xlabel("Years since entry", fontsize=9)
    ax_le.set_title("D.  Cumulative Earnings ($)", fontsize=10, fontweight='bold')
    ax_le.set_xticks(range(LE_DISPLAY_MIN_YEAR, MAX_FOLLOWUP_YEARS + 1, _le_tick_step))
    ax_le.grid(alpha=0.3)
    fig.suptitle("AI-Exposure Demand Shock: Shock − No Shock", fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    _handles = [Line2D([0], [0], color=SUMMARY_POLICY_COLOR, linewidth=LW, linestyle='-',
                       label='Top-quartile exposure (A-C)'),
                Line2D([0], [0], color=EXPOSURE_REST_COLOR, linewidth=LW, linestyle='--',
                       label='Rest of occupations (A-C)'),
                Line2D([0], [0], color=ENTRY_COHORT_COLOR, linewidth=LW, linestyle='-', marker='o',
                       markersize=SUMMARY_MS, label='All entrants (D)')]
    add_shared_legend(fig, {}, extra_handles=_handles, ncol=3)
    # No caption text on the figure itself -- captions live in exposure_shock_writeup.tex.
    fname = f"fig_exposure_shock_{mc}_summary_delta.png"
    save_plot_standard(fig, os.path.join(SOURCE_OUTPUT_PATH, fname), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    save_plot_standard(fig, os.path.join(SOURCE_OUTPUT_PATH, fname.replace('.png', '.pdf')),
                       aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
    plt.show()
    print(f"Saved: {fname} (PNG + PDF)")


for _mc in (CM_SPECS if EXERCISE == "shock" else [SUMMARY_MODEL]):
    if _mc in CM_SPECS:
        if EXERCISE == "shock":
            _shock_exposure_delta_figure(_mc)
            _summary_figure('levels', [_mc])
        else:
            for _version in ('delta', 'levels'):
                _summary_figure(_version, [_mc])
if len(CM_SPECS) > 1:
    # Same six panels for all cross-model variants (policy - no_policy only: levels would need
    # two lines per model per scope)
    _summary_figure('delta', list(CM_SPECS))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16: lifetime earnings validation against the CPS ASEC
#
# Mean annual and mean cumulative earnings of the entry cohort by years since
# entry (levels; full and from-year-5 versions) and their growth (index and
# year-on-year; base years 1 and 5), no-policy scenario, one line per model in
# CM_SPECS, against a CPS ASEC
# synthetic-cohort profile built in R (data/occ_macro_vars/lifetime_earnings_cps/,
# see its README): civilian labour force, zeros included, nominal dollars averaged
# over income years 2000-2023 -- the same dollar basis as the model's OEWS-based
# occupation wages. Means rather than medians, because a synthetic cohort's
# cumulative mean equals the sum of its annual means (comparable with the model's
# mean cumulative earnings); cumulative medians are not comparable.
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 16: Lifetime earnings validation (CPS ASEC) ===")

CPS_EARNINGS_CSV = os.path.join(PROJECT_ROOT, "..", "data", "occ_macro_vars", "lifetime_earnings_cps",
                                "cps_earnings_by_experience.csv")
VALIDATION_SCENARIO = BASE_SCEN
VALIDATION_LATE_START = 5   # second version: years 5+ only, cumulating from year 5 (leaves out the early-career
                            # gap, where entrants in the model start near full occupational pay)


def _model_annual_per_sim(wage_age_df):
    """Per-sim cohort mean annual dollar earnings by years since entry (zeros included), and cumulative."""
    per_sim = (_entrant_cumulative_earnings_by_sim(wage_age_df, OCC_WAGE_INFO)
               [['sim_id', 'years_since_entry', 'cum_dollars_mean']]
               .sort_values(['sim_id', 'years_since_entry']))
    per_sim['annual'] = per_sim.groupby('sim_id')['cum_dollars_mean'].diff()
    per_sim['annual'] = per_sim['annual'].fillna(per_sim['cum_dollars_mean'])   # year 0: annual = cumulative
    return per_sim


def _model_earnings_profile(wage_age_df, start=VALIDATION_LATE_START):
    """Mean (and SEM across sims) of the cohort's mean annual and mean cumulative dollar earnings by years
    since entry, plus cumulative earnings counted from year `start` ('cum_late'; NaN before `start`).
    Per-sim cohort means come from _entrant_cumulative_earnings_by_sim (zeros included)."""
    per_sim = _model_annual_per_sim(wage_age_df)
    per_sim['cum_late'] = (per_sim['annual'].where(per_sim['years_since_entry'] >= start)
                           .groupby(per_sim['sim_id']).cumsum()
                           .where(per_sim['years_since_entry'] >= start))
    agg = per_sim.groupby('years_since_entry')[['annual', 'cum_dollars_mean', 'cum_late']].agg(['mean', 'sem'])
    agg.columns = [f'{c}_{st}' for c, st in agg.columns]
    return agg.reset_index()


if os.path.exists(CPS_EARNINGS_CSV):
    _cps = pd.read_csv(CPS_EARNINGS_CSV)
    _cps = _cps[_cps['years_since_entry'] <= MAX_FOLLOWUP_YEARS]
    _profiles = {mc: _model_earnings_profile(CM_SHOCK_WAGE_AGE[mc][VALIDATION_SCENARIO]) for mc in CM_SPECS}

    _cps['cum_late'] = (_cps['mean_annual_earnings'].where(_cps['years_since_entry'] >= VALIDATION_LATE_START)
                        .cumsum().where(_cps['years_since_entry'] >= VALIDATION_LATE_START))

    def _validation_figure(late):
        """late=False: years 0+, cumulating from entry. late=True: years VALIDATION_LATE_START+, cumulating
        from that year on both sides."""
        s0 = VALIDATION_LATE_START if late else 0
        cum_m, cum_c = ('cum_late', 'cum_late') if late else ('cum_dollars_mean', 'cum_mean_earnings')
        fig_val, (ax_val_A, ax_val_B) = plt.subplots(1, 2, figsize=(TEXTWIDTH, ROW_HEIGHT * 1.1))
        for mc, prof in _profiles.items():
            prof = prof[prof['years_since_entry'] >= s0]
            _c = MODEL_COLORS_BY_ID.get(mc, '#aaaaaa')
            _band(ax_val_A, prof['years_since_entry'], prof['annual_mean'] / 1000, prof['annual_sem'] / 1000, _c,
                  lw=SUMMARY_LW, marker='o', markersize=SUMMARY_MS)
            _band(ax_val_B, prof['years_since_entry'], prof[f'{cum_m}_mean'] / 1000, prof[f'{cum_m}_sem'] / 1000,
                  _c, lw=SUMMARY_LW, marker='o', markersize=SUMMARY_MS)
        _cp = _cps[_cps['years_since_entry'] >= s0]
        ax_val_A.plot(_cp['years_since_entry'], _cp['mean_annual_earnings'] / 1000, color='black', linestyle='--',
                      linewidth=SUMMARY_LW, marker='s', markersize=SUMMARY_MS)
        ax_val_B.plot(_cp['years_since_entry'], _cp[cum_c] / 1000, color='black', linestyle='--',
                      linewidth=SUMMARY_LW, marker='s', markersize=SUMMARY_MS)
        cum_title = f"B.  Cumulative earnings from year {s0}" if late else "B.  Cumulative earnings"
        for ax, ylab, title in [(ax_val_A, "Mean annual earnings ($000)", "A.  Annual earnings"),
                                (ax_val_B, "Mean cumulative earnings ($000)", cum_title)]:
            ax.set_xlabel("Years since entry", fontsize=9)
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks(range(s0, MAX_FOLLOWUP_YEARS + 1, _le_tick_step))
            ax.grid(alpha=0.3)
        plt.tight_layout()
        _val_legend = {f"{CM_DISPLAY[mc].replace(chr(10), ' ')} ({BASE_LABEL.lower()})": MODEL_COLORS_BY_ID.get(mc, '#aaaaaa')
                       for mc in CM_SPECS}
        add_shared_legend(fig_val, _val_legend, ncol=2,
                          extra_handles=[Line2D([0], [0], color='black', linestyle='--', marker='s',
                                                markersize=SUMMARY_MS, label='CPS ASEC, income years 2000-2023')])
        fname = f"fig_lifetime_earnings_validation{'_from_year' + str(s0) if late else ''}.png"
        save_plot_standard(fig_val, os.path.join(SOURCE_OUTPUT_PATH, fname), aspect_ratio=(ROW_HEIGHT * 1.1) / TEXTWIDTH)
        save_plot_standard(fig_val, os.path.join(SOURCE_OUTPUT_PATH, fname.replace('.png', '.pdf')),
                           aspect_ratio=(ROW_HEIGHT * 1.1) / TEXTWIDTH)
        plt.show()
        print(f"Saved: {fname} (PNG + PDF)")

    _validation_figure(late=False)
    _validation_figure(late=True)

    # Earnings growth: the model's level is too high early in careers (entrants start near full occupational
    # pay), so compare the SHAPE of the trajectory instead -- (A) earnings indexed to a base year and
    # (B) year-on-year growth. Model index/growth computed per sim, then mean and 95% band across sims.
    _per_sim_annual = {mc: _model_annual_per_sim(CM_SHOCK_WAGE_AGE[mc][VALIDATION_SCENARIO]) for mc in CM_SPECS}
    _cps_annual = _cps.set_index('years_since_entry')['mean_annual_earnings']
    _growth_rows = []

    def _growth_figure(base_year):
        fig_g, (ax_g_A, ax_g_B) = plt.subplots(1, 2, figsize=(TEXTWIDTH, ROW_HEIGHT * 1.1))
        years = range(base_year, MAX_FOLLOWUP_YEARS + 1)
        for mc, ps in _per_sim_annual.items():
            w = ps.pivot(index='years_since_entry', columns='sim_id', values='annual').loc[list(years)]
            idx = w / w.loc[base_year] * 100
            gr = w.pct_change() * 100
            _c = MODEL_COLORS_BY_ID.get(mc, '#aaaaaa')
            for ax, frame, skip in [(ax_g_A, idx, 0), (ax_g_B, gr, 1)]:
                f = frame.iloc[skip:]
                n = f.shape[1]
                _band(ax, f.index, f.mean(axis=1), f.std(axis=1, ddof=1) / np.sqrt(n), _c,
                      lw=SUMMARY_LW, marker='o', markersize=SUMMARY_MS)
            _growth_rows.append(pd.DataFrame({'model': mc, 'base_year': base_year, 'years_since_entry': list(years),
                                              'index_mean': idx.mean(axis=1).values,
                                              'growth_pct_mean': gr.mean(axis=1).values}))
        c = _cps_annual.loc[list(years)]
        c_idx, c_gr = c / c.loc[base_year] * 100, c.pct_change() * 100
        _growth_rows.append(pd.DataFrame({'model': 'CPS', 'base_year': base_year, 'years_since_entry': list(years),
                                          'index_mean': c_idx.values, 'growth_pct_mean': c_gr.values}))
        ax_g_A.plot(c_idx.index, c_idx.values, color='black', linestyle='--', linewidth=SUMMARY_LW,
                    marker='s', markersize=SUMMARY_MS)
        ax_g_B.plot(c_gr.index[1:], c_gr.values[1:], color='black', linestyle='--', linewidth=SUMMARY_LW,
                    marker='s', markersize=SUMMARY_MS)
        ax_g_A.axhline(100, color='black', linewidth=0.8, linestyle=':')
        ax_g_B.axhline(0, color='black', linewidth=0.8, linestyle=':')
        for ax, ylab, title in [(ax_g_A, f"Mean annual earnings (year {base_year} = 100)", "A.  Earnings index"),
                                (ax_g_B, "Year-on-year growth (%)", "B.  Earnings growth")]:
            ax.set_xlabel("Years since entry", fontsize=9)
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks(list(years)[::max(1, round(len(years) / 8))])
            ax.grid(alpha=0.3)
        plt.tight_layout()
        _leg = {f"{CM_DISPLAY[mc].replace(chr(10), ' ')} ({BASE_LABEL.lower()})": MODEL_COLORS_BY_ID.get(mc, '#aaaaaa')
                for mc in CM_SPECS}
        add_shared_legend(fig_g, _leg, ncol=2,
                          extra_handles=[Line2D([0], [0], color='black', linestyle='--', marker='s',
                                                markersize=SUMMARY_MS, label='CPS ASEC, income years 2000-2023')])
        fname = f"fig_lifetime_earnings_growth_base_year{base_year}.png"
        save_plot_standard(fig_g, os.path.join(SOURCE_OUTPUT_PATH, fname), aspect_ratio=(ROW_HEIGHT * 1.1) / TEXTWIDTH)
        save_plot_standard(fig_g, os.path.join(SOURCE_OUTPUT_PATH, fname.replace('.png', '.pdf')),
                           aspect_ratio=(ROW_HEIGHT * 1.1) / TEXTWIDTH)
        plt.show()
        print(f"Saved: {fname} (PNG + PDF)")

    for _base in (1, VALIDATION_LATE_START):
        _growth_figure(_base)
    pd.concat(_growth_rows).to_csv(os.path.join(SOURCE_OUTPUT_PATH, "lifetime_earnings_growth.csv"), index=False)
    print("Saved: lifetime_earnings_growth.csv")

    _val_rows = [prof.assign(model=mc) for mc, prof in _profiles.items()]
    _val_table = (pd.concat(_val_rows)
                  .merge(_cps[['years_since_entry', 'mean_annual_earnings', 'cum_mean_earnings', 'cum_late']]
                         .rename(columns={'mean_annual_earnings': 'cps_annual', 'cum_mean_earnings': 'cps_cumulative',
                                          'cum_late': f'cps_cumulative_from_year{VALIDATION_LATE_START}'}),
                         on='years_since_entry', how='left'))
    _val_table = _val_table.rename(columns={'cum_late_mean': f'cum_from_year{VALIDATION_LATE_START}_mean',
                                            'cum_late_sem': f'cum_from_year{VALIDATION_LATE_START}_sem'})
    _val_table.to_csv(os.path.join(SOURCE_OUTPUT_PATH, "lifetime_earnings_validation.csv"), index=False)
    print("Saved: lifetime_earnings_validation.csv")
    print(_val_table[_val_table['years_since_entry'].isin([0, 1, 5, 10, MAX_FOLLOWUP_YEARS])]
          [['model', 'years_since_entry', 'annual_mean', 'cps_annual', 'cum_dollars_mean_mean', 'cps_cumulative']]
          .round(0).to_string(index=False))
else:
    print(f"  CPS benchmark not found ({CPS_EARNINGS_CSV}); run "
          "data/occ_macro_vars/lifetime_earnings_cps/lifetime_earnings_cps.R first.")

print(f"\n✓ Done. Output written to {SOURCE_OUTPUT_PATH}")
