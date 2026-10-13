"""
opt_search_benchmark.py

Diagnostic runs for the optimizing-search benchmark (referee response).

Runs the reference model (otj_cyclical_e_disc, calibrated params held fixed) with
alternative rules for unemployed application effort:
  - "empirical":   the CPS schedule draw (current ABM)
  - "sticky":      marginal-benefit rule, sticky beliefs (eq. 10, h = 0), reservation wage
  - "bayes_month": marginal-benefit rule, Bayesian beliefs (one failure per month), reservation wage
  - "fixed":       non-behavioural benchmark: beliefs never erode, no reservation wage
  - "rational":    non-behavioural benchmark: full information on last period's competition,
                   no reservation wage
The two non-behavioural benchmarks run at the behavioural cost (kappa = 0.015) and level-matched
("_lm": kappa set so mean applications match the empirical scenario: offline guess, then
refined with single-run ABM probes; probe log saved to level_match_<scenario>.json).
See OPT_SEARCH_DEFAULTS / optimal_application_set in abm_funs.py and the
"Optimizing benchmark for the ABM" section of job_search_notebook.ipynb.

Outputs (figures A1-A8, B1 duration Wasserstein, B3 applications by duration, fit summary)
are regenerated after every finished scenario, so partial results are usable if the run stops.
Each scenario is cached under a hash of its overrides.

WRITES ONLY INSIDE OUTPUT_PATH (see _out below).

Run interactively in VS Code "Run in Interactive Window", or from calibration_remote/:
    ~/.venvs/transition_abm/bin/python opt_search_benchmark.py
"""

import os
import sys
import json
import time
import pickle
import random
import hashlib
import traceback
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# This script writes ONLY to OUTPUT_PATH below. Two shared modules have side-effect writes:
# collate_macro_vars re-saves its collated CSVs on import, and network_input_builder("full_omn")
# saves occupational_va_shocks.png to the Overleaf Dropbox folder. Both are disabled here.
_NO_WRITES = [mock.patch('pandas.DataFrame.to_csv'), mock.patch('matplotlib.pyplot.savefig'),
              mock.patch('matplotlib.figure.Figure.savefig')]
for _p in _NO_WRITES:
    _p.start()
from network_input_builder import network_input_builder
from model_fun import run_single_local
from collate_macro_vars import gdp_dat, macro_observations, tightness
from abm_funs import (duration_to_prob_dict, applications_sent, OPT_SEARCH_DEFAULTS,
                      optimal_application_set, rho_tilde_row)
from plot_style import TEXTWIDTH, ROW_HEIGHT, save_plot_standard, add_shared_legend
from plot_funs import (plot_uer_vac, plot_ltuer, plot_bev_curve_color, plot_cd_vs_td, plot_trans_rates,
                       plot_seeker_comp_line, plot_applications_and_composition, plot_market_tightness,
                       plot_gender_pay_gap_ts, plot_duration_wasserstein_rank_heatmap,
                       plot_duration_wasserstein_over_time, plot_duration_wasserstein_rank_heatmap_monthly,
                       plot_duration_wasserstein_over_time_monthly)

# ── Settings ──────────────────────────────────────────────────────────────────
NW          = "full_omn"
CALIB_DATE  = ["2001-01-01", "2024-05-01"]
CALIB_END   = "2019-05-01"   # in-sample window used for the fit summary
CALIB_V_LINE = ["2000-12-01", "2019-05-01"]
DELAY       = 25
N_SIMS      = int(os.environ.get('OPT_N_SIMS', 10))   # OPT_N_SIMS=1 for a quick smoke test
FORCE_RERUN = False
CONFIDENCE_LEVEL = 0.95
KAPPA_BEHAV = 0.015          # cost used by the behavioural rules (notebook round 3)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    f"output/calib_final/{NW}/exog_vac/figures/covid_oos/opt_search_benchmark/"
)
RUN_DIR = os.path.join(OUTPUT_PATH, f"overnight_N{N_SIMS}/")
# Outputs (figures, fit summary) can go to a subfolder of RUN_DIR, e.g. OPT_OUT_SUBDIR=core for a re-plot of a
# scenario subset from the cached runs; caches and level-match logs are always read from RUN_DIR
OUT_DIR = os.path.join(RUN_DIR, os.environ["OPT_OUT_SUBDIR"] + "/") if os.environ.get("OPT_OUT_SUBDIR") else RUN_DIR
FIG_DIR = os.path.join(OUT_DIR, "figures/")


def _out(*parts):
    """Path inside OUTPUT_PATH; refuses anything else."""
    p = os.path.abspath(os.path.join(*parts))
    assert p.startswith(os.path.abspath(OUTPUT_PATH)), f"refusing to write outside OUTPUT_PATH: {p}"
    return p


os.makedirs(_out(FIG_DIR), exist_ok=True)

# Scenarios in run order (most important first). "kappa": "level_match" is resolved after the
# empirical scenario has run (mean applications per seeker of the empirical scenario).
SCENARIOS = {
    'empirical': {},
    'sticky': {'u_search_rule': 'sticky',
               'opt_search_params': {'p_bar': 0.20, 'kappa': KAPPA_BEHAV,
                                     'beta0': 1.0, 'omega': 0.10, 'r': 0.10}},
    'bayes_month': {'u_search_rule': 'bayes_month',
                    'opt_search_params': {'p_bar': 0.30, 'kappa': KAPPA_BEHAV, 'N0': 2.0}},
    'nb_fixed_lm': {'u_search_rule': 'fixed',
                    'opt_search_params': {'p_bar': 0.20, 'beta0': 1.0, 'kappa': 'level_match',
                                          'use_res_wage': False}},
    'nb_rational_lm': {'u_search_rule': 'rational',
                       'opt_search_params': {'p_bar': 0.20, 'kappa': 'level_match',
                                             'use_res_wage': False}},
    'nb_fixed': {'u_search_rule': 'fixed',
                 'opt_search_params': {'p_bar': 0.20, 'beta0': 1.0, 'kappa': KAPPA_BEHAV,
                                       'use_res_wage': False}},
    'nb_rational': {'u_search_rule': 'rational',
                    'opt_search_params': {'p_bar': 0.20, 'kappa': KAPPA_BEHAV,
                                          'use_res_wage': False}},
    'sticky_r105': {'u_search_rule': 'sticky',
                    'opt_search_params': {'p_bar': 0.30, 'kappa': KAPPA_BEHAV,
                                          'beta0': 1.0, 'omega': 0.10, 'r': 0.105}},
}
SCENARIO_LABELS = {
    'empirical':      'CPS Schedule (Current ABM)',
    'sticky':         'Optimizing: Sticky Beliefs',
    'bayes_month':    'Optimizing: Bayesian Beliefs (per Month)',
    'nb_fixed_lm':    'Non-Behavioural Optimizing: Fixed Beliefs',
    'nb_rational_lm': 'Non-Behavioural Optimizing: Full Information',
    'nb_fixed':       'Non-Behavioural Optimizing: Fixed Beliefs (kappa = 0.015)',
    'nb_rational':    'Non-Behavioural Optimizing: Full Information (kappa = 0.015)',
    'sticky_r105':    'Optimizing: Sticky Beliefs (r = 0.105)',
}
SCENARIO_COLORS = {
    'empirical': '#4D4D4D', 'sticky': '#1F78B4', 'bayes_month': '#33A02C',
    'nb_fixed_lm': '#FF7F00', 'nb_rational_lm': '#6A3D9A',
    'nb_fixed': '#FDBF6F', 'nb_rational': '#CAB2D6', 'sticky_r105': '#A6CEE3',
}

# ── Reference model inputs (mirrors entry_policy_exercise.py) ────────────────
print("Building network...")
_mod_data, _net_temp, _vacs, _, _occ_shocks_dat = network_input_builder(NW, False, CALIB_DATE, delay=DELAY)
for _p in _NO_WRITES:
    _p.stop()   # writes re-enabled: everything below saves only into OUTPUT_PATH (via _out)
plt.close('all')

CALIB_CSV = os.path.join(
    PROJECT_ROOT,
    "output/calib_final/full_omn/exog_vac/"
    "calibrated_params_all_theta_signal_exog_wage_update_True.csv"
)
_param_df = pd.read_csv(CALIB_CSV)
_ref = (_param_df[_param_df['model_cat'] == 'otj_cyclical_e_disc']
        .sort_values('Timestamp', ascending=False))

_obs = macro_observations.loc[
    (macro_observations['DATE'] >= CALIB_DATE[0]) &
    (macro_observations['DATE'] <= CALIB_DATE[1])
].dropna(subset=['VACRATE'])
_vac_dat = _obs['VACRATE'].to_numpy()

REF_BASE_PARAMS = {
    'mod_data':           _mod_data,
    'net_temp':           _net_temp,
    'vacs':               _vacs,
    'time_steps':         int(_occ_shocks_dat.shape[1]),
    'occ_shocks_data':    _occ_shocks_dat,
    'app_effort_dat':     duration_to_prob_dict,
    'gdp_data':           gdp_dat,
    'bus_confidence_dat': gdp_dat,
    'vac_data':           _vac_dat,
    'delay':              DELAY,
    'simple_res':         False,
    'dumb_hire':          False,
    'mistake_rate':       0.1,
    'vac_method':         'exog',
    'vac_wage_update':    True,
    'steady_state':       False,
    # model spec: Behavioural w. Cyc. OTJ w. RW (otj_cyclical_e_disc)
    'otj':                True,
    'cyc_otj':            True,
    'cyc_ue':             False,
    'disc':               True,
    'wage_prefs':         True,
    'emp_apps':           1,
    'strict_rw':          False,
    'd_v':                None,
    'gamma_v':            None,
    # calibrated params, held fixed across scenarios
    'gamma_u': float(_ref.loc[_ref['Parameter'] == 'gamma_u', 'Value'].iloc[0]),
    'd_u':     float(_ref.loc[_ref['Parameter'] == 'd_u',     'Value'].iloc[0]),
    'theta':   float(_ref.loc[_ref['Parameter'] == 'theta',   'Value'].iloc[0]),
}
print(f"  Reference params: gamma_u={REF_BASE_PARAMS['gamma_u']:.4f}, "
      f"d_u={REF_BASE_PARAMS['d_u']:.5f}, theta={REF_BASE_PARAMS['theta']:.4f}")


# ── Run / load scenarios ──────────────────────────────────────────────────────
def _monthly_dates(n):
    # One model period per month from CALIB_DATE[0], aligned with vac_data (month-start dates)
    return pd.date_range(start=CALIB_DATE[0], periods=n, freq='MS')


def _cache_path(name, overrides):
    key = hashlib.md5(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:8]
    return _out(RUN_DIR, f"{name}_{key}_N{N_SIMS}.pkl")


def run_scenario(name, overrides, n_sims=N_SIMS):
    """n_sims runs of the reference model with overrides. Returns a dict of per-run lists /
    concatenated frames (each tagged with 'sim_id')."""
    grouped_runs, seekers_runs, stock_runs, abd_runs = [], [], [], []
    for i in range(n_sims):
        t0 = time.time()
        params = deepcopy(REF_BASE_PARAMS)
        params.update(overrides)
        (_, sim_grouped, _, data, seekers_rec, _, _, _, _, _, ue_stock_df, _) = run_single_local(**params)
        g = sim_grouped.copy()
        g['DATE'] = _monthly_dates(len(g))
        g['LTUER'] = g['LT Unemployed Persons'] / g['Unemployment']
        g['AVGWAGE'] = g['Total_Wages'] / g['Employment']
        g['Seeker Composition'] = g['Employed Seekers'] / (g['Employed Seekers'] + g['Unemployed Seekers'])
        grouped_runs.append(g)
        s = seekers_rec.copy()
        s['DATE'] = _monthly_dates(len(s))
        s['Application Effort (U)'] = s['Applications Sent (Unemployed)'] / s['Unemployed Seekers']
        s['Application Effort (E)'] = s['Applications Sent (Employed)'] / s['Employed Seekers']
        seekers_runs.append(s)
        st = ue_stock_df[['Time Step', 'UEDuration']].astype('int32')
        st['sim_id'] = np.int16(i)
        stock_runs.append(st)
        abd = data['apps_by_duration'].copy()
        abd['sim_id'] = i
        abd_runs.append(abd)
        print(f"    [{name}] {i + 1}/{n_sims} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return {'grouped': grouped_runs, 'seekers': seekers_runs,
            'ue_stock': pd.concat(stock_runs, ignore_index=True),
            'apps_by_duration': pd.concat(abd_runs, ignore_index=True)}


# ── Level matching for the non-behavioural benchmarks (before their ABM runs) ─
def _offline_searchers(n_agents=600, n_draws=3, seed=2026):
    """Unemployed workers from the initial network, each with n_draws monthly vacancy samples
    drawn exactly as search_and_apply does (30 draws, weighted by similarity, deduplicated)."""
    rng = random.Random(seed)
    vac_by_occ = {}
    for v in _vacs:
        vac_by_occ.setdefault(v.occupation_id, []).append(v)
    workers = [w for occ in _net_temp for w in occ.list_of_unemployed]
    out = []
    for w in rng.sample(workers, min(n_agents, len(workers))):
        occ = _net_temp[w.occupation_id]
        pool, wts = [], []
        for oid, wt in occ.neigh_nonzero:
            vs = vac_by_occ.get(oid, [])
            pool.extend(vs)
            wts.extend([wt] * len(vs))
        if not pool:
            continue
        rho = rho_tilde_row(occ.list_of_neigh_weights, occ.occupation_id)
        for _ in range(n_draws):
            draw = rng.choices(pool, weights=wts, k=min(30, len(pool)))
            seen = set()
            cands = [v for v in draw if not (id(v) in seen or seen.add(id(v)))]
            out.append((SimpleNamespace(wage=w.wage, time_unemployed=1, n_fail=0), cands, rho))
    return out


def level_match_kappa(rule, opt_params, target_apps, p_typical):
    """kappa such that mean A* over offline searchers equals target_apps (bisection in log kappa).
    For "rational", every occupation is given the typical pick probability p_typical."""
    searchers = _offline_searchers()
    fake_net = [SimpleNamespace(competition_last=p_typical) for _ in _net_temp]

    def mean_apps(kappa):
        prm = {**OPT_SEARCH_DEFAULTS, **opt_params, 'kappa': kappa}
        return np.mean([optimal_application_set(w, c, rho, None, rule, prm, net=fake_net)[1]
                        for w, c, rho in searchers])

    lo, hi = 1e-5, 1.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if mean_apps(mid) > target_apps:
            lo = mid
        else:
            hi = mid
    kappa = float(f"{np.sqrt(lo * hi):.4g}")
    print(f"  level-matched kappa for '{rule}': {kappa} (offline mean A* = {mean_apps(kappa):.2f}, "
          f"target {target_apps:.2f})")
    return kappa


def _mean_apps(res):
    abd = res['apps_by_duration']
    return abd['Applications Sent'].sum() / abd['Unemployed Seekers'].sum()


def abm_level_match_kappa(name, overrides, kappa0, target_apps, tol=0.03, max_probes=6):
    """Refine kappa inside the ABM (single-run probes) so mean applications per unemployed seeker
    match target_apps. The offline guess misses because vacancy stocks, wages and (for "rational")
    competition evolve in the model. Secant steps in log-log space; bracketing keeps them safe."""
    log = []
    k = kappa0
    lo, hi = None, None          # kappa giving too many / too few applications
    for i in range(max_probes):
        ov = deepcopy(overrides)
        ov['opt_search_params'] = {**ov['opt_search_params'], 'kappa': k}
        apps = _mean_apps(run_scenario(f"{name} probe {i + 1}", ov, n_sims=1))
        log.append((k, apps))
        print(f"  [{name}] probe {i + 1}: kappa = {k:.5g} -> {apps:.2f} applications (target {target_apps:.2f})",
              flush=True)
        if abs(apps / target_apps - 1) <= tol:
            break
        if apps > target_apps:
            lo = k
        else:
            hi = k
        if len(log) >= 2 and log[-1][1] != log[-2][1] and log[-1][0] != log[-2][0]:
            (k1, a1), (k2, a2) = log[-2], log[-1]
            slope = (np.log(a2) - np.log(a1)) / (np.log(k2) - np.log(k1))
            k_new = np.exp(np.log(k2) + (np.log(target_apps) - np.log(a2)) / slope) if slope < 0 else np.nan
        else:
            # first step: elasticity guess of -0.5 (offline A*(kappa) is roughly this steep)
            k_new = np.exp(np.log(k) + (np.log(target_apps) - np.log(apps)) / -0.5)
        if not np.isfinite(k_new) or (lo is not None and k_new <= lo) or (hi is not None and k_new >= hi):
            k_new = np.sqrt(lo * hi) if (lo is not None and hi is not None) else (k * 0.5 if hi is not None and lo is None else k * 2.0)
        k = float(k_new)
    best_k, best_apps = min(log, key=lambda x: abs(np.log(x[1] / target_apps)))
    best_k = float(f"{best_k:.4g}")
    print(f"  level-matched kappa for '{name}' (in ABM): {best_k} -> {best_apps:.2f} applications", flush=True)
    with open(_out(RUN_DIR, f"level_match_{name}.json"), 'w') as f:
        json.dump({'target_apps': target_apps, 'probes': log, 'kappa': best_k}, f, indent=1)
    return best_k


def _empirical_targets(res):
    abd = res['apps_by_duration']
    target = abd['Applications Sent'].sum() / abd['Unemployed Seekers'].sum()
    # Typical chance of being picked at a vacancy: vacancies per application, averaged over months
    vac = np.array([g['Vacancies'].to_numpy() for g in res['grouped']]).mean(axis=0)
    s = res['seekers']
    apps = np.array([(x['Applications Sent (Unemployed)'] + x['Applications Sent (Employed)']).to_numpy()
                     for x in s]).mean(axis=0)
    n = min(len(vac), len(apps))
    p_typ = float(np.clip(np.nanmean(vac[:n] / np.maximum(apps[:n], 1)), 0.01, 1.0))
    return target, p_typ


# ── Outputs (regenerated after every finished scenario) ──────────────────────
def aggregate_runs(run_list):
    """Mean / std / CI across runs (mirrors aggregate_simulation_results in auto_run_model_modified.py)."""
    from scipy.stats import norm
    n_min = min(len(d) for d in run_list)
    template = run_list[0].iloc[:n_min]
    cols = [c for c in template.select_dtypes(include=[np.number]).columns if c not in ('Time Step', 'sim_id')]
    stacked = np.stack([d[cols].iloc[:n_min].to_numpy(dtype=float) for d in run_list], axis=0)
    mean, std = np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)
    margin = norm.ppf((1 + CONFIDENCE_LEVEL) / 2) * std / np.sqrt(len(run_list))
    out = {'DATE': template['DATE'].to_numpy()}
    if 'Time Step' in template:
        out['Time Step'] = template['Time Step'].to_numpy()
    for k, c in enumerate(cols):
        out[c] = mean[:, k]
        out[f'{c}_std'] = std[:, k]
        out[f'{c}_ci_lower'] = mean[:, k] - margin[:, k]
        out[f'{c}_ci_upper'] = mean[:, k] + margin[:, k]
    return pd.DataFrame(out)


def _observed_inputs():
    obs = _obs.reset_index(drop=True).copy()
    obs['DATE'] = pd.to_datetime(obs['DATE'])
    rates = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/transition_rates_96_24.csv'))
    rates = rates[(rates['date'] >= CALIB_DATE[0]) & (rates['date'] <= CALIB_DATE[1])].copy()
    rates['DATE'] = pd.to_datetime(rates['date'])
    comp = pd.read_csv(os.path.join(PROJECT_ROOT, '../data/behav_params/Eeckhout_Replication/'
                                    'comp_searchers_s_series_abm_validation.csv'))
    qmap = {"Q1": "-01-01", "Q2": "-04-01", "Q3": "-07-01", "Q4": "-10-01"}
    comp['DATE'] = comp['date'].replace(qmap, regex=True).pipe(pd.to_datetime)
    comp = comp.set_index('DATE').resample('MS').asfreq()
    comp['comp_searchers_s'] = comp['comp_searchers_s'].interpolate(method='linear')
    comp = comp.reset_index().rename(columns={"comp_searchers_s": "Seeker Composition"})
    comp = comp[(comp['DATE'] >= CALIB_DATE[0]) & (comp['DATE'] <= CALIB_DATE[1])]
    return obs, rates, comp


def _try(label, fn):
    try:
        fn()
        print(f"    ✓ {label}")
    except Exception:
        print(f"    ✗ {label} failed:\n{traceback.format_exc()}")
    finally:
        plt.close('all')


def make_outputs(results):
    names = list(results.keys())
    colors = {SCENARIO_LABELS[n]: SCENARIO_COLORS[n] for n in names}
    res = {SCENARIO_LABELS[n]: aggregate_runs(results[n]['grouped']) for n in names}
    seek = {SCENARIO_LABELS[n]: aggregate_runs(results[n]['seekers']) for n in names}
    stock = {SCENARIO_LABELS[n]: results[n]['ue_stock'] for n in names}
    obs, rates, comp = _observed_inputs()
    fd = _out(FIG_DIR) + os.sep
    print(f"  Writing outputs for {len(names)} scenario(s) to {fd}")

    # Fit summary with run-to-run spread (RMSE per run vs observed, then mean and sd across runs)
    def _rmse_per_run(g, window_end=None):
        m = g[['DATE', 'UER', 'VACRATE', 'LTUER']].merge(obs[['DATE', 'UER', 'VACRATE', 'LTUER']],
                                                          on='DATE', suffixes=('', '_obs'))
        assert len(m) > 0.95 * len(g), "simulated and observed dates do not line up"
        if window_end is not None:
            m = m[m['DATE'] <= window_end]
        return {k: np.sqrt(np.nanmean((m[k] - m[k + '_obs']) ** 2)) for k in ['UER', 'VACRATE', 'LTUER']}

    rows = []
    for n in names:
        runs = results[n]['grouped']
        r_in = pd.DataFrame([_rmse_per_run(g, CALIB_END) for g in runs]).add_suffix('_rmse_in')
        r_all = pd.DataFrame([_rmse_per_run(g) for g in runs]).add_suffix('_rmse_all')
        abd = results[n]['apps_by_duration']
        rows.append({'scenario': n, 'n_runs': len(runs),
                     **{c: r_in[c].mean() for c in r_in}, **{c + '_sd': r_in[c].std() for c in r_in},
                     **{c: r_all[c].mean() for c in r_all},
                     'mean_UER': np.mean([g['UER'].mean() for g in runs]),
                     'mean_apps_per_u': abd['Applications Sent'].sum() / abd['Unemployed Seekers'].sum(),
                     'opt_search_params': json.dumps(SCENARIOS_RESOLVED[n].get('opt_search_params', {}))})
    summary = pd.DataFrame(rows).set_index('scenario')
    summary.to_csv(_out(OUT_DIR, "fit_summary.csv"))
    print(summary.drop(columns='opt_search_params').round(4).to_string())

    # A1-A8 (existing plot_funs figures)
    _try("A1 UER and vacancy rate", lambda: plot_uer_vac(res, obs, calib_date=CALIB_V_LINE, sep=False,
                                                         save=True, path=fd, plot_colors=colors))
    _try("A2 LTUER", lambda: plot_ltuer(res, obs, sep=False, save=True, path=fd, colors=colors,
                                        calib_date=CALIB_V_LINE))
    _try("A3 Beveridge curve", lambda: plot_bev_curve_color(res, obs, sep=False, smooth=3, save=True,
                                                            path=fd, colors=colors))
    _try("A4 Transition rates", lambda: plot_trans_rates(res, observation=rates, save=True, path=fd,
                                                         colors=colors, calib_v_line=CALIB_V_LINE))
    _try("A5 Seeker composition", lambda: plot_seeker_comp_line(res, comp, save=True, path=fd, colors=colors,
                                                                calib_v_line=CALIB_V_LINE))
    _try("A5 Applications and composition", lambda: plot_applications_and_composition(
        seek, res, comp, save=True, path=fd, colors=colors, calib_v_line=CALIB_V_LINE))
    _try("A6 Market tightness", lambda: plot_market_tightness(res, tightness, save=True, path=fd, colors=colors,
                                                              adjusted=False, calib_v_line=CALIB_V_LINE))
    _try("A6 Market tightness (adjusted)", lambda: plot_market_tightness(
        res, tightness, save=True, path=fd, colors=colors, adjusted=True, calib_v_line=CALIB_V_LINE))
    _try("A7 Current vs target demand", lambda: plot_cd_vs_td(res, save=True, path=fd, colors=colors,
                                                              calib_v_line=CALIB_V_LINE))
    _try("A8 Gender pay gap", lambda: plot_gender_pay_gap_ts(res, save=True, path=fd, colors=colors))

    # B1 unemployment-duration distribution vs CPS (no dropbox_path: nothing leaves FIG_DIR)
    survey = pd.read_csv(os.path.join(PROJECT_ROOT, "data/ltuer_cps_data_new.csv"))
    _try("B1 Duration Wasserstein ranking", lambda: plot_duration_wasserstein_rank_heatmap(
        ue_stock_dfs=stock, survey_df=survey, calib_date=CALIB_DATE, delay=DELAY, colors=colors,
        save=True, path=fd))
    _try("B1 Duration Wasserstein over time", lambda: plot_duration_wasserstein_over_time(
        ue_stock_dfs=stock, survey_df=survey, calib_date=CALIB_DATE, delay=DELAY, colors=colors,
        save=True, path=fd, calib_v_line=CALIB_V_LINE))
    monthly_path = os.path.join(PROJECT_ROOT, "data/ltuer_cps_data_monthly.csv")
    if os.path.exists(monthly_path):
        survey_m = pd.read_csv(monthly_path)
        _try("B1 Duration Wasserstein ranking (monthly)", lambda: plot_duration_wasserstein_rank_heatmap_monthly(
            ue_stock_dfs=stock, survey_df=survey_m, calib_date=CALIB_DATE, delay=DELAY, colors=colors,
            save=True, path=fd))
        _try("B1 Duration Wasserstein over time (monthly)", lambda: plot_duration_wasserstein_over_time_monthly(
            ue_stock_dfs=stock, survey_df=survey_m, calib_date=CALIB_DATE, delay=DELAY, colors=colors,
            save=True, path=fd, calib_v_line=CALIB_V_LINE))

    # B3 applications by unemployment duration vs CPS schedule (untargeted validation)
    def _b3():
        max_dur = 36
        cps = pd.Series({d: applications_sent(d, duration_to_prob_dict, expectation=True)
                         for d in range(1, max_dur + 1)})
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(TEXTWIDTH, ROW_HEIGHT * 2.2), sharex=True)
        rows_b3 = []
        for n in names:
            abd = results[n]['apps_by_duration']
            abd = abd[(abd['Duration'] >= 1) & (abd['Duration'] <= max_dur)]

            def _ratio(col):
                per = abd.assign(r=abd[col] / abd['Unemployed Seekers']).pivot_table(
                    index='Duration', columns='sim_id', values='r')
                return per.mean(axis=1), 1.96 * per.std(axis=1) / np.sqrt(per.shape[1])
            mean, half = _ratio('Applications Sent')
            ax.plot(mean.index, mean.values, color=SCENARIO_COLORS[n])
            ax.fill_between(mean.index, mean - half, mean + half, color=SCENARIO_COLORS[n], alpha=0.2, lw=0)
            row = pd.DataFrame({'scenario': n, 'Duration': mean.index, 'mean_apps': mean.values,
                                'ci_half_width': half.values})
            if 'Zero-Application Seekers' in abd:      # caches from before the diagnostic lack it
                z, _ = _ratio('Zero-Application Seekers')
                na, _ = _ratio('No-Acceptable-Vacancy Seekers')
                acc, _ = _ratio('Acceptable Vacancies')
                ax2.plot(z.index, z.values, color=SCENARIO_COLORS[n])
                ax2.plot(na.index, na.values, color=SCENARIO_COLORS[n], linestyle=':')
                row['share_zero_apps'] = z.values
                row['share_no_acceptable'] = na.values
                row['mean_acceptable_vacancies'] = acc.values
            rows_b3.append(row)
        ax.plot(cps.index, cps.values, color='black', linestyle='--')
        ax.set_ylabel("Applications per Seeker")
        ax.set_title("Applications by Unemployment Duration: Model vs CPS", fontweight='bold')
        ax2.set_ylabel("Share of Seekers")
        ax2.set_xlabel("Months Unemployed")
        ax2.set_title("Seekers Sending No Applications (Solid) and Finding No Acceptable Vacancy (Dotted)",
                      fontweight='bold', fontsize=9)
        plt.tight_layout()
        add_shared_legend(fig, colors, ncol=2,
                          extra_handles=[Line2D([0], [0], color='black', linestyle='--', label='CPS Schedule')])
        save_plot_standard(fig, _out(FIG_DIR, "applications_by_duration_vs_cps.png"),
                           aspect_ratio=ROW_HEIGHT * 2.2 / TEXTWIDTH)
        pd.concat(rows_b3).to_csv(_out(OUT_DIR, "applications_by_duration.csv"), index=False)
    _try("B3 Applications by duration", _b3)

    # Overview: UER, vacancy rate, applications per seeker on a common date axis
    def _overview():
        fig, axes = plt.subplots(3, 1, figsize=(TEXTWIDTH, ROW_HEIGHT * 3), sharex=True)
        for lbl, df in res.items():
            axes[0].plot(df['DATE'], df['UER'], color=colors[lbl])
            axes[1].plot(df['DATE'], df['VACRATE'], color=colors[lbl])
            sd = seek[lbl]
            axes[2].plot(sd['DATE'], sd['Application Effort (U)'], color=colors[lbl])
        axes[0].plot(obs['DATE'], obs['UER'], color='black', linestyle=':')
        axes[1].plot(obs['DATE'], obs['VACRATE'], color='black', linestyle=':')
        for ax, t, y in zip(axes, ["Unemployment Rate", "Vacancy Rate", "Applications per Unemployed Seeker"],
                            ["UER", "Vacancy Rate", "Applications"]):
            ax.set_title(t, fontweight='bold')
            ax.set_ylabel(y)
        plt.tight_layout()
        add_shared_legend(fig, colors, ncol=2,
                          extra_handles=[Line2D([0], [0], color='black', linestyle=':', label='Observed')])
        save_plot_standard(fig, _out(FIG_DIR, "opt_search_benchmark_fit.png"),
                           aspect_ratio=ROW_HEIGHT * 3 / TEXTWIDTH)
    _try("Overview figure", _overview)


# ── Main loop ─────────────────────────────────────────────────────────────────
results = {}
SCENARIOS_RESOLVED = {}
_only = [x for x in os.environ.get('OPT_SCENARIOS', '').split(',') if x]   # e.g. OPT_SCENARIOS=empirical,sticky
for name, overrides in SCENARIOS.items():
    if _only and name not in _only:
        continue
    overrides = deepcopy(overrides)
    prm = overrides.get('opt_search_params', {})
    if prm.get('kappa') == 'level_match':
        if 'empirical' not in results:
            print(f"Skipping '{name}': level matching needs the empirical scenario first")
            continue
        try:
            target, p_typ = _empirical_targets(results['empirical'])
            lm_json = _out(RUN_DIR, f"level_match_{name}.json")
            if os.path.exists(lm_json) and not FORCE_RERUN:
                with open(lm_json) as f:
                    prm['kappa'] = json.load(f)['kappa']
                print(f"  using cached level-matched kappa for '{name}': {prm['kappa']}")
            else:
                kappa0 = level_match_kappa(overrides['u_search_rule'], prm, target, p_typ)   # offline guess
                prm['kappa'] = abm_level_match_kappa(name, overrides, kappa0, target)
        except Exception:
            print(f"✗ Level matching for '{name}' failed:\n{traceback.format_exc()}")
            continue
    SCENARIOS_RESOLVED[name] = overrides
    pkl = _cache_path(name, overrides)
    try:
        if os.path.exists(pkl) and not FORCE_RERUN:
            print(f"Loading '{name}' from {pkl}")
            with open(pkl, 'rb') as f:
                results[name] = pickle.load(f)
        else:
            print(f"Running '{name}': {overrides}", flush=True)
            results[name] = run_scenario(name, overrides)
            with open(pkl, 'wb') as f:
                pickle.dump(results[name], f)
    except Exception:
        print(f"✗ Scenario '{name}' failed:\n{traceback.format_exc()}")
        continue
    make_outputs(results)

print(f"\nDone. Outputs in {RUN_DIR}")
