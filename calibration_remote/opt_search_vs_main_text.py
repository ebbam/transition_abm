"""
opt_search_vs_main_text.py

Compares the optimizing unemployed-search agent (opt_search_benchmark.py) with the main-text
behavioural model, Behavioural w. Cyc. OTJ w. RW (otj_cyclical_e_disc). No new simulations:
  - main text:   the N25 cache behind the paper figures (cached_results/model_results_N25.pkl)
  - optimizing:  the N10 runs of OPT_SCENARIO from opt_search_benchmark.py (overnight_N10/)
Both use the otj_cyclical_e_disc base model and calibrated parameters; only the unemployed search
rule differs. The optimizing runs were made in Oct 2026 and the main-text runs in June-July 2026.

The first run loads the 9.6 GB N25 cache once and saves a small extract (main_text_extract.pkl)
in the output folder; later runs read only the extract.

Outputs (same figure functions as the paper and the optimizing-search summary):
  uer_vac_single_panel, ltuer_line_single_panel, duration_wasserstein_rank_heatmap,
  duration_wasserstein_over_time (each as .jpg/.png and .pdf), fit_summary.csv

WRITES ONLY INSIDE OUT_DIR.

From calibration_remote/:
    ~/.venvs/transition_abm/bin/python opt_search_vs_main_text.py
"""

import os
import sys
import glob
import pickle
from unittest import mock

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# collate_macro_vars re-saves its collated CSVs on import; block that write.
with mock.patch('pandas.DataFrame.to_csv'):
    from collate_macro_vars import macro_observations
import plot_funs
from plot_funs import (plot_uer_vac, plot_ltuer, plot_duration_wasserstein_rank_heatmap,
                       plot_duration_wasserstein_over_time)

# Every figure below is saved through plot_funs.save_plot_standard; also write a PDF next to each one.
_save_plot_standard = plot_funs.save_plot_standard


def _save_with_pdf(fig, path, *args, **kwargs):
    _save_plot_standard(fig, path, *args, **kwargs)
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches='tight')


plot_funs.save_plot_standard = _save_with_pdf

# ── Settings ──────────────────────────────────────────────────────────────────
OPT_SCENARIO = 'sticky'      # blue line in the summary figures; 'sticky_cps' for the CPS-calibrated rule
MAIN_MODEL_ID = 'otj_cyclical_e_disc'
MAIN_MODEL_DISPLAY = 'Behavioural w. Cyc. OTJ w. RW'    # key in the N25 cache
CALIB_DATE = ["2001-01-01", "2024-05-01"]
CALIB_END = "2019-05-01"
CALIB_V_LINE = ["2000-12-01", "2019-05-01"]
DELAY = 25
CONFIDENCE_LEVEL = 0.95

BASE = os.path.join(PROJECT_ROOT, "output/calib_final/full_omn/exog_vac/figures/covid_oos/")
N25_PKL = os.path.join(BASE, "cached_results/model_results_N25.pkl")
OPT_RUN_DIR = os.path.join(BASE, "opt_search_benchmark/overnight_N10/")
OUT_DIR = os.path.join(BASE, f"opt_search_benchmark/vs_main_text/{OPT_SCENARIO}/")
FIG_DIR = os.path.join(OUT_DIR, "figures/")
EXTRACT_PKL = os.path.join(BASE, "opt_search_benchmark/vs_main_text/main_text_extract.pkl")

OPT_COLORS = {'sticky': '#1F78B4', 'sticky_cps': '#E31A1C'}   # as in opt_search_benchmark.py
MAIN_LABEL = "Heuristic Agent"            # main-text behavioural model (MAIN_MODEL_DISPLAY)
OPT_LABEL = "Optimizing Agent"            # OPT_SCENARIO; the output folder records which rule
# Darker orange than the main-text colour for this model (light orange is hard to see against the blue)
COLORS = {MAIN_LABEL: "#FF7F00", OPT_LABEL: OPT_COLORS[OPT_SCENARIO]}


def _out(*parts):
    """Path inside the vs_main_text folder; refuses anything else."""
    p = os.path.abspath(os.path.join(*parts))
    assert p.startswith(os.path.abspath(os.path.join(BASE, "opt_search_benchmark/vs_main_text"))), \
        f"refusing to write outside vs_main_text: {p}"
    return p


os.makedirs(_out(FIG_DIR), exist_ok=True)


def _monthly_dates(n):
    # The main-text runs stamp DATE with date_range(start, end, periods), which is not month-start;
    # re-stamp by position (one period per month) so both sources line up with the observed series.
    return pd.date_range(start=CALIB_DATE[0], periods=n, freq='MS')


def aggregate_runs(run_list):
    """Mean / std / CI across runs (as aggregate_runs in opt_search_benchmark.py)."""
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


# ── Main-text model (extract from the N25 cache once) ─────────────────────────
if os.path.exists(EXTRACT_PKL):
    print(f"Loading main-text extract from {EXTRACT_PKL}")
    with open(EXTRACT_PKL, 'rb') as f:
        main = pickle.load(f)
else:
    print(f"Loading the N25 cache (9.6 GB, several minutes): {N25_PKL}")
    with open(N25_PKL, 'rb') as f:
        cached = pickle.load(f)
    main = {'grouped': cached['model_results_all_sims'][MAIN_MODEL_ID],
            'ue_stock': cached['filtered_ue_stock_dfs'][MAIN_MODEL_DISPLAY][['Time Step', 'UEDuration', 'sim_id']].copy(),
            'n_simulations': cached['n_simulations']}
    del cached
    with open(_out(EXTRACT_PKL), 'wb') as f:
        pickle.dump(main, f)
    print(f"  saved extract to {EXTRACT_PKL}")

main_runs = []
for g in main['grouped']:
    g = g.copy()
    g['DATE'] = _monthly_dates(len(g))
    if 'LTUER' not in g:
        g['LTUER'] = g['LT Unemployed Persons'] / g['Unemployment']
    main_runs.append(g)
print(f"  main text: {len(main_runs)} runs")

# ── Optimizing agent (N10 runs from opt_search_benchmark.py) ──────────────────
opt_pkls = glob.glob(os.path.join(OPT_RUN_DIR, f"{OPT_SCENARIO}_????????_N10.pkl"))
assert len(opt_pkls) == 1, f"expected one cache for '{OPT_SCENARIO}', found {opt_pkls}"
print(f"Loading optimizing runs from {opt_pkls[0]}")
with open(opt_pkls[0], 'rb') as f:
    opt = pickle.load(f)
opt_runs = opt['grouped']
print(f"  optimizing: {len(opt_runs)} runs")

res = {MAIN_LABEL: aggregate_runs(main_runs), OPT_LABEL: aggregate_runs(opt_runs)}
stock = {MAIN_LABEL: main['ue_stock'], OPT_LABEL: opt['ue_stock']}

# ── Observed series (as in opt_search_benchmark.py) ───────────────────────────
obs = macro_observations.loc[(macro_observations['DATE'] >= CALIB_DATE[0]) &
                             (macro_observations['DATE'] <= CALIB_DATE[1])].dropna(subset=['VACRATE'])
obs = obs.reset_index(drop=True).copy()
obs['DATE'] = pd.to_datetime(obs['DATE'])


# ── Fit summary: RMSE per run against observed, mean and sd across runs ───────
def _rmse_per_run(g, window_end=None):
    m = g[['DATE', 'UER', 'VACRATE', 'LTUER']].merge(obs[['DATE', 'UER', 'VACRATE', 'LTUER']],
                                                      on='DATE', suffixes=('', '_obs'))
    assert len(m) > 0.95 * len(g), "simulated and observed dates do not line up"
    if window_end is not None:
        m = m[m['DATE'] <= window_end]
    return {k: np.sqrt(np.nanmean((m[k] - m[k + '_obs']) ** 2)) for k in ['UER', 'VACRATE', 'LTUER']}


rows = []
for lbl, runs in [(MAIN_LABEL, main_runs), (OPT_LABEL, opt_runs)]:
    r_in = pd.DataFrame([_rmse_per_run(g, CALIB_END) for g in runs]).add_suffix('_rmse_in')
    r_all = pd.DataFrame([_rmse_per_run(g) for g in runs]).add_suffix('_rmse_all')
    rows.append({'model': lbl, 'n_runs': len(runs),
                 **{c: r_in[c].mean() for c in r_in}, **{c + '_sd': r_in[c].std() for c in r_in},
                 **{c: r_all[c].mean() for c in r_all},
                 'mean_UER': np.mean([g['UER'].mean() for g in runs])})
summary = pd.DataFrame(rows).set_index('model')
summary.to_csv(_out(OUT_DIR, "fit_summary.csv"))
print(summary.round(4).to_string())

# ── Figures ───────────────────────────────────────────────────────────────────
fd = _out(FIG_DIR) + os.sep
plot_uer_vac(res, obs, calib_date=CALIB_V_LINE, sep=False, save=True, path=fd, plot_colors=COLORS)
plt.close('all')
plot_ltuer(res, obs, sep=False, save=True, path=fd, colors=COLORS, calib_date=CALIB_V_LINE)
plt.close('all')
survey = pd.read_csv(os.path.join(PROJECT_ROOT, "data/ltuer_cps_data_new.csv"))
plot_duration_wasserstein_rank_heatmap(ue_stock_dfs=stock, survey_df=survey, calib_date=CALIB_DATE, delay=DELAY,
                                       colors=COLORS, save=True, path=fd)
plt.close('all')
plot_duration_wasserstein_over_time(ue_stock_dfs=stock, survey_df=survey, calib_date=CALIB_DATE, delay=DELAY,
                                    colors=COLORS, save=True, path=fd, calib_v_line=CALIB_V_LINE)
plt.close('all')

print(f"\nDone. Outputs in {OUT_DIR}")
