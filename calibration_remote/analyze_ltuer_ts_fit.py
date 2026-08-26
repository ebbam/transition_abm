"""
analyze_ltuer_ts_fit.py

LTUER time-series fit, occupation by occupation: for each (model, occupation
group) pair, compares the annual model LTUER series to the annual CPS LTUER
series via NMSE (level fit, normalized by the group's squared mean
observed level) and Pearson correlation of year-over-year changes
(co-movement fit).
Produces one scatter figure per SOC grouping level, mirroring how
plot_wasserstein_summary is produced for each grouping level in
diagnose_wasserstein.py.

Loops over all three alternative networks (onet, onet_wage_asym,
single_node) at N10, producing the full set of figures for each.

Run interactively in VS Code "Run in Interactive Window".
"""

import importlib
import pickle
import sys
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import plot_funs as pf
importlib.reload(pf)

CALIB_DATE = ["2001-01-01", "2024-05-01"]
DELAY      = 25
suffix     = "_N25"
covid_oos  = True
vac_rate_setting = "exog_vac"
NETWORKS   = ["full_omn"]

DATA_PATH = os.path.join(PROJECT_ROOT, "data/")
SAVE      = True

SAVE_UER_LTUER_PANEL_TO_DROPBOX = True

from model_colors import MODEL_COLORS
_DESIRED_ORDER = list(MODEL_COLORS.keys())
_EXCLUDE = ["Behavioural w. Cyc. OTJ w. Strict RW",
            "Behavioural w.o. Cyc. OTJ w. Strict RW"]


def _filter_models(d, exclude):
    return {k: v for k, v in d.items() if k not in exclude}


def _reorder(d):
    return {k: d[k] for k in _DESIRED_ORDER if k in d}


occ_crosswalk = pd.read_csv(
    DATA_PATH + "occ_uer_ltuer_observed_w_labels.csv",
    dtype={'SOC_major_adj': str, 'SOC_minor_adj': str,
           'SOC_broad_adj': str, 'SOC2010_adj': str})
occ_crosswalk['SOC_major_adj'] = pd.to_numeric(occ_crosswalk['SOC_major_adj'], errors='coerce')

_CPS_FILE = {
    'SOC_major_adj': 'unemp_duration_dist_yearly_SOC_major.csv',
    'SOC_minor_adj': 'unemp_duration_dist_yearly_SOC_minor.csv',
    'SOC_broad_adj': 'unemp_duration_dist_yearly_SOC_broad.csv',
    'SOC2010_adj':   'unemp_duration_dist_yearly_SOC2010.csv',
}

# Annual occupation-level CPS UER series, columns ['YEAR', 'group_id', 'uer'],
# expected to come from the IPUMS UER pipeline (not yet built). Until each file
# exists, plot_uer_ltuer_ts_fit_panel renders a "data pending" placeholder for
# the UER row and still produces the LTUER row in full.
_CPS_UER_FILE = {
    'SOC_major_adj': 'unemp_rate_yearly_SOC_major.csv',
    'SOC_minor_adj': 'unemp_rate_yearly_SOC_minor.csv',
    'SOC_broad_adj': 'unemp_rate_yearly_SOC_broad.csv',
    'SOC2010_adj':   'unemp_rate_yearly_SOC2010.csv',
}

# Monthly occupation-level CPS UER series, columns ['YEAR', 'MONTH', 'group_id',
# 'uer']. Only built out for the two most aggregated levels -- monthly cells at
# SOC_broad/SOC2010 are too sparse to be usable.
_CPS_UER_MONTHLY_FILE = {
    'SOC_major_adj': 'unemp_rate_monthly_SOC_major.csv',
    'SOC_minor_adj': 'unemp_rate_monthly_SOC_minor.csv',
}

_GROUPING_COLS = ['SOC_major_adj', 'SOC_minor_adj', 'SOC_broad_adj', 'SOC2010_adj']

for nw in NETWORKS:
    print(f"\n{'=' * 70}\nNetwork: {nw}  (suffix={suffix})\n{'=' * 70}")

    PKL_PATH    = os.path.join(PROJECT_ROOT,
        f"output/calib_final/{nw}/{vac_rate_setting}/figures/cached_results/model_results{suffix}.pkl")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, f"output/calib_final/{nw}/{vac_rate_setting}/figures")
    if covid_oos:
        OUTPUT_PATH = os.path.join(PROJECT_ROOT, f"output/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos")
        PKL_PATH    = os.path.join(PROJECT_ROOT,
            f"output/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos/cached_results/model_results{suffix}.pkl")

    _n_dir = suffix.lstrip('_') if suffix else ''
    if _n_dir:
        OUTPUT_PATH = os.path.join(OUTPUT_PATH, _n_dir)
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    DROPBOX_UER_LTUER_PANEL_PATH = os.path.expanduser(
        f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos"
        + (f"/{_n_dir}" if _n_dir else "")
    )

    print(f"Loading cache from {PKL_PATH} ...")
    with open(PKL_PATH, 'rb') as f:
        cached_data = pickle.load(f)

    filtered_sim_results  = _reorder(_filter_models(cached_data['filtered_sim_results'], _EXCLUDE))
    filtered_ue_stock_dfs = _reorder(_filter_models(cached_data.get('filtered_ue_stock_dfs', {}), _EXCLUDE))

    all_results = {}
    for GROUPING_COL in _GROUPING_COLS:
        cps_dur_dist = pd.read_csv(DATA_PATH + _CPS_FILE[GROUPING_COL])

        results = pf.plot_ltuer_ts_fit_scatter(
            filtered_ue_stock_dfs, cps_dur_dist, occ_crosswalk, CALIB_DATE, DELAY,
            sim_results_dict=filtered_sim_results, grouping_col=GROUPING_COL,
            colors=MODEL_COLORS, save=SAVE, path=OUTPUT_PATH)
        all_results[GROUPING_COL] = results
        n = 0 if results is None else len(results)
        print(f"✓ {GROUPING_COL}: {n} (model, group) points plotted.")

        pf.plot_ltuer_ts_fit_hist(
            filtered_ue_stock_dfs, cps_dur_dist, occ_crosswalk, CALIB_DATE, DELAY,
            sim_results_dict=filtered_sim_results, grouping_col=GROUPING_COL,
            colors=MODEL_COLORS, save=SAVE, path=OUTPUT_PATH,
            dropbox_path=DROPBOX_UER_LTUER_PANEL_PATH if SAVE_UER_LTUER_PANEL_TO_DROPBOX else None)
        print(f"✓ {GROUPING_COL}: histogram overlay saved.")

        # Lead/lag cross-correlation only at the two most aggregated levels:
        # shifting by a year on top of differencing costs another observation,
        # and annual CPS cells get too thin to support that at finer levels.
        if GROUPING_COL in ('SOC_major_adj', 'SOC_minor_adj'):
            pf.plot_ltuer_xcorr_hist(
                filtered_ue_stock_dfs, cps_dur_dist, occ_crosswalk, CALIB_DATE, DELAY,
                grouping_col=GROUPING_COL, lags=(-1, 0, 1),
                colors=MODEL_COLORS, save=SAVE, path=OUTPUT_PATH)
            print(f"✓ {GROUPING_COL}: cross-correlation lag histogram saved.")

            # Monthly UER fit (12-month-lag differencing) -- only where the CPS
            # monthly series is dense enough to be meaningful.
            cps_uer_monthly_path = DATA_PATH + _CPS_UER_MONTHLY_FILE[GROUPING_COL]
            if os.path.exists(cps_uer_monthly_path):
                cps_uer_monthly = pd.read_csv(cps_uer_monthly_path)
                pf.plot_uer_ts_fit_hist_monthly(
                    filtered_sim_results, cps_uer_monthly, occ_crosswalk, CALIB_DATE, DELAY,
                    grouping_col=GROUPING_COL, lag_months=12,
                    colors=MODEL_COLORS, save=SAVE, path=OUTPUT_PATH,
                    dropbox_path=DROPBOX_UER_LTUER_PANEL_PATH if SAVE_UER_LTUER_PANEL_TO_DROPBOX else None)
                print(f"✓ {GROUPING_COL}: monthly UER histogram saved.")

                # Combined monthly UER+LTUER panel (the same top-row/bottom-row
                # layout as plot_uer_ltuer_ts_fit_panel, but at monthly
                # resolution), comparing raw (not deseasonalized) series. The
                # exogenous vacancy series driving the model
                # (vac_rate_setting='exog_vac') is itself raw/seasonal, so the
                # model's own UER/LTUER inherit genuine seasonal variation from
                # the same real-world source as the CPS series -- seasonality
                # here is a real signal the model should be expected to
                # reproduce, not a confound to strip out, so both sides are
                # left as-is. UER row uses a 12-month lag to match
                # plot_uer_ts_fit_hist_monthly's standalone UER fit exactly;
                # LTUER row keeps a 1-month lag.
                for series_label, uer_col, ltuer_col, uer_lag, ltuer_lag in [
                    ('raw', 'uer', 'ltuer', 12, 1),
                ]:
                    pf.plot_uer_ltuer_ts_fit_panel_monthly(
                        filtered_ue_stock_dfs, filtered_sim_results, cps_uer_monthly,
                        occ_crosswalk, CALIB_DATE, DELAY, grouping_col=GROUPING_COL,
                        uer_col=uer_col, ltuer_col=ltuer_col,
                        uer_lag_months=uer_lag, ltuer_lag_months=ltuer_lag,
                        series_label=series_label, colors=MODEL_COLORS, save=SAVE,
                        path=OUTPUT_PATH,
                        dropbox_path=DROPBOX_UER_LTUER_PANEL_PATH if SAVE_UER_LTUER_PANEL_TO_DROPBOX else None)
                    print(f"✓ {GROUPING_COL}: monthly UER+LTUER panel ({series_label}) saved.")
            else:
                print(f"  (note: {_CPS_UER_MONTHLY_FILE[GROUPING_COL]} not found yet -- skipping monthly UER fit)")

        # Final writeup figure: UER row (top) + LTUER row (bottom). UER row is a
        # placeholder until DATA_PATH + _CPS_UER_FILE[GROUPING_COL] exists.
        cps_uer_path = DATA_PATH + _CPS_UER_FILE[GROUPING_COL]
        cps_uer = pd.read_csv(cps_uer_path) if os.path.exists(cps_uer_path) else None
        if cps_uer is None:
            print(f"  (note: {_CPS_UER_FILE[GROUPING_COL]} not found yet -- UER row will be a placeholder)")

        pf.plot_uer_ltuer_ts_fit_panel(
            filtered_ue_stock_dfs, cps_dur_dist, filtered_sim_results, cps_uer,
            occ_crosswalk, CALIB_DATE, DELAY, grouping_col=GROUPING_COL,
            colors=MODEL_COLORS, save=SAVE, path=OUTPUT_PATH,
            dropbox_path=DROPBOX_UER_LTUER_PANEL_PATH if SAVE_UER_LTUER_PANEL_TO_DROPBOX else None)
        print(f"✓ {GROUPING_COL}: UER+LTUER combined panel saved.")

    print(f"\nDone with {nw}. Figures saved to:", OUTPUT_PATH)

print("\nAll networks complete:", ", ".join(NETWORKS))
