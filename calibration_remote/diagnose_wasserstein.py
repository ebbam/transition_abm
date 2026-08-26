"""
diagnose_wasserstein.py

Diagnostic script: identifies SOC groups where the simulated unemployment
duration distribution diverges most from CPS (high Wasserstein distance) and
characterises those groups by occupation properties and network structure.

Run interactively in VS Code "Run in Interactive Window".

Steps
-----
1. Compute mean Wasserstein distance per (model, SOC group) across years.
2. Characterise outlier groups by static occupation properties (wage level,
   occupation size, entry-level share) from filtered_net_results and
   filtered_sim_results.
3. Characterise outlier groups by network position: degree (number of
   neighbours), mean neighbour size, mean neighbour tightness.

Changing the SOC grouping
-------------------------
Set GROUPING_COL to any column in occ_crosswalk that maps individual
occupations to a group (e.g. 'SOC_major_adj', 'SOC_minor_adj',
'SOC_broad_adj').  Note: the Wasserstein computation (Step 1) is always at
SOC_major level because that is what the CPS duration data is aggregated to.
The GROUPING_COL only affects the characteristics tables in Steps 2 and 3,
so set it to 'SOC_major_adj' for the merge with Wasserstein results to be
consistent.
"""

import importlib
import pickle
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import plot_funs as pf
importlib.reload(pf)

from plot_style import (TEXTWIDTH, DEFAULT_ASPECT_RATIO, ROW_HEIGHT, save_plot_standard,
                         add_shared_legend, add_suptitle_spacing)

from network_input_builder import network_input_builder

# ── settings (keep in sync with explore_results.py) ──────────────────────────
CALIB_DATE   = ["2001-01-01", "2024-05-01"]
DELAY        = 25

# ── build static occupation characteristics once from the network structure ───
# Model-independent: same adjacency and entry_level for all model variants.
NETWORKS = ["full_omn"]
suffix   = "_N25"

for nw in NETWORKS:
    print(f"\n{'#' * 70}\n# NETWORK: {nw}\n{'#' * 70}")

    # ── build static occupation characteristics once per network from the
    # network structure. Rebuilt here (not cached once globally) because
    # adjacency/entry_level are network-specific -- a single global build
    # would silently characterize the wrong topology for every network
    # after the first one.
    print("Building static occupation characteristics from network...")
    _, _net_temp, _, _, _ = network_input_builder(nw, False, CALIB_DATE, delay=DELAY)
    _static_rows = []
    for _occ in _net_temp:
        _neigh_bool = np.array(_occ.list_of_neigh_bool, dtype=bool)
        _neigh_ids  = [j for j, b in enumerate(_neigh_bool) if b and j != _occ.occupation_id]
        _static_rows.append({
            'Occupation':   _occ.occupation_id,
            'entry_level':  int(_occ.entry_level),
            'n_neighbors':  int(_neigh_bool.sum()),
            'neighbor_ids': _neigh_ids,
        })
    STATIC_OCC_DF = pd.DataFrame(_static_rows)
    print(f"✓ Static characteristics built for {len(STATIC_OCC_DF)} occupations.")


    #covid_oos = True
    for covid_oos in [True]: #, False]:
        for vac_rate_setting in ["exog_vac"]: # exog_vac # "endog_cost_red", "endog_vac_drift", 

            PKL_PATH    = os.path.join(PROJECT_ROOT,
                            f"output/calib_final/{nw}/{vac_rate_setting}/figures/cached_results/model_results{suffix}.pkl")
            OUTPUT_PATH = os.path.join(PROJECT_ROOT, f"output/calib_final/{nw}/{vac_rate_setting}/figures")
            DROPBOX_OUTPUT_PATH = os.path.expanduser(
                                f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures"
                            )
            if covid_oos:
                OUTPUT_PATH = os.path.join(PROJECT_ROOT, f"output/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos")
                PKL_PATH    = os.path.join(PROJECT_ROOT,
                            f"output/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos/cached_results/model_results{suffix}.pkl")
                DROPBOX_OUTPUT_PATH = os.path.expanduser(
                                f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos"
                )
            _n_dir = suffix.lstrip('_') if suffix else ''
            if _n_dir:
                OUTPUT_PATH         = os.path.join(OUTPUT_PATH, _n_dir)
                DROPBOX_OUTPUT_PATH = os.path.join(DROPBOX_OUTPUT_PATH, _n_dir)
                os.makedirs(OUTPUT_PATH, exist_ok=True)
            dropbox = True
            DATA_PATH    = os.path.join(PROJECT_ROOT, "data/")
            SAVE         = True
            # GROUPING_COL = 'SOC_broad_adj'  # 'SOC_major_adj' | 'SOC_minor_adj' | 'SOC_broad_adj' | 'SOC2010_adj'

            # ── plot toggles — set False to skip individual plot groups ───────────────────
            RUN_PLOTS = {
                'D1':  False,   # Wasserstein outlier bars + scatter vs characteristics
                'D2b': False,   # Directional bias vs all characteristics (grid)
                'D2c': False,   # Rank vs all characteristics (grid)
                'D2d': True,   # Bias vs wage — continuous scatter (log x-axis)
                'D2e': True,   # Bias vs wage — quartile bins
                'D3':  False,   # Neighbour tightness outliers
                'D4':  False,   # Wasserstein heatmap (top2 + binary)
                'D5':  False,   # Wasserstein + mean_diff over time per SOC group
                'D6':  False,   # Manuscript summary figure (combined 3-panel)
                'D6b': True,   # Individual panels A / B / C as separate PDFs
            }

            # Maps each crosswalk grouping column to the corresponding CPS duration CSV.
            # All CSVs share the same structure with a 'group_id' column (generated by the
            # loop in ipums_cps.R).  Re-run that loop whenever you add a new grouping level.
            _CPS_FILE = {
                'SOC_major_adj': 'unemp_duration_dist_yearly_SOC_major.csv',
                'SOC_minor_adj': 'unemp_duration_dist_yearly_SOC_minor.csv',
                'SOC_broad_adj': 'unemp_duration_dist_yearly_SOC_broad.csv',
                'SOC2010_adj':   'unemp_duration_dist_yearly_SOC2010.csv',
            }

            from model_colors import MODEL_COLORS
            _DESIRED_ORDER = list(MODEL_COLORS.keys())

            # ── load cache ────────────────────────────────────────────────────────────────
            def _filter_models(d, exclude):
                return {k: v for k, v in d.items() if k not in exclude}

            def _reorder(d):
                return {k: d[k] for k in _DESIRED_ORDER if k in d}

            _EXCLUDE = ["Behavioural w. Cyc. OTJ w. Strict RW",
                        "Behavioural w.o. Cyc. OTJ w. Strict RW"]

            print(f"Loading cache from {PKL_PATH} ...")
            with open(PKL_PATH, 'rb') as f:
                cached_data = pickle.load(f)

            filtered_sim_results  = _reorder(_filter_models(cached_data['filtered_sim_results'],  _EXCLUDE))
            filtered_net_results  = _reorder(_filter_models(cached_data['filtered_net_results'],  _EXCLUDE))
            filtered_ue_stock_dfs = _reorder(_filter_models(cached_data.get('filtered_ue_stock_dfs', {}), _EXCLUDE))

            _CROSSWALK_BY_NW = {
                'full_omn':       DATA_PATH + "occ_uer_ltuer_observed_w_labels.csv",
                'onet':           DATA_PATH + "occ_uer_ltuer_observed_onet_w_labels.csv",
                'onet_wage_asym': DATA_PATH + "occ_uer_ltuer_observed_onet_w_labels.csv",
            }
            occ_crosswalk = pd.read_csv(
                _CROSSWALK_BY_NW.get(nw, DATA_PATH + "occ_uer_ltuer_observed_w_labels.csv"),
                dtype={'SOC_major_adj': str, 'SOC_minor_adj': str,
                       'SOC_broad_adj': str, 'SOC2010_adj': str})
            # Only SOC_major_adj is a plain integer — convert it to numeric.
            # The finer levels ('11-1000', '11-1010', ...) must stay as strings.
            occ_crosswalk['SOC_major_adj'] = pd.to_numeric(occ_crosswalk['SOC_major_adj'], errors='coerce')

            def _build_label_map(crosswalk, grouping_col):
                """Build {group_id: label} from the *_description column matching grouping_col."""
                desc_col = grouping_col.replace('_adj', '_description')
                if desc_col not in crosswalk.columns:
                    return {}
                return (crosswalk[[grouping_col, desc_col]]
                        .dropna(subset=[grouping_col, desc_col])
                        .drop_duplicates(grouping_col)
                        .set_index(grouping_col)[desc_col]
                        .to_dict())

            _GROUPING_COLS = (['SOC_major_adj', 'SOC_minor_adj'] if nw == 'onet_wage_asym'
                              else ['SOC_major_adj', 'SOC_minor_adj', 'SOC_broad_adj', 'SOC2010_adj'])
            for GROUPING_COL in _GROUPING_COLS:

                cps_dur_dist = pd.read_csv(DATA_PATH + _CPS_FILE[GROUPING_COL])

                print("CPS group_id sample:", cps_dur_dist['group_id'].dropna().unique()[:5])
                print("Crosswalk SOC_minor_adj sample:", occ_crosswalk['SOC_minor_adj'].dropna().unique()[:5])
                print(occ_crosswalk)
                print("✓ Data loaded.")

                # ── shared helpers ────────────────────────────────────────────────────────────

                def _make_cw(occ_crosswalk, grouping_col):
                    """Return DataFrame (Occupation [0-indexed], group_id) after dropping NaN groups.
                    group_id preserves the original type of grouping_col (int for SOC_major_adj,
                    string for SOC_minor_adj / SOC_broad_adj / SOC2010_adj)."""
                    cw = occ_crosswalk[['X', grouping_col]].drop_duplicates('X').copy()
                    cw['Occupation'] = cw['X'].astype(int) - 1
                    cw['group_id']   = cw[grouping_col]
                    return cw[['Occupation', 'group_id']].dropna(subset=['group_id'])


                def _ts_to_year(ts, calib_date, delay):
                    start = pd.Timestamp(calib_date[0])
                    return (start + pd.DateOffset(months=int(ts) - delay - 1)).year




                # ═══════════════════════════════════════════════════════════════════════════════
                # STEP 1 — Wasserstein summary
                # ═══════════════════════════════════════════════════════════════════════════════

                wass_results = pf._compute_wasserstein_all_years(
                    filtered_ue_stock_dfs, cps_dur_dist, occ_crosswalk,
                    CALIB_DATE, DELAY, sim_results_dict=filtered_sim_results,
                    grouping_col=GROUPING_COL)

                # unweighted mean across years per (model, SOC group) for ranking outliers
                _multisim_wass = 'Wasserstein_std' in wass_results.columns and wass_results['Wasserstein_std'].notna().any()
                _agg_dict = dict(mean_wass=('Wasserstein', 'mean'),
                                 mean_diff=('mean_diff',   'mean'),
                                 n_years  =('Year',        'nunique'))
                if _multisim_wass:
                    _agg_dict['mean_wass_std'] = ('Wasserstein_std', 'mean')
                    _agg_dict['mean_diff_std'] = ('mean_diff_std',   'mean')
                wass_mean = (wass_results
                            .groupby(['Model', 'SOC_major', 'label'])
                            .agg(**_agg_dict)
                            .reset_index()
                            .rename(columns={'SOC_major': 'group_id'}))

                print("\nMean Wasserstein by model and SOC group:")
                print(wass_mean.sort_values(['Model', 'mean_wass'], ascending=[True, False])
                            .to_string(index=False))


                # ═══════════════════════════════════════════════════════════════════════════════
                # STEP 2 — Occupation characteristics
                # ═══════════════════════════════════════════════════════════════════════════════

                def build_occ_characteristics(filtered_net_results, filtered_sim_results,
                                            occ_crosswalk, calib_date, delay,
                                            grouping_col='SOC_major_adj',
                                            static_occ_df=None):
                    """
                    Build per-group occupation characteristics.

                    Static (from net_temp objects or static_occ_df fallback):
                        entry_level_share, mean_n_neighbors
                    Dynamic (from sim_results — time-averaged over post-burn-in period):
                        mean_workers, mean_employment, mean_avg_wage, mean_tightness

                    Parameters
                    ----------
                    grouping_col : str
                        Column in occ_crosswalk mapping occupation_id → group.
                        Use 'SOC_major_adj' to match the Wasserstein results.
                    static_occ_df : DataFrame, optional
                        Pre-built static characteristics (Occupation, entry_level, n_neighbors,
                        neighbor_ids) from network_input_builder. Used as fallback when
                        filtered_net_results is empty (multi-sim mode).

                    Returns
                    -------
                    DataFrame with columns [Model, group_id, label, <characteristics>].
                    """
                    cw      = _make_cw(occ_crosswalk, grouping_col)
                    lab_map = _build_label_map(occ_crosswalk, grouping_col)

                    use_static_df = (not filtered_net_results) and (static_occ_df is not None)
                    model_names   = (list(filtered_sim_results.keys()) if use_static_df
                                     else list(filtered_net_results.keys()))

                    records = []
                    for model_name in model_names:
                        sim_rec = filtered_sim_results.get(model_name)

                        # -- static characteristics ----------------------------------------
                        if use_static_df:
                            static_df = (static_occ_df[['Occupation', 'entry_level', 'n_neighbors']]
                                         .copy()
                                         .merge(cw[['Occupation', 'group_id']], on='Occupation', how='inner'))
                        else:
                            occ_list = filtered_net_results[model_name]
                            static_rows = []
                            for occ in occ_list:
                                grp = cw.loc[cw['Occupation'] == occ.occupation_id, 'group_id']
                                if grp.empty:
                                    continue
                                neigh = np.array(occ.list_of_neigh_bool, dtype=bool)
                                static_rows.append({
                                    'Occupation':  occ.occupation_id,
                                    'group_id':    grp.iloc[0],
                                    'entry_level': int(occ.entry_level),
                                    'n_neighbors': int(neigh.sum()),
                                })
                            static_df = pd.DataFrame(static_rows)

                        # -- dynamic characteristics from sim_results (time-averaged) -------
                        if sim_rec is not None:
                            dyn = sim_rec[sim_rec['Time Step'] >= delay].copy()
                            dyn['avg_wage']  = dyn['Total_Wages'] / dyn['Employment'].clip(lower=1)
                            dyn['tightness'] = dyn['Vacancies'] / (dyn['Vacancies'] + dyn['Employment']).clip(lower=1)
                            dyn_agg = (dyn.groupby('Occupation')
                                        .agg(mean_workers    =('Workers',    'mean'),
                                            mean_employment =('Employment', 'mean'),
                                            mean_avg_wage   =('avg_wage',   'mean'),
                                            mean_tightness  =('tightness',  'mean'))
                                        .reset_index())
                            static_df = static_df.merge(dyn_agg, on='Occupation', how='left')

                        # -- aggregate to group level --------------------------------------
                        agg_cols = {c: 'mean' for c in static_df.columns
                                    if c not in ('Occupation', 'group_id')}
                        agg_cols['entry_level'] = 'mean'   # proportion entry-level
                        grp_agg = (static_df.groupby('group_id')
                                            .agg(**{k: (k, v) for k, v in agg_cols.items()})
                                            .reset_index()
                                            .rename(columns={'entry_level': 'entry_level_share'}))
                        grp_agg['label'] = grp_agg['group_id'].map(lab_map)
                        grp_agg['Model'] = model_name
                        records.append(grp_agg)

                    return pd.concat(records, ignore_index=True)


                occ_chars = build_occ_characteristics(
                    filtered_net_results, filtered_sim_results, occ_crosswalk,
                    CALIB_DATE, DELAY, grouping_col=GROUPING_COL,
                    static_occ_df=STATIC_OCC_DF)

                print("\nOccupation characteristics by group (first model):")
                first_model = (list(filtered_net_results.keys()) or list(filtered_sim_results.keys()))[0]
                print(occ_chars[occ_chars['Model'] == first_model]
                    .sort_values('mean_avg_wage', ascending=False)
                    .to_string(index=False))


                # ═══════════════════════════════════════════════════════════════════════════════
                # STEP 3 — Network characteristics
                # ═══════════════════════════════════════════════════════════════════════════════

                def build_network_characteristics(filtered_net_results, filtered_sim_results,
                                                occ_crosswalk, calib_date, delay,
                                                grouping_col='SOC_major_adj',
                                                static_occ_df=None):
                    """
                    For each occupation group, compute:
                        mean_degree            — mean number of neighbours
                        mean_neigh_tightness   — mean tightness (vacancy rate) of neighbour occupations
                        mean_neigh_size        — mean number of workers in neighbour occupations

                    Tightness and size of neighbours are time-averaged over the post-burn-in
                    period from filtered_sim_results.

                    Parameters
                    ----------
                    grouping_col : str
                        Column in occ_crosswalk mapping occupation_id → group.
                    static_occ_df : DataFrame, optional
                        Pre-built static characteristics with a 'neighbor_ids' column.
                        Used as fallback when filtered_net_results is empty (multi-sim mode).

                    Returns
                    -------
                    DataFrame with columns [Model, group_id, mean_degree,
                                            mean_neigh_tightness, mean_neigh_size].
                    """
                    cw = _make_cw(occ_crosswalk, grouping_col)

                    use_static_df = (not filtered_net_results) and (static_occ_df is not None)
                    model_names   = (list(filtered_sim_results.keys()) if use_static_df
                                     else list(filtered_net_results.keys()))

                    records = []
                    for model_name in model_names:
                        sim_rec = filtered_sim_results.get(model_name)
                        if sim_rec is None:
                            continue

                        # time-averaged tightness and size per occupation
                        dyn = sim_rec[sim_rec['Time Step'] >= delay].copy()
                        dyn['tightness'] = dyn['Vacancies'] / (dyn['Vacancies'] + dyn['Employment']).clip(lower=1)
                        occ_ts = (dyn.groupby('Occupation')
                                    .agg(mean_tightness=('tightness', 'mean'),
                                        mean_workers  =('Workers',   'mean'))
                                    .reset_index())

                        neigh_rows = []
                        if use_static_df:
                            for _, occ_row in static_occ_df.iterrows():
                                grp = cw.loc[cw['Occupation'] == occ_row['Occupation'], 'group_id']
                                if grp.empty:
                                    continue
                                neigh_ids   = occ_row['neighbor_ids']
                                neigh_stats = occ_ts[occ_ts['Occupation'].isin(neigh_ids)]
                                neigh_rows.append({
                                    'Occupation':           occ_row['Occupation'],
                                    'group_id':             grp.iloc[0],
                                    'degree':               len(neigh_ids),
                                    'mean_neigh_tightness': neigh_stats['mean_tightness'].mean()
                                                            if not neigh_stats.empty else np.nan,
                                    'mean_neigh_size':      neigh_stats['mean_workers'].mean()
                                                            if not neigh_stats.empty else np.nan,
                                })
                        else:
                            occ_list = filtered_net_results[model_name]
                            for occ in occ_list:
                                grp = cw.loc[cw['Occupation'] == occ.occupation_id, 'group_id']
                                if grp.empty:
                                    continue
                                neigh_ids = [j for j, b in enumerate(occ.list_of_neigh_bool)
                                            if b and j != occ.occupation_id]
                                neigh_stats = occ_ts[occ_ts['Occupation'].isin(neigh_ids)]
                                neigh_rows.append({
                                    'Occupation':           occ.occupation_id,
                                    'group_id':             grp.iloc[0],
                                    'degree':               len(neigh_ids),
                                    'mean_neigh_tightness': neigh_stats['mean_tightness'].mean()
                                                            if not neigh_stats.empty else np.nan,
                                    'mean_neigh_size':      neigh_stats['mean_workers'].mean()
                                                            if not neigh_stats.empty else np.nan,
                                })

                        neigh_df = pd.DataFrame(neigh_rows)
                        grp_agg  = (neigh_df.groupby('group_id')
                                            .agg(mean_degree           =('degree',               'mean'),
                                                mean_neigh_tightness  =('mean_neigh_tightness', 'mean'),
                                                mean_neigh_size       =('mean_neigh_size',       'mean'))
                                            .reset_index())
                        grp_agg['Model'] = model_name
                        records.append(grp_agg)

                    return pd.concat(records, ignore_index=True)


                net_chars = build_network_characteristics(
                    filtered_net_results, filtered_sim_results, occ_crosswalk,
                    CALIB_DATE, DELAY, grouping_col=GROUPING_COL,
                    static_occ_df=STATIC_OCC_DF)

                # ── merge all three into one diagnostic table ─────────────────────────────────
                diag_df = (wass_mean
                        .merge(occ_chars.drop(columns=['label'], errors='ignore'),
                                on=['Model', 'group_id'], how='left')
                        .merge(net_chars,
                                on=['Model', 'group_id'], how='left'))

                print("\nFull diagnostic table:")
                print(diag_df.sort_values(['Model', 'mean_wass'], ascending=[True, False])
                            .to_string(index=False))


                # ═══════════════════════════════════════════════════════════════════════════════
                # PLOTS
                # ═══════════════════════════════════════════════════════════════════════════════

                def plot_wass_outlier_bars(diag_df, colors=None, grouping_col='', save=False, path=None):
                    """
                    Horizontal bar chart: mean Wasserstein per SOC group per model.
                    2×2 grid: non-behavioural models on top row, behavioural on bottom row.
                    Sorted from highest to lowest within each panel.
                    """
                    if colors is None:
                        colors = MODEL_COLORS
                    model_names = [m for m in _DESIRED_ORDER if m in diag_df['Model'].unique()]
                    n_models    = len(model_names)
                    n_cols      = 2
                    n_rows      = -(-n_models // n_cols)
                    grid_positions = {name: divmod(i, n_cols) for i, name in enumerate(model_names)}

                    fig, axes = plt.subplots(n_rows, n_cols,
                                             figsize=(12, 5 * n_rows),
                                             sharex=True, sharey=False)
                    axes = np.array(axes).reshape(n_rows, n_cols)

                    for model_name in model_names:
                        row, col_idx = grid_positions[model_name]
                        ax  = axes[row][col_idx]
                        sub = (diag_df[diag_df['Model'] == model_name]
                            .sort_values('mean_wass', ascending=True))
                        col = colors.get(model_name, 'steelblue')
                        ax.barh(sub['label'], sub['mean_wass'], color=col, alpha=0.8)
                        ax.set_xlabel('Mean Wasserstein distance (months)', fontsize=8)
                        ax.set_title(model_name, fontsize=9, fontweight='bold')
                        ax.grid(axis='x', alpha=0.3)
                        ax.axvline(sub['mean_wass'].median(), color='black',
                                linewidth=0.8, linestyle='--', label='Median')
                        ax.legend(fontsize=7)

                    for i in range(n_models, n_rows * n_cols):
                        r, c = divmod(i, n_cols)
                        axes[r][c].set_visible(False)

                    fig.suptitle('Mean Wasserstein distance by SOC major group',
                                fontweight='bold', fontsize=12)
                    plt.tight_layout()
                    if save and path:
                        fig.savefig(os.path.join(path, f'wass_outlier_bars_{grouping_col}.jpg'),
                                    dpi=200, bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)


                def plot_wass_vs_characteristics(diag_df, colors=None, grouping_col='', save=False, path=None):
                    """
                    Scatter plots: mean Wasserstein vs four occupation characteristics,
                    one row per characteristic, one column per model.
                    Points are sized by mean_workers (occupation size) and labelled by SOC group.
                    Y-axis is shared across all panels for visual comparability.
                    A horizontal dashed line marks the global median Wasserstein distance.
                    """
                    if colors is None:
                        colors = MODEL_COLORS
                    model_names = [m for m in _DESIRED_ORDER if m in diag_df['Model'].unique()]
                    n_models    = len(model_names)

                    char_vars = [
                        ('mean_workers',           'Mean workers (occupation size)'),
                        ('mean_avg_wage',          'Mean average wage'),
                        ('mean_degree',            'Mean network degree (# neighbours)'),
                        ('mean_neigh_tightness',   'Mean neighbour tightness'),
                    ]
                    n_chars = len(char_vars)

                    # global y range and median across all models and groups
                    global_median = diag_df['mean_wass'].median()
                    y_min = diag_df['mean_wass'].min()
                    y_max = diag_df['mean_wass'].max()
                    y_pad = (y_max - y_min) * 0.08
                    y_lim = (y_min - y_pad, y_max + y_pad)

                    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * n_chars
                    fig, axes = plt.subplots(n_chars, n_models,
                                            figsize=(_fig_w, _fig_h),
                                            squeeze=False)

                    for col_i, model_name in enumerate(model_names):
                        sub = diag_df[diag_df['Model'] == model_name].copy()
                        col = colors.get(model_name, 'steelblue')
                        size_norm = (sub['mean_workers'] / sub['mean_workers'].max() * 300).fillna(50)

                        for row_i, (var, xlabel) in enumerate(char_vars):
                            ax = axes[row_i][col_i]
                            if var not in sub.columns or sub[var].isna().all():
                                ax.set_visible(False)
                                continue

                            ax.scatter(sub[var], sub['mean_wass'],
                                    s=size_norm, color=col, alpha=0.7, edgecolors='white', linewidths=0.5)
                            for _, row in sub.iterrows():
                                if pd.notna(row.get(var)) and pd.notna(row['mean_wass']):
                                    ax.annotate(str(row['label'])[:20],
                                                (row[var], row['mean_wass']),
                                                fontsize=5.5, ha='left', va='bottom',
                                                xytext=(3, 2), textcoords='offset points')
                            model_median = sub['mean_wass'].median()
                            ax.axhline(global_median, color='black', linewidth=0.8, linestyle='--')
                            ax.axhline(model_median, color=col, linewidth=0.8, linestyle=':')
                            ax.set_ylim(y_lim)
                            ax.set_xlabel(xlabel)
                            ax.set_ylabel('Mean Wasserstein\n(months)' if col_i == 0 else '')
                            if row_i == 0:
                                ax.set_title(model_name, fontweight='bold')
                            ax.grid(alpha=0.3)

                    fig.suptitle('Mean Wasserstein vs occupation characteristics\n'
                                '(bubble size ∝ occupation size)',
                                fontweight='bold')
                    plt.tight_layout()
                    add_suptitle_spacing(fig)
                    add_shared_legend(fig, colors,
                            extra_handles=[plt.Line2D([0], [0], color='black', linewidth=0.8, linestyle='--',
                                                label='Global median'),
                                        plt.Line2D([0], [0], color='grey',  linewidth=0.8, linestyle=':',
                                                label='Model median (model colour)')],
                            ncol=min(len(colors) + 2, 5))
                    if save and path:
                        save_plot_standard(fig, os.path.join(path, f'wass_vs_characteristics_{grouping_col}.jpg'),
                                            aspect_ratio=_fig_h / _fig_w)
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)


                def plot_bias_vs_characteristics(diag_df, colors=None, grouping_col='', save=False, path=None):
                    """
                    Scatter plots: mean directional bias (model − CPS, months) vs occupation
                    characteristics, one row per characteristic, one column per model.
                    Positive values = model over-estimates unemployment duration.
                    Points are sized by occupation size; y=0 marks perfect fit.
                    """
                    if colors is None:
                        colors = MODEL_COLORS
                    model_names = [m for m in _DESIRED_ORDER if m in diag_df['Model'].unique()]
                    n_models    = len(model_names)

                    if 'mean_diff' not in diag_df.columns or diag_df['mean_diff'].isna().all():
                        print("⚠ plot_bias_vs_characteristics skipped — mean_diff not in diag_df.")
                        return

                    char_vars = [
                        ('mean_workers',         'Mean workers (occupation size)'),
                        ('mean_avg_wage',        'Mean average wage'),
                        ('mean_degree',          'Mean network degree (# neighbours)'),
                        ('mean_neigh_tightness', 'Mean neighbour tightness'),
                    ]
                    n_chars = len(char_vars)

                    # shared y range across all models and characteristics
                    y_abs = diag_df['mean_diff'].abs().max()
                    y_pad = y_abs * 0.12
                    y_lim = (-(y_abs + y_pad), y_abs + y_pad)

                    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * n_chars
                    fig, axes = plt.subplots(n_chars, n_models,
                                             figsize=(_fig_w, _fig_h),
                                             squeeze=False)

                    for col_i, model_name in enumerate(model_names):
                        sub = diag_df[diag_df['Model'] == model_name].copy()
                        col = colors.get(model_name, 'steelblue')
                        size_norm = (sub['mean_workers'] / sub['mean_workers'].max() * 300).fillna(50)

                        for row_i, (var, xlabel) in enumerate(char_vars):
                            ax = axes[row_i][col_i]
                            if var not in sub.columns or sub[var].isna().all():
                                ax.set_visible(False)
                                continue

                            ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
                            model_mean = sub['mean_diff'].mean()
                            ax.axhline(model_mean, color=col, linewidth=0.9, linestyle=':')

                            ax.scatter(sub[var], sub['mean_diff'],
                                       s=size_norm, color=col, alpha=0.7,
                                       edgecolors='white', linewidths=0.5)
                            # show 95% CI error bars when multi-sim std is available
                            if 'mean_diff_std' in sub.columns and sub['mean_diff_std'].notna().any():
                                ci = 1.96 * sub['mean_diff_std'].fillna(0)
                                ax.errorbar(sub[var], sub['mean_diff'],
                                            yerr=ci, fmt='none', color=col,
                                            alpha=0.4, linewidth=0.8, capsize=2)
                            for _, row in sub.iterrows():
                                if pd.notna(row.get(var)) and pd.notna(row['mean_diff']):
                                    ax.annotate(str(row['label'])[:20],
                                                (row[var], row['mean_diff']),
                                                fontsize=5.5, ha='left', va='bottom',
                                                xytext=(3, 2), textcoords='offset points')

                            ax.set_ylim(y_lim)
                            ax.set_xlabel(xlabel)
                            if col_i == 0:
                                ax.set_ylabel('Mean bias\n(model − CPS, months)')
                            if row_i == 0:
                                ax.set_title(model_name, fontweight='bold')
                            ax.spines[['top', 'right']].set_visible(False)
                            ax.grid(alpha=0.25, lw=0.5)

                    fig.suptitle('Directional bias vs occupation characteristics\n'
                                 '(positive = model over-estimates duration; bubble size ∝ occupation size)',
                                 fontweight='bold')
                    plt.tight_layout()
                    add_suptitle_spacing(fig)
                    add_shared_legend(fig, colors,
                               extra_handles=[
                                   plt.Line2D([0], [0], color='black', lw=1.0, ls='--', alpha=0.6, label='No bias (0)'),
                                   plt.Line2D([0], [0], color='grey',  lw=0.9, ls=':', label='Model mean bias (model colour)'),
                               ],
                               ncol=min(len(colors) + 2, 5))
                    if save and path:
                        save_plot_standard(fig, os.path.join(path, f'bias_vs_characteristics_{grouping_col}.jpg'),
                                            aspect_ratio=_fig_h / _fig_w)
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)


                def plot_rank_vs_characteristics(diag_df, colors=None, grouping_col='',
                                                 save=False, path=None):
                    """
                    Scatter: per-occupation model rank (1 = best Wasserstein fit, N = worst)
                    vs occupation characteristics. One row per characteristic, one column per
                    model. A linear trend line shows whether better-fit occupations tend to
                    have higher/lower characteristic values. Rank axis is inverted so rank 1
                    (best) sits at the top of each panel. Bubble size ∝ occupation size.
                    """
                    if colors is None:
                        colors = MODEL_COLORS
                    model_names = [m for m in _DESIRED_ORDER if m in diag_df['Model'].unique()]
                    n_models    = len(model_names)

                    # rank each model within each occupation group (1 = lowest Wasserstein)
                    df = diag_df.copy()
                    df['rank'] = (df.groupby('group_id')['mean_wass']
                                  .rank(method='min', ascending=True)
                                  .astype(int))
                    max_rank = int(df['rank'].max())

                    char_vars = [
                        ('mean_workers',         'Mean occupation size'),
                        ('mean_avg_wage',        'Mean average wage'),
                        ('mean_degree',          'Network degree (# neighbours)'),
                        ('mean_neigh_tightness', 'Mean neighbour tightness'),
                    ]
                    n_chars = len(char_vars)

                    _rng = np.random.default_rng(42)
                    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * n_chars
                    fig, axes = plt.subplots(n_chars, n_models,
                                             figsize=(_fig_w, _fig_h),
                                             squeeze=False)

                    for col_i, model_name in enumerate(model_names):
                        sub = df[df['Model'] == model_name].copy()
                        col = colors.get(model_name, 'steelblue')

                        for row_i, (var, xlabel) in enumerate(char_vars):
                            ax = axes[row_i][col_i]
                            if var not in sub.columns or sub[var].isna().all():
                                ax.set_visible(False)
                                continue

                            valid = sub.dropna(subset=[var, 'rank'])
                            if valid.empty:
                                ax.set_visible(False)
                                continue

                            x      = valid[var].values
                            y      = valid['rank'].values.astype(float)
                            sz     = (valid['mean_workers'] / valid['mean_workers'].max() * 200).fillna(30).values
                            jitter = _rng.uniform(-0.18, 0.18, len(y))

                            ax.scatter(x, y + jitter, s=sz, color=col, alpha=0.7,
                                       edgecolors='white', linewidths=0.5, zorder=2)
                            # for _, row in valid.iterrows():
                            #     ax.annotate(str(row['label'])[:18],
                            #                 (row[var], row['rank']),
                            #                 fontsize=5.0, ha='left', va='center',
                            #                 xytext=(3, 0), textcoords='offset points',
                            #                 alpha=0.7)

                            # linear trend line
                            if len(x) >= 3:
                                try:
                                    coef  = np.polyfit(x, y, 1)
                                    x_fit = np.linspace(x.min(), x.max(), 100)
                                    ax.plot(x_fit, np.polyval(coef, x_fit),
                                            color=col, linewidth=1.4, linestyle='-',
                                            alpha=0.85, zorder=3)
                                except np.linalg.LinAlgError:
                                    pass

                            tick_labels = [str(r) for r in range(1, max_rank + 1)]
                            tick_labels[0]  = f'1\n(best)'
                            tick_labels[-1] = f'{max_rank}\n(worst)'
                            ax.set_yticks(range(1, max_rank + 1))
                            ax.set_yticklabels(tick_labels)
                            ax.set_ylim(max_rank + 0.5, 0.5)   # inverted: rank 1 at top
                            ax.set_xlabel(xlabel)
                            if col_i == 0:
                                ax.set_ylabel('Rank (1 = best fit)')
                            if row_i == 0:
                                ax.set_title(model_name, fontweight='bold')
                            ax.spines[['top', 'right']].set_visible(False)
                            ax.grid(alpha=0.25, lw=0.5)

                    fig.suptitle('Model rank vs occupation characteristics\n'
                                 '(rank 1 = lowest Wasserstein for that occupation; '
                                 'bubble size ∝ occupation size)',
                                 fontweight='bold')
                    plt.tight_layout()
                    add_suptitle_spacing(fig)
                    add_shared_legend(fig, colors, ncol=min(len(colors), 5))
                    if save and path:
                        save_plot_standard(fig, os.path.join(path, f'rank_vs_characteristics_{grouping_col}.jpg'),
                                            aspect_ratio=_fig_h / _fig_w)
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)


                def plot_bias_vs_wage_scatter(diag_df, colors=None, grouping_col='',
                                              save=False, path=None, show_ci=False):
                    """
                    Option A: single panel, all models overlaid.
                    X = mean occupational wage; Y = directional bias (model − CPS, months).
                    Bubble size ∝ occupation size. Linear trend line per model.
                    y=0 reference line marks perfect fit.
                    """
                    if colors is None:
                        colors = MODEL_COLORS
                    if 'mean_diff' not in diag_df.columns or diag_df['mean_diff'].isna().all():
                        print("⚠ plot_bias_vs_wage_scatter skipped — mean_diff not in diag_df.")
                        return
                    if 'mean_avg_wage' not in diag_df.columns or diag_df['mean_avg_wage'].isna().all():
                        print("⚠ plot_bias_vs_wage_scatter skipped — mean_avg_wage not in diag_df.")
                        return

                    model_names = [m for m in _DESIRED_ORDER if m in diag_df['Model'].unique()]
                    _fig_w, _fig_h = TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO
                    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.5, zorder=1)

                    for model_name in model_names:
                        sub = diag_df[diag_df['Model'] == model_name].dropna(
                            subset=['mean_avg_wage', 'mean_diff'])
                        sub = sub[sub['mean_avg_wage'] > 0]
                        if sub.empty:
                            continue
                        col  = colors.get(model_name, 'steelblue')
                        sz   = (sub['mean_workers'] / sub['mean_workers'].max() * 250).fillna(40)
                        ax.scatter(sub['mean_avg_wage'], sub['mean_diff'],
                                   s=sz, color=col, alpha=0.65,
                                   edgecolors='white', linewidths=0.5, zorder=2)
                        if show_ci and 'mean_diff_std' in sub.columns and sub['mean_diff_std'].notna().any():
                            ci = 1.96 * sub['mean_diff_std'].fillna(0)
                            ax.errorbar(sub['mean_avg_wage'], sub['mean_diff'],
                                        yerr=ci, fmt='none', color=col,
                                        alpha=0.3, linewidth=0.7, capsize=2, zorder=2)
                        x     = np.log(sub['mean_avg_wage'].values)
                        y     = sub['mean_diff'].values
                        if len(x) >= 3:
                            try:
                                coef  = np.polyfit(x, y, 1)
                                x_fit = np.linspace(x.min(), x.max(), 100)
                                ax.plot(np.exp(x_fit), np.polyval(coef, x_fit),
                                        color=col, linewidth=1.5, linestyle='-', alpha=0.85, zorder=3)
                            except np.linalg.LinAlgError:
                                pass

                    ax.set_xscale('log')
                    ax.set_xlabel('Mean Occupational Wage (log Scale)')
                    ax.set_ylabel('Mean Bias (Simulated − Observed, months)')
                    ax.set_title('Simulated Duration Distribution Bias and Occupational Wage', fontweight='bold')
                    ax.grid(alpha=0.2, lw=0.5)
                    plt.tight_layout()
                    add_shared_legend(fig, colors,
                                       extra_handles=[plt.Line2D([0], [0], color='black', linewidth=1.0,
                                                                  linestyle='--', alpha=0.5, label='No bias')],
                                       ncol=3)
                    if save and path:
                        _ci_suffix = '' if show_ci else '_noci'
                        save_plot_standard(fig, os.path.join(path, f'bias_vs_wage_scatter{_ci_suffix}_{grouping_col}.pdf'),
                                            aspect_ratio=_fig_h / _fig_w)
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)


                def plot_bias_vs_wage_binned(diag_df, colors=None, grouping_col='',
                                             n_bins=4, save=False, path=None):
                    """
                    Option C: occupations binned into wage quartiles; mean bias per bin
                    per model shown as connected dots with ±1 SEM error bars.
                    """
                    if colors is None:
                        colors = MODEL_COLORS
                    if 'mean_diff' not in diag_df.columns or diag_df['mean_diff'].isna().all():
                        print("⚠ plot_bias_vs_wage_binned skipped — mean_diff not in diag_df.")
                        return
                    if 'mean_avg_wage' not in diag_df.columns or diag_df['mean_avg_wage'].isna().all():
                        print("⚠ plot_bias_vs_wage_binned skipped — mean_avg_wage not in diag_df.")
                        return

                    model_names = [m for m in _DESIRED_ORDER if m in diag_df['Model'].unique()]

                    # build wage bins on the pooled wage values (consistent across models)
                    wage_vals = diag_df['mean_avg_wage'].dropna()
                    _, bin_edges = pd.qcut(wage_vals, q=n_bins, retbins=True, duplicates='drop')
                    bin_labels = [f'Q{i+1}' for i in range(len(bin_edges) - 1)]
                    n_actual = len(bin_labels)

                    # label bin midpoints with approximate wage range for x-tick labels
                    tick_labels = []
                    for i in range(n_actual):
                        lo = bin_edges[i]
                        hi = bin_edges[i + 1]
                        tick_labels.append(f'Q{i+1}\n(${lo/1000:.0f}k–${hi/1000:.0f}k)')

                    _fig_w, _fig_h = TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO
                    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.5, zorder=1)

                    n_models = len(model_names)
                    offsets  = np.linspace(-0.2, 0.2, n_models) if n_models > 1 else [0]

                    for m_i, model_name in enumerate(model_names):
                        sub = diag_df[diag_df['Model'] == model_name].dropna(
                            subset=['mean_avg_wage', 'mean_diff']).copy()
                        if sub.empty:
                            continue
                        sub['wage_bin'] = pd.cut(sub['mean_avg_wage'], bins=bin_edges,
                                                 labels=bin_labels, include_lowest=True)
                        grp = (sub.groupby('wage_bin', observed=True)['mean_diff']
                                 .agg(['mean', 'sem', 'count'])
                                 .reindex(bin_labels))

                        col     = colors.get(model_name, 'steelblue')
                        x_pos   = np.arange(n_actual) + offsets[m_i]
                        means   = grp['mean'].values
                        sems    = grp['sem'].fillna(0).values

                        ax.errorbar(x_pos, means, yerr=sems,
                                    fmt='o-', color=col, markersize=5, linewidth=1.4,
                                    capsize=3, capthick=1.0, zorder=3)

                    ax.set_xticks(np.arange(n_actual))
                    ax.set_xticklabels(tick_labels)
                    ax.set_xlabel('Occupational wage quartile')
                    ax.set_ylabel('Mean bias (model − CPS, months)')
                    ax.set_title('Directional bias by wage quartile\n(±1 SEM across occupations in bin)',
                                 fontweight='bold')
                    ax.grid(axis='y', alpha=0.2, lw=0.5)
                    plt.tight_layout()
                    add_shared_legend(fig, colors, ncol=min(len(colors), 5))
                    if save and path:
                        save_plot_standard(fig, os.path.join(path, f'bias_vs_wage_binned_{grouping_col}.pdf'),
                                            aspect_ratio=_fig_h / _fig_w)
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)


                def plot_neigh_tightness_outliers(filtered_net_results, filtered_sim_results,
                                                occ_crosswalk, wass_mean,
                                                calib_date, delay,
                                                colors=None, grouping_col='SOC_major_adj',
                                                static_occ_df=None,
                                                save=False, path=None):
                    """
                    One panel per model. Plots neighbour tightness over time for every
                    individual occupation as a line. Occupations belonging to SOC groups whose
                    mean Wasserstein exceeds the model-specific median are drawn in the model
                    colour; all others are drawn in light grey behind them.

                    'Neighbour tightness' for occupation i in year y = mean vacancy rate of
                    all occupations adjacent to i in the mobility network.

                    static_occ_df : DataFrame, optional
                        Pre-built static characteristics with a 'neighbor_ids' column.
                        Used as fallback when filtered_net_results is empty (multi-sim mode).
                    """
                    if colors is None:
                        colors = MODEL_COLORS

                    cw = _make_cw(occ_crosswalk, grouping_col)

                    def ts_to_year(ts):
                        return _ts_to_year(ts, calib_date, delay)

                    use_static_df = (not filtered_net_results) and (static_occ_df is not None)
                    model_names   = ([m for m in _DESIRED_ORDER if m in filtered_sim_results]
                                     if use_static_df
                                     else [m for m in _DESIRED_ORDER if m in filtered_net_results])
                    n_models    = len(model_names)
                    fig, axes   = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
                    if n_models == 1:
                        axes = [axes]

                    for ax, model_name in zip(axes, model_names):
                        sim_rec = filtered_sim_results.get(model_name)
                        if sim_rec is None:
                            ax.set_title(model_name, fontsize=8, fontweight='bold')
                            continue

                        # identify outlier SOC groups (above model-specific 75th percentile Wasserstein)
                        model_wass     = wass_mean[wass_mean['Model'] == model_name]
                        threshold_wass = model_wass['mean_wass'].quantile(0.75)
                        outlier_groups = set(model_wass.loc[model_wass['mean_wass'] > threshold_wass, 'group_id'])
                        outlier_occ_ids = set(
                            cw.loc[cw['group_id'].isin(outlier_groups), 'Occupation'].tolist())

                        # tightness per (year, occupation)
                        dyn = sim_rec[sim_rec['Time Step'] >= delay].copy()
                        dyn['Year']      = dyn['Time Step'].apply(ts_to_year)
                        dyn['tightness'] = dyn['Vacancies'] / (dyn['Vacancies'] + dyn['Employment']).clip(lower=1)
                        occ_year_tight   = dyn.groupby(['Year', 'Occupation'])['tightness'].mean()
                        all_years        = sorted(dyn['Year'].unique())

                        col = colors.get(model_name, 'steelblue')

                        # build iterable of (occ_id, neigh_ids) from either source
                        if use_static_df:
                            occ_iter = [(int(r['Occupation']), r['neighbor_ids'])
                                        for _, r in static_occ_df.iterrows()
                                        if r['neighbor_ids']]
                        else:
                            occ_list = filtered_net_results.get(model_name, [])
                            occ_iter = [(occ.occupation_id,
                                         [j for j, b in enumerate(occ.list_of_neigh_bool)
                                          if b and j != occ.occupation_id])
                                        for occ in occ_list]

                        # draw grey lines first so coloured outliers appear on top
                        for occ_id, neigh_ids in occ_iter:
                            if not neigh_ids:
                                continue
                            series = [np.nanmean([occ_year_tight.get((yr, nid), np.nan)
                                                for nid in neigh_ids])
                                    for yr in all_years]
                            if occ_id not in outlier_occ_ids:
                                ax.plot(all_years, series, color='#d3d3d3', linewidth=0.6,
                                        alpha=0.7, zorder=1)

                        # draw outlier lines on top
                        added_label = False
                        for occ_id, neigh_ids in occ_iter:
                            if occ_id not in outlier_occ_ids or not neigh_ids:
                                continue
                            series = [np.nanmean([occ_year_tight.get((yr, nid), np.nan)
                                                for nid in neigh_ids])
                                    for yr in all_years]
                            ax.plot(all_years, series, color=col, linewidth=1.2, alpha=0.85,
                                    zorder=2, label='Above-median Wasserstein' if not added_label else None)
                            added_label = True

                        ax.set_title(model_name, fontsize=8, fontweight='bold')
                        ax.set_xlabel('Year', fontsize=8)
                        if ax is axes[0]:
                            ax.set_ylabel('Mean neighbour tightness', fontsize=8)
                        ax.grid(alpha=0.3)
                        ax.tick_params(axis='x', rotation=45, labelsize=7)
                        ax.legend(fontsize=7)

                    fig.suptitle('Neighbour tightness over time — per occupation\n'
                                '(coloured = top-25% Wasserstein group, grey = below)',
                                fontweight='bold', fontsize=11)
                    plt.tight_layout()
                    if save and path:
                        fig.savefig(os.path.join(path, f'wass_neigh_tightness_outliers_{grouping_col}.jpg'),
                                    dpi=200, bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)


                # ── run plots ─────────────────────────────────────────────────────────────────
                if RUN_PLOTS['D1']:
                    plot_wass_outlier_bars(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                        save=SAVE, path=OUTPUT_PATH)
                    print("✓ Plot D1 (outlier bar chart) complete.")

                    plot_wass_vs_characteristics(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                                save=SAVE, path=OUTPUT_PATH)
                    print("✓ Plot D2 (Wasserstein vs characteristics scatter) complete.")
                    if dropbox:
                        plot_wass_vs_characteristics(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                                    save=True, path=DROPBOX_OUTPUT_PATH)
                        print("✓ Plot D2 saved to Dropbox.")

                if RUN_PLOTS['D2b']:
                    plot_bias_vs_characteristics(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                                 save=SAVE, path=OUTPUT_PATH)
                    print("✓ Plot D2b (directional bias vs characteristics scatter) complete.")
                    if dropbox:
                        plot_bias_vs_characteristics(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                                     save=True, path=DROPBOX_OUTPUT_PATH)
                        print("✓ Plot D2b saved to Dropbox.")

                if RUN_PLOTS['D2c']:
                    plot_rank_vs_characteristics(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                                 save=SAVE, path=OUTPUT_PATH)
                    print("✓ Plot D2c (model rank vs characteristics scatter) complete.")
                    if dropbox:
                        plot_rank_vs_characteristics(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                                     save=True, path=DROPBOX_OUTPUT_PATH)
                        print("✓ Plot D2c saved to Dropbox.")

                if RUN_PLOTS['D2d']:
                    plot_bias_vs_wage_scatter(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                              save=SAVE, path=OUTPUT_PATH)
                    print("✓ Plot D2d (bias vs wage — scatter, all models overlaid) complete.")
                    if dropbox:
                        plot_bias_vs_wage_scatter(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                                  save=True, path=DROPBOX_OUTPUT_PATH)
                        print("✓ Plot D2d saved to Dropbox.")
                    plot_bias_vs_wage_scatter(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                              save=SAVE, path=OUTPUT_PATH, show_ci=False)
                    print("✓ Plot D2d (no CI) complete.")
                    if dropbox:
                        plot_bias_vs_wage_scatter(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                                  save=True, path=DROPBOX_OUTPUT_PATH, show_ci=False)
                        print("✓ Plot D2d (no CI) saved to Dropbox.")

                if RUN_PLOTS['D2e']:
                    plot_bias_vs_wage_binned(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                             save=SAVE, path=OUTPUT_PATH)
                    print("✓ Plot D2e (bias vs wage — quartile bins) complete.")
                    if dropbox:
                        plot_bias_vs_wage_binned(diag_df, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                                 save=True, path=DROPBOX_OUTPUT_PATH)
                        print("✓ Plot D2e saved to Dropbox.")

                if RUN_PLOTS['D3']:
                    plot_neigh_tightness_outliers(
                        filtered_net_results, filtered_sim_results, occ_crosswalk,
                        wass_mean, CALIB_DATE, DELAY,
                        colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                        static_occ_df=STATIC_OCC_DF,
                        save=SAVE, path=OUTPUT_PATH)
                    print("✓ Plot D3 (neighbour tightness — per occupation, outliers highlighted) complete.")

                # ═══════════════════════════════════════════════════════════════════════════════
                # PLOT D4 — Heatmap: Wasserstein by SOC group × year (reuses plot_funs)
                # ═══════════════════════════════════════════════════════════════════════════════

                _wass_diag_args = dict(
                    ue_stock_dfs     = filtered_ue_stock_dfs,
                    cps_dur_dist     = cps_dur_dist,
                    occ_crosswalk    = occ_crosswalk,
                    calib_date       = CALIB_DATE,
                    delay            = DELAY,
                    colors           = MODEL_COLORS,
                    sim_results_dict = filtered_sim_results,
                    grouping_col     = GROUPING_COL,
                    save             = SAVE,
                    path             = OUTPUT_PATH,
                )

                if RUN_PLOTS['D4']:
                    pf.plot_wasserstein_heatmap(**_wass_diag_args, mode='top2')
                    pf.plot_wasserstein_heatmap(**_wass_diag_args, mode='binary')
                    print("✓ Plot D4 (Wasserstein heatmap) complete.")


                # ═══════════════════════════════════════════════════════════════════════════════
                # PLOT D5 — Line plot: Wasserstein + mean_diff over time per SOC group per model
                # ═══════════════════════════════════════════════════════════════════════════════

                def plot_wass_over_time(wass_results, colors=None, grouping_col='', save=False, path=None):
                    """
                    Two-row grid, one column per model.

                    Top row    — Wasserstein distance over time, one line per SOC group.
                                Shows magnitude of distributional mismatch.
                    Bottom row — mean_diff (mean model duration − mean CPS duration) over time,
                                one line per SOC group.
                                Positive = model over-estimates unemployment duration (under-estimates re-entry).
                                Negative = model under-estimates unemployment duration (over-estimates re-entry).

                    Both rows share the same x-axis (year). Y-axis is shared within each row
                    across all model panels so magnitudes are directly comparable.
                    SOC groups share a common colour legend below the figure.
                    """
                    if colors is None:
                        colors = MODEL_COLORS

                    model_names = [m for m in _DESIRED_ORDER if m in wass_results['Model'].unique()]
                    n_models    = len(model_names)
                    soc_groups  = sorted(wass_results['SOC_major'].unique())

                    # assign a distinct colour to each SOC group
                    soc_palette = plt.cm.tab10.colors
                    soc_colors  = {soc: soc_palette[i % len(soc_palette)]
                                for i, soc in enumerate(soc_groups)}
                    soc_labels  = (wass_results[['SOC_major', 'label']]
                                .drop_duplicates()
                                .set_index('SOC_major')['label']
                                .to_dict())

                    fig, axes = plt.subplots(2, n_models,
                                            figsize=(4.5 * n_models, 7),
                                            sharex=True, sharey='row',
                                            squeeze=False)

                    for col_i, model_name in enumerate(model_names):
                        sub = wass_results[wass_results['Model'] == model_name]

                        for soc in soc_groups:
                            soc_sub = sub[sub['SOC_major'] == soc].sort_values('Year')
                            if soc_sub.empty:
                                continue
                            col = soc_colors[soc]

                            # top row: Wasserstein magnitude
                            axes[0][col_i].plot(soc_sub['Year'], soc_sub['Wasserstein'],
                                                color=col, linewidth=1.3, alpha=0.85)

                            # bottom row: signed mean difference
                            axes[1][col_i].plot(soc_sub['Year'], soc_sub['mean_diff'],
                                                color=col, linewidth=1.3, alpha=0.85)

                        # zero reference line on mean_diff panel
                        axes[1][col_i].axhline(0, color='black', linewidth=0.7, linestyle='--')

                        for row_i in range(2):
                            axes[row_i][col_i].grid(alpha=0.3)
                            axes[row_i][col_i].tick_params(axis='x', rotation=45, labelsize=7)

                        axes[0][col_i].set_title(model_name, fontsize=8, fontweight='bold')
                        axes[1][col_i].set_xlabel('Year', fontsize=8)

                    axes[0][0].set_ylabel('Wasserstein distance (months)', fontsize=8)
                    axes[1][0].set_ylabel('Mean duration difference\n(model − CPS, months)', fontsize=8)

                    # shared SOC group legend below figure
                    legend_handles = [plt.Line2D([0], [0], color=soc_colors[soc], linewidth=1.5,
                                                label=soc_labels.get(soc, str(soc)))
                                    for soc in soc_groups]
                    fig.legend(handles=legend_handles, loc='lower center',
                            ncol=min(len(soc_groups), 5), fontsize=7,
                            bbox_to_anchor=(0.5, -0.08), frameon=True)

                    fig.suptitle('Wasserstein distance and mean duration gap over time by SOC group',
                                fontweight='bold', fontsize=11)
                    plt.tight_layout()
                    if save and path:
                        fig.savefig(os.path.join(path, f'wass_over_time_by_soc_{grouping_col}.jpg'),
                                    dpi=200, bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)


                if RUN_PLOTS['D5']:
                    plot_wass_over_time(wass_results, colors=MODEL_COLORS, grouping_col=GROUPING_COL,
                                        save=SAVE, path=OUTPUT_PATH)
                    print("✓ Plot D5 (Wasserstein + mean_diff over time by SOC group) complete.")

                # ═══════════════════════════════════════════════════════════════════════════════
                # PLOT D6 — Manuscript summary figure: CDF + boxplot over time + mean_diff violin
                # ═══════════════════════════════════════════════════════════════════════════════

                if RUN_PLOTS['D6']:
                    pf.plot_wasserstein_summary(**_wass_diag_args)
                    print("✓ Plot D6 (manuscript summary: CDF / boxplot / mean_diff violin) complete.")
                    if dropbox:
                        pf.plot_wasserstein_summary(**{**_wass_diag_args, 'path': DROPBOX_OUTPUT_PATH})
                        print("✓ Plot D6 (manuscript summary: CDF / boxplot / mean_diff violin) saved to Dropbox.")

                # ═══════════════════════════════════════════════════════════════════════════════
                # PLOT D6b — Individual panels A / B / C saved as separate PDFs
                # ═══════════════════════════════════════════════════════════════════════════════
                if RUN_PLOTS['D6b']:
                    pf.plot_wasserstein_summary_panels(
                        **{**_wass_diag_args,
                           'dropbox_path': DROPBOX_OUTPUT_PATH if dropbox else None})
                    print("✓ Plot D6b (individual panels A/B/C) complete.")

