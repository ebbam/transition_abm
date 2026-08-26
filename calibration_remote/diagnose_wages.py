"""
diagnose_wages.py

Diagnostic script focused on re-employment wages across models.
Complements diagnose_wasserstein.py (which covers time to re-employment).

No external wage benchmark is used; all comparisons are cross-model,
relative to the non-behavioural with OTJ baseline.

Run interactively in VS Code "Run in Interactive Window".

Plots
-----
W1  — Accepted wage at U→E over time + distribution by business-cycle phase.
W2a — Economy-wide re-employment wage heatmap vs benchmark (quarterly bins).
W2b — Re-employment wage heatmap split by UER quartile.
W3  — Wage scarring by occupation stress category.
W4  — Occupation heterogeneity: mean U→E relative wage by SOC major group
       (ranked bar chart + scatter vs occupation characteristics).
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

from collate_macro_vars import recession_spans

# ── settings (keep in sync with explore_results.py) ──────────────────────────
CALIB_DATE  = ["2001-01-01", "2024-05-01"]
DELAY       = 25
BENCHMARK   = 'Non-behavioural w. OTJ'   # cross-model reference
covid_oos = True
suffix   = "_N25"
NETWORKS = ["full_omn"]

for nw in NETWORKS:
    print(f"\n{'#' * 70}\n# NETWORK: {nw}\n{'#' * 70}")
    if nw == "onet_wage_asym":
        occ_shocks_suffix = "onet_related_occs"
        nw_short = "onet"
    elif nw == "onet":
        occ_shocks_suffix = "onet_related_occs"
        nw_short = "onet"
    elif nw == "full_omn":
        occ_shocks_suffix = "full_omn"
        nw_short = "full_omn"
    elif nw == "single_node":
        # single_node collapses all occupations into a single node (confirmed
        # via cached sim results: Occupation.nunique() == 1), so occupation-
        # level demand-shock data and SOC-grouped seps-rate data don't apply.
        occ_shocks_suffix = None
        nw_short = None

    for vac_rate_setting in ["exog_vac"]: #, "endog_cost_red", "endog_vac_drift"]:
        PKL_PATH    = os.path.join(PROJECT_ROOT,
                        f"output/calib_final/{nw}/{vac_rate_setting}/figures/cached_results/model_results.pkl")
        OUTPUT_PATH = os.path.join(PROJECT_ROOT, f"output/calib_final/{nw}/{vac_rate_setting}/figures")
        if covid_oos:
            OUTPUT_PATH = os.path.join(PROJECT_ROOT, f"output/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos")
            PKL_PATH    = os.path.join(PROJECT_ROOT,
                        f"output/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos/cached_results/model_results{suffix}.pkl")

        _n_dir = suffix.lstrip('_') if suffix else ''
        if _n_dir:
            OUTPUT_PATH = os.path.join(OUTPUT_PATH, _n_dir)
            os.makedirs(OUTPUT_PATH, exist_ok=True)

        DATA_PATH   = os.path.join(PROJECT_ROOT, "data/")
        SAVE        = True
        dropbox = True
        #GROUPING_COL = 'SOC2010_adj'   # 'SOC_major_adj' | 'SOC_minor_adj' | 'SOC_broad_adj' | 'SOC2010_adj'

        from model_colors import MODEL_COLORS
        _DESIRED_ORDER = list(MODEL_COLORS.keys())
        _EXCLUDE = ["Behavioural w. Cyc. OTJ w. Strict RW",
                    "Behavioural w.o. Cyc. OTJ w. Strict RW"]

        # ── load cache ────────────────────────────────────────────────────────────────
        def _filter_models(d, exclude):
            return {k: v for k, v in d.items() if k not in exclude}

        def _reorder(d):
            return {k: d[k] for k in _DESIRED_ORDER if k in d}

        print(f"Loading cache from {PKL_PATH} ...")
        with open(PKL_PATH, 'rb') as f:
            cached_data = pickle.load(f)

        filtered_model_results  = _reorder(_filter_models(cached_data['filtered_model_results'],  _EXCLUDE))
        filtered_sim_results    = _reorder(_filter_models(cached_data['filtered_sim_results'],    _EXCLUDE))
        filtered_net_results    = _reorder(_filter_models(cached_data['filtered_net_results'],    _EXCLUDE))
        filtered_wage_age_dfs   = _reorder(_filter_models(cached_data['filtered_wage_age_dfs'],   _EXCLUDE))
        filtered_transition_dfs = _reorder(_filter_models(cached_data.get('filtered_transition_dfs', {}), _EXCLUDE))
        filtered_ue_stock_dfs   = _reorder(_filter_models(cached_data.get('filtered_ue_stock_dfs',   {}), _EXCLUDE))

        # Multi-sim caches built before the Time Step fix lack that column in
        # filtered_model_results; reconstruct it from sequential row position.
        filtered_model_results = {
            m: (df.assign(**{'Time Step': range(len(df))}) if 'Time Step' not in df.columns else df)
            for m, df in filtered_model_results.items()
        }

        # Network-specific crosswalk: onet_wage_asym uses a different set of occupations
        # and requires its own crosswalk file with SOC grouping columns.
        # Until that file is built, SOC-grouped plots (W4, Wasserstein) are skipped for ONET.
        _CROSSWALK_BY_NW = {
            'full_omn':       DATA_PATH + "occ_uer_ltuer_observed_w_labels.csv",
            'onet':           DATA_PATH + "occ_uer_ltuer_observed_onet_w_labels.csv",
            'onet_wage_asym': DATA_PATH + "occ_uer_ltuer_observed_onet_w_labels.csv",  # build from acs_onet_2010_soc_cw.csv
            # single_node intentionally has no entry: it collapses all occupations
            # into one node (confirmed via cached sim results), so SOC-grouped
            # analysis doesn't apply. No fallback default here either -- silently
            # defaulting an unlisted network to full_omn's crosswalk would
            # misalign occupation IDs/labels rather than cleanly skipping.
        }
        _crosswalk_path  = _CROSSWALK_BY_NW.get(nw)
        _has_crosswalk   = _crosswalk_path is not None and os.path.exists(_crosswalk_path)
        if _has_crosswalk:
            occ_crosswalk = pd.read_csv(_crosswalk_path)
            occ_crosswalk['SOC_major_adj'] = pd.to_numeric(occ_crosswalk['SOC_major_adj'], errors='coerce')
        else:
            print(f"⚠ No SOC crosswalk found for nw='{nw}' at {_crosswalk_path}.")
            print("  SOC-grouped plots (W4, Wasserstein diagnostics) will be skipped.")
            print("  Build the crosswalk from acs_onet_2010_soc_cw.csv to enable them.")
            occ_crosswalk = pd.DataFrame()  # empty — guards below will skip SOC-grouped plots

        cps_dur_dist = pd.read_csv(DATA_PATH + "unemp_duration_dist_yearly_SOC_major.csv")

        # ── occupational demand shock data ────────────────────────────────────────────
        # Skipped entirely for single_node (occ_shocks_suffix=None): with only one
        # occupation, occupation-level demand-shock data doesn't apply.
        if occ_shocks_suffix is not None:
            _shock_raw  = pd.read_csv(os.path.join(PROJECT_ROOT, f"data/occupational_va_shocks_{occ_shocks_suffix}.csv"), index_col=0)
            _shock_id_col    = 'acs_2010_code' if 'acs_2010_code' in _shock_raw.columns else 'X'
            _shock_date_cols = [c for c in _shock_raw.columns if c != _shock_id_col]
            # quarterly log-growth rates; +1 → growth factors; cumprod → cumulative demand level
            _shock_cum         = (_shock_raw[_shock_date_cols] + 1).cumprod(axis=1)
            _shock_cum.index   = range(len(_shock_cum))           # 0-indexed = OriginOccupation
            _shock_cum.columns = pd.to_datetime(_shock_cum.columns)
            _shock_info = _shock_raw[[_shock_id_col]].copy().reset_index(drop=True)
            if _shock_id_col == 'acs_2010_code':
                _shock_info = _shock_info.merge(
                    occ_crosswalk[['acs_occ_code', 'label', 'SOC_major_description']].drop_duplicates('acs_occ_code'),
                    left_on='acs_2010_code', right_on='acs_occ_code', how='left'
                )
            else:
                # onet_related_occs: X is a row index with no ACS code — labels not available
                _shock_info['label'] = _shock_info[_shock_id_col].astype(str)
                _shock_info['SOC_major_description'] = ''
        else:
            print(f"⚠ No occupational demand-shock data for nw='{nw}' — shock diagnostics (W5/W5b) skipped.")

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

        print("✓ Data loaded.")


        _GROUPING_COLS = (['SOC_major_adj', 'SOC_minor_adj'] if nw == 'onet_wage_asym'
                          else ['SOC2010_adj', 'SOC_major_adj', 'SOC_minor_adj', 'SOC_broad_adj'])
        for GROUPING_COL in _GROUPING_COLS:

            # ═══════════════════════════════════════════════════════════════════════════════
            # PLOT W1 — Accepted wage at U→E: time series + business-cycle distribution
            # ═══════════════════════════════════════════════════════════════════════════════
            # Uses economy-wide filtered_model_results (U_REL_WAGE_MEAN column).
            # Shows whether behavioural models produce systematically higher or lower
            # accepted wages, and how the distribution shifts across the business cycle.

            pf.plot_accepted_wages(
                sim_results_dict = filtered_model_results,
                calib_date       = CALIB_DATE,
                delay            = DELAY,
                recessions       = recession_spans,
                colors           = MODEL_COLORS,
                lowess_frac      = 0.15,
                save             = SAVE,
                path             = OUTPUT_PATH,
            )
            print("✓ Plot W1 (accepted wage at U→E — time series + cycle distribution) complete.")
            if dropbox:
                if covid_oos:
                    DROPBOX_OUTPUT_PATH = os.path.expanduser(
                        f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos/{_n_dir}"
                    )
                else:
                    DROPBOX_OUTPUT_PATH = os.path.expanduser(
                        f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures/{_n_dir}"
                    )
                pf.plot_accepted_wages(
                    sim_results_dict = filtered_model_results,
                    calib_date       = CALIB_DATE,
                    delay            = DELAY,
                    recessions       = recession_spans,
                    colors           = MODEL_COLORS,
                    lowess_frac      = 0.15,
                    save             = True,
                    path             = DROPBOX_OUTPUT_PATH,
                )
                print("✓ Plot W1 saved to Dropbox.")


            # ═══════════════════════════════════════════════════════════════════════════════
            # PLOT W1b — Corrected wage decomposition using individual transition records
            # ═══════════════════════════════════════════════════════════════════════════════
            # Uses filtered_transition_dfs (available after re-running the simulations).
            # Panel A: mean pre-separation wage of UE hires — the anchor workers fall from.
            # Panel B: mean accepted wage at U→E — where they actually land.
            # Panel C: ratio = accepted / pre-separation = U_REL_WAGE_MEAN (cross-check W1).
            #
            # Double-bind prediction:
            #   RW model → higher Panel A (wage preferences inflate the pre-sep wage anchor)
            #              lower  Panel B (long unemployment duration worsens the match)
            #              lower  Panel C (both effects compound)

            if filtered_transition_dfs:
                pf.plot_wage_decomposition_corrected(
                    transition_dfs_dict = filtered_transition_dfs,
                    calib_date          = CALIB_DATE,
                    delay               = DELAY,
                    recessions          = recession_spans,
                    colors              = MODEL_COLORS,
                    lowess_frac         = 0.15,
                    grouping_col        = GROUPING_COL,
                    save                = SAVE,
                    path                = OUTPUT_PATH,
                )
                print("✓ Plot W1b (corrected wage decomposition: pre-sep vs accepted wage) complete.")
            else:
                print("⚠ Plot W1b skipped — filtered_transition_dfs not in cache. Re-run simulations first.")


            # ═══════════════════════════════════════════════════════════════════════════════
            # PLOT W1c — UE wage scarring vs EE wage gains
            # ═══════════════════════════════════════════════════════════════════════════════
            # Left panel:  U→E wage ratio — do workers recover their pre-separation wage?
            # Right panel: E→E wage ratio — do on-the-job movers improve their wage?
            # Both expressed as AcceptedWage / PreSepWage; reference at 1.0.
            # Non-behavioural model (no OTJ search) is absent from the EE panel.

            _soc_label_map = _build_label_map(occ_crosswalk, GROUPING_COL) if _has_crosswalk else {}

            if filtered_transition_dfs and _has_crosswalk:
                pf.plot_wage_ue_ee_comparison(
                    transition_dfs_dict = filtered_transition_dfs,
                    calib_date          = CALIB_DATE,
                    delay               = DELAY,
                    recessions          = recession_spans,
                    colors              = MODEL_COLORS,
                    lowess_frac         = 0.15,
                    occ_crosswalk       = occ_crosswalk,
                    show_ts=False,
                    grouping_col        = GROUPING_COL,
                    soc_labels          = _soc_label_map,
                    latex_ready         = SAVE,
                    save                = SAVE,
                    path                = OUTPUT_PATH,
                )
                print("✓ Plot W1c (UE scarring vs EE gains comparison) complete.")

                if dropbox:
                    if covid_oos:
                        DROPBOX_OUTPUT_PATH = os.path.expanduser(
                            f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos/{_n_dir}"
                        )
                    else:
                        DROPBOX_OUTPUT_PATH = os.path.expanduser(
                            f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures/{_n_dir}"
                        )
                    pf.plot_wage_ue_ee_comparison(
                        transition_dfs_dict = filtered_transition_dfs,
                        calib_date          = CALIB_DATE,
                        delay               = DELAY,
                        recessions          = recession_spans,
                        show_ts=False,
                        colors              = MODEL_COLORS,
                        lowess_frac         = 0.15,
                        occ_crosswalk       = occ_crosswalk,
                        grouping_col        = GROUPING_COL,
                        soc_labels          = _soc_label_map,
                        latex_ready         = SAVE,
                        save                = SAVE,
                        path                = DROPBOX_OUTPUT_PATH,
                    )

                    print("✓ Plot W1c (UE scarring vs EE gains comparison) saved to Dropbox.")
                    pf.plot_wage_scarring_by_duration(
                        transition_dfs_dict = filtered_transition_dfs,
                        calib_date          = CALIB_DATE,
                        delay               = DELAY,
                        colors              = MODEL_COLORS,
                        latex_ready         = SAVE,
                        save                = SAVE,
                        path                = DROPBOX_OUTPUT_PATH,
                        fname_suffix        = '_stacked',
                    )
                    print("✓ Plot W1d (wage scarring by unemployment duration) saved to Dropbox.")

                    pf.plot_wage_scarring_by_duration(
                        transition_dfs_dict = filtered_transition_dfs,
                        calib_date          = CALIB_DATE,
                        delay               = DELAY,
                        colors              = MODEL_COLORS,
                        latex_ready         = SAVE,
                        save                = SAVE,
                        path                = DROPBOX_OUTPUT_PATH,
                        model_groups = [
                            ['Non-behavioural', 'Non-behavioural w. OTJ'],
                            ['Behavioural w.o. Cyc. OTJ w.o RW', 'Behavioural w. Cyc. OTJ w.o RW'],
                            ['Behavioural w.o. Cyc. OTJ w. RW', 'Behavioural w. Cyc. OTJ w. RW']
                        ],
                        nrows        = 1,
                        fname_suffix = '_grouped_1row',
                    )

                    pf.plot_wage_scarring_by_duration(
                        transition_dfs_dict = filtered_transition_dfs,
                        calib_date          = CALIB_DATE,
                        delay               = DELAY,
                        colors              = MODEL_COLORS,
                        latex_ready         = SAVE,
                        save                = SAVE,
                        path                = DROPBOX_OUTPUT_PATH,
                        nrows        = 2,
                        model_groups = [
                            ['Non-behavioural', 'Non-behavioural w. OTJ'],
                            ['Behavioural w.o. Cyc. OTJ w.o RW', 'Behavioural w. Cyc. OTJ w.o RW'],
                            ['Behavioural w.o. Cyc. OTJ w. RW', 'Behavioural w. Cyc. OTJ w. RW']
                        ],
                        fname_suffix = '_grouped_2row',
                    )




            else:
                print("⚠ Plot W1c skipped — filtered_transition_dfs not in cache. Re-run simulations first.")


            # ═══════════════════════════════════════════════════════════════════════════════
            # PLOT W1d — Wage scarring as a function of unemployment duration
            # ═══════════════════════════════════════════════════════════════════════════════
            # x-axis: unemployment duration bins (0–3, 3–6, 6–12, 12–24, 24+ months)
            # y-axis: mean AcceptedWage / PreSepWage
            # Solid line = expansion-entry spells; dashed = recession-entry spells.
            # Ribbon = 25th–75th percentile across workers in each bin.
            # Shows whether the wage penalty steepens with duration, and whether
            # recession-era layoffs compound that scarring.

            if filtered_transition_dfs:
                pf.plot_wage_scarring_by_duration(
                    transition_dfs_dict = filtered_transition_dfs,
                    calib_date          = CALIB_DATE,
                    delay               = DELAY,
                    colors              = MODEL_COLORS,
                    latex_ready         = SAVE,
                    save                = SAVE,
                    path                = OUTPUT_PATH,
                )
                print("✓ Plot W1d (wage scarring by unemployment duration) complete.")

                pf.plot_wage_scarring_by_duration(
                    transition_dfs_dict = filtered_transition_dfs,
                    calib_date          = CALIB_DATE,
                    delay               = DELAY,
                    colors              = MODEL_COLORS,
                    latex_ready         = SAVE,
                    save                = SAVE,
                    path                = OUTPUT_PATH,
                    fname_suffix        = '_stacked',
                )
                print("✓ Plot W1d (wage scarring by unemployment duration, stacked) complete.")

                pf.plot_wage_scarring_by_duration(
                    transition_dfs_dict = filtered_transition_dfs,
                    calib_date          = CALIB_DATE,
                    delay               = DELAY,
                    colors              = MODEL_COLORS,
                    latex_ready         = SAVE,
                    save                = SAVE,
                    path                = OUTPUT_PATH,
                    model_groups = [
                        ['Non-behavioural', 'Non-behavioural w. OTJ'],
                        ['Behavioural w.o. Cyc. OTJ w.o RW', 'Behavioural w. Cyc. OTJ w.o RW'],
                        ['Behavioural w.o. Cyc. OTJ w. RW', 'Behavioural w. Cyc. OTJ w. RW']
                    ],
                    nrows        = 1,
                    fname_suffix = '_grouped_1row',
                )
                print("✓ Plot W1d (wage scarring by unemployment duration, grouped 1-row) complete.")

                pf.plot_wage_scarring_by_duration(
                    transition_dfs_dict = filtered_transition_dfs,
                    calib_date          = CALIB_DATE,
                    delay               = DELAY,
                    colors              = MODEL_COLORS,
                    latex_ready         = SAVE,
                    save                = SAVE,
                    path                = OUTPUT_PATH,
                    nrows        = 2,
                    model_groups = [
                        ['Non-behavioural', 'Non-behavioural w. OTJ'],
                        ['Behavioural w.o. Cyc. OTJ w.o RW', 'Behavioural w. Cyc. OTJ w.o RW'],
                        ['Behavioural w.o. Cyc. OTJ w. RW', 'Behavioural w. Cyc. OTJ w. RW']
                    ],
                    fname_suffix = '_grouped_2row',
                )
                print("✓ Plot W1d (wage scarring by unemployment duration, grouped 2-row) complete.")
            else:
                print("⚠ Plot W1d skipped — filtered_transition_dfs not in cache. Re-run simulations first.")


            # ═══════════════════════════════════════════════════════════════════════════════
            # PLOT W1e — Wage ratio vs unemployment duration (cross-model comparison)
            # ═══════════════════════════════════════════════════════════════════════════════
            # Shows the mechanical link between spell length and wage penalty.
            # One line per model — directly tests whether the RW model's lower wage ratio
            # is driven by longer spells or by lower wages conditional on duration.
            # Ribbon = 25th–75th pct (single sim) or 95% CI across sims (multi-sim).

            if filtered_transition_dfs:
                pf.plot_wage_ratio_by_duration(
                    transition_dfs_dict = filtered_transition_dfs,
                    delay               = DELAY,
                    colors              = MODEL_COLORS,
                    latex_ready         = SAVE,
                    save                = SAVE,
                    path                = OUTPUT_PATH,
                )
                print("✓ Plot W1e (wage ratio vs unemployment duration) complete.")
                if dropbox:
                    pf.plot_wage_ratio_by_duration(
                        transition_dfs_dict = filtered_transition_dfs,
                        delay               = DELAY,
                        colors              = MODEL_COLORS,
                        latex_ready         = SAVE,
                        save                = True,
                        path                = DROPBOX_OUTPUT_PATH,
                    )
                    print("✓ Plot W1e saved to Dropbox.")
            else:
                print("⚠ Plot W1e skipped — filtered_transition_dfs not in cache. Re-run simulations first.")


            # ═══════════════════════════════════════════════════════════════════════════════
            # PLOT W2a — Economy-wide re-employment wage heatmap vs benchmark
            # ═══════════════════════════════════════════════════════════════════════════════
            # Cell colour = model re-employment wage minus benchmark (BENCHMARK).
            # Blue = higher wage than benchmark; Red = lower.
            # Uses per-occupation filtered_sim_results (U_Rel_Wage, UE_Transitions columns).

            pf.plot_wage_rehire_heatmap(
                sim_results_dict = filtered_sim_results,
                calib_date       = CALIB_DATE,
                benchmark        = BENCHMARK,
                delay            = DELAY,
                freq             = 'Q',
                recession_spans  = recession_spans,
                save             = SAVE,
                path             = OUTPUT_PATH,
            )
            print("✓ Plot W2a (economy-wide re-employment wage heatmap vs benchmark) complete.")


            # ═══════════════════════════════════════════════════════════════════════════════
            # PLOT W2b — Re-employment wage heatmap by UER quartile
            # ═══════════════════════════════════════════════════════════════════════════════
            # Same as W2a but split into four panels by the origin-occupation UER quartile.
            # Reveals whether wage gaps between models are concentrated in tight or slack
            # labour markets.

            pf.plot_wage_rehire_by_uer_quantile(
                sim_results_dict = filtered_sim_results,
                calib_date       = CALIB_DATE,
                benchmark        = BENCHMARK,
                delay            = DELAY,
                freq             = 'Q',
                recession_spans  = recession_spans,
                save             = SAVE,
                path             = OUTPUT_PATH,
            )
            print("✓ Plot W2b (re-employment wage heatmap by UER quartile) complete.")


            # ═══════════════════════════════════════════════════════════════════════════════
            # PLOT W3 — Wage scarring by occupation stress category
            # ═══════════════════════════════════════════════════════════════════════════════
            # Occupations are ranked by UER stress relative to the benchmark model.
            # Top-k (highest-stress) and bottom-k (lowest-stress) are plotted separately.
            # Shows whether behavioural assumptions change the wage penalty of unemployment
            # in high-stress vs low-stress occupations.

            if _has_crosswalk:
                pf.plot_wage_scar_by_stress(
                    sim_results_dict = filtered_sim_results,
                    calib_date       = CALIB_DATE,
                    benchmark        = BENCHMARK,
                    topk             = 15,
                    delay            = DELAY,
                    lowess_frac      = 0.15,
                    colors           = MODEL_COLORS,
                    recession_spans  = recession_spans,
                    occ_crosswalk    = occ_crosswalk,
                    save             = SAVE,
                    path             = OUTPUT_PATH,
                )
                print("✓ Plot W3 (wage scarring by occupation stress) complete.")
            else:
                print("⚠ Plot W3 skipped — no SOC crosswalk for this network.")


            # ═══════════════════════════════════════════════════════════════════════════════
            # PLOT W4 — Occupation heterogeneity: mean U→E relative wage by SOC major group
            # ═══════════════════════════════════════════════════════════════════════════════

            def _build_sep_rate_df(occ_crosswalk, seps_rates_path, grouping_col='SOC_major_adj'):
                """
                Build occupation-group-level separation rates from the empirical IPUMS panel.

                Returns two DataFrames:
                sep_avg_df  — one row per group_id: mean seps_rate across years
                sep_ts_df   — one row per (group_id, year): mean seps_rate in that year
                Both have columns: group_id, <sep column>
                """
                sr = pd.read_csv(seps_rates_path)

                # merge occupation crosswalk on acs_occ_code
                cw = (occ_crosswalk[['acs_occ_code', grouping_col]]
                    .dropna(subset=[grouping_col])
                    .drop_duplicates('acs_occ_code'))
                sr = sr.merge(cw, on='acs_occ_code', how='left').dropna(subset=[grouping_col])
                sr['group_id'] = sr[grouping_col]

                sep_avg_df = (sr.groupby('group_id')['seps_rate']
                                .mean()
                                .reset_index()
                                .rename(columns={'seps_rate': 'mean_sep_rate'}))

                sep_ts_df = (sr.groupby(['group_id', 'year'])['seps_rate']
                            .mean()
                            .reset_index()
                            .rename(columns={'seps_rate': 'sep_rate_yr'}))

                return sep_avg_df, sep_ts_df


            def _build_occ_wage_ts_df(filtered_sim_results, occ_crosswalk,
                                    calib_date, delay, grouping_col='SOC_major_adj'):
                """
                Per (Model, SOC group, year): mean accepted relative wage at U→E.
                Used for the per-year scatter against empirical separation rates.
                Returns columns: Model, group_id, year, rel_wage_yr, n_trans_yr
                """
                start = pd.Timestamp(calib_date[0])
                cw = occ_crosswalk[['X', grouping_col]].drop_duplicates('X').copy()
                cw['Occupation'] = cw['X'].astype(int) - 1
                cw['group_id']   = cw[grouping_col]
                cw = cw[['Occupation', 'group_id']].dropna(subset=['group_id'])

                records = []
                for model_name, sim_rec in filtered_sim_results.items():
                    sub = sim_rec[sim_rec['Time Step'] >= delay].copy()
                    sub = sub.merge(cw, on='Occupation', how='left').dropna(subset=['group_id'])
                    sub['DATE'] = start + pd.to_timedelta(
                        (sub['Time Step'] - delay) * 30.44, unit='D')
                    sub['year'] = sub['DATE'].dt.year

                    for (grp, yr), grp_df in sub.groupby(['group_id', 'year']):
                        n_trans  = grp_df['UE_Transitions'].sum()
                        rel_wage = (grp_df['U_Rel_Wage'].sum() / n_trans
                                    if n_trans > 0 else np.nan)
                        records.append({
                            'Model':        model_name,
                            'group_id':     grp,
                            'year':         yr,
                            'rel_wage_yr':  rel_wage,
                            'n_trans_yr':   n_trans,
                        })

                return pd.DataFrame(records)


            def _build_occ_wage_df(filtered_sim_results, occ_crosswalk,
                                delay, grouping_col='SOC_major_adj',
                                transition_dfs_dict=None):
                """
                For each (model, SOC group), compute the time-averaged mean accepted
                relative wage at U→E grouped by ORIGIN occupation.

                mean_rel_wage = mean(AcceptedWage / PreSepWage) across U→E transitions
                originating from that occupation group, from individual transition records.
                Requires transition_dfs_dict; raises ValueError if not provided.

                Returns a long DataFrame with columns:
                    Model, group_id, label, mean_rel_wage, n_transitions,
                    mean_workers, mean_employment, mean_avg_wage, mean_tightness
                """
                if not transition_dfs_dict:
                    raise ValueError(
                        "_build_occ_wage_df requires transition_dfs_dict (filtered_transition_dfs). "
                        "Re-run simulations to generate the cache with transition records."
                    )

                cw = occ_crosswalk[['X', grouping_col]].drop_duplicates('X').copy()
                cw['Occupation'] = cw['X'].astype(int) - 1
                cw['group_id']   = cw[grouping_col]
                cw = cw[['Occupation', 'group_id']].dropna(subset=['group_id'])
                cw_occ = cw.rename(columns={'Occupation': 'occ_id'})

                lab_map = _build_label_map(occ_crosswalk, grouping_col)

                # origin-grouped wage ratios from individual transition records
                origin_ratios = {}
                origin_ntrans = {}
                for model_name, tdf in transition_dfs_dict.items():
                    ue = (tdf[(tdf['TransitionType'] == 'UE') &
                            (tdf['Time Step'] >= delay)]
                        .dropna(subset=['PreSepWage', 'AcceptedWage'])
                        .copy())
                    ue['ratio'] = ue['AcceptedWage'] / ue['PreSepWage'].clip(lower=1e-9)
                    ue = (ue.rename(columns={'OriginOccupation': 'occ_id'})
                            .merge(cw_occ, on='occ_id', how='left')
                            .dropna(subset=['group_id']))
                    grp = ue.groupby('group_id')['ratio'].agg(['sum', 'count'])
                    origin_ratios[model_name] = (grp['sum'] / grp['count'])
                    origin_ntrans[model_name] = grp['count']

                records = []
                for model_name, sim_rec in filtered_sim_results.items():
                    sub = sim_rec[sim_rec['Time Step'] >= delay].copy()
                    sub = sub.merge(cw, on='Occupation', how='left').dropna(subset=['group_id'])
                    sub['avg_wage']  = sub['Total_Wages'] / sub['Employment'].clip(lower=1)
                    sub['tightness'] = sub['Vacancies'] / (sub['Vacancies'] + sub['Employment']).clip(lower=1)

                    for grp, grp_df in sub.groupby('group_id'):
                        rel_wage = origin_ratios[model_name].get(grp, np.nan) if model_name in origin_ratios else np.nan
                        n_trans  = int(origin_ntrans[model_name].get(grp, 0)) if model_name in origin_ntrans else 0
                        try:
                            label = lab_map.get(int(grp), str(grp))
                        except (ValueError, TypeError):
                            label = str(grp)
                        records.append({
                            'Model':            model_name,
                            'group_id':         grp,
                            'label':            label,
                            'mean_rel_wage':    rel_wage,
                            'n_transitions':    n_trans,
                            'mean_workers':     grp_df['Workers'].mean(),
                            'mean_employment':  grp_df['Employment'].mean(),
                            'mean_avg_wage':    grp_df['avg_wage'].mean(),
                            'mean_tightness':   grp_df['tightness'].mean(),
                        })

                return pd.DataFrame(records)


            def plot_wage_occ_bars(wage_occ_df, colors=None, save=False, path=None):
                """
                W4a: Horizontal bar chart of mean U→E relative wage per SOC major group
                per model. Sorted highest to lowest within each panel.
                A value > 1 means accepted wages exceed the occupation mean wage.
                """
                if colors is None:
                    colors = MODEL_COLORS
                model_names = [m for m in _DESIRED_ORDER if m in wage_occ_df['Model'].unique()]
                n_models    = len(model_names)
                n_cols      = 2
                n_rows      = -(-n_models // n_cols)
                grid_pos    = {name: divmod(i, n_cols) for i, name in enumerate(model_names)}

                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(12, 5 * n_rows),
                                         sharex=False, sharey=False)
                axes = np.array(axes).reshape(n_rows, n_cols)

                for model_name in model_names:
                    row, col = grid_pos[model_name]
                    ax  = axes[row][col]
                    sub = (wage_occ_df[wage_occ_df['Model'] == model_name]
                        .dropna(subset=['mean_rel_wage'])
                        .sort_values('mean_rel_wage', ascending=True))
                    col_c = colors.get(model_name, 'steelblue')
                    ax.barh(sub['label'], sub['mean_rel_wage'], color=col_c, alpha=0.8)
                    ax.axvline(1.0, color='black', linewidth=0.8, linestyle='--', label='= occupation mean')
                    ax.set_xlabel('Mean accepted wage / occupation mean wage', fontsize=8)
                    ax.set_title(model_name, fontsize=9, fontweight='bold')
                    ax.tick_params(axis='y', labelsize=7)
                    ax.tick_params(axis='x', labelsize=7)
                    ax.grid(axis='x', alpha=0.3)
                    ax.legend(fontsize=7)

                for i in range(n_models, n_rows * n_cols):
                    r, c = divmod(i, n_cols)
                    axes[r][c].set_visible(False)

                fig.suptitle('Mean accepted relative wage at U→E by SOC major group',
                            fontweight='bold', fontsize=11)
                plt.tight_layout()
                if save and path:
                    fig.savefig(os.path.join(path, f'wage_occ_bars_{GROUPING_COL}.jpg'),
                                dpi=200, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
                    plt.close(fig)


            def plot_wage_occ_scatter(wage_occ_df, sep_avg_df=None, wage_occ_ts_df=None,
                                    sep_ts_df=None, colors=None, save=False, path=None):
                """
                W4b: Scatter of mean U→E relative wage vs occupation characteristics.
                One row per characteristic, one column per model.
                Bubble size ∝ occupation size (mean_workers).

                Optional extra rows:
                sep_avg_df + wage_occ_df  → row: time-avg empirical separation rate (one point per occ group)
                wage_occ_ts_df + sep_ts_df → row: per-year empirical sep rate (one point per occ group × year)
                """
                if colors is None:
                    colors = MODEL_COLORS
                model_names = [m for m in _DESIRED_ORDER if m in wage_occ_df['Model'].unique()]
                n_models    = len(model_names)

                char_vars = [
                    ('mean_workers',    'Mean workers (occupation size)'),
                    ('mean_avg_wage',   'Mean average wage'),
                    ('mean_tightness',  'Mean market tightness'),
                ]

                # merge time-averaged sep rate into wage_occ_df if available
                if sep_avg_df is not None:
                    wage_occ_df = wage_occ_df.merge(sep_avg_df, on='group_id', how='left')
                    char_vars.append(('mean_sep_rate', 'Empirical separation rate (time-avg)'))

                n_static_rows = len(char_vars)
                n_rows = n_static_rows + (1 if (wage_occ_ts_df is not None and sep_ts_df is not None) else 0)

                global_median = wage_occ_df['mean_rel_wage'].median()
                y_min = wage_occ_df['mean_rel_wage'].min()
                y_max = wage_occ_df['mean_rel_wage'].max()
                y_pad = (y_max - y_min) * 0.08
                y_lim = (y_min - y_pad, y_max + y_pad)

                fig, axes = plt.subplots(n_rows, n_models,
                                        figsize=(4.5 * n_models, 3.5 * n_rows),
                                        squeeze=False)

                # ── static rows (one point per occupation group) ──────────────────────────
                for col_i, model_name in enumerate(model_names):
                    sub       = wage_occ_df[wage_occ_df['Model'] == model_name].copy()
                    col_c     = colors.get(model_name, 'steelblue')
                    size_norm = (sub['mean_workers'] / sub['mean_workers'].max() * 300).fillna(50)

                    for row_i, (var, xlabel) in enumerate(char_vars):
                        ax = axes[row_i][col_i]
                        if var not in sub.columns or sub[var].isna().all():
                            ax.set_visible(False)
                            continue
                        ax.scatter(sub[var], sub['mean_rel_wage'],
                                s=size_norm, color=col_c, alpha=0.7,
                                edgecolors='white', linewidths=0.5)
                        for _, row in sub.iterrows():
                            if pd.notna(row.get(var)) and pd.notna(row['mean_rel_wage']):
                                ax.annotate(str(row['label'])[:20],
                                            (row[var], row['mean_rel_wage']),
                                            fontsize=5.5, ha='left', va='bottom',
                                            xytext=(3, 2), textcoords='offset points')
                        model_median = sub['mean_rel_wage'].median()
                        ax.axhline(global_median, color='black', linewidth=0.8, linestyle='--')
                        ax.axhline(1.0,           color='grey',  linewidth=0.6, linestyle=':')
                        ax.axhline(model_median,  color=col_c,   linewidth=0.8, linestyle=':')
                        ax.set_ylim(y_lim)
                        ax.set_xlabel(xlabel, fontsize=8)
                        ax.set_ylabel('Mean accepted\nrelative wage' if col_i == 0 else '', fontsize=8)
                        if row_i == 0:
                            ax.set_title(model_name, fontsize=8, fontweight='bold')
                        ax.grid(alpha=0.3)

                # ── per-year row (one point per occ group × year) ─────────────────────────
                if wage_occ_ts_df is not None and sep_ts_df is not None:
                    ts_merged = wage_occ_ts_df.merge(sep_ts_df, on=['group_id', 'year'], how='inner')
                    y_ts_min = ts_merged['rel_wage_yr'].quantile(0.02)
                    y_ts_max = ts_merged['rel_wage_yr'].quantile(0.98)
                    y_ts_pad = (y_ts_max - y_ts_min) * 0.08

                    for col_i, model_name in enumerate(model_names):
                        ax    = axes[n_static_rows][col_i]
                        col_c = colors.get(model_name, 'steelblue')
                        sub   = ts_merged[ts_merged['Model'] == model_name].dropna(
                                    subset=['sep_rate_yr', 'rel_wage_yr'])
                        size_norm = (sub['n_trans_yr'] / sub['n_trans_yr'].max() * 60).clip(lower=5)
                        ax.scatter(sub['sep_rate_yr'], sub['rel_wage_yr'],
                                s=size_norm, color=col_c, alpha=0.35,
                                edgecolors='none')
                        ax.axhline(global_median, color='black', linewidth=0.8, linestyle='--')
                        ax.axhline(1.0,           color='grey',  linewidth=0.6, linestyle=':')
                        ax.set_ylim(y_ts_min - y_ts_pad, y_ts_max + y_ts_pad)
                        ax.set_xlabel('Empirical separation rate (per year)', fontsize=8)
                        ax.set_ylabel('Accepted relative wage\n(per year)' if col_i == 0 else '',
                                    fontsize=8)
                        ax.grid(alpha=0.3)

                fig.legend(loc='upper right', fontsize=7,
                        handles=[
                            plt.Line2D([0], [0], color='black', linewidth=0.8, linestyle='--',
                                        label='Global median'),
                            plt.Line2D([0], [0], color='grey',  linewidth=0.6, linestyle=':',
                                        label='= occupation mean wage'),
                            plt.Line2D([0], [0], color='grey',  linewidth=0.8, linestyle=':',
                                        label='Model median (model colour)'),
                        ])
                fig.suptitle('Mean accepted relative wage vs occupation characteristics\n'
                            '(bubble size ∝ occupation size; bottom row: per-year empirical sep rate)',
                            fontweight='bold', fontsize=11)
                plt.tight_layout()
                if save and path:
                    fig.savefig(os.path.join(path, f'wage_occ_scatter_{GROUPING_COL}.jpg'),
                                dpi=200, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
                    plt.close(fig)


            if _has_crosswalk:
                # build the shared occupation-level wage tables
                wage_occ_df = _build_occ_wage_df(
                    filtered_sim_results, occ_crosswalk,
                    DELAY, grouping_col=GROUPING_COL,
                    transition_dfs_dict=filtered_transition_dfs if filtered_transition_dfs else None)

                SEPS_PATH = os.path.join(PROJECT_ROOT,
                    f"dRC_Replication/data/ipums_variables_w_seps_rate_{nw_short}.csv")
                sep_avg_df, sep_ts_df = _build_sep_rate_df(occ_crosswalk, SEPS_PATH,
                                                            grouping_col=GROUPING_COL)
                wage_occ_ts_df = _build_occ_wage_ts_df(
                    filtered_sim_results, occ_crosswalk,
                    CALIB_DATE, DELAY, grouping_col=GROUPING_COL)

                print("\nMean accepted relative wage by SOC group (first model):")
                first_model = _DESIRED_ORDER[0]
                print(wage_occ_df[wage_occ_df['Model'] == first_model]
                    .sort_values('mean_rel_wage', ascending=False)
                    .to_string(index=False))

                plot_wage_occ_bars(wage_occ_df, colors=MODEL_COLORS, save=SAVE, path=OUTPUT_PATH)
                print("✓ Plot W4a (accepted wage ranked bar by SOC group) complete.")

                plot_wage_occ_scatter(wage_occ_df,
                                    sep_avg_df=sep_avg_df,
                                    wage_occ_ts_df=wage_occ_ts_df,
                                    sep_ts_df=sep_ts_df,
                                    colors=MODEL_COLORS, save=SAVE, path=OUTPUT_PATH)
                print("✓ Plot W4b (accepted wage vs occupation characteristics scatter) complete.")
            else:
                print("⚠ Plots W4a/W4b skipped — no SOC crosswalk for this network.")

        # ═══════════════════════════════════════════════════════════════════════════════
        # PLOT W5 — Re-employment and wage trajectories after demand shocks
        # ═══════════════════════════════════════════════════════════════════════════════
        # Identifies the top-N occupations with the largest cumulative demand decline
        # in SHOCK_WINDOW (data-driven, no sector pre-selection).
        # Compares wage-by-duration and UE-duration for workers separated from those
        # occupations during SHOCK_HORIZON vs all other workers in the same period.
        # Wage ratio is demeaned relative to each occupation's own baseline (outside
        # SHOCK_HORIZON), so y=1 means "same outcome as in non-shock times".

        SHOCK_WINDOW  = ("2000-01-01", "2003-07-01")   # window used to rank shock severity
        SHOCK_HORIZON = ("2000-01-01", "2006-01-01")   # separation window for worker filtering
        TOP_N_SHOCKED = 20

        if occ_shocks_suffix is not None:
            _cols       = _shock_cum.columns
            _in_win     = _cols[(_cols >= pd.to_datetime(SHOCK_WINDOW[0])) & (_cols <= pd.to_datetime(SHOCK_WINDOW[1]))]
            _cum_chg    = _shock_cum[_in_win[-1]] / _shock_cum[_in_win[0]] - 1
            shocked_ids = _cum_chg.nsmallest(TOP_N_SHOCKED).index.tolist()

            print(f"\nTop-{TOP_N_SHOCKED} demand-shocked occupations ({SHOCK_WINDOW[0][:4]}–{SHOCK_WINDOW[1][:4]}):")
            _disp = _shock_info.copy()
            _disp['cum_chg_pct'] = _cum_chg.values * 100
            _disp_cols = [c for c in [_shock_id_col, 'label', 'SOC_major_description', 'cum_chg_pct']
                          if c in _disp.columns]
            print(_disp.loc[shocked_ids, _disp_cols].to_string())
        else:
            shocked_ids = []

        if filtered_transition_dfs:
            # pf.plot_shock_episode_wages(
            #     transition_dfs_dict = filtered_transition_dfs,
            #     shocked_occ_ids     = shocked_ids,
            #     shock_horizon       = SHOCK_HORIZON,
            #     shock_info_df       = _shock_info,
            #     shock_window        = SHOCK_WINDOW,
            #     shock_label         = f"Top-{TOP_N_SHOCKED} demand-shocked",
            #     calib_date          = CALIB_DATE,
            #     delay               = DELAY,
            #     colors              = MODEL_COLORS,
            #     latex_ready         = SAVE,
            #     save                = SAVE,
            #     path                = OUTPUT_PATH,
            # )
            # print("✓ Plot W5 (demand-shock re-employment and wage trajectories) complete.")

            if dropbox:
                if covid_oos:
                    DROPBOX_OUTPUT_PATH = os.path.expanduser(
                        f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos/{_n_dir}"
                    )
                else:
                    DROPBOX_OUTPUT_PATH = os.path.expanduser(
                        f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures/{_n_dir}"
                    )
                # pf.plot_shock_episode_wages(
                #     transition_dfs_dict = filtered_transition_dfs,
                #     shocked_occ_ids     = shocked_ids,
                #     shock_horizon       = SHOCK_HORIZON,
                #     shock_info_df       = _shock_info,
                #     shock_window        = SHOCK_WINDOW,
                #     shock_label         = f"Top-{TOP_N_SHOCKED} demand-shocked",
                #     calib_date          = CALIB_DATE,
                #     delay               = DELAY,
                #     colors              = MODEL_COLORS,
                #     latex_ready         = SAVE,
                #     save                = SAVE,
                #     path                = DROPBOX_OUTPUT_PATH,
                # )
                # print("✓ Plot W5 saved to Dropbox.")
        else:
            print("⚠ Plot W5 skipped — filtered_transition_dfs not in cache.")


        # ═══════════════════════════════════════════════════════════════════════════════
        # PLOT W5b — Episode-specific trajectories across models
        # ═══════════════════════════════════════════════════════════════════════════════
        # For each named shock episode, compare post-separation wage and duration
        # outcomes for workers from the matched occupations laid off DURING the episode
        # vs the SAME occupations outside the episode window.  One figure per episode,
        # one panel per model — holds occupation composition fixed, varies timing.
        #
        # Occupation ids (0-indexed, from analyze_occ_shocks.py output):

        # Occupation IDs are network-specific (0-indexed from analyze_occ_shocks.py).
        # Add an entry for each network as needed; leave empty to skip W5b for that network.
        _W5B_EPISODES_BY_NW = {
            "full_omn": {
                "Coal mining (2011–2016)": {
                    "occ_ids": [86, 335, 367, 370, 371, 373, 374],
                    "window":  ("2011-01-01", "2016-12-01"),
                },
                "Construction trades (2008–2010)": {
                    "occ_ids": [15, 335, 337, 338, 340, 341, 344, 349, 352, 358, 359, 366],
                    "window":  ("2008-01-01", "2010-12-01"),
                },
                "Motor vehicles & assemblers (2008–2009)": {
                    "occ_ids": [413, 414, 415, 417, 382, 386, 387, 388],
                    "window":  ("2008-01-01", "2009-12-01"),
                },
            },
        }
        W5B_EPISODES = _W5B_EPISODES_BY_NW.get(nw, {})

        if not W5B_EPISODES:
            print(f"⚠ W5b episodes not defined for nw='{nw}' — Plot W5b skipped.")


        if filtered_transition_dfs:
            #for ep_name, ep_cfg in W5B_EPISODES.items():
                # pf.plot_episode_trajectories(
                #     transition_dfs_dict = filtered_transition_dfs,
                #     episode_occ_ids     = ep_cfg['occ_ids'],
                #     episode_window      = ep_cfg['window'],
                #     episode_name        = ep_name,
                #     calib_date          = CALIB_DATE,
                #     delay               = DELAY,
                #     colors              = MODEL_COLORS,
                #     cps_dur_dist        = cps_dur_dist,
                #     occ_crosswalk       = occ_crosswalk,
                #     ue_stock_dfs        = filtered_ue_stock_dfs,
                #     latex_ready         = SAVE,
                #     save                = SAVE,
                #     path                = OUTPUT_PATH,
                # )
                # print(f"✓ Plot W5b — {ep_name}")

            if dropbox:
                if covid_oos:
                    DROPBOX_OUTPUT_PATH = os.path.expanduser(
                        f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures/covid_oos"
                    )
                else:
                    DROPBOX_OUTPUT_PATH = os.path.expanduser(
                        f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{nw}/{vac_rate_setting}/figures"
                    )
                # for ep_name, ep_cfg in W5B_EPISODES.items():
                #     pf.plot_episode_trajectories(
                #         transition_dfs_dict = filtered_transition_dfs,
                #         episode_occ_ids     = ep_cfg['occ_ids'],
                #         episode_window      = ep_cfg['window'],
                #         episode_name        = ep_name,
                #         calib_date          = CALIB_DATE,
                #         delay               = DELAY,
                #         colors              = MODEL_COLORS,
                #         cps_dur_dist        = cps_dur_dist,
                #         occ_crosswalk       = occ_crosswalk,
                #         ue_stock_dfs        = filtered_ue_stock_dfs,
                #         latex_ready         = SAVE,
                #         save                = SAVE,
                #         path                = DROPBOX_OUTPUT_PATH,
                #     )
                #     print(f"✓ Plot W5b — {ep_name} saved to Dropbox.")
        else:
            print("⚠ Plot W5b skipped — filtered_transition_dfs not in cache.")
