# %%
# CPU LIMITING CONFIGURATION - ADD THESE AT THE VERY TOP
# ============================================================================
import os
import time

# Method 1: Limit threads for NumPy/SciPy (MOST IMPORTANT)
os.environ["OMP_NUM_THREADS"] = "2"  # Limit OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # Limit OpenBLAS threads
os.environ["MKL_NUM_THREADS"] = "2"  # Limit Intel MKL threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"  # Limit Accelerate framework (macOS)
os.environ["NUMEXPR_NUM_THREADS"] = "2"  # Limit NumExpr threads

# Method 2: Set process priority to low (niceness)
try:
    os.nice(10)  # Increase niceness (lower priority)
    print("Process priority set to low (niceness: 10)")
except:
    print("Could not set process niceness (may need sudo)")

# %%
# Import packages
import numpy as np
from abm_funs import *
from plot_funs import *
from plot_style import TEXTWIDTH, save_plot_standard, apply_minor_gridlines, add_shared_legend
# from us_input import *
from copy import deepcopy 
from network_input_builder import *
import pandas as pd
import random as random
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from params_to_latex import *
from scipy.stats import truncnorm
from model_fun import *
from scipy.stats import pearsonr, linregress
import math as math
import networkx as nx
from matplotlib.lines import Line2D
import importlib
from copy import deepcopy
from pandas import Series
import seaborn as sns
import matplotlib.gridspec as gridspec
import datetime

from collate_macro_vars import *
from statsmodels.tsa.filters import hp_filter, bk_filter, cf_filter
from quantecon import hamilton_filter
from statsmodels.tsa.seasonal import seasonal_decompose
import csv

rng = np.random.default_rng()
test_fun()

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"

print(f"Total CPU cores: {os.cpu_count()}")
print(f"Limited to 2 threads for numerical operations")

save_button = True
suffix = ""
SIMPLE_RES = False # Set True to run fast with simple_res=True and only plot UER/vacancy rate

calib_v_line = ["2000-12-01", "2019-05-01"]

complete_nw = False
steady_state_run = True
dropbox = True
#vac_rate_setting = "endog_vac_drift" # Options: "endog_vac_drift", "endog_cost_red", "gdp_growth", "g_prime_norm_weighted"
for vac_rate_setting in ["exog_vac"]:#, "endog_cost_red", "endog_vac_drift"]:
    calib_source_folder = "calib_final" #"calib_w_wages" #"cos_calib_smart_hire_5" # This now includes the wage growth dimension
    if calib_source_folder in ["cos_calib_smart_hire_5", "cos_calib_smart_hire_4", "calib_w_wages"]:
        param_source_folder = calib_source_folder
        if calib_source_folder == "cos_calib_smart_hire_5": # This then uses the calibrated parameter valuesnot calibrated with the wage growth parameter - its almost the same. "cos_calib_w_wages" is the real one calibrated with the wage mechanism on.
            param_source_folder = "cos_calib_smart_hire_4"
        #suffix = "smart_hire"
        suffix = "_wages_exog_vac"
        dumb_hire_spec = False
    elif calib_source_folder == "cos_calib":
        suffix = ""
        dumb_hire_spec = True
    elif calib_source_folder == "calib_final":
        if vac_rate_setting == "endog_vac_drift":
            #suffix = "_wages_replace_early_shocks_endog_vac_distance_weighted_vr"
            vac_method_arg = "g_prime_norm"
            suffix = "_g_prime_norm_wage_update_True"
        elif vac_rate_setting == "endog_cost_red":
            #suffix = "_wages_replace_early_shocks_cost_red_distance_weighted_vr"
            vac_method_arg = "cost_red"
            suffix = "_cost_red_wage_update_True"
        elif vac_rate_setting == "exog_vac":
            #suffix = "_wages_replace_early_shocks_exog_vac"
            suffix = "_exog_wage_update_True"
            vac_method_arg = "exog"

        param_source_folder = calib_source_folder
        dumb_hire_spec = False

    else: 
        raise KeyError("Need to specify a dumb_hire_spec with the param source folder!.")

    # ============================================================================
    # HELPER FUNCTIONS FOR MULTI-SIMULATION MODE
    # ============================================================================

    def aggregate_simulation_results(sim_list, metric_cols=None):
        """
        Aggregate multiple simulation runs into mean, std, and confidence intervals.
        
        OPTIMIZED: Builds DataFrame all at once to avoid fragmentation warnings
        
        Parameters:
        -----------
        sim_list : list of DataFrames
            Each DataFrame has same structure (DATE + metric columns)
        metric_cols : list or None
            Columns to aggregate. If None, aggregates all numeric columns
        
        Returns:
        --------
        DataFrame with DATE, metric (mean), metric_std, metric_ci_lower, metric_ci_upper
        """
        if not sim_list:
            return pd.DataFrame()
        
        template = sim_list[0].copy()
        
        if metric_cols is None:
            metric_cols = template.select_dtypes(include=[np.number]).columns.tolist()
            metric_cols = [c for c in metric_cols if c not in ['Time Step']]
        
        # Stack: (n_sims, n_timesteps, n_metrics)
        all_sims = [sim_df[metric_cols].values for sim_df in sim_list]
        stacked = np.stack(all_sims, axis=0)
        
        # Statistics across simulations
        mean_vals = np.mean(stacked, axis=0)
        std_vals = np.std(stacked, axis=0)
        
        # Confidence intervals
        from scipy.stats import norm
        z_score = norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
        ci_margin = z_score * (std_vals / np.sqrt(len(sim_list)))
        ci_lower = mean_vals - ci_margin
        ci_upper = mean_vals + ci_margin
        
        # Build result dictionary - ALL COLUMNS AT ONCE (prevents fragmentation)
        result_dict = {}
        
        # Add identifier columns from template (not aggregated, same across all sims)
        if 'DATE' in template.columns:
            result_dict['DATE'] = template['DATE'].values
        if 'Time Step' in template.columns:
            result_dict['Time Step'] = template['Time Step'].values

        # Add all metric columns at once
        for i, col in enumerate(metric_cols):
            result_dict[col] = mean_vals[:, i]
            result_dict[f'{col}_std'] = std_vals[:, i]
            result_dict[f'{col}_ci_lower'] = ci_lower[:, i]
            result_dict[f'{col}_ci_upper'] = ci_upper[:, i]
        
        # Create DataFrame once from complete dictionary (no fragmentation!)
        result = pd.DataFrame(result_dict)
        
        return result


    def run_model_multiple_times(model_name, base_params, calib_params,
                                calib_date, occ_ids, n_sims):
        """
        Run a single model configuration multiple times and aggregate results.
        NOW WITH CPU THROTTLING

        Returns:
        --------
        Tuple of (aggregated_grouped, grouped_list, ue_stock_df_combined, transition_df_combined)
        where ue_stock_df_combined and transition_df_combined are the per-worker DataFrames
        from all runs concatenated with a 'sim_id' column (0-indexed).
        """
        print(f"  Running {n_sims} simulations for {model_name}...")

        grouped_list      = []
        ue_stock_runs     = []
        transition_runs   = []
        sim_record_runs   = []
        seekers_runs      = []
        wage_age_runs     = []
        # Incremental mean for sim_record: avoids holding N copies in memory at once
        _rec_sum     = None   # running sum of numeric columns (ndarray)
        _rec_count   = 0
        _rec_template = None  # one sim_record to carry id/label columns

        for sim_i in range(n_sims):
            if (sim_i + 1) % 10 == 0:
                print(f"    Simulation {sim_i + 1}/{n_sims}")

            # Add throttling delay BEFORE running simulation
            if sim_i > 0 and THROTTLE_DELAY > 0:
                time.sleep(THROTTLE_DELAY)

            test_params = deepcopy(base_params)
            test_params.update(calib_params)

            # Run model — simple_res=True returns just a data dict; False returns 10 objects
            result = run_single_local(**test_params)
            if test_params.get('simple_res', False):
                # Convert simple data dict to a minimal grouped DataFrame for aggregation
                sim_grouped = pd.DataFrame({
                    'UER': result['UER'],
                    'VACRATE': result['VACRATE']
                })
                sim_grouped['DATE'] = pd.date_range(start=calib_date[0],
                                                    periods=len(sim_grouped), freq='MS')
            else:
                sim_record, sim_grouped, sim_net, sum_stats, seekers_rec, \
                    avg_wage_off_diff, app_loads_df, time_to_emp, vacs_final, wage_age_df, ue_stock_df, transition_df = result

                # Add DATE and derived metrics to grouped
                sim_grouped['DATE'] = pd.date_range(start=calib_date[0],
                                                    end=calib_date[1],
                                                    periods=len(sim_grouped))
                sim_grouped['LTUER'] = (sim_grouped['LT Unemployed Persons'] /
                                    sim_grouped['Unemployment'])
                sim_grouped['AVGWAGE'] = (sim_grouped['Total_Wages'] /
                                        sim_grouped['Employment'])
                sim_grouped["Seeker Composition"] = (
                    sim_grouped["Employed Seekers"] /
                    (sim_grouped["Employed Seekers"] + sim_grouped["Unemployed Seekers"])
                )

                # Derive per-occupation metrics on sim_record (mirrors single-sim path)
                # Use incremental mean to avoid holding N copies in memory simultaneously.
                if not sim_record.empty:
                    sim_record = sim_record.copy()
                    sim_record['LTUER']           = sim_record['LT Unemployed Persons'] / sim_record['Unemployment']
                    sim_record['UER']             = sim_record['Unemployment'] / sim_record['Workers']
                    sim_record['AVGWAGE']         = sim_record['Total_Wages'] / sim_record['Employment']
                    sim_record['VACRATE1']        = sim_record['Vacancies'] / sim_record['Target_Demand']
                    sim_record['VACRATE']         = sim_record['Vacancies'] / (sim_record['Vacancies'] + sim_record['Employment'])
                    sim_record['U_REL_WAGE_MEAN'] = sim_record['U_Rel_Wage'] / sim_record['UE_Transitions']
                    sim_record['E_REL_WAGE_MEAN'] = sim_record['E_Rel_Wage'] / sim_record['EE_Transitions']
                    sim_record['UE_Trans_Rate']   = sim_record['UE_Transitions'] / sim_record['Workers']
                    sim_record['EE_Trans_Rate']   = sim_record['EE_Transitions'] / sim_record['Workers']
                    sim_record = sim_record.merge(occ_ids, left_on='Occupation', right_on='id', how='left')
                    _num_cols = sim_record.select_dtypes(include=[np.number]).columns.tolist()
                    vals = sim_record[_num_cols].values
                    if _rec_sum is None:
                        _rec_sum      = vals.copy()
                        _rec_template = sim_record.copy()
                    else:
                        _rec_sum     += vals
                    _rec_count += 1
                    _sr = sim_record.copy()
                    _sr['sim_id'] = sim_i
                    sim_record_runs.append(_sr)

                # Tag with sim_id and collect for cross-run aggregation
                if not ue_stock_df.empty:
                    ue_stock_df = ue_stock_df.copy()
                    ue_stock_df['sim_id'] = sim_i
                    ue_stock_runs.append(ue_stock_df)
                if not transition_df.empty:
                    transition_df = transition_df.copy()
                    transition_df['sim_id'] = sim_i
                    transition_runs.append(transition_df)
                if not seekers_rec.empty:
                    _s = seekers_rec.copy()
                    _s['DATE'] = pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(_s))
                    _s['Application Effort (U)'] = _s['Applications Sent (Unemployed)'] / _s['Unemployed Seekers']
                    _s['Application Effort (E)'] = _s['Applications Sent (Employed)'] / _s['Employed Seekers']
                    seekers_runs.append(_s)
                if not wage_age_df.empty:
                    _wsum = (wage_age_df.groupby('Year')['Wage']
                                       .quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
                                       .unstack()
                                       .rename(columns={0.01: 'p1', 0.05: 'p5', 0.10: 'p10',
                                                        0.25: 'p25', 0.50: 'p50', 0.75: 'p75',
                                                        0.90: 'p90', 0.95: 'p95', 0.99: 'p99'})
                                       .reset_index())
                    _wsum['mean'] = (wage_age_df.groupby('Year')['Wage'].mean().values)
                    _wsum['sim_id'] = sim_i
                    wage_age_runs.append(_wsum)

            grouped_list.append(sim_grouped)

        print(f"  Aggregating {n_sims} simulations...")
        aggregated = aggregate_simulation_results(grouped_list)

        ue_stock_combined      = pd.concat(ue_stock_runs,   ignore_index=True) if ue_stock_runs   else pd.DataFrame()
        transition_combined    = pd.concat(transition_runs, ignore_index=True) if transition_runs else pd.DataFrame()
        sim_record_all_sims    = pd.concat(sim_record_runs, ignore_index=True) if sim_record_runs else pd.DataFrame()
        seekers_combined       = aggregate_simulation_results(seekers_runs)    if seekers_runs    else pd.DataFrame()
        wage_age_combined      = pd.concat(wage_age_runs,   ignore_index=True) if wage_age_runs   else pd.DataFrame()

        # Finalise incremental sim_record mean
        if _rec_template is not None and _rec_count > 0:
            _num_cols = _rec_template.select_dtypes(include=[np.number]).columns.tolist()
            sim_record_combined = _rec_template.copy()
            sim_record_combined[_num_cols] = _rec_sum / _rec_count
        else:
            sim_record_combined = pd.DataFrame()

        return aggregated, grouped_list, ue_stock_combined, transition_combined, sim_record_combined, seekers_combined, wage_age_combined, sim_record_all_sims

    def combine_caches(cache_paths, output_path, name_map=None):
        """
        Combine two or more multi-simulation cache files into a single cache
        with N = sum of all input Ns.

        Parameters
        ----------
        cache_paths : list of str
            Paths to the pickle files to combine (must all be multi-sim caches).
        output_path : str
            Where to write the merged pickle file.
        name_map : dict, optional
            Mapping from raw model IDs (keys of model_results_all_sims) to
            display names (keys of filtered_* dicts). Required when the caches
            were saved with raw IDs in model_results_all_sims.

        Economy-wide grouped results are re-aggregated from the raw per-sim
        lists (exact CIs). Occupation-level sim_results are combined via
        weighted mean. Worker-level transition_dfs and ue_stock_dfs are
        concatenated with renumbered sim_ids.
        """
        caches = []
        for p in cache_paths:
            with open(p, 'rb') as f:
                caches.append(pickle.load(f))

        for i, c in enumerate(caches):
            if not c.get('model_results_all_sims'):
                raise ValueError(
                    f"Cache {cache_paths[i]} has no 'model_results_all_sims'. "
                    "Re-run with the current version of auto_run_model_modified.py first."
                )

        ns      = [c['n_simulations'] for c in caches]
        n_total = sum(ns)
        raw_ids = list(caches[0]['model_results_all_sims'].keys())

        # Build raw→display mapping. If model_results_all_sims is already keyed
        # by display names (no name_map needed), id_to_display is identity.
        if name_map:
            id_to_display = {raw: name_map.get(raw, raw) for raw in raw_ids}
        else:
            id_to_display = {k: k for k in raw_ids}

        # Re-aggregate economy-wide grouped (exact CIs); output keyed by display name
        filtered_model_results = {}
        for raw_id in raw_ids:
            display = id_to_display[raw_id]
            all_runs = []
            for c in caches:
                all_runs.extend(c['model_results_all_sims'].get(raw_id, []))
            filtered_model_results[display] = aggregate_simulation_results(all_runs)

        # Weighted mean for occupation-level sim_results; look up by display name
        filtered_sim_results = {}
        for raw_id in raw_ids:
            display = id_to_display[raw_id]
            base = caches[0]['filtered_sim_results'].get(display)
            if base is None:
                continue
            num_cols = base.select_dtypes(include=[np.number]).columns
            combined = base.copy()
            combined[num_cols] = sum(
                n * c['filtered_sim_results'][display][num_cols].values
                for n, c in zip(ns, caches)
            ) / n_total
            filtered_sim_results[display] = combined

        # Concatenate worker-level frames; renumber sim_ids; look up by display name
        filtered_transition_dfs       = {}
        filtered_ue_stock_dfs         = {}
        filtered_sim_results_all_sims = {}
        filtered_wage_age_dfs         = {}
        for raw_id in raw_ids:
            display = id_to_display[raw_id]
            t_parts, u_parts, s_parts, w_parts = [], [], [], []
            offset = 0
            for n, c in zip(ns, caches):
                t = c['filtered_transition_dfs'].get(display)
                u = c['filtered_ue_stock_dfs'].get(display)
                s = c.get('filtered_sim_results_all_sims', {}).get(display)
                w = c.get('filtered_wage_age_dfs', {}).get(display)
                if t is not None and not t.empty:
                    t = t.copy(); t['sim_id'] = t['sim_id'] + offset
                    t_parts.append(t)
                if u is not None and not u.empty:
                    u = u.copy(); u['sim_id'] = u['sim_id'] + offset
                    u_parts.append(u)
                if s is not None and not s.empty:
                    s = s.copy(); s['sim_id'] = s['sim_id'] + offset
                    s_parts.append(s)
                if w is not None and not w.empty:
                    w = w.copy()
                    if 'sim_id' in w.columns:
                        w['sim_id'] = w['sim_id'] + offset
                    w_parts.append(w)
                offset += n
            if t_parts:
                filtered_transition_dfs[display]       = pd.concat(t_parts, ignore_index=True)
            if u_parts:
                filtered_ue_stock_dfs[display]         = pd.concat(u_parts, ignore_index=True)
            if s_parts:
                filtered_sim_results_all_sims[display] = pd.concat(s_parts, ignore_index=True)
            if w_parts:
                filtered_wage_age_dfs[display]         = pd.concat(w_parts, ignore_index=True)

        # Merge raw grouped lists keyed by raw_id so the cache can be combined again
        model_results_all_sims = {}
        for raw_id in raw_ids:
            model_results_all_sims[raw_id] = []
            for c in caches:
                model_results_all_sims[raw_id].extend(c['model_results_all_sims'].get(raw_id, []))

        # Weighted mean for seekers_recs (raw per-sim data not stored, so approximate)
        seekers_recs = {}
        for raw_id in raw_ids:
            display = id_to_display[raw_id]
            base = caches[0].get('seekers_recs', {}).get(display)
            if base is None:
                continue
            num_cols = base.select_dtypes(include=[np.number]).columns
            combined = base.copy()
            combined[num_cols] = sum(
                n * c['seekers_recs'][display][num_cols].values
                for n, c in zip(ns, caches)
            ) / n_total
            seekers_recs[display] = combined

        cache_out = {
            'filtered_model_results':        filtered_model_results,
            'filtered_sim_results':          filtered_sim_results,
            'filtered_sim_results_all_sims': filtered_sim_results_all_sims,
            'filtered_transition_dfs':       filtered_transition_dfs,
            'filtered_ue_stock_dfs':         filtered_ue_stock_dfs,
            'model_results_all_sims':        model_results_all_sims,
            'n_simulations':                 n_total,
            'confidence_level':              caches[0].get('confidence_level', 0.95),
            'filtered_net_results':          {},
            'filtered_sum_stats':            {},
            'seekers_recs':                  seekers_recs,
            'filtered_app_loads':            {},
            'filtered_time_to_emps':         {},
            'filtered_wage_age_dfs':         filtered_wage_age_dfs,
            'filtered_avg_wage_off_diffs':   {},
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(cache_out, f)
        print(f"✓ Combined cache saved: N={n_total} ({' + '.join(str(n) for n in ns)}) → {output_path}")

    # ============================================================================
    # COMBINE EXISTING CACHES  (set COMBINE_CACHES = True to use)
    # ============================================================================
    # Set COMBINE_CACHES = True and edit the cache_paths list to point to the
    # model_results_N*.pkl files you want to merge.  The script will combine them
    # and write a new file; it will NOT rerun any simulations.
    CONFIDENCE_LEVEL = 0.95  # Used by aggregate_simulation_results inside combine_caches
    COMBINE_CACHES = True
    if COMBINE_CACHES:
        _base = f'output/{calib_source_folder}/full_omn/{vac_rate_setting}/figures'
        combine_caches(
            cache_paths=[
                f'{_base}/covid_oos/cached_results/model_results_N10.pkl',
                f'{_base}/covid_oos/cached_results/model_results_N15.pkl',
            ],
            output_path=f'{_base}/covid_oos/cached_results/model_results_N25.pkl',
            name_map={
                "nonbehav":                  "Non-behavioural",
                "otj_nonbehav":              "Non-behavioural w. OTJ",
                "otj_cyclical_e_disc":       "Behavioural w. Cyc. OTJ w. RW",
                "otj_disc":                  "Behavioural w.o. Cyc. OTJ w. RW",
                "otj_cyclical_e_disc_no_rw": "Behavioural w. Cyc. OTJ w.o RW",
                "otj_disc_no_rw":            "Behavioural w.o. Cyc. OTJ w.o RW",
            },
        )

    for N in [10]:  # Change to [25] for post-lunch run, then set COMBINE_CACHES=True
        # ============================================================================
        # MULTI-SIMULATION CONFIGURATION
        # ============================================================================
        # REDUCED from 25 to 10 for lower CPU usage
        N_SIMULATIONS = N # Reduced from 25 to 10
        CONFIDENCE_LEVEL = 0.95  # For confidence intervals when N_SIMULATIONS is not None

        # Add throttling delay between simulations (in seconds)
        THROTTLE_DELAY = 0.5  # Wait 0.5 seconds between simulations

        # Rest of the file continues exactly as before...
        if dropbox:
            output_prefix = f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/{calib_source_folder}/"
            cached_output_path = f'output/{calib_source_folder}/'
        else:
            output_prefix = f'output/{calib_source_folder}/'
            cached_output_path = f'output/{calib_source_folder}/'


        for nw in ["onet", "onet_wage_asym", "single_node"]: # ]:#,"full_omn"]:#
            print(nw)  
            which_params = f'{param_source_folder}/{nw}/'
            if calib_source_folder == "calib_final":
                which_params = f'{param_source_folder}/{nw}/{vac_rate_setting}/'
            for run in ["covid_oos"]:#["steady_state"]: # , ]: #, "covid_oos"]: #, 'covid_oos', "steady_state"]:
                print(run)
                if run == "covid_oos":
                    calib_date = ["2001-01-01", "2024-05-01"]
                    if calib_source_folder == "calib_final":
                        output_path = os.path.expanduser(f'{output_prefix}{nw}/{vac_rate_setting}/figures/covid_oos/')
                    else:
                        output_path = os.path.expanduser(f'{output_prefix}{nw}/figures/covid_oos/')

                elif run == "steady_state":
                    calib_date = ["2001-01-01", "2019-05-01"]
                    if calib_source_folder == "calib_final":
                        output_path = os.path.expanduser(f'{output_prefix}{nw}/{vac_rate_setting}/figures/steady_state/')
                    else:
                        output_path = os.path.expanduser(f'{output_prefix}{nw}/figures/steady_state/')

                else:
                    calib_date = ["2001-01-01", "2019-05-01"]
                    
                    if calib_source_folder == "calib_final":
                        output_path = os.path.expanduser(f'{output_prefix}{nw}/{vac_rate_setting}/figures/')
                    else:
                        output_path = os.path.expanduser(f'{output_prefix}{nw}/figures/')

                if N_SIMULATIONS is not None and N_SIMULATIONS > 1:
                    output_path = output_path.rstrip('/')
                    output_path = f"{output_path}/N{N_SIMULATIONS}/"

                if not dropbox:
                    os.makedirs(output_path, exist_ok=True)

                # Macro observations
                observation = macro_observations.loc[
                    (macro_observations['DATE'] >= calib_date[0]) & 
                    (macro_observations['DATE'] <= calib_date[1])].dropna(subset=["UNRATE", "VACRATE"]).reset_index()

                mod_data, net_temp, vacs, occ_ids, occ_shocks_dat = network_input_builder(nw, complete_nw, calib_date, delay=25)
                print(occ_shocks_dat.shape[1])

                param_df1 = pd.read_csv(path + f"output/{which_params}calibrated_params_all{suffix}.csv")
                print(param_df1)
                if calib_source_folder == "calib_final":
                    param_df2 = pd.read_csv(path + f"output/{which_params}calibrated_params_all_theta_signal{suffix}.csv")
                    param_df = pd.concat([param_df1, param_df2], ignore_index=True)
                elif calib_source_folder == "calib_w_wages":
                    param_df1 = param_df1[~param_df1['model_cat'].isin(['nonbehav', 'otj_disc'])]
                    param_df2 = pd.read_csv(path + f"output/{which_params}calibrated_params_all_theta_signal{suffix}.csv")
                    param_df_backup = pd.read_csv(path + f"output/{which_params}calibrated_params_all{suffix}_backup.csv")
                    param_df = pd.concat([param_df1, param_df2, param_df_backup], ignore_index=True)
                else:
                    param_df2 = pd.read_csv(path + f"output/{which_params}calibrated_params_all_theta_signal{suffix}.csv")
                    param_df = pd.concat([param_df1, param_df2], ignore_index=True)

                #print(param_df1.shape, param_df2.shape, param_df.shape)

                # Sort by Timestamp in descending order
                param_df = param_df.sort_values(by='Timestamp', ascending=False)
                    
                name_map = {
                    "nonbehav": "Non-behavioural",
                    "otj_nonbehav": "Non-behavioural w. OTJ",
                    "otj_cyclical_e_disc": "Behavioural w. Cyc. OTJ w. RW",
                    "otj_disc": "Behavioural w.o. Cyc. OTJ w. RW",
                    #"otj_disc_strict_rw": "Behavioural w.o. Cyc. OTJ w. Strict RW",
                    #"otj_cyclical_e_disc_strict_rw": "Behavioural w. Cyc. OTJ w. Strict RW",
                    "otj_cyclical_e_disc_no_rw": "Behavioural w. Cyc. OTJ w.o RW",
                    "otj_disc_no_rw": "Behavioural w.o. Cyc. OTJ w.o RW"
                }

                desired_order = [
                    "Non-behavioural",
                    "Non-behavioural w. OTJ",
                    "Behavioural w. Cyc. OTJ w.o RW",
                    "Behavioural w.o. Cyc. OTJ w.o RW",
                    "Behavioural w. Cyc. OTJ w. RW",
                    "Behavioural w.o. Cyc. OTJ w. RW"#,
                    #"Behavioural w. Cyc. OTJ w. Strict RW",
                    #"Behavioural w.o. Cyc. OTJ w. Strict RW"
                ]

                if run in ["base", "covid_oos"]:
                    params_to_latex(
                        param_df,
                        out_tex_path=f"{output_path}params_table.tex",
                        param_col="Parameter",
                        value_col="Value",
                        model_col="model_cat",
                        timestamp_col="Timestamp",
                        model_name_map=name_map,
                        desired_model_order=desired_order,
                        decimals=3,
                        label_suffix = nw
                    )

                params = {
                    'mod_data': mod_data,
                    'dumb_hire': dumb_hire_spec,
                    'net_temp': net_temp,
                    'vacs': vacs,
                    'time_steps': occ_shocks_dat.shape[1],
                    'delay': 25,
                    'gdp_data': gdp_dat,
                    'app_effort_dat': duration_to_prob_dict,
                    'occ_shocks_data': occ_shocks_dat,
                    'simple_res': SIMPLE_RES,
                    'mistake_rate': 0.1,
                    'vac_method': vac_method_arg
                }

                # vac_dat must cover the calibration window only (not the 24+delay prefix).
                # model_fun indexes vac_data as vac_data[t-delay] for t>=delay, so it
                # needs exactly calib_length entries — use len(observation) as the cap.
                vac_df = observation['VACRATE'].to_numpy()
                calib_length = len(observation)
                if len(vac_df) > calib_length:
                    print("smoothing vac_df")
                    window = len(vac_df) // calib_length
                    vac_dat = Series(vac_df).rolling(window=window, min_periods=1).mean()[window-1::window].reset_index(drop=True)
                    vac_dat = vac_dat[:calib_length]
                else:
                    vac_dat = vac_df[:calib_length]

                fig, ax = plt.subplots()
                ax.hist(mod_data['demand_target'], bins = 100)
                # Add percentile vertical lines
                ax.axvline(np.percentile(mod_data['demand_target'], 25), color='blue', linestyle='--', linewidth=2, label='25th percentile')
                ax.axvline(np.percentile(mod_data['demand_target'], 50), color='red', linestyle='--', linewidth=2, label='Median (50th)')
                ax.axvline(np.percentile(mod_data['demand_target'], 75), color='green', linestyle='--', linewidth=2, label='75th percentile')
                ax.set_title("Occupational Employment Levels")
                ax.set_ylabel("Count")
                ax.set_xlabel("Target demand (10,000 Persons)")
                save_plot_standard(fig, os.path.join(output_path, "dist_occ_size.png"))
                plt.close(fig)

                def filter_models(model_dict, include=None, exclude=None, pattern=None):
                    """
                    Filter models by name.
                    
                    Parameters:
                    - include: List of exact names to include
                    - exclude: List of exact names to exclude
                    - pattern: String pattern that must be in the name
                    """
                    result = model_dict.copy()
                    
                    if include is not None:
                        result = {k: v for k, v in result.items() if k in include}
                    
                    if exclude is not None:
                        result = {k: v for k, v in result.items() if k not in exclude}
                    
                    if pattern is not None:
                        result = {k: v for k, v in result.items() if pattern in k}
                    
                    return result
                
                exclude_models = ["Behavioural w. Cyc. OTJ w. Strict RW",
                    "Behavioural w.o. Cyc. OTJ w. Strict RW"]
                    #"Behavioural w. Cyc. OTJ w.o RW",
                    #"Behavioural w.o. Cyc. OTJ w. RW",
                    #"Behavioural w.o. Cyc. OTJ w.o RW"]
                
                sep_strings_filtered = [("Non-behavioural", "Non-behavioural w. and w.o OTJ"), 
                                            ("w.o RW", "Behavioural w.o Res. Wages"),
                                            ("w. RW", "Behavioural w. Res Wages")]
                                        # ("w. Strict RW", "Behavioural w. Strict Res Wages")]

                # ===== CONFIGURATION =====
                FORCE_RERUN = False # Set to True to ignore cache and rerun models
                if dropbox:
                    FORCE_RERUN = False

                # Define where to save the results
                if calib_source_folder == "calib_final":
                    if run == "base":
                        results_cache_dir = f'{cached_output_path}{nw}/{vac_rate_setting}/figures/cached_results/'
                    else:
                        results_cache_dir = f'{cached_output_path}{nw}/{vac_rate_setting}/figures/{run}/cached_results/'
                else:
                    if run == "base":
                        results_cache_dir = f'{cached_output_path}{nw}/figures/cached_results/'
                    else:
                        results_cache_dir = f'{cached_output_path}{nw}/figures/{run}/cached_results/'

                if not dropbox:
                    Path(results_cache_dir).mkdir(parents=True, exist_ok=True)

                # Suggested (separate files per mode):
                if N_SIMULATIONS is not None and N_SIMULATIONS > 1:
                    cache_file = f'{results_cache_dir}model_results_N{N_SIMULATIONS}.pkl'
                else:
                    cache_file = f'{results_cache_dir}model_results.pkl'
                print(cache_file)

                # ===== LOAD OR RUN MODELS =====
                if os.path.exists(cache_file) and not FORCE_RERUN:
                    print(f"Loading cached results from {cache_file}...")
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Unpack the cached objects
                    filtered_model_results = filter_models(cached_data['filtered_model_results'], exclude = exclude_models)
                    filtered_net_results = filter_models(cached_data['filtered_net_results'], exclude = exclude_models)
                    filtered_sim_results = filter_models(cached_data['filtered_sim_results'], exclude = exclude_models)
                    filtered_sum_stats = filter_models(cached_data['filtered_sum_stats'], exclude = exclude_models)
                    seekers_recs = filter_models(cached_data['seekers_recs'], exclude = exclude_models)
                    filtered_app_loads = filter_models(cached_data['filtered_app_loads'], exclude = exclude_models)
                    filtered_time_to_emps = filter_models(cached_data['filtered_time_to_emps'], exclude = exclude_models)
                    filtered_wage_age_dfs = filter_models(cached_data['filtered_wage_age_dfs'], exclude = exclude_models)
                    filtered_avg_wage_off_diffs = filter_models(cached_data.get('filtered_avg_wage_off_diffs', {}), exclude = exclude_models)
                    filtered_ue_stock_dfs = filter_models(cached_data.get('filtered_ue_stock_dfs', {}), exclude=exclude_models)
                    model_results_all_sims = {k: v for k, v in cached_data.get('model_results_all_sims', {}).items()
                                              if k not in exclude_models}
                    filtered_sim_results_all_sims = filter_models(cached_data.get('filtered_sim_results_all_sims', {}), exclude=exclude_models)

                    # Enforce desired display order on all loaded dicts
                    _reorder = lambda d: {k: d[k] for k in desired_order if k in d}
                    filtered_model_results      = _reorder(filtered_model_results)
                    filtered_net_results        = _reorder(filtered_net_results)
                    filtered_sim_results        = _reorder(filtered_sim_results)
                    filtered_sum_stats          = _reorder(filtered_sum_stats)
                    seekers_recs                = _reorder(seekers_recs)
                    filtered_app_loads          = _reorder(filtered_app_loads)
                    filtered_time_to_emps       = _reorder(filtered_time_to_emps)
                    filtered_wage_age_dfs       = _reorder(filtered_wage_age_dfs)
                    filtered_avg_wage_off_diffs = _reorder(filtered_avg_wage_off_diffs)
                    filtered_ue_stock_dfs         = _reorder(filtered_ue_stock_dfs)
                    # model_results_all_sims is keyed by raw IDs, not display names —
                    # reorder using name_map key order, not desired_order
                    _raw_order = list(name_map.keys())
                    model_results_all_sims        = {k: model_results_all_sims[k] for k in _raw_order if k in model_results_all_sims}
                    filtered_sim_results_all_sims = _reorder(filtered_sim_results_all_sims)

                    print("✓ Results loaded successfully from cache!")
                    
                else:
                    if dropbox:
                        raise KeyError("Dropbox should not run new models!")
                    if FORCE_RERUN:
                        print("FORCE_RERUN is True - running models from scratch...")
                    else:
                        print("No cached results found - running models...")
                    
                    # %%
                    calib_list = {
                        "nonbehav": {"otj": False, # has been run
                                            "cyc_otj": False, 
                                            "cyc_ue": False, 
                                            "disc": False,
                                            "delay": 25,
                                            "bus_confidence_dat": gdp_dat,
                                            "emp_apps": 1,
                                            'wage_prefs': False,
                                            'vac_data': vac_dat,
                                            'strict_rw': True},
                                "otj_nonbehav": {"otj": True, # has been run
                                            "cyc_otj": False, 
                                            "cyc_ue": False, 
                                            "disc": False, 
                                            "delay": 25,
                                            'wage_prefs': False,
                                            'emp_apps': 1,
                                            "bus_confidence_dat": gdp_dat,
                                                "vac_data": vac_dat,
                                                'strict_rw': True},
                                "otj_cyclical_e_disc": {"otj": True,
                                            "cyc_otj": True,
                                            "cyc_ue": False,
                                            "disc": True,
                                            "delay": 25,
                                            'emp_apps': 1,
                                            'wage_prefs': True,
                                            "bus_confidence_dat": gdp_dat,
                                            "vac_data": vac_dat,
                                            'strict_rw': False},
                                # "otj_cyclical_e_disc_strict_rw": {"otj": True,
                                #             "cyc_otj": True,
                                #             "cyc_ue": False,
                                #             "disc": True,
                                #             "delay": 25,
                                #             'emp_apps': 1,
                                #             'wage_prefs': True,
                                #             "bus_confidence_dat": gdp_dat,
                                #             "vac_data": vac_dat,
                                #             'strict_rw': True},
                                "otj_cyclical_e_disc_no_rw": {"otj": True,
                                            "cyc_otj": True,
                                            "cyc_ue": False,
                                            "disc": True,
                                            "delay": 25,
                                            'wage_prefs': False,
                                            'emp_apps': 1,
                                            "bus_confidence_dat": gdp_dat,
                                            "vac_data": vac_dat,
                                            'strict_rw': True},
                                "otj_disc": {"otj": True,
                                            "cyc_otj": False, 
                                            "cyc_ue": False, 
                                            "disc": True,
                                            "delay": 25,
                                            'emp_apps': 1,
                                            'wage_prefs': True,
                                            "bus_confidence_dat": gdp_dat,
                                            "vac_data": vac_dat,
                                            'strict_rw': False},
                                # "otj_disc_strict_rw": {"otj": True,
                                #             "cyc_otj": False, 
                                #             "cyc_ue": False, 
                                #             "disc": True,
                                #             "delay": 25,
                                #             'emp_apps': 1,
                                #             'wage_prefs': True,
                                #             "bus_confidence_dat": gdp_dat,
                                #             "vac_data": vac_dat,
                                #             'strict_rw': True},
                                "otj_disc_no_rw": {"otj": True,
                                            "cyc_otj": False, 
                                            "cyc_ue": False, 
                                            "disc": True,
                                            "delay": 25,
                                            "wage_prefs": False,
                                            'emp_apps': 1,
                                            "bus_confidence_dat": gdp_dat,
                                            "vac_data": vac_dat,
                                            'strict_rw': True}
                                }

                    # Initialize the results dictionaries
                    model_results = {}
                    net_results = {}
                    sim_results = {}
                    sim_results_all_sims = {}
                    sum_stats_list = {}
                    seekers_recs_list = {}
                    avg_wage_off_diffs = {}
                    app_loads = {}
                    time_to_emps = {}
                    transition_dfs = {}
                    vacs_finals = {}
                    ue_stock_dfs = {}

                    # ================================================================
                    # MAIN MODEL RUNNING LOOP (Single or Multi-simulation mode)
                    # ================================================================
                    
                    # Check if multi-simulation mode is enabled
                    if N_SIMULATIONS is not None and N_SIMULATIONS > 1:
                        if run == "steady_state":
                            continue
                        # ========== MULTI-SIMULATION MODE ==========
                        print(f"\n{'='*60}")
                        print(f"MULTI-SIMULATION MODE: Running each model {N_SIMULATIONS} times")
                        print(f"{'='*60}\n")
                        
                        # Initialize the results dictionaries
                        model_results = {}
                        model_results_all_sims = {}  # Store all individual runs
                        seekers_recs_multisim = {}
                        sim_results = {}
                        sim_results_all_sims = {}
                        transition_dfs = {}
                        ue_stock_dfs = {}
                        wage_age_dfs = {}
                        
                        # Loop through each model configuration
                        for name, item in calib_list.items():
                            print(f"\n{'='*60}")
                            print(f"Model: {name}")
                            print(f"{'='*60}")
                            
                            # Create a deep copy of the base parameters
                            base_params = deepcopy(params)
                            

                            if 'cyclical_e' in name:
                                calib_params = {'gamma_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_u"),"Value"].iloc[0],
                                            'd_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_u"),"Value"].iloc[0],
                                            'theta': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "theta"),"Value"].iloc[0],
                                            'gamma_v': None,
                                            'd_v': None}#,
                                            #'d_v':param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_v"),"Value"].iloc[0],
                                            #'gamma_v':param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_v"),"Value"].iloc[0]}
                            else:
                                calib_params = {'gamma_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_u"),"Value"].iloc[0],
                                            'd_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_u"),"Value"].iloc[0],
                                            'theta': None,
                                            'gamma_v': None,
                                            'd_v': None}#
                                            #'d_v': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_v"),"Value"].iloc[0], #0.01,
                                            #'gamma_v':param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_v"),"Value"].iloc[0]} #0.16}
                                
                            if vac_rate_setting in ["endog_vac_drift", "endog_cost_red"]:
                                calib_params.update({'d_v': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_v"),"Value"].iloc[0],
                                                    'gamma_v':param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_v"),"Value"].iloc[0]})

                            
                            # Update with the values from the calib_list
                            calib_params.update(item)
                            if run == "steady_state":
                                calib_params.update({"steady_state": True,
                                                    "time_steps": 1000,
                                                    "delay": 25})
                            else:
                                calib_params.update({"steady_state": False})
                            
                            # RUN MODEL MULTIPLE TIMES AND AGGREGATE
                            aggregated_grouped, grouped_list, ue_stock_combined, transition_combined, sim_record_combined, seekers_combined, wage_age_combined, sim_record_all_sims = run_model_multiple_times(
                                model_name=name,
                                base_params=base_params,
                                calib_params=calib_params,
                                calib_date=calib_date,
                                occ_ids=occ_ids,
                                n_sims=N_SIMULATIONS
                            )

                            # Store aggregated results
                            model_results[name] = aggregated_grouped
                            model_results_all_sims[name] = grouped_list
                            ue_stock_dfs[name] = ue_stock_combined
                            transition_dfs[name] = transition_combined
                            sim_results[name] = sim_record_combined
                            sim_results_all_sims[name] = sim_record_all_sims
                            seekers_recs_multisim[name] = seekers_combined
                            wage_age_dfs[name] = wage_age_combined
                        
                        print(f"\n{'='*60}")
                        print(f"Multi-simulation complete! Each model run {N_SIMULATIONS} times")
                        print(f"Results show mean ± {int(CONFIDENCE_LEVEL*100)}% CI")
                        print(f"{'='*60}\n")
                        
                        # For multi-simulation mode, we don't populate these detailed single-run results
                        net_results = {}
                        sum_stats_list = {}
                        seekers_recs_list = {}
                        avg_wage_off_diffs = {}
                        app_loads = {}
                        time_to_emps = {}
                        vacs_finals = {}
                        
                    else:
                        # ========== SINGLE-SIMULATION MODE (ORIGINAL BEHAVIOR) ==========
                        print(f"\n{'='*60}")
                        print(f"SINGLE-SIMULATION MODE: Running each model once")
                        print(f"{'='*60}\n")
                        
                        # Initialize the results dictionaries
                        model_results = {}
                        net_results = {}
                        sim_results = {}
                        sum_stats_list = {}
                        seekers_recs_list = {}
                        avg_wage_off_diffs = {}
                        app_loads = {}
                        time_to_emps = {}
                        vacs_finals = {}
                        wage_age_dfs = {}
                        ue_stock_dfs = {}

                        # Loop through each model configuration
                        for name, item in calib_list.items():
                            print(name)
                            # Create a deep copy of the base parameters
                            test_params = deepcopy(params)

                            if 'cyclical_e' in name:
                                calib_params = {'gamma_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_u"),"Value"].iloc[0],
                                            'd_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_u"),"Value"].iloc[0],
                                            'theta': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "theta"),"Value"].iloc[0],
                                            'gamma_v': None,
                                            'd_v': None}#,
                                            #'d_v':param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_v"),"Value"].iloc[0],
                                            #'gamma_v':param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_v"),"Value"].iloc[0]}
                            else:
                                calib_params = {'gamma_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_u"),"Value"].iloc[0],
                                            'd_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_u"),"Value"].iloc[0],
                                            'theta': None,
                                            'gamma_v': None,
                                            'd_v': None}#
                                            #'d_v': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_v"),"Value"].iloc[0], #0.01,
                                            #'gamma_v':param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_v"),"Value"].iloc[0]} #0.16}
                                
                            if vac_rate_setting in ["endog_vac_drift", "endog_cost_red"]:
                                calib_params.update({'d_v': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_v"),"Value"].iloc[0],
                                                    'gamma_v':param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_v"),"Value"].iloc[0]})

                            test_params.update(calib_params)

                            # Update with the values from the calib_list
                            test_params.update(item)
                            if run == "steady_state":
                                test_params.update({"steady_state": True,
                                                    "time_steps": 1000,
                                                    "delay": 25})
                            else:
                                test_params.update({"steady_state": False})

                            # Run the model — simple_res=True returns just a data dict; False returns 10 objects
                            result = run_single_local(**test_params)
                            if SIMPLE_RES:
                                sim_grouped = pd.DataFrame({
                                    'UER': result['UER'],
                                    'VACRATE': result['VACRATE']
                                })
                                sim_grouped['DATE'] = pd.date_range(start=calib_date[0],
                                                                    periods=len(sim_grouped), freq='MS')
                                # Set unused objects to empty/None so downstream storage works
                                sim_record = pd.DataFrame()
                                sim_net = None
                                sum_stats = None
                                seekers_rec = pd.DataFrame()
                                avg_wage_off_diff = pd.DataFrame()
                                app_loads_df = pd.DataFrame()
                                time_to_emp = pd.DataFrame()
                                transition_df = pd.DataFrame()
                                vacs_final = []
                                wage_age_df = pd.DataFrame()
                                ue_stock_df = pd.DataFrame()
                            else:
                                sim_record, sim_grouped, sim_net, sum_stats, seekers_rec, avg_wage_off_diff, app_loads_df, time_to_emp, vacs_final, wage_age_df, ue_stock_df, transition_df = result

                            if not SIMPLE_RES:
                                sim_grouped['DATE'] = pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(sim_grouped))
                                sim_grouped['LTUER'] = sim_grouped['LT Unemployed Persons'] / sim_grouped['Unemployment']
                                sim_grouped['AVGWAGE'] = sim_grouped['Total_Wages'] / sim_grouped['Employment']
                                sim_grouped["Seeker Composition"] = sim_grouped["Employed Seekers"] / (sim_grouped["Employed Seekers"] + sim_grouped["Unemployed Seekers"])
                                sim_record['LTUER'] = sim_record['LT Unemployed Persons'] / sim_record['Unemployment']
                                sim_record['UER'] = sim_record['Unemployment'] / sim_record['Workers']
                                sim_record['AVGWAGE'] = sim_record['Total_Wages'] / sim_record['Employment']
                                sim_record['VACRATE1'] = sim_record['Vacancies'] / sim_record['Target_Demand']
                                sim_record['VACRATE'] = sim_record['Vacancies'] / (sim_record['Vacancies'] + sim_record['Employment'])
                                sim_record['U_REL_WAGE_MEAN'] = sim_record['U_Rel_Wage'] / sim_record['UE_Transitions']
                                sim_record['E_REL_WAGE_MEAN'] = sim_record['E_Rel_Wage'] / sim_record['EE_Transitions']
                                sim_record['UE_Trans_Rate'] = sim_record['UE_Transitions'] / sim_record['Workers']
                                sim_record['EE_Trans_Rate'] = sim_record['EE_Transitions'] / sim_record['Workers']
                                sim_record = sim_record.merge(occ_ids, left_on='Occupation', right_on="id", how='left')
                                seekers_rec['DATE'] = pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(seekers_rec))
                                seekers_rec['Application Effort (U)'] = seekers_rec['Applications Sent (Unemployed)'] / seekers_rec['Unemployed Seekers']
                                seekers_rec['Application Effort (E)'] = seekers_rec['Applications Sent (Employed)'] / seekers_rec['Employed Seekers']

                            # Store the results
                            model_results[name] = sim_grouped
                            sim_results[name] = sim_record
                            net_results[name] = sim_net
                            sum_stats_list[name] = sum_stats
                            seekers_recs_list[name] = seekers_rec
                            avg_wage_off_diffs[name] = avg_wage_off_diff
                            app_loads[name] = app_loads_df
                            time_to_emps[name] = time_to_emp
                            transition_dfs[name] = transition_df
                            vacs_finals[name] = vacs_final
                            wage_age_dfs[name] = wage_age_df
                            ue_stock_dfs[name] = ue_stock_df
                        sum_stats_list[name] = sum_stats
                        seekers_recs_list[name] = seekers_rec
                        avg_wage_off_diffs[name] = avg_wage_off_diff
                        app_loads[name] = app_loads_df
                        time_to_emps[name] = time_to_emp
                        transition_dfs[name] = transition_df
                        vacs_finals[name] = vacs_final

                    # %%
                    # --- Apply plot labels to models ---
                    # name_map = {
                    #     "nonbehav": "Non-behavioural",
                    #     "otj_nonbehav": "Non-behavioural w. OTJ",
                    #     "otj_cyclical_e_disc": "Behavioural w. Cyc. OTJ w. RW",
                    #     "otj_disc": "Behavioural w.o. Cyc. OTJ w. RW",
                    #     "otj_cyclical_e_disc_no_rw": "Behavioural w. Cyc. OTJ w.o RW",
                    #     "otj_disc_no_rw": "Behavioural w.o. Cyc. OTJ w.o RW",
                    # }
                    desired_order = list(name_map.keys())

                    def filter_and_relabel(d, mapping, order):
                        return {mapping[k]: d[k] for k in order if k in d}

                    # Filter and relabel the main model results (works for both modes)
                    filtered_model_results = filter_and_relabel(model_results, name_map, desired_order)
                    
                    # Only filter detailed results if in single-simulation mode
                    if N_SIMULATIONS is None or N_SIMULATIONS <= 1:
                        filtered_net_results   = filter_and_relabel(net_results,   name_map, desired_order)
                        filtered_sim_results   = filter_and_relabel(sim_results,   name_map, desired_order)
                        filtered_sum_stats     = filter_and_relabel(sum_stats_list, name_map, desired_order)
                        seekers_recs    = filter_and_relabel(seekers_recs_list, name_map, desired_order)
                        filtered_app_loads    = filter_and_relabel(app_loads, name_map, desired_order)
                        filtered_time_to_emps = filter_and_relabel(time_to_emps, name_map, desired_order)
                        filtered_transition_dfs = filter_and_relabel(transition_dfs, name_map, desired_order)
                        filtered_wage_age_dfs = filter_and_relabel(wage_age_dfs, name_map, desired_order)
                        filtered_avg_wage_off_diffs = filter_and_relabel(avg_wage_off_diffs, name_map, desired_order)
                        filtered_ue_stock_dfs = filter_and_relabel(ue_stock_dfs, name_map, desired_order)

                    else:
                        # In multi-simulation mode, per-run details are not available
                        filtered_net_results = {}
                        filtered_sum_stats = {}
                        seekers_recs = filter_and_relabel(seekers_recs_multisim, name_map, desired_order)
                        filtered_app_loads = {}
                        filtered_time_to_emps = {}
                        filtered_wage_age_dfs = filter_and_relabel(wage_age_dfs, name_map, desired_order)
                        filtered_avg_wage_off_diffs = {}
                        # sim_results, ue_stock_dfs and transition_dfs are populated per model in the loop above
                        filtered_sim_results          = filter_and_relabel(sim_results,          name_map, desired_order)
                        filtered_sim_results_all_sims = filter_and_relabel(sim_results_all_sims, name_map, desired_order)
                        filtered_transition_dfs       = filter_and_relabel(transition_dfs,       name_map, desired_order)
                        filtered_ue_stock_dfs         = filter_and_relabel(ue_stock_dfs,         name_map, desired_order)

                    
                    # ===== SAVE ALL OBJECTS TO CACHE =====
                    if save_button:
                        print(f"Saving results to {cache_file}...")
                        cache_data = {
                            'filtered_model_results': filtered_model_results,
                            'filtered_net_results': filtered_net_results,
                            'filtered_sim_results': filtered_sim_results,
                            'filtered_sum_stats': filtered_sum_stats,
                            'seekers_recs': seekers_recs,
                            'filtered_app_loads': filtered_app_loads,
                            'filtered_time_to_emps': filtered_time_to_emps,
                            'filtered_transition_dfs': filtered_transition_dfs,
                            'filtered_wage_age_dfs': filtered_wage_age_dfs,
                            'filtered_avg_wage_off_diffs': filtered_avg_wage_off_diffs,
                            'filtered_ue_stock_dfs': filtered_ue_stock_dfs,
                            'n_simulations': N_SIMULATIONS,
                            'confidence_level': CONFIDENCE_LEVEL if N_SIMULATIONS else None,
                            # Raw per-sim economy-wide grouped DataFrames (small; enables combining caches)
                            'model_results_all_sims': model_results_all_sims if (N_SIMULATIONS and N_SIMULATIONS > 1) else {},
                            # Per-sim occupation-level DataFrames with sim_id (enables CIs for occ-level plots)
                            'filtered_sim_results_all_sims': filtered_sim_results_all_sims if (N_SIMULATIONS and N_SIMULATIONS > 1) else {},
                        }
                        
                        with open(cache_file, 'wb') as f:
                            pickle.dump(cache_data, f)
                        
                        if N_SIMULATIONS is not None and N_SIMULATIONS > 1:
                            print(f"✓ Results saved (Multi-simulation mode: {N_SIMULATIONS} runs per model)")
                        else:
                            print("✓ Results saved (Single-simulation mode)")



                # Skip all figure generation in multi-simulation mode —
                # the cache has been saved; plots are produced by the
                # diagnose_*.py scripts which read from the cache directly.
                # if N_SIMULATIONS is not None and N_SIMULATIONS > 1:
                #     continue

                # palette & colors
                from model_colors import MODEL_COLORS
                plot_colors = {k: v for k, v in MODEL_COLORS.items()
                            if k in name_map.values()}

                # %%

                if run == "steady_state":
                    # Call with smoothing
                    plot_uer_vac_steady_state(filtered_model_results, observation, 
                            plot_colors=plot_colors,
                            save=save_button, 
                            path=output_path,
                            delay_ref=50,
                            smooth=12,  # Will show "Smoothing over 10 points"
                            free_date_scale=True,
                            calib_date=calib_date,
                            suffix = "combined")
                    
                    plot_uer_vac_steady_state(filtered_model_results, observation, 
                            plot_colors=plot_colors,
                            sep_strings=sep_strings_filtered, 
                            sep=True, 
                            save=save_button, 
                            path=output_path,
                            delay_ref=50,
                            smooth=12,
                            free_date_scale=True,
                            calib_date=calib_date,
                            suffix = "single")

                else:
                    seekers_comp_obs_full = pd.read_csv('../data/behav_params/Eeckhout_Replication/comp_searchers_s_series_abm_validation.csv')

                    # Map quarters to first day of quarter
                    quarter_map = {"Q1": "-01-01", "Q2": "-04-01", "Q3": "-07-01", "Q4": "-10-01"}

                    # Convert quarterly to monthly time series and interpolate missing values

                    # Replace and convert
                    seekers_comp_obs_full['DATE'] = (
                        seekers_comp_obs_full['date']
                        .replace(quarter_map, regex=True)
                        .pipe(pd.to_datetime)
                    )

                    # Set DATE as index for resampling
                    seekers_comp_obs_full = seekers_comp_obs_full.set_index('DATE')

                    # Resample to monthly frequency, keeping the value at the start of each quarter
                    monthly = seekers_comp_obs_full.resample('MS').asfreq()

                    # Interpolate missing values linearly
                    monthly['comp_searchers_s'] = monthly['comp_searchers_s'].interpolate(method='linear')

                    # Reset index to restore DATE as a column
                    seekers_comp_obs_full = monthly.reset_index()

                    seekers_comp_obs_full = seekers_comp_obs_full.rename(columns={"comp_searchers_s": "Seeker Composition"})
                    seekers_comp_obs_full['DATE'] = pd.to_datetime(seekers_comp_obs_full['DATE'])
                    seekers_comp_obs = seekers_comp_obs_full[(seekers_comp_obs_full['DATE'] >= calib_date[0]) & (seekers_comp_obs_full['DATE'] <= calib_date[1])]

                    ###############################################################################################################
                    #################### UER-Vac w Error ##########################################################################
                    ###############################################################################################################
                    plot_uer_vac(filtered_model_results, observation,
                            sep_strings=sep_strings_filtered,
                                sep=True, save=save_button, path=output_path,
                                free_date_scale=(run == "steady_state"),
                                calib_date=calib_v_line,
                                plot_colors=plot_colors)
                    
                    plot_uer_vac(filtered_model_results, observation,
                            #sep_strings=sep_strings_filtered,
                                sep=False, save=save_button, path=output_path,
                                free_date_scale=(run == "steady_state"),
                                calib_date=calib_v_line,
                                plot_colors=plot_colors)

                    if not SIMPLE_RES:
                        ###############################################################################################################
                        #################### Relative Wages ###########################################################################
                        ###############################################################################################################
                        #plot_rel_wages(filtered_model_results, save = save_button, path = output_path, freq = 'ME', colors = plot_colors, unemp_only = True, suffix = "")
                        #for lowess_fit in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
                        #   plot_rel_wages(filtered_model_results, save = save_button, path = output_path, freq = 'ME', colors = plot_colors, unemp_only = True, lowess = True, lowess_frac = lowess_fit, suffix = f"lowess_{lowess_fit}")
        
        
                        ###############################################################################################################
                        #################### LTUER Time Series ########################################################################
                        ###############################################################################################################
                        plot_ltuer(filtered_model_results, observation,
                                sep_strings=sep_strings_filtered,
                                sep=True, save=save_button, path=output_path,
                                calib_date=calib_v_line,
                                colors=plot_colors)

                        plot_ltuer(filtered_model_results, observation,
                                #sep_strings=sep_strings_filtered,
                                sep=False, save=save_button, path=output_path,
                                calib_date=calib_v_line,
                                colors=plot_colors)
                        
                        ###############################################################################################################
                        #################### Beveridge Curve ##########################################################################
                        ###############################################################################################################
                        plot_bev_curve_color(filtered_model_results, observation,
                                #sep_strings=sep_strings_filtered,
                                    sep=False, save=save_button, path=output_path, smooth = 3, colors=plot_colors)
                        
        
                        ###############################################################################################################
                        #################### Current vs. Target Demand ################################################################
                        ###############################################################################################################
                        plot_cd_vs_td(filtered_model_results, save=save_button, path=output_path, colors = plot_colors, calib_v_line=calib_v_line)
        
                        ###############################################################################################################
                        #################### Transition Rates #########################################################################
                        ###############################################################################################################
                        all_rates_new = pd.read_csv('data/transition_rates_96_24.csv')
                        all_rates_new = all_rates_new[(all_rates_new['date'] >= calib_date[0]) & (all_rates_new['date'] <= calib_date[1])]
                        all_rates_new['DATE'] = pd.to_datetime(all_rates_new['date'])
        
                        plot_trans_rates(filtered_model_results, observation = all_rates_new, save = save_button, path = output_path, colors = plot_colors, calib_v_line=calib_v_line)
        
        
                        ###############################################################################################################
                        #################### Hires-Seps Rates #########################################################################
                        ###############################################################################################################
        
                        hires_seps_rate(filtered_model_results, jolts = jolts, save = save_button, path = output_path, colors = plot_colors, calib_v_line=calib_v_line)
        
                        ###############################################################################################################
                        #################### Seeker Composition - Line ################################################################
                        ###############################################################################################################
                        plot_seeker_comp_line(filtered_model_results, seekers_comp_obs, save=save_button , path=output_path, colors = plot_colors, calib_v_line=calib_v_line)

                        ###############################################################################################################
                        #################### Applications + Seeker Composition (combined, kept alongside the individual figures) #####
                        ###############################################################################################################
                        plot_applications_and_composition(seekers_recs, filtered_model_results, seekers_comp_obs,
                                                           save=save_button, path=output_path, colors=plot_colors,
                                                           calib_v_line=calib_v_line)

                        ###############################################################################################################
                        #################### Seeker Composition #######################################################################
                        ###############################################################################################################
                        plot_seeker_comp(seekers_recs, seekers_comp_obs, sep=True, share=True,
                                            save=save_button , path=output_path, colors = plot_colors,
                                            calib_v_line=calib_v_line)
        
                        ###############################################################################################################
                        #################### Market Tightness ################################################################
                        ###############################################################################################################
                    
                        plot_market_tightness(filtered_model_results,
                                tightness,
                                save = save_button,
                                path = output_path,
                                colors = plot_colors,
                                adjusted = False,
                                calib_v_line=calib_v_line)

                        plot_market_tightness(filtered_model_results,
                                tightness,
                                save = save_button,
                                path = output_path,
                                colors = plot_colors,
                                adjusted = True,
                                calib_v_line=calib_v_line)

                        plot_wage_comparison(filtered_sim_results, wages,
                            calib_date,
                            save = save_button,
                            path = output_path, colors = plot_colors,
                            calib_v_line=calib_v_line)


                        # ── Run ────────────────────────────────────────────────────────────────────────
                        plot_wage_comparison_dev(
                            filtered_sim_results,
                            wages,
                            calib_date,
                            save=save_button,
                            path = output_path,
                            colors=plot_colors,
                            calib_v_line=calib_v_line,
                        )

                        plot_wage_comparison_dev_error(
                            filtered_sim_results,
                            wages,
                            calib_date,
                            save=save_button,
                            path=output_path,
                            colors=plot_colors,
                            calib_v_line=calib_v_line,
                        )
                        plot_wage_comparison_comoment(
                            filtered_sim_results,
                            wages,
                            calib_date,
                            save=save_button,
                            path=output_path,
                            colors=plot_colors,
                            rolling_window=24,
                            calib_v_line=calib_v_line,
                        )


                        plot_wage_percentile_ts(filtered_wage_age_dfs,
                            calib_date=calib_date,
                            save=save_button, path=output_path,
                            colors=plot_colors,
                            calib_v_line=calib_v_line)



                        # Load your yearly survey data
                        survey_df = pd.read_csv("data/ltuer_cps_data_new.csv")
        
                        # Annual ranking of models by aggregate unemployment-duration
                        # fit to CPS (Wasserstein distance, the same gap-between-CDFs
                        # quantity plot_ltuer_cdf_with_observed shows informally), plus
                        # the raw-magnitude companion over time.
                        plot_duration_wasserstein_rank_heatmap(
                            ue_stock_dfs=filtered_ue_stock_dfs,
                            survey_df=survey_df,
                            calib_date=calib_date,
                            delay=25,
                            colors=plot_colors,
                            save=save_button,
                            path=output_path
                        )

                        plot_duration_wasserstein_over_time(
                            ue_stock_dfs=filtered_ue_stock_dfs,
                            survey_df=survey_df,
                            calib_date=calib_date,
                            delay=25,
                            colors=plot_colors,
                            save=save_button,
                            path=output_path,
                            calib_v_line=calib_v_line
                        )

                        # Monthly CPS duration data (finer-grained Wasserstein diagnostics)
                        monthly_dur_path = "data/ltuer_cps_data_monthly.csv"
                        if os.path.exists(monthly_dur_path):
                            survey_df_monthly = pd.read_csv(monthly_dur_path)
                            plot_duration_wasserstein_rank_heatmap_monthly(
                                ue_stock_dfs=filtered_ue_stock_dfs,
                                survey_df=survey_df_monthly,
                                calib_date=calib_date,
                                delay=25,
                                colors=plot_colors,
                                save=save_button,
                                path=output_path
                            )
                            plot_duration_wasserstein_over_time_monthly(
                                ue_stock_dfs=filtered_ue_stock_dfs,
                                survey_df=survey_df_monthly,
                                calib_date=calib_date,
                                delay=25,
                                colors=plot_colors,
                                save=save_button,
                                path=output_path,
                                calib_v_line=calib_v_line
                            )


                        # ###############################################################################################################
                        # #################### Applications per unemployed job-seeker ###################################################
                        # ###############################################################################################################
                        
    
                        if N_SIMULATIONS is not None and N_SIMULATIONS > 1:
                            # Multi-sim: plot mean line + 95% CI ribbon
                            fig_u, ax_u = plt.subplots()
                            fig_e, ax_e = plt.subplots()
                            for name, item in seekers_recs.items():
                                col = plot_colors.get(name)
                                ax_u.plot(item['DATE'], item['Application Effort (U)'], color=col, label=name)
                                if 'Application Effort (U)_ci_lower' in item.columns:
                                    ax_u.fill_between(item['DATE'],
                                                        item['Application Effort (U)_ci_lower'],
                                                        item['Application Effort (U)_ci_upper'],
                                                        color=col, alpha=0.15)
                                ax_e.plot(item['DATE'], item['Application Effort (E)'], color=col, label=name)
                                if 'Application Effort (E)_ci_lower' in item.columns:
                                    ax_e.fill_between(item['DATE'],
                                                        item['Application Effort (E)_ci_lower'],
                                                        item['Application Effort (E)_ci_upper'],
                                                        color=col, alpha=0.15)
                            _seeker_colors = {name: plot_colors.get(name) for name in seekers_recs.keys()}
                            for ax, ylabel, fname in [
                                (ax_u, 'Applications per Unemployed Seeker', 'applications_per_unemployed_seeker.png'),
                                (ax_e, 'Applications per Employed Seeker',   'applications_per_employed_seeker.png'),
                            ]:
                                ax.set_xlabel('Year'); ax.set_ylabel(ylabel)
                                ax.set_title('Application Rate Over Time')
                                add_shared_legend(ax.get_figure(), _seeker_colors, ncol=3)
                                save_plot_standard(ax.get_figure(), os.path.join(output_path, fname))
                                plt.close(ax.get_figure())

        
                        if N_SIMULATIONS is None or N_SIMULATIONS <= 1:

                            ###############################################################################################################
                            #################### Applications per unemployed job-seeker ###################################################
                            ###############################################################################################################

                            fig, ax = plt.subplots()
                            for i, (name, item) in enumerate(seekers_recs.items()):
                                if name == "Non-behavioural":
                                    ax.plot(item['DATE'][3::4], item['Application Effort (U)'][3::4], color=plot_colors[name], marker='o', label=f"{name}", linestyle="")
                                elif name == "Non-behavioural w. OTJ":
                                    ax.plot(item['DATE'][1::4], item['Application Effort (U)'][1::4], color=plot_colors[name], marker='o', label=f"{name}", linestyle="")
                                else:
                                    ax.plot(item['DATE'], item['Application Effort (U)'], marker='o', label=name, color=plot_colors[name])
                            ax.set_xlabel('Year')
                            ax.set_ylabel('Applications per Unemployed Seeker')
                            ax.set_title('Application Rate Over Time')
                            add_shared_legend(fig, {name: plot_colors[name] for name in seekers_recs.keys()}, ncol=3)
                            save_plot_standard(fig, os.path.join(output_path, "applications_per_unemployed_seeker.png"))
                            plt.close(fig)

                            fig, ax = plt.subplots()
                            for i, (name, item) in enumerate(seekers_recs.items()):
                                ax.plot(item['DATE'], item['Application Effort (E)'], marker='o', label=name, color=plot_colors[name])
                            ax.set_xlabel('Year')
                            ax.set_ylabel('Applications per Employed Seeker')
                            ax.set_title('Application Rate Over Time')
                            add_shared_legend(fig, {name: plot_colors[name] for name in seekers_recs.keys()}, ncol=3)
                            save_plot_standard(fig, os.path.join(output_path, "applications_per_employed_seeker.png"))
                            plt.close(fig)
        
                            ###############################################################################################################
                            #################### Occupation-Level Market Tightness ########################################################
                            ###############################################################################################################
                            plot_occ_market_tightness(filtered_sim_results,
                                                    calib_date=calib_date,
                                                    save=save_button,
                                                    path=output_path)
        
                            plot_tightness_wage_corr(filtered_sim_results,
                                                    calib_date=calib_date,
                                                    save=save_button,
                                                    path=output_path)
        
                            plot_wage_comparison(filtered_sim_results, wages,
                                                calib_date,
                                                save = save_button,
                                                path = output_path, colors = plot_colors,
                                                calib_v_line=calib_v_line)


                            plot_wage_vacancy_grid(filtered_sim_results, calib_date, save = save_button,
                                                path = output_path, colors = plot_colors,
                                                calib_v_line=calib_v_line)
        
                            
                            ###############################################################################################################
                            #################### LTUER Distributions ########################################################################
                            ###############################################################################################################
        
                            plot_ltuer_dist(filtered_net_results, gender=False, 
                                            names=list(filtered_net_results.keys()), 
                                            save=save_button, path=output_path, colors = plot_colors)
                            
                            plot_ltuer_cdf(filtered_net_results, 
                                            names=list(filtered_net_results.keys()), 
                                            save=save_button, path=output_path, colors = plot_colors)
                            
                        
                            # Load your yearly survey data
                            survey_df = pd.read_csv("data/ltuer_cps_data_new.csv")
        
                            if run == "covid_oos":
                                survey_reference_year = 2024
                            elif run == "base":
                                survey_reference_year = 2019
                            # Use specific year
                            plot_ltuer_cdf_with_observed(
                                net_dict=filtered_net_results,
                                names=list(filtered_net_results.keys()),
                                survey_df=survey_df,
                                survey_year=survey_reference_year,  # Changed from survey_date
                                save=True,
                                path=output_path,
                                colors=plot_colors
                            )


                            try:
                                plot_ltuer_kde(filtered_net_results, 
                                                names=list(filtered_net_results.keys()), 
                                                save=save_button, path=output_path, colors = plot_colors)
                            except np.linalg.LinAlgError as e:
                                print("ATTN: SKIPPING KDE PLOT")
        
                            try:
                                plot_ltuer_hist_kde(filtered_net_results, 
                                                names=list(filtered_net_results.keys()), 
                                                save=save_button, path=output_path, colors = plot_colors)
                            except np.linalg.LinAlgError as e:
                                print("ATTN: SKIPPING HISTOGRAM & KDE PLOT")
        
                            plot_ltuer_boxplot(filtered_net_results, 
                                            names=list(filtered_net_results.keys()), 
                                            save=save_button, path=output_path, colors = plot_colors)
                            
                            plot_ltuer_log(filtered_net_results, 
                                            names=list(filtered_net_results.keys()), 
                                            save=save_button, path=output_path, colors = plot_colors)
        
        
                            ###############################################################################################################
                            #################### Seeker Composition #######################################################################
                            ###############################################################################################################
                            plot_seeker_comp(seekers_recs, seekers_comp_obs, sep=True, share=True,
                                            save=save_button , path=output_path, colors = plot_colors,
                                            calib_v_line=calib_v_line)
        

                        ###############################################################################################################
                        #################### Wage & Age Plots (base and covid_oos only) ##################################################
                        ###############################################################################################################
                            if run in ("base", "covid_oos"):
                                plot_wage_dist_by_year(filtered_wage_age_dfs,
                                                    save=save_button, path=output_path)


                                plot_age_dist_by_year(filtered_wage_age_dfs,
                                                    save=save_button, path=output_path)

                                plot_occ_wage_growth(filtered_wage_age_dfs,
                                                    save=save_button, path=output_path)

                                plot_wage_vs_age_growth_rates(filtered_wage_age_dfs,
                                                            save=save_button, path=output_path, colors=plot_colors)

                                plot_wage_age_corr_over_time(filtered_wage_age_dfs,
                                                            save=save_button, path=output_path, colors=plot_colors,
                                                            calib_date=calib_date, calib_v_line=calib_v_line)

                                plot_wage_age_growth_scatter(filtered_wage_age_dfs,
                                                            save=save_button, path=output_path, colors=plot_colors)

                        ###############################################################################################################
                        #################### Retirement Rate Grid (base and covid_oos only; runs at any N) #############################
                        ###############################################################################################################
                        if run in ("base", "covid_oos"):
                            # model_results_all_sims only exists (and is keyed by raw model
                            # IDs, not display names) in multi-simulation mode -- relabel to
                            # display names so it lines up with filtered_model_results' keys.
                            # Inlined rather than calling filter_and_relabel: that helper is
                            # only defined in the fresh-run branch of the cache-load if/else
                            # above, so it's undefined whenever results come from cache.
                            if N_SIMULATIONS is not None and N_SIMULATIONS > 1 and model_results_all_sims:
                                _retirement_all_sims = {name_map[k]: model_results_all_sims[k]
                                                         for k in name_map.keys() if k in model_results_all_sims}
                            else:
                                _retirement_all_sims = None
                            plot_retirement_rate_grid(filtered_model_results, calib_date,
                                                    save=save_button, path=output_path, colors=plot_colors,
                                                    calib_v_line=calib_v_line,
                                                    model_results_all_sims=_retirement_all_sims)

            # time_opens = [job.time_open for job in vacs_finals['otj_disc']]

            # print(vacs_finals['otj_disc'][0].time_open)
            # print(time_opens)
            # plt.hist(time_opens)
            # plt.axvline(np.mean(time_opens), color='red', linestyle='--')#, label=f"Mean = {mean_wage:.2f}")
            # plt.legend()
            # plt.close()

            # # Build a DataFrame from your objects
            # df = pd.DataFrame([
            #     {"occupation": job.occupation_id, "time_open": job.time_open}
            #     for job in vacs_finals["otj_disc"]
            # ])

            # order = df.groupby("occupation")["time_open"].median().sort_values().index

            # plt.figure(figsize=(12, 6))
            # sns.stripplot(    data=df,
            #     x="occupation",
            #     y="time_open",
            #     order=order,
            #     jitter=True,            # jitter to avoid vertical overlaps
            #     alpha=0.6,
            #     size=3,
            #     color="steelblue")
            # plt.ylim(0, 6) 
            # plt.xticks(rotation=90)
            # plt.tight_layout()
            # plt.close()

                        ###############################################################################################################
                        #################### Time to Re-Employment ####################################################################
                        ###############################################################################################################
                        # Prepare the models to plot (skip "Non-behavioural")
                        plots = [(name, df) for name, df in filtered_time_to_emps.items()] # if "Non-behavioural" not in name]

                        # Calculate the maximum value across all models
                        max_value = 0
                        for name, df in plots:
                            current_max = df['UEDuration'].max()
                            if current_max > max_value:
                                max_value = current_max

                        # Add some padding (optional, e.g., 5% extra)
                        y_max = max_value * 1.05  # or just use max_value without padding

                        # If you want to force exactly 4 subplots (2x2)
                        nrows, ncols = 2, 4
                        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
                        axes = axes.flatten()

                        # If there are more than 4 items, only plot first 4 (change if you want different behavior)
                        if len(plots) > len(axes):
                            print(f"Warning: {len(plots)} series found but only {len(axes)} subplots available — plotting first {len(axes)}.")
                        plots_to_plot = plots[:len(axes)]

                        for ax, (name, df) in zip(axes, plots_to_plot):
                            # Prepare data: list of arrays, one per time step
                            time_steps = sorted(df['Time Step'].unique())
                            data = [df[df['Time Step'] == t]['UEDuration'].values for t in time_steps]

                            threshold = 13

                            # Shade regions (adjust the upper bound to y_max)
                            ax.axhspan(0, threshold, color="lightgreen", alpha=0.3)
                            ax.axhspan(threshold, y_max, color="lightyellow", alpha=0.3)

                            # Violin plot
                            ax.violinplot(data, positions=time_steps, showmeans=False, showmedians=True, widths=1.2)

                            # Horizontal line at threshold
                            xmin = min(time_steps) - 0.5
                            xmax = max(time_steps) + 0.5
                            ax.hlines(y=threshold, xmin=xmin, xmax=xmax)

                            ax.set_xlabel('Time Step')
                            ax.set_ylabel('Duration Until Re-Employment')
                            ax.set_ylim(0, y_max)  # Use calculated max instead of 80
                            ax.set_title(f'{name}')

                            # Extend x-limits a bit to make room for annotation to the right
                            ax.set_xlim(xmin, max(time_steps) + 12)

                            # Annotations (adjust y-positions based on y_max)
                            ax.text(
                                x=max(time_steps) + 5, y=(threshold + 5) / 2,
                                s="Effort High", color="green", fontsize=12, va="center"
                            )
                            ax.text(
                                x=max(time_steps) + 5, y=threshold / 2,
                                s="Res. Wage Declines", color="green", fontsize=12, va="center"
                            )
                            ax.text(
                                x=max(time_steps) + 5, y=(threshold + y_max) / 2,  # Adjusted
                                s="Effort Low", color="orange", fontsize=12, va="center"
                            )
                            ax.text(
                                x=max(time_steps) + 5, y=(threshold + y_max * 0.95) / 2,  # Adjusted
                                s="Res. Wage Declines", color="green", fontsize=12, va="center"
                            )

                        # Turn off any unused subplots (if less than 4 series)
                        for ax in axes[len(plots_to_plot):]:
                            ax.axis('off')

                        fig.suptitle("Distribution of Duration (mos.) Until Re-Employment", size=18)
                        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave room for suptitle

                        if save_button:
                            save_plot_standard(fig, os.path.join(output_path, "behavioural_regimes_time_to_reemp.png"),
                                                aspect_ratio=10/18, width_frac=18/TEXTWIDTH)
                            plt.close(fig)
                        else:
                            plt.show()
                        
                        ###############################################################################################################
                        #################### Time to Re-Employment Distribution #######################################################
                        ###############################################################################################################
                        tmp = pd.read_csv(path + "dRC_Replication/data/ipums_variables.csv")
                        fig, axes = plt.subplots(3, 8, figsize=(24, 18))
                        model_names = filtered_time_to_emps.keys()
                        for col, i in enumerate(model_names):
                            # 1. Bar plot (all occupations)
                            mean_ue_by_origin = (
                                filtered_time_to_emps[i]
                                .groupby('OriginOccupation', as_index=True)['UEDuration']
                                .mean()
                                .sort_values(ascending=False)
                            )
                            x = np.arange(len(mean_ue_by_origin))
                            axes[0, col].scatter(x, mean_ue_by_origin.values)
                            axes[0, col].set_xticks(x)
                            axes[0, col].set_xticklabels(mean_ue_by_origin.index.astype(str), rotation=90)
                            axes[0, col].set_ylabel('Mean periods unemployed before hire')
                            axes[0, col].set_xlabel('Origin Occupation')
                            axes[0, col].set_title(f'All Occupations')
                            axes[0,col].set_ylim(0,10)

                            # 2. Bar plot (top 30 occupations)
                            mean_ue_by_origin_short = mean_ue_by_origin.iloc[0:30]
                            x_short = np.arange(len(mean_ue_by_origin_short))
                            # Get labels from tmp for each occupation, fallback to Occ {id}
                            labels_short = [tmp.loc[occ, 'label'] if occ in tmp.index else f'Occ {occ}' for occ in mean_ue_by_origin_short.index]
                            axes[2, col].bar(x_short, mean_ue_by_origin_short.values, color="violet")
                            axes[2, col].set_xticks(x_short)
                            axes[2, col].set_xticklabels(labels_short, rotation=90)
                            axes[2, col].set_ylabel('Mean periods unemployed before hire')
                            axes[2, col].set_xlabel('Origin Occupation')
                            axes[2, col].set_title(f'Top 30 Occupations')
                            axes[2,col].set_ylim(0,10)


                            # 3. Line plot (mean UE duration by origin occupation and time step)
                            mean_ue_by_origin_time = (
                                filtered_time_to_emps[i]
                                .groupby(['OriginOccupation', 'Time Step'])['UEDuration']
                                .mean()
                                .reset_index()
                            )
                            # mean_ue_by_origin_time = mean_ue_by_origin_time.loc[
                            #     mean_ue_by_origin_time['OriginOccupation'].isin(mean_ue_by_origin_short.index)
                            # ]
                            for occ in mean_ue_by_origin_time['OriginOccupation'].unique():
                                group = mean_ue_by_origin_time[mean_ue_by_origin_time['OriginOccupation'] == occ]
                                label = tmp.loc[occ, 'label'] if occ in tmp.index else f'Occ {occ}'
                                axes[1, col].plot(group['Time Step'], group['UEDuration'], label=label)
                            axes[1, col].set_xlabel('Time Step')
                            axes[1, col].set_ylabel('Mean UE→E Spell Length')
                            axes[1, col].set_title(f'Mean UE Duration Over Time')
                            axes[1,col].set_ylim(0,70)

                            #if col == 0:
                            #    axes[2, col].legend(ncol=1, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
                            # Column label
                            axes[0, col].annotate(i, xy=(0.5, 1.15), xycoords='axes fraction', ha='center', fontsize=16, fontweight='bold')

                        fig.suptitle("Time to Re-Employment (Months Unemployed Prior to Hiring) by Origin Occupation")
                        fig.tight_layout()
                        if save_button:
                            save_plot_standard(fig, os.path.join(output_path, "time_to_reemp_dist.png"),
                                                aspect_ratio=18/24, width_frac=24/TEXTWIDTH)
                            plt.close(fig)
                        else:
                            plt.show()



                        ###############################################################################################################
                        #################### Plot Network #############################################################################
                        ###############################################################################################################
                        plot_network_from_matrix(mod_data['A'], 
                                                    layout = "kamada_kawai", directed=False, threshold=0.05,
                                                    #color_key=color_key,
                                                    save = save_button,
                                                    path=output_path,
                                                    title="Occupation Mobility Network")



                        ###############################################################################################################
                        #################### Net Entry Plots ##########################################################################
                        ###############################################################################################################
                        # Plot heatmaps for all models
                        plot_net_entry_heatmap_grid(
                            filtered_time_to_emps, 
                            label_map=tmp, 
                            title="Net entry heatmap (no self-flows) - All Models",
                            save=save_button, 
                            path=output_path
                        )

                        # Plot stability for all models
                        plot_net_entry_stability_grid(
                            filtered_time_to_emps, 
                            label_map=tmp, 
                            topk=100, 
                            title="Net entry stability (mean vs std) - All Models",
                            save=save_button, 
                            path=output_path
                        )


                        ###############################################################################################################
                        #################### UER - LTUER Scatterplot comparison #######################################################
                        ###############################################################################################################
                        if nw == "full_omn_currently_not_working":
                            # Example usage:
                            # occ_ltuers = pd.read_csv(path+"data/highlev_occ_ltuers.csv")
                            # occ_ltuers = occ_ltuers.groupby('occupation')['ltuer'].mean().reset_index()
                            # Pasted from data/occ_macro_vars/CPS_LTUER/occ_ltuer_observed.csv
                            occ_ltuer_obs = pd.read_csv(path + "data/occ_uer_ltuer_observed.csv",
                                dtype={
                                        'SOC_major_adj': str,
                                        'SOC_minor_adj': str,
                                        'SOC_broad_adj': str,
                                        'SOC2010_adj': str,
                                    }
                                )

                            soc_labs = pd.read_csv('/Users/ebbamark/Documents/Documents - Nuff-Malham/GitHub/transition_abm//data/occ_macro_vars/OEWS/soc_major_labels.csv', dtype={"soc_code": str})

                            plot_occupation_uer_grid2(filtered_sim_results, occ_ltuer_obs, soc_labs, save=save_button, path=output_path)

                            # %%
                            for i in {'absolute', 'percentage'}:
                                for j in {True, False}:
                                    plot_ltuer_difference_heatmap(filtered_sim_results, occ_ltuer_obs, difference_type = i, abs_value = j, save=save_button, path=output_path)

                        ###############################################################################################################
                        #################### Gender Wage Gaps #########################################################################
                        ###############################################################################################################
                        gender_income = pd.read_csv(path+"data/gender_income_distribution_usa_2022.csv", delimiter=',', thousands = ",")

                        gender_income['Income Bracket'] = gender_income['Income Bracket'].str.replace(' or loss', '')
                        gender_income['Income Bracket'] = gender_income['Income Bracket'].str.replace(' or more', '')
                        gender_income['Ceiling'] = gender_income['Income Bracket'].str[-7:]
                        gender_income['Ceiling'] = gender_income['Ceiling'].str.replace(',', '')
                        gender_income['Ceiling'] = gender_income['Ceiling'].str.replace('$', '')
                        gender_income['Ceiling'] = gender_income['Ceiling'].str.replace(' ', '')
                        gender_income['Ceiling'] = gender_income['Ceiling'].astype('Int64')
                        gender_income.loc[0:7, 'Bracket'] = "20k "
                        gender_income.loc[8:12, 'Bracket'] = "20k - 39k"
                        gender_income.loc[13:14, 'Bracket'] = "40k - 49k"
                        gender_income.loc[15, 'Bracket'] = "50k - 54k"
                        gender_income.loc[16, 'Bracket'] = "55k - 64k"
                        gender_income.loc[17, 'Bracket'] = "65 - 74k"
                        gender_income.loc[18, 'Bracket'] = "75k - 95k"
                        gender_income.loc[19, 'Bracket'] = ">= 100k"
                        brackets = gender_income.groupby(['Bracket']).sum().reset_index()

                        barWidth = 1
                        bls_fig, bls_ax = plt.subplots()
                        br1 = np.arange(len(brackets['Bracket'])) + 1
                        br2 = br1

                        ############## GENDER WAGE GAPS ####################
                        ####################################################
                        def plot_gender_gaps(net_dict, sep = False, save = False, path = None):
                            """
                            Function to plot the gender wage disttribution across models
                            """
                            n = len(net_dict)
                            if sep:
                                cols = 2
                            else:
                                max_cols = 2
                                cols = min(n, max_cols)
                            
                            rows = math.ceil(n / cols)

                            # Find the global max wage across all models
                            global_max_wage = 0
                            for net in net_dict.values():
                                for occ in net:
                                    w_wages = [wrkr.wage for wrkr in occ.list_of_employed if wrkr.female]
                                    m_wages = [wrkr.wage for wrkr in occ.list_of_employed if not(wrkr.female)]
                                    max_wage = max(w_wages + m_wages) if (w_wages + m_wages) else 0
                                    if max_wage > global_max_wage:
                                        global_max_wage = max_wage

                            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False, sharey = True)
                            _gg_width_frac = (5 * cols) / TEXTWIDTH
                            _gg_aspect = rows / cols
                            axes = axes.flatten()  # Flatten to make indexing easy

                            for i, (name, net) in enumerate(net_dict.items()):
                                ax = axes[i]
                                emp_counter = 0
                                women = 0
                                men = 0

                                w_wages = []
                                m_wages = []

                                w_wage = 0
                                m_wage = 0

                                for occ in net:
                                    emp_counter += len(occ.list_of_employed)
                                    women += len([wrkr for wrkr in occ.list_of_employed if wrkr.female])
                                    men += len([wrkr for wrkr in occ.list_of_employed if not(wrkr.female)])
                                    w_wages.extend([wrkr.wage for wrkr in occ.list_of_employed if wrkr.female])
                                    m_wages.extend([wrkr.wage for wrkr in occ.list_of_employed if not(wrkr.female)])
                                    w_wage += np.nansum([wrkr.wage for wrkr in occ.list_of_employed if wrkr.female])
                                    m_wage += np.nansum([wrkr.wage for wrkr in occ.list_of_employed if not wrkr.female])


                                women_arr = np.array(w_wages, dtype=float)
                                men_arr = np.array(m_wages, dtype=float)
                                # Count NaNs
                                num_nan_women = np.isnan(women_arr).sum()
                                num_nan_men = np.isnan(men_arr).sum()

                                # Print warnings if any are found
                                if num_nan_women > 0:
                                    print(f"{num_nan_women} NA value(s) found in women's wages for '{name}' — ignored in calculations.")
                                if num_nan_men > 0:
                                    print(f"{num_nan_men} NA value(s) found in men's wages for '{name}' — ignored in calculations.")

                                t= " \n" + " \n" +  "Female share of employed: " + str(round((women/emp_counter)*100)) + "% \n" + "Mean Female Wage: $" + str(round(w_wage/women)) + "\n" + "Mean Male Wage: $" + str(round(m_wage/men)) + "\n" + "Gender wage gap: " + str(round(100*(1 - (w_wage/women)/(m_wage/men)))) + "%" + "\n" + "--------------------"

                                n_bins = 10
                                women_arr = np.array(w_wages)
                                men_arr = np.array(m_wages)

                                # We can set the number of bins with the *bins* keyword argument.
                                ax.hist(women_arr, bins=n_bins, alpha = 0.3, color = 'purple', label = 'Women', fill = True, hatch = '.')
                                ax.hist(men_arr, bins=n_bins, alpha = 0.3, label = 'Men', color = 'green', fill = True, hatch = '.')  
                                ax.axvline(np.nanmean(women_arr), color='purple', linestyle='dashed', linewidth=1, label = 'Women Avg.')
                                ax.axvline(np.nanmean(men_arr), color='green', linestyle='dashed', linewidth=1, label = 'Men Avg.')
                                #ax.legend(loc='upper right') 
                                ax.annotate(
                                        t,
                                        xy=(0.5, 0.8),
                                        xycoords='axes fraction',
                                        fontsize=7,
                                        verticalalignment='center',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
                                    )
                                ax.set_title(name)
                                ax.set_xlim([0, global_max_wage*0.9])
                                if i == 0:
                                    handles, labels = ax.get_legend_handles_labels()

                            fig.supxlabel("Wage")  # Shared x-axis label
                            fig.suptitle('Distribution of Male and Female Wages', fontsize = 15) 
                            fig.subplots_adjust(bottom=0.1)
                            # Add single legend below the title, above the plots
                            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
                                    ncol=len(net_dict)//2, frameon=True, fontsize=10)


                            if save:
                                save_plot_standard(fig, f'{path}gender_wage_gaps.jpg',
                                                    aspect_ratio=_gg_aspect, width_frac=_gg_width_frac)
                            plt.close(fig)


                        # Make the plot
                        bls_ax.bar(br1, brackets['Full-Time Females'], color ='lightblue', width = barWidth, alpha = 0.8,
                                label ='Women')
                        bls_ax.bar(br2, brackets['Full-Time Males'], color ='orange', width = barWidth, alpha = 0.3,
                                label ='Men')

                        # Adding Xticks
                        bls_ax.set_xlabel('Income Bracket')
                        bls_ax.set_ylabel('Full-Time Employees per Bracket')
                        bls_ax.ticklabel_format(useOffset=False, style='plain')
                        bls_ax.set_xticks([r + barWidth for r in range(len(brackets['Bracket']))])
                        bls_ax.set_xticklabels(brackets['Bracket'])

                        bls_ax.set_title('Distribution of Male and Female Wages in US Labour Market')
                        bls_ax.legend()
                        save_plot_standard(bls_fig, f'{output_path}bls_data_gender_wage_gaps.jpg')
                        plt.close(bls_fig)

                        womens_wage = (gender_income['Ceiling'] * gender_income['Full-Time Females']).sum()/(gender_income['Full-Time Females'].sum())
                        mens_wage = (gender_income['Ceiling'] * gender_income['Full-Time Males']).sum()/(gender_income['Full-Time Males'].sum())

                        if N_SIMULATIONS == 1:
                            plot_gender_gaps(filtered_net_results, save = True, path = output_path)

                        ############## GENDER PAY GAP TIME SERIES ####################
                        def plot_gender_pay_gap_ts(mod_results_dict, save=False, path=None,
                                                colors=None, freq='ME', lowess=False, lowess_frac=0.2):
                            """
                            Plots the mean and median gender pay gap as time series for each model run.
                            Gap = 1 - (female wage / male wage), expressed as %.
                            Solid lines = mean-based gap; dashed lines = median-based gap.
                            """
                            from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_smooth

                            if colors is None:
                                default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
                                colors = dict(zip(mod_results_dict.keys(), default_colors))

                            fig, ax = plt.subplots()

                            for name, res in mod_results_dict.items():
                                if 'Gender_Pay_Gap' not in res.columns:
                                    print(f"Warning: 'Gender_Pay_Gap' not found for '{name}'. Re-run the model.")
                                    continue

                                data = res.set_index('DATE').resample(freq).mean() if freq != 'ME' else res.copy()
                                dates = np.array(data.index if freq != 'ME' else data['DATE'])
                                mean_gap  = data['Gender_Pay_Gap'] * 100
                                has_median = 'Gender_Median_Pay_Gap' in data.columns
                                if has_median:
                                    median_gap = data['Gender_Median_Pay_Gap'] * 100

                                if lowess:
                                    sm = lowess_smooth(mean_gap.values, np.arange(len(mean_gap)), frac=lowess_frac)
                                    ax.plot(dates, mean_gap.values, color=colors[name], alpha=0.15, linewidth=0.8)
                                    ax.plot(dates, sm[:, 1], color=colors[name], label=f"{name} (mean)", linewidth=2)
                                    if has_median:
                                        sm2 = lowess_smooth(median_gap.values, np.arange(len(median_gap)), frac=lowess_frac)
                                        ax.plot(dates, median_gap.values, color=colors[name], alpha=0.15, linewidth=0.8, linestyle='--')
                                        ax.plot(dates, sm2[:, 1], color=colors[name], label=f"{name} (median)", linewidth=2, linestyle='--')
                                else:
                                    ax.plot(dates, mean_gap.values, color=colors[name], label=f"{name} (mean)", linewidth=2)
                                    if has_median:
                                        ax.plot(dates, median_gap.values, color=colors[name], label=f"{name} (median)", linewidth=2, linestyle='--')

                            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
                            ax.set_ylabel("Gender Pay Gap (%)\n(% female wage below male wage)")
                            ax.set_xlabel("Date")
                            ax.set_title("Gender Pay Gap Over Time (solid = mean, dashed = median)")
                            fig.tight_layout()
                            add_shared_legend(fig, colors, ncol=3)

                            if save:
                                save_plot_standard(fig, f'{path}gender_pay_gap_ts.jpg')
                            else:
                                plt.show()
                            plt.close(fig)

                        plot_gender_pay_gap_ts(filtered_model_results, save=save_button, path=output_path, colors=plot_colors)


                        ###############################################################################################################
                        #################### TS Metrics Table #########################################################################
                        ###############################################################################################################
                        if run in ["base", "covid_oos"]:#, "covid_oos"]:
                            mukoyama_obs_full = pd.read_csv('../data/behav_params/Mukoyama_Replication/monthly_search_ts.csv')
                            mukoyama_obs_full['DATE'] = mukoyama_obs_full['year'].astype(str) + "-" + mukoyama_obs_full['month'].astype(str).str.zfill(2)  + "-01"

                            mukoyama_obs = mukoyama_obs_full[(mukoyama_obs_full['DATE'] >= calib_date[0]) & (mukoyama_obs_full['DATE'] <= calib_date[1])]
                            mukoyama_obs = mukoyama_obs.rename(columns={"value_smooth": "Application Effort (U)"})
                            mukoyama_obs['DATE'] = pd.to_datetime(mukoyama_obs['DATE'])

                            plt.plot(mukoyama_obs_full['DATE'], mukoyama_obs_full['value_smooth'], linestyle='-', color='orange', label="Observed Search Effort - Smoothed")
                            plt.plot(mukoyama_obs_full['DATE'], mukoyama_obs_full['value'], linestyle='-', color='purple', label="Observed Search Effort - Raw")
                            plt.title("Mukoyama Imputed Search Effort - Raw & Smoothed Minutes")
                            plt.close()

                            # %%
                            comp_series = observation
                            jolts_obs = jolts[(jolts['DATE'] >= calib_date[0]) & (jolts['DATE'] <= calib_date[1])].reset_index()
                            jolts_obs = jolts_obs.rename(columns={"HIRESRATE": "Hires Rate", "SEPSRATE": "Separations Rate"})
                            all_rates_new = all_rates_new.rename(columns={"UE": "UE_Trans_Rate", "EE": "EE_Trans_Rate"})
                            comp_series = comp_series.merge(jolts_obs, on='DATE', how='left')
                            comp_series = comp_series.merge(all_rates_new, on = 'DATE')
                            comp_series = comp_series.merge(mukoyama_obs, on = 'DATE')
                            comp_series = comp_series.merge(seekers_comp_obs, on = 'DATE', how = 'left')


                            def _diff_corr(sim_s, obs_s):
                                """
                                Pearson correlation of period-over-period changes, not raw
                                levels. Raw-level correlation is inflated by any shared trend
                                (e.g. the COVID unemployment spike) even when the series don't
                                actually move together period to period — same convention used
                                for the occupation-level LTUER fit in plot_funs.py.
                                """
                                diff_sim = np.diff(np.asarray(sim_s, dtype=float))
                                diff_obs = np.diff(np.asarray(obs_s, dtype=float))
                                if len(diff_sim) < 2 or np.std(diff_sim) == 0 or np.std(diff_obs) == 0:
                                    return float('nan')
                                return round(float(pearsonr(diff_sim, diff_obs)[0]), 3)

                            def _sign_conc(sim_s, obs_s):
                                """
                                Fraction of periods where sim and observed move in the same
                                direction. Periods where either series has zero change are
                                excluded (direction is undefined). Returns a value in [0, 1];
                                0.5 is chance level.
                                """
                                diff_sim = np.diff(np.asarray(sim_s, dtype=float))
                                diff_obs = np.diff(np.asarray(obs_s, dtype=float))
                                mask = (diff_sim != 0) & (diff_obs != 0)
                                if mask.sum() < 2:
                                    return float('nan')
                                return round(float(np.mean(np.sign(diff_sim[mask]) == np.sign(diff_obs[mask]))), 3)

                            def compute_time_series_metrics(res_dict, seekers_info_dict, obs_dict, variables, oos = False):
                                if oos:
                                    obs_dict = obs_dict[(obs_dict['DATE'] >= "2019-05-01") & (obs_dict['DATE'] <= calib_date[1])].reset_index()
                                    print(len(obs_dict))
                                rows = []
                                for model_name, sim_df in res_dict.items():
                                    if oos:
                                        sim_df = sim_df[(sim_df['DATE'] >= "2019-05-01") & (sim_df['DATE'] <= calib_date[1])].reset_index()
                                        print(len(sim_df))
                                    for var in variables:
                                        print(var)
                                        if var == "Application Effort (U)":
                                            if oos:
                                                sim_df_temp = seekers_info_dict[model_name]
                                                sim_df_temp = sim_df_temp[(sim_df_temp['DATE'] >= "2019-05-01") & (sim_df_temp['DATE'] <= calib_date[1])].reset_index()
                                                print(len(sim_df_temp))
                                            else:
                                                sim_df_temp = seekers_info_dict[model_name]
                                            sim_series = sim_df_temp[var].values if var in sim_df_temp else None
                                        else:
                                            print(sim_df.columns)
                                            sim_series = sim_df[var].values if var in sim_df else None
                                        print(sim_series)
                                        obs_series = obs_dict[var].values if var in obs_dict else None
                                        # TRIM ALL SIMULATED SERIES TO MATCH OBSERVED LENGTH
                                        if sim_series is not None and obs_series is not None and len(sim_series) != len(obs_series):
                                            sim_series = sim_series[:len(obs_series)]
                                            # VACANCY RATE DF IS TYPICALLY 1 UNIT LONGER THAN THE OBSERVED SERIES
                                        if sim_series is None or obs_series is None or len(sim_series) != len(obs_series):
                                            # Shorten vac_df to the same length as gdp_dat using a moving average (if needed)
                                            print(len(sim_series))
                                            print(len(obs_series))
                                            #continue
                                            raise KeyError(f"Missing or mismatched data for {var} in {model_name}")
                                        if var == "Application Effort (U)":
                                            # Only keep correlation
                                            row = {
                                                'Model': model_name,
                                                'Variable': var,
                                                'Correlation': _diff_corr(sim_series, obs_series),
                                                'Concordance': _sign_conc(sim_series, obs_series)
                                            }
                                        else:
                                            # Full set of metrics
                                            row = {
                                                'Model': model_name,
                                                'Variable': var,
                                                'Sim. Mean': round(float(np.mean(sim_series)), 3),
                                                'Obs. Mean': round(float(np.mean(obs_series)), 3),
                                                #'Variance (Sim)': round(float(np.var(sim_series)), 3),
                                                #'Variance (Obs)': round(float(np.var(obs_series)), 3),
                                                'SSE': round(float(np.sum((sim_series - obs_series) ** 2)), 3),
                                                'Correlation': _diff_corr(sim_series, obs_series),
                                                'Concordance': _sign_conc(sim_series, obs_series)
                                            }

                                        rows.append(row)
                                return pd.DataFrame(rows)

                            OOS_SPLIT = "2019-05-01"

                            def _sim_obs_metrics(sim_s, obs_s, var):
                                """Return dict of metrics for one sim-obs pair."""
                                if sim_s is None or obs_s is None or len(sim_s) == 0:
                                    return {}
                                min_len = min(len(sim_s), len(obs_s))
                                sim_s, obs_s = sim_s[:min_len], obs_s[:min_len]
                                metrics = {}
                                if var != "Application Effort (U)":
                                    metrics['Mean Dist.'] = round(abs(float(np.mean(sim_s)) - float(np.mean(obs_s))), 3)
                                metrics['Correlation']  = _diff_corr(sim_s, obs_s)
                                metrics['Concordance']  = _sign_conc(sim_s, obs_s)
                                return metrics

                            def _extract_series(sim_df, seekers_df, obs_df, var, date_lo=None, date_hi=None):
                                """Extract aligned sim and obs arrays, optionally filtered by date range."""
                                s = sim_df.copy()
                                o = obs_df.copy()
                                k = seekers_df.copy() if seekers_df is not None else None
                                if date_lo is not None:
                                    s = s[s['DATE'] >= date_lo]
                                    o = o[o['DATE'] >= date_lo]
                                    if k is not None:
                                        k = k[k['DATE'] >= date_lo]
                                if date_hi is not None:
                                    s = s[s['DATE'] < date_hi]
                                    o = o[o['DATE'] < date_hi]
                                    if k is not None:
                                        k = k[k['DATE'] < date_hi]
                                if var == "Application Effort (U)":
                                    if k is None or var not in k.columns:
                                        return None, None
                                    return k[var].values, o[var].values if var in o.columns else None
                                return (s[var].values if var in s.columns else None,
                                        o[var].values if var in o.columns else None)

                            def compute_time_series_metrics_multisim(all_sims_dict, seekers_info_dict, obs_dict, variables, oos=False):
                                """
                                Compute fit metrics per simulation run then aggregate (mean ± SD).
                                When oos=True, produces separate IS (pre-2019-05-01) and OOS columns.
                                all_sims_dict: dict of model_name -> list of per-sim grouped DataFrames
                                """
                                rows = []
                                for model_name, sim_list in all_sims_dict.items():
                                    seekers_df = seekers_info_dict.get(model_name)
                                    if seekers_df is not None and seekers_df.empty:
                                        seekers_df = None

                                    for var in variables:
                                        is_vals = {}   # metric -> list of per-sim values
                                        oos_vals = {}

                                        for sim_df in sim_list:
                                            if oos:
                                                # In-sample window
                                                s_is, o_is = _extract_series(sim_df, seekers_df, obs_dict, var,
                                                                               date_hi=OOS_SPLIT)
                                                for metric, val in _sim_obs_metrics(s_is, o_is, var).items():
                                                    is_vals.setdefault(metric, []).append(val)
                                                # Out-of-sample window
                                                s_oos, o_oos = _extract_series(sim_df, seekers_df, obs_dict, var,
                                                                                date_lo=OOS_SPLIT)
                                                for metric, val in _sim_obs_metrics(s_oos, o_oos, var).items():
                                                    oos_vals.setdefault(metric, []).append(val)
                                            else:
                                                s, o = _extract_series(sim_df, seekers_df, obs_dict, var)
                                                for metric, val in _sim_obs_metrics(s, o, var).items():
                                                    is_vals.setdefault(metric, []).append(val)

                                        row = {'Model': model_name, 'Variable': var}
                                        if oos:
                                            for metric, vals in is_vals.items():
                                                if vals:
                                                    row[f'IS {metric}'] = round(float(np.mean(vals)), 3)
                                                    row[f'IS {metric} SD'] = round(float(np.std(vals)), 3)
                                            for metric, vals in oos_vals.items():
                                                if vals:
                                                    row[f'OOS {metric}'] = round(float(np.mean(vals)), 3)
                                                    row[f'OOS {metric} SD'] = round(float(np.std(vals)), 3)
                                        else:
                                            for metric, vals in is_vals.items():
                                                if vals:
                                                    row[metric] = round(float(np.mean(vals)), 3)
                                                    row[f'{metric} SD'] = round(float(np.std(vals)), 3)
                                        rows.append(row)
                                return pd.DataFrame(rows)

                            BEST_COLOR = "yellow!25"
                            EXCELLENT_COLOR = "red!25"
                            BEST_GREY = "black!30"
                            EXCELLENT_GREY = "black!15"



                            def _fmt_num(x):
                                return "--" if pd.isna(x) else f"{x:.3f}"

                            def _color_val(x, color=BEST_COLOR):
                                return "--" if pd.isna(x) else f"\\cellcolor{{{color}}}{x:.3f}"

                            def highlight_best(df: pd.DataFrame, threshold_pct=0.05) -> pd.DataFrame:
                                """
                                Highlight cells with two-tier coloring:
                                - RED: Best value that outperforms second-best by more than threshold_pct
                                - YELLOW: Values within threshold_pct of the best

                                Parameters:
                                -----------
                                df : pd.DataFrame
                                    The metrics dataframe
                                threshold_pct : float
                                    Percentage threshold (default 0.05 = 5%)
                                """
                                df = df.copy()

                                # columns we may format
                                metric_cols = [
                                    "Sim. Mean", "Obs. Mean",
                                    "Variance (Sim)", "Variance (Obs)",
                                    "Correlation", "Mean Dist.",
                                    "IS Correlation", "IS Mean Dist.",
                                    "OOS Correlation", "OOS Mean Dist.",
                                ]

                                # format all numeric cells first (as strings)
                                for c in df.columns:
                                    if c in metric_cols:
                                        df[c] = df[c].apply(_fmt_num)

                                # now compute winners per Variable and recolor
                                for var in df["Variable"].unique():
                                    sub = df[df["Variable"] == var]

                                    # Mean difference -> color Mean (Sim) with smallest |Mean(Sim)-Mean(Obs)|
                                    if {"Mean (Sim)", "Mean (Obs)"}.issubset(sub.columns):
                                        mean_sim = pd.to_numeric(sub["Mean (Sim)"].replace("--", np.nan))
                                        mean_obs = pd.to_numeric(sub["Mean (Obs)"].replace("--", np.nan))
                                        diffs = (mean_sim - mean_obs).abs()
                                    
                                        valid_diffs = diffs.dropna()
                                        if not valid_diffs.empty and len(valid_diffs) >= 2:
                                            best_diff = valid_diffs.min()
                                            best_idx = valid_diffs.idxmin()
                                            second_best = valid_diffs.nsmallest(2).iloc[-1]
                                        
                                            # Check if best outperforms second-best by more than threshold
                                            is_excellent = second_best > best_diff * (1 + threshold_pct)
                                        
                                            # Color the best
                                            raw_val = mean_sim.loc[best_idx]
                                            if var == "UER" or var == 'Unemployment Rate':
                                                df.loc[best_idx, "Mean (Sim)"] = _color_val(raw_val, EXCELLENT_GREY if is_excellent else BEST_GREY)
                                            else:
                                                df.loc[best_idx, "Mean (Sim)"] = _color_val(raw_val, EXCELLENT_COLOR if is_excellent else BEST_COLOR)

                                            # Color others within threshold (yellow only)
                                            threshold = best_diff * (1 + threshold_pct)
                                            for idx in valid_diffs.index:
                                                if idx != best_idx and diffs.loc[idx] <= threshold:
                                                    raw_val = mean_sim.loc[idx]
                                                    if var == "UER" or var == 'Unemployment Rate':
                                                        df.loc[idx, "Mean (Sim)"] = _color_val(raw_val, BEST_GREY)
                                                    else:
                                                        df.loc[idx, "Mean (Sim)"] = _color_val(raw_val, BEST_COLOR)
                                        elif len(valid_diffs) == 1:
                                            # Only one value - make it yellow
                                            idx = valid_diffs.index[0]
                                            raw_val = mean_sim.loc[idx]
                                            if var == "UER" or var == 'Unemployment Rate':
                                                df.loc[idx, "Mean (Sim)"] = _color_val(raw_val, BEST_GREY)
                                            else:
                                                df.loc[idx, "Mean (Sim)"] = _color_val(raw_val, BEST_COLOR)

                                    # Variance difference -> color Variance (Sim)
                                    if {"Variance (Sim)", "Variance (Obs)"}.issubset(sub.columns):
                                        var_sim = pd.to_numeric(sub["Variance (Sim)"].replace("--", np.nan))
                                        var_obs = pd.to_numeric(sub["Variance (Obs)"].replace("--", np.nan))
                                        diffs = (var_sim - var_obs).abs()
                                    
                                        valid_diffs = diffs.dropna()
                                        if not valid_diffs.empty and len(valid_diffs) >= 2:
                                            best_diff = valid_diffs.min()
                                            best_idx = valid_diffs.idxmin()
                                            second_best = valid_diffs.nsmallest(2).iloc[-1]
                                        
                                            is_excellent = second_best > best_diff * (1 + threshold_pct)
                                        
                                            raw_val = var_sim.loc[best_idx]
                                            if var == 'UER':
                                                df.loc[best_idx, "Variance (Sim)"] = _color_val(raw_val, EXCELLENT_GREY if is_excellent else BEST_GREY)
                                            else:
                                                df.loc[best_idx, "Variance (Sim)"] = _color_val(raw_val, EXCELLENT_COLOR if is_excellent else BEST_COLOR)
                                        
                                            threshold = best_diff * (1 + threshold_pct)
                                            for idx in valid_diffs.index:
                                                if idx != best_idx and diffs.loc[idx] <= threshold:
                                                    raw_val = var_sim.loc[idx]
                                                    if var == "UER" or var == 'Unemployment Rate':
                                                        df.loc[idx, "Variance (Sim)"] = _color_val(raw_val, BEST_GREY)
                                                    else:
                                                        df.loc[idx, "Variance (Sim)"] = _color_val(raw_val, BEST_COLOR)
                                        elif len(valid_diffs) == 1:
                                            idx = valid_diffs.index[0]
                                            raw_val = var_sim.loc[idx]
                                            if var == "UER" or var == 'Unemployment Rate':
                                                df.loc[idx, "Variance (Sim)"] = _color_val(raw_val, BEST_GREY)
                                            else:
                                                df.loc[idx, "Variance (Sim)"] = _color_val(raw_val, BEST_COLOR)

                                    # Mean Dist. -> lowest wins
                                    if "Mean Dist." in sub.columns:
                                        _md = pd.to_numeric(sub["Mean Dist."].replace("--", np.nan))
                                        _valid_md = _md.dropna()
                                        if not _valid_md.empty and len(_valid_md) >= 2:
                                            _best_md = _valid_md.min()
                                            _best_md_idx = _valid_md.idxmin()
                                            _second_md = _valid_md.nsmallest(2).iloc[-1]
                                            _exc_md = _second_md > _best_md * (1 + threshold_pct)
                                            df.loc[_best_md_idx, "Mean Dist."] = _color_val(float(_md.loc[_best_md_idx]), EXCELLENT_COLOR if _exc_md else BEST_COLOR)
                                            _thr_md = _best_md * (1 + threshold_pct)
                                            for _idx_md in _valid_md.index:
                                                if _idx_md != _best_md_idx and _md.loc[_idx_md] <= _thr_md:
                                                    df.loc[_idx_md, "Mean Dist."] = _color_val(float(_md.loc[_idx_md]), BEST_COLOR)
                                        elif len(_valid_md) == 1:
                                            df.loc[_valid_md.index[0], "Mean Dist."] = _color_val(float(_valid_md.iloc[0]), BEST_COLOR)

                                    # Correlation -> highest wins
                                    if "Correlation" in sub.columns:
                                        corr = pd.to_numeric(sub["Correlation"].replace("--", np.nan))
                                        valid_corr = corr.dropna()
                                    
                                        if not valid_corr.empty and len(valid_corr) >= 2:
                                            best_corr = valid_corr.max()
                                            best_idx = valid_corr.idxmax()
                                            second_best = valid_corr.nlargest(2).iloc[-1]
                                        
                                            # Best is "excellent" if second-best is more than threshold% lower
                                            is_excellent = second_best < best_corr * (1 - threshold_pct)
                                        
                                            # Color the best
                                            raw_val = float(corr.loc[best_idx])
                                            if var == "UER" or var == 'Unemployment Rate':
                                                df.loc[best_idx, "Correlation"] = _color_val(raw_val, EXCELLENT_GREY if is_excellent else BEST_GREY)
                                            else:
                                                df.loc[best_idx, "Correlation"] = _color_val(raw_val, EXCELLENT_COLOR if is_excellent else BEST_COLOR)
                                        
                                            # Color others within threshold (yellow only)
                                            threshold = best_corr * (1 - threshold_pct)
                                            for idx in valid_corr.index:
                                                if idx != best_idx and corr.loc[idx] >= threshold:
                                                    raw_val = float(corr.loc[idx])
                                                    if var == "UER" or var == 'Unemployment Rate':
                                                        df.loc[idx, "Correlation"] = _color_val(raw_val, BEST_GREY)
                                                    else:
                                                        df.loc[idx, "Correlation"] = _color_val(raw_val, BEST_COLOR)
                                        elif len(valid_corr) == 1:
                                            idx = valid_corr.index[0]
                                            raw_val = float(corr.loc[idx])
                                            if var == "UER" or var == 'Unemployment Rate':
                                                df.loc[idx, "Correlation"] = _color_val(raw_val, BEST_GREY)
                                            else:
                                                df.loc[idx, "Correlation"] = _color_val(raw_val, BEST_COLOR)

                                    # IS/OOS Correlation and Mean Dist. (multisim OOS path)
                                    for _prefix in ("IS ", "OOS "):
                                        _corr_col  = f"{_prefix}Correlation"
                                        _md_col    = f"{_prefix}Mean Dist."

                                        # Mean Dist. -> lowest wins
                                        if _md_col in sub.columns:
                                            _md = pd.to_numeric(sub[_md_col].replace("--", np.nan))
                                            _valid = _md.dropna()
                                            if len(_valid) >= 2:
                                                _best = _valid.min(); _best_idx = _valid.idxmin()
                                                _second = _valid.nsmallest(2).iloc[-1]
                                                _exc = _second > _best * (1 + threshold_pct)
                                                df.loc[_best_idx, _md_col] = _color_val(float(_md.loc[_best_idx]), EXCELLENT_COLOR if _exc else BEST_COLOR)
                                                _thr = _best * (1 + threshold_pct)
                                                for _idx in _valid.index:
                                                    if _idx != _best_idx and _md.loc[_idx] <= _thr:
                                                        df.loc[_idx, _md_col] = _color_val(float(_md.loc[_idx]), BEST_COLOR)
                                            elif len(_valid) == 1:
                                                df.loc[_valid.index[0], _md_col] = _color_val(float(_valid.iloc[0]), BEST_COLOR)

                                        # Correlation -> highest wins
                                        if _corr_col in sub.columns:
                                            _hv = pd.to_numeric(sub[_corr_col].replace("--", np.nan))
                                            _valid = _hv.dropna()
                                            if len(_valid) >= 2:
                                                _best = _valid.max(); _best_idx = _valid.idxmax()
                                                _second = _valid.nlargest(2).iloc[-1]
                                                _exc = _second < _best * (1 - threshold_pct)
                                                df.loc[_best_idx, _corr_col] = _color_val(float(_hv.loc[_best_idx]), EXCELLENT_COLOR if _exc else BEST_COLOR)
                                                _thr = _best * (1 - threshold_pct)
                                                for _idx in _valid.index:
                                                    if _idx != _best_idx and _hv.loc[_idx] >= _thr:
                                                        df.loc[_idx, _corr_col] = _color_val(float(_hv.loc[_idx]), BEST_COLOR)
                                            elif len(_valid) == 1:
                                                df.loc[_valid.index[0], _corr_col] = _color_val(float(_valid.iloc[0]), BEST_COLOR)

                                return df

                            # ---- use it ----
                            _vars = ['LTUER', "Hires Rate", "Separations Rate",
                                     'Application Effort (U)', 'Seeker Composition']
                            if N_SIMULATIONS is not None and N_SIMULATIONS > 1 and model_results_all_sims:
                                # model_results_all_sims uses raw IDs; seekers_recs uses display names
                                # after filter_and_relabel — remap so the lookup inside the function works
                                _inv_name_map = {v: k for k, v in name_map.items()}
                                _seekers_by_raw_id = {_inv_name_map.get(k, k): v
                                                      for k, v in seekers_recs.items()}
                                ts_table = compute_time_series_metrics_multisim(
                                    model_results_all_sims, _seekers_by_raw_id, comp_series,
                                    _vars, oos=run == "covid_oos"
                                )
                            else:
                                ts_table = compute_time_series_metrics(
                                    filtered_model_results, seekers_recs, comp_series,
                                    _vars, oos=run == "covid_oos"
                                )

                            # Map raw variable names to pretty/display names (minimal change)
                            var_display_map = {
                                'VACRATE': 'Vacancy Rate',
                                'UER': 'Unemployment Rate',
                                'LTUER': 'Long-term Unemployment Rate',
                                'Hires Rate': 'Hires Rate',
                                'Separations Rate': 'Separations Rate',
                                'UE_Trans_Rate': 'UE Transition Rate',
                                'EE_Trans_Rate': 'EE Transition Rate',
                                'Application Effort (U)': 'Application Effort (U)',  # keep or shorten to 'App Effort (U)'
                                'Seeker Composition': 'Seeker Composition'
                            }

                            # apply the mapping (keeps original name if not found in map)
                            ts_table['Variable'] = ts_table['Variable'].map(var_display_map).fillna(ts_table['Variable'])
                            # multisim path uses raw keys from model_results_all_sims; single-sim path already has clean names
                            if N_SIMULATIONS is not None and N_SIMULATIONS > 1 and model_results_all_sims:
                                ts_table['Model'] = ts_table['Model'].map(name_map).fillna(ts_table['Model'])

                            # Sort rows by desired_order so models appear in the correct sequence
                            _order_index = {name: i for i, name in enumerate(desired_order)}
                            ts_table['_model_order'] = ts_table['Model'].map(_order_index).fillna(len(desired_order))
                            ts_table = (ts_table.sort_values('_model_order', kind='stable')
                                                .drop(columns=['_model_order'])
                                                .reset_index(drop=True))

                            # Compute per-variable ranks for each metric column (1 = best)
                            _rank_lower  = {'Mean Dist.', 'IS Mean Dist.',  'OOS Mean Dist.'}
                            _rank_higher = {'Correlation',
                                            'IS Correlation',
                                            'OOS Correlation'}
                            # Stores (rank, n_valid_models) per (row_index, column) so the
                            # rank-to-color mapping below knows how many models it's ranking
                            # against (e.g. Application Effort lacks a Mean Dist. column).
                            _rank_store = {}
                            for _var in ts_table['Variable'].unique():
                                _vmask = ts_table['Variable'] == _var
                                for _rcol in ts_table.columns:
                                    if _rcol in _rank_lower or _rcol in _rank_higher:
                                        _asc   = _rcol in _rank_lower
                                        _rvals = pd.to_numeric(ts_table.loc[_vmask, _rcol], errors='coerce')
                                        _n_valid = int(_rvals.notna().sum())
                                        _ranks = _rvals.rank(method='min', ascending=_asc, na_option='keep')
                                        for _ri, _rr in _ranks.items():
                                            if pd.notna(_rr):
                                                _rank_store[(_ri, _rcol)] = (int(_rr), _n_valid)

                            # Save full model assignment before deduplication (needed for color lookup)
                            _model_col_full = ts_table['Model'].copy()

                            # Replace repeated model names with empty string after first occurrence per model
                            ts_table['Model'] = ts_table['Model'].where(~ts_table['Model'].duplicated(), '')

                            highlighted = highlight_best(ts_table)

                            # Apply model color shading to the Model column
                            from model_colors import MODEL_COLORS as _MC_TABLE

                            def _latex_model_color_name(n):
                                return 'model' + n.replace(' ', '').replace('.', '').replace('-', '').replace('/', '')

                            _model_definecolors = []
                            for _mn in _model_col_full.unique():
                                if _mn in _MC_TABLE:
                                    _r, _g, _b = [int(x * 255) for x in _MC_TABLE[_mn]]
                                    _model_definecolors.append(
                                        f'\\definecolor{{{_latex_model_color_name(_mn)}}}{{RGB}}{{{_r},{_g},{_b}}}'
                                    )

                            for _idx in highlighted.index:
                                _full_name = _model_col_full.loc[_idx]
                                _cell_text = highlighted.loc[_idx, 'Model']
                                if _full_name in _MC_TABLE:
                                    _cn = _latex_model_color_name(_full_name)
                                    highlighted.loc[_idx, 'Model'] = f'\\cellcolor{{{_cn}!25}}{_cell_text}'

                            # Multisim: fold SD into \makecell{value \\ (sd)}
                            _is_multisim_table = (N_SIMULATIONS is not None and N_SIMULATIONS > 1
                                                  and model_results_all_sims)
                            if _is_multisim_table:
                                import re as _re

                                def _merge_makecell(val, sd):
                                    if val == '--' or (isinstance(val, float) and pd.isna(val)):
                                        return '--'
                                    try:
                                        sd_str = f'({float(sd):.3f})'
                                    except Exception:
                                        return str(val)
                                    # Hoist \cellcolor{} outside \makecell so background covers full cell
                                    _m = _re.match(r'(\\cellcolor\{[^}]+\})(.*)', str(val))
                                    if _m:
                                        return f'{_m.group(1)}\\makecell{{{_m.group(2)} \\\\ {sd_str}}}'
                                    return f'\\makecell{{{val} \\\\ {sd_str}}}'

                                # Drop obs-mean SD cols (always 0, not informative)
                                _drop = [c for c in highlighted.columns if 'Obs. Mean SD' in c]
                                highlighted.drop(columns=_drop, inplace=True)

                                # Merge each remaining (metric, metric SD) pair
                                _sd_pairs = [(c, c + ' SD') for c in list(highlighted.columns)
                                             if c + ' SD' in highlighted.columns]
                                for _mc, _sc in _sd_pairs:
                                    highlighted[_mc] = highlighted.apply(
                                        lambda r, c=_mc, s=_sc: _merge_makecell(r[c], r[s]), axis=1
                                    )
                                    highlighted.drop(columns=[_sc], inplace=True)

                            # Color metric cells by rank: top 3 green, then the remaining
                            # ranks split evenly across yellow / orange / red (worst).
                            # Overrides any \cellcolor highlight_best already applied, since
                            # the rank gradient replaces the binary best/second-best scheme.
                            import re as _re_rank

                            # Top-3 shades: rank 1 = pure pastel green, ranks 2-3 progressively
                            # mixed with yellow (but capped below the actual yellow band).
                            _GREEN_SHADES = {1: 'green!25', 2: 'green!70!yellow!25', 3: 'green!40!yellow!25'}

                            def _rank_color(r, n):
                                if n <= 3 or r <= 3:
                                    return _GREEN_SHADES.get(r, 'green!25')
                                _remaining = max(n - 3, 1)
                                _pos = r - 3
                                _idx = min(2, (_pos - 1) * 3 // _remaining)
                                return ['yellow!25', 'orange!25', 'red!25'][_idx]

                            def _apply_rank_color(s, r, n):
                                if not isinstance(s, str) or s in ('--', ''):
                                    return s
                                _stripped = _re_rank.sub(r'^\\cellcolor\{[^}]*\}', '', s)
                                return f'\\cellcolor{{{_rank_color(r, n)}}}{_stripped}'

                            _all_rank_cols = _rank_lower | _rank_higher
                            for _rcol in highlighted.columns:
                                if _rcol in _all_rank_cols:
                                    for _ridx in highlighted.index:
                                        _rkey = (_ridx, _rcol)
                                        if _rkey in _rank_store:
                                            _rank, _n = _rank_store[_rkey]
                                            highlighted.loc[_ridx, _rcol] = _apply_rank_color(
                                                str(highlighted.loc[_ridx, _rcol]), _rank, _n
                                            )

                            save_button = True

                            if save_button:
                                latex_table = highlighted.fillna('').to_latex(
                                    index=False,
                                    escape=False,                 # required so \cellcolor works
                                    longtable=False,
                                    float_format="%.3f"
                                )

                                # OOS multisim: inject IS / OOS multicolumn header row
                                if _is_multisim_table and run == "covid_oos":
                                    _is_cols  = [c for c in highlighted.columns if c.startswith('IS ')]
                                    _oos_cols = [c for c in highlighted.columns if c.startswith('OOS ')]
                                    _n_fixed  = 2  # Model + Variable
                                    _n_is, _n_oos = len(_is_cols), len(_oos_cols)
                                    _multi = ' & ' * _n_fixed
                                    if _n_is:
                                        _multi += f'\\multicolumn{{{_n_is}}}{{c}}{{\\textbf{{In-Sample}}}}'
                                    if _n_oos:
                                        _multi += f' & \\multicolumn{{{_n_oos}}}{{c}}{{\\textbf{{Out-of-Sample}}}}'
                                    _multi += ' \\\\\n'
                                    _cmi = ''
                                    if _n_is:
                                        _cmi += f'\\cmidrule(lr){{{_n_fixed+1}-{_n_fixed+_n_is}}}'
                                    if _n_oos:
                                        _s2 = _n_fixed + _n_is + 1
                                        _cmi += f'\\cmidrule(lr){{{_s2}-{_s2+_n_oos-1}}}'
                                    _cmi += '\n'
                                    latex_table = latex_table.replace(
                                        '\\toprule\n', f'\\toprule\n{_multi}{_cmi}'
                                    )

                                with open(f'{output_path}ts_metrics_table.tex', "w") as f:
                                    f.write("\\begin{adjustbox}{width=\\textwidth}\n")
                                    f.write(latex_table)
                                    f.write("\\end{adjustbox}\n")

                                print(f"LaTeX table saved to {output_path}ts_metrics_table.tex")

                    # ###############################################################################################################
                    # #################### Occupational Outcomes ####################################################################
                    # ###############################################################################################################
                    # obs = observation['DATE']
                    # filt = filtered_model_results['Non-behavioural']['DATE']
                    # only_in_observation = set(obs) - set(filt)
                    # only_in_filtered = set(filt) - set(obs)

                    # print("Values in observation but not in filtered:", only_in_observation)
                    # print("Values in filtered but not in observation:", only_in_filtered)

                    # # ========= Helpers =========
                    # def _build_occ_metrics_from_record_df(record_df: pd.DataFrame) -> pd.DataFrame:
                    #     """Create per-time, per-occupation metrics from record_df, then average over time."""
                    #     df = record_df.copy()

                    #     # Safe denominators
                    #     emp   = df['Employment'].replace(0, np.nan)
                    #     unemp = df['Unemployment'].replace(0, np.nan)

                    #     # Per-time-step rates (per occupation)
                    #     df['HireRate_emp']        = df['Hires'] / emp
                    #     df['SepRate_emp']         = df['Separations'] / emp
                    #     df['MatchRate_perUnemp']  = df['Hires'] / unemp  # job-finding rate
                    #     df['Vacancy_Rate']  = df['Vacancies'] / emp  # vacancy rate

                    #     # Application effort proxies already logged in record_df
                    #     df['UnempSeekers'] = df['Unemployed Seekers']
                    #     df['EmpSeekers']   = df['Employed Seekers']

                    #     # Mean over the whole simulation by occupation (ignoring NaNs)
                    #     metrics = [
                    #         'UnempSeekers',
                    #         'EmpSeekers',
                    #         'Vacancies',
                    #         'Vacancy_Rate',
                    #         'HireRate_emp',
                    #         'SepRate_emp',
                    #         'MatchRate_perUnemp',
                    #     ]
                    #     keep = ['Occupation'] + metrics
                    #     mean_occ = df[keep].groupby('Occupation', as_index=True).mean(numeric_only=True)

                    #     if mod_data is not None and 'separation_rates' in mod_data:
                    #         sep_rates = mod_data['separation_rates']
                    #         # If sep_rates is a dict: convert to Series/DataFrame
                    #         if isinstance(sep_rates, dict):
                    #             sep_rates = pd.Series(sep_rates, name='SeparationRate_exog')
                    #         # If sep_rates is already a Series, ensure name
                    #         if isinstance(sep_rates, pd.Series):
                    #             if sep_rates.name is None:
                    #                 sep_rates.name = 'SeparationRate_exog'
                    #             sep_rates.index = sep_rates.index.astype(mean_occ.index.dtype)
                    #             mean_occ = mean_occ.merge(sep_rates, left_index=True, right_index=True, how='left')
                    #         # If sep_rates is a DataFrame, assume it has 'Occupation' column
                    #         elif isinstance(sep_rates, pd.DataFrame):
                    #             mean_occ = mean_occ.merge(
                    #                 sep_rates.rename(columns={sep_rates.columns[0]: 'SeparationRate_exog'}),
                    #                 left_index=True, right_on='Occupation', how='left'
                    #             ).set_index('Occupation')

                    #     return mean_occ

                    # def _build_econ_means_from_grouped(grouped: pd.DataFrame) -> pd.Series:
                    #     """If record_df not available, make economy-wide means over time from grouped."""
                    #     g = grouped.copy()
                    #     # These rates already exist in your code; recompute defensively if absent.
                    #     if 'Hires Rate' not in g.columns:
                    #         g['Hires Rate'] = g['Hires'] / g['Employment'].replace(0, np.nan)
                    #     if 'Separations Rate' not in g.columns:
                    #         g['Separations Rate'] = g['Separations'] / g['Employment'].replace(0, np.nan)
                    #     # Job-finding rate (hires per unemployed)
                    #     if 'MatchRate_perUnemp' not in g.columns:
                    #         g['MatchRate_perUnemp'] = g['Hires'] / g['Unemployment'].replace(0, np.nan)

                    #     # Application effort proxies at economy level (sum across occupations already in grouped)
                    #     rename_map = {'Unemployed Seekers': 'UnempSeekers', 'Employed Seekers': 'EmpSeekers'}
                    #     g = g.rename(columns=rename_map)

                    #     econ_means = pd.Series({
                    #         'UnempSeekers':       g['UnempSeekers'].mean(skipna=True) if 'UnempSeekers' in g else np.nan,
                    #         'EmpSeekers':         g['EmpSeekers'].mean(skipna=True) if 'EmpSeekers' in g else np.nan,
                    #         'Vacancies':          g['Vacancies'].mean(skipna=True) if 'Vacancies' in g else np.nan,
                    #         'HireRate_emp':       g['Hires Rate'].mean(skipna=True),
                    #         'SepRate_emp':        g['Separations Rate'].mean(skipna=True),
                    #         'MatchRate_perUnemp': g['MatchRate_perUnemp'].mean(skipna=True),
                    #     })
                    #     return econ_means

                    # def plot_metric_mean_by_occ(mean_df: pd.DataFrame, metric: str, sort_by_value=True, obs_mean= None, topk=None, title=None):
                    #     """Scatter: one point per occupation, y = mean over time of chosen metric."""
                    #     df = mean_df[[metric]].dropna().copy()
                    #     if metric == "SepRate_emp":
                    #         df = mean_df[[metric, 'seps_rate']].dropna().copy()
                    #     if df.empty:
                    #         print(f"No data to plot for metric '{metric}'.")
                    #         return

                    #     if sort_by_value:
                    #         df = df.sort_values(metric, ascending=True)

                    #     if isinstance(topk, int):
                    #         df = df.nlargest(topk, metric)

                    #     plt.figure(figsize=(12, 6))
                    #     xs = np.arange(len(df))
                    #     colors = ['red' if occ in trending_occs else 'blue' for occ in df.index]
                    #     alphs = [1 if occ in trending_occs else 0.02 for occ in df.index]
                    #     plt.scatter(xs, df[metric], c=colors, alpha=alphs)

                    #     if obs_mean is not None:
                    #         plt.plot(range(len(xs)), [obs_mean]*len(xs), color='red', linestyle='--', label='Observed Mean')
                    #     if metric == "SepRate_emp":
                    #         plt.scatter(range(len(xs)), df['seps_rate'], color='grey', alpha = 0.5, label='Observed Separation Rates')

                    #     plt.xticks(xs, df.index.astype(str), rotation=90)
                    #     plt.xlabel('Occupation')
                    #     plt.ylabel(f"Mean {metric}")
                    #     plt.title(title if title else f"Mean {metric} across simulation")
                    #     plt.legend()
                    #     plt.tight_layout()
                    #     plt.close()

                    # def plot_hire_sep_ordered(mean_df: pd.DataFrame, topk=None, ascending=True):
                    #     """
                    #     One plot with BOTH mean hiring and separations rates per occupation,
                    #     ordered by mean hiring rate.
                    #     """
                    #     df = mean_df[['HireRate_emp', 'SepRate_emp', 'MatchRate_perUnemp', 'Vacancy_Rate']].dropna().copy()
                    #     if df.empty:
                    #         print("No data to plot for HireRate_emp / SepRate_emp.")
                    #         return

                    #     # Order by hiring rate
                    #     df = df.sort_values('HireRate_emp', ascending=ascending)

                    #     # Optionally keep only the top-k by hiring rate (after sorting)
                    #     if isinstance(topk, int):
                    #         df = df.tail(topk) if ascending else df.head(topk)

                    #     xs = np.arange(len(df))
                    #     # small horizontal offsets to avoid perfect overlap
                    #     offset = 0.15

                    #     plt.figure(figsize=(14, 6))
                    #     plt.scatter(xs - offset, df['HireRate_emp'], alpha=0.8, label='HireRate_emp')
                    #     #plt.scatter(xs - offset, df['MatchRate_perUnemp'], alpha=0.8, label='MatchRate_perUnemp')
                    #     plt.scatter(xs + offset, df['SepRate_emp'],  alpha=0.8, label='SepRate_emp')
                    #     plt.scatter(xs - offset, df['Vacancy_Rate'], alpha=0.8, label='Vacancy_Rate')


                    #     plt.xticks(xs, df.index.astype(str), rotation=90)
                    #     plt.xlabel('Occupation (sorted by mean hiring rate)')
                    #     plt.ylabel('Mean rate')
                    #     plt.title('Mean Hiring vs Separations Rates by Occupation (ordered by Hiring Rate)')
                    #     plt.legend()
                    #     plt.tight_layout()
                    #     plt.close()


                    # # Build per-occupation means (this is what you want for one point per occupation)
                    # mean_occ = _build_occ_metrics_from_record_df(filtered_sim_results['Behavioural w. Cyc. OTJ w. RW'])  # or any other model's record_df
                    # # Plot: one dot per occupation (mean over time)
                    # #plot_metric_mean_by_occ(mean_occ, 'UnempSeekers',       sort_by_value=True, topk=None, title='Mean Unemployed Seekers by Occupation')
                    # #plot_metric_mean_by_occ(mean_occ, 'EmpSeekers',         sort_by_value=True, topk=None, title='Mean Employed Seekers (OTJ) by Occupation')
                    # plot_metric_mean_by_occ(mean_occ, 'Vacancy_Rate',        sort_by_value=True, obs_mean=observation['VACRATE'].mean(), topk=None, title='Mean Vacancy Rate by Occupation')
                    # plot_metric_mean_by_occ(mean_occ, 'HireRate_emp',       sort_by_value=True, obs_mean = jolts['HIRESRATE'].mean(), topk=None, title='Mean Hiring Rate (per Employed)')
                    # plot_metric_mean_by_occ(mean_occ, 'SepRate_emp',        sort_by_value=True, obs_mean = jolts['SEPSRATE'].mean(), topk=None, title='Mean Separations Rate (per Employed)')
                    # plot_metric_mean_by_occ(mean_occ, 'MatchRate_perUnemp', sort_by_value=True, obs_mean = jolts['HIRESRATE'].mean(),topk=None, title='Mean Job-Finding Rate (Hires per Unemployed)')

                    # # NEW: both rates on the same plot, ordered by mean hiring rate
                    # plot_hire_sep_ordered(mean_occ, topk=None, ascending=True)  # set topk=50 to show the 50 lowest/highest by hiring rat