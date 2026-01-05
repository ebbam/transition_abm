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

complete_nw = False
steady_state_run = False
dropbox = True

# ============================================================================
# MULTI-SIMULATION CONFIGURATION
# ============================================================================
# REDUCED from 25 to 10 for lower CPU usage
N_SIMULATIONS = 25 # Reduced from 25 to 10
CONFIDENCE_LEVEL = 0.95  # For confidence intervals when N_SIMULATIONS is not None

# Add throttling delay between simulations (in seconds)
THROTTLE_DELAY = 0.5  # Wait 0.5 seconds between simulations

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
    
    # Add DATE column if it exists
    if 'DATE' in template.columns:
        result_dict['DATE'] = template['DATE'].values
    
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
    Tuple of (aggregated_grouped, grouped_list)
    """
    print(f"  Running {n_sims} simulations for {model_name}...")
    
    grouped_list = []
    
    for sim_i in range(n_sims):
        if (sim_i + 1) % 10 == 0:
            print(f"    Simulation {sim_i + 1}/{n_sims}")
        
        # Add throttling delay BEFORE running simulation
        if sim_i > 0 and THROTTLE_DELAY > 0:
            time.sleep(THROTTLE_DELAY)
        
        test_params = deepcopy(base_params)
        test_params.update(calib_params)
        
        # Run model
        sim_record, sim_grouped, sim_net, sum_stats, seekers_rec, \
            avg_wage_off_diff, app_loads_df, time_to_emp, vacs_final = \
            run_single_local(**test_params)
        
        # Add DATE and derived metrics
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
        
        grouped_list.append(sim_grouped)
    
    print(f"  Aggregating {n_sims} simulations...")
    aggregated = aggregate_simulation_results(grouped_list)
    
    return aggregated, grouped_list

# Rest of the file continues exactly as before...
if dropbox:
    output_prefix = "~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/"
    cached_output_path = 'output/cos_calib/'
else:
    output_prefix = 'output/cos_calib/'
    cached_output_path = 'output/cos_calib/'


for nw in ["full_omn", "single_node", "onet", "onet_wage_asym"]: 
    print(nw)  
    which_params = f'cos_calib/{nw}/'
    for run in ["base", "covid_oos"]:#, "steady_state"]:
        print(run)
        if run == "covid_oos":
           calib_date = ["2000-12-01", "2024-05-01"]
           output_path = os.path.expanduser(f'{output_prefix}{nw}/figures/covid_oos/')

        elif run == "steady_state":
            calib_date = ["2000-12-01", "2019-05-01"]
            output_path = os.path.expanduser(f'{output_prefix}{nw}/figures/steady_state/')

        else:
            calib_date = ["2000-12-01", "2019-05-01"]
            output_path = os.path.expanduser(f'{output_prefix}{nw}/figures/')

        if N_SIMULATIONS is not None and N_SIMULATIONS > 1:
            output_path = output_path.rstrip('/')
            output_path = f"{output_path}_N{N_SIMULATIONS}/"

        if not dropbox:
            os.makedirs(output_path, exist_ok=True)

        # Macro observations
        observation = macro_observations.loc[
            (macro_observations['DATE'] >= calib_date[0]) & 
            (macro_observations['DATE'] <= calib_date[1])].dropna(subset=["UNRATE", "VACRATE"]).reset_index()

        mod_data, net_temp, vacs, occ_ids, occ_shocks_dat = network_input_builder(nw, complete_nw, calib_date)
        print(occ_shocks_dat.shape[1])

        param_df1 = pd.read_csv(path + f"output/{which_params}calibrated_params_all{suffix}.csv")
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
            "otj_disc_strict_rw": "Behavioural w.o. Cyc. OTJ w. Strict RW",
            "otj_cyclical_e_disc_strict_rw": "Behavioural w. Cyc. OTJ w. Strict RW",
            "otj_cyclical_e_disc_no_rw": "Behavioural w. Cyc. OTJ w.o RW",
            "otj_disc_no_rw": "Behavioural w.o. Cyc. OTJ w.o RW"
        }

        desired_order = [
            "Non-behavioural",
            "Non-behavioural w. OTJ",
            "Behavioural w. Cyc. OTJ w.o RW",
            "Behavioural w.o. Cyc. OTJ w.o RW",
            "Behavioural w. Cyc. OTJ w. RW",
            "Behavioural w.o. Cyc. OTJ w. RW",
            "Behavioural w. Cyc. OTJ w. Strict RW",
            "Behavioural w.o. Cyc. OTJ w. Strict RW"
        ]


        params_to_latex(
            param_df,
            out_tex_path=f"{output_path}params_table.tex",
            param_col="Parameter",
            value_col="Value",
            model_col="model_cat",
            timestamp_col="Timestamp",
            model_name_map=name_map,
            desired_model_order=desired_order,
            decimals=3
        )

        params = {
            'mod_data': mod_data,
            'net_temp': net_temp,
            'vacs': vacs,
            'time_steps': occ_shocks_dat.shape[1],
            'delay': 0,
            'gdp_data': gdp_dat,
            'app_effort_dat': duration_to_prob_dict,
            'occ_shocks_data': occ_shocks_dat,
            'simple_res': False,
            'mistake_rate': 0.1
        }

        # Shorten vac_df to the same length as gdp_dat using a moving average (if needed)
        vac_df = observation['VACRATE'].to_numpy()
        if len(vac_df) > occ_shocks_dat.shape[1]:
            print("smoothing vac_df")
            # Apply moving average with window to smooth and match length
            window = len(vac_df) // occ_shocks_dat.shape[1]
            vac_dat = Series(vac_df).rolling(window=window, min_periods=1).mean()[window-1::window].reset_index(drop=True)
            vac_dat = vac_dat[:occ_shocks_dat.shape[1]]
        else:
            vac_dat = vac_df[:occ_shocks_dat.shape[1]]

        plt.figure(figsize=(10, 6))
        plt.hist(mod_data['demand_target'], bins = 100)
        #plt.hist(mod_data['employment'], bins = 100)
        # plt.figure(figsize=(10, 6)
        # Add percentile vertical lines
        plt.axvline(np.percentile(mod_data['demand_target'], 25), color='blue', linestyle='--', linewidth=2, label='25th percentile')
        plt.axvline(np.percentile(mod_data['demand_target'], 50), color='red', linestyle='--', linewidth=2, label='Median (50th)')
        plt.axvline(np.percentile(mod_data['demand_target'], 75), color='green', linestyle='--', linewidth=2, label='75th percentile')
        plt.title("Occupational Employment Levels")
        plt.ylabel("Count")
        plt.xlabel("Target demand (10,000 Persons)")
        plt.savefig(os.path.join(output_path, "dist_occ_size.png"), bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()

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
        
        sep_strings_filtered = [("Non-behavioural", "Non-behavioural w. and w.o OTJ"), 
                                    ("w.o RW", "Behavioural w.o Res. Wages"),
                                    ("w. RW", "Behavioural w. Loose Res Wages")]
                                   # ("w. Strict RW", "Behavioural w. Strict Res Wages")]

        # ===== CONFIGURATION =====
        FORCE_RERUN = False  # Set to True to ignore cache and rerun models
        if dropbox:
            FORCE_RERUN = False

        # Define where to save the results
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
                        "otj_cyclical_e_disc_strict_rw": {"otj": True,
                                    "cyc_otj": True, 
                                    "cyc_ue": False, 
                                    "disc": True,
                                    "delay": 25,
                                    'emp_apps': 1,
                                    'wage_prefs': True,
                                    "bus_confidence_dat": gdp_dat,
                                    "vac_data": vac_dat,
                                    'strict_rw': True},
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
                        "otj_disc_strict_rw": {"otj": True,
                                    "cyc_otj": False, 
                                    "cyc_ue": False, 
                                    "disc": True,
                                    "delay": 25,
                                    'emp_apps': 1,
                                    'wage_prefs': True,
                                    "bus_confidence_dat": gdp_dat,
                                    "vac_data": vac_dat,
                                    'strict_rw': True},
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
            sum_stats_list = {}
            seekers_recs_list = {}
            avg_wage_off_diffs = {}
            app_loads = {}
            time_to_emps = {}
            vacs_finals = {}

            # ================================================================
            # MAIN MODEL RUNNING LOOP (Single or Multi-simulation mode)
            # ================================================================
            
            # Check if multi-simulation mode is enabled
            if N_SIMULATIONS is not None and N_SIMULATIONS > 1:
                # ========== MULTI-SIMULATION MODE ==========
                print(f"\n{'='*60}")
                print(f"MULTI-SIMULATION MODE: Running each model {N_SIMULATIONS} times")
                print(f"{'='*60}\n")
                
                # Initialize the results dictionaries
                model_results = {}
                model_results_all_sims = {}  # Store all individual runs
                
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
                                    'd_v':0.01,
                                    'gamma_v':0.16}
                    else:
                        calib_params = {'gamma_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_u"),"Value"].iloc[0],
                                    'd_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_u"),"Value"].iloc[0],
                                    'theta': None, 
                                    'd_v':0.01,
                                    'gamma_v':0.16}
                    
                    # Update with the values from the calib_list
                    calib_params.update(item)
                    if run == "steady_state":
                        calib_params.update({"steady_state": True,
                                            "time_steps": 1000,
                                            "delay": 25})
                    else:
                        calib_params.update({"steady_state": False})
                    
                    # RUN MODEL MULTIPLE TIMES AND AGGREGATE
                    aggregated_grouped, grouped_list = run_model_multiple_times(
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
                
                print(f"\n{'='*60}")
                print(f"Multi-simulation complete! Each model run {N_SIMULATIONS} times")
                print(f"Results show mean ± {int(CONFIDENCE_LEVEL*100)}% CI")
                print(f"{'='*60}\n")
                
                # For multi-simulation mode, we don't populate these detailed results
                # (they're only from single runs)
                net_results = {}
                sim_results = {}
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

                # Loop through each model configuration
                for name, item in calib_list.items():
                    print(name)
                    # Create a deep copy of the base parameters
                    test_params = deepcopy(params)

                    if 'cyclical_e' in name:
                        calib_params = {'gamma_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_u"),"Value"].iloc[0],
                                    'd_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_u"),"Value"].iloc[0],
                                    'theta': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "theta"),"Value"].iloc[0],
                                    'd_v':0.01,
                                    'gamma_v':0.16}
                    else:
                        calib_params = {'gamma_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_u"),"Value"].iloc[0],
                                    'd_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_u"),"Value"].iloc[0],
                                    'theta': None, 
                                    'd_v':0.01,
                                    'gamma_v':0.16}

                    test_params.update(calib_params)

                    # Update with the values from the calib_list
                    test_params.update(item)
                    if run == "steady_state":
                        test_params.update({"steady_state": True,
                                            "time_steps": 1000,
                                            "delay": 25})
                    else:
                        test_params.update({"steady_state": False})

                    # Run the model
                    sim_record, sim_grouped, sim_net, sum_stats, seekers_rec, avg_wage_off_diff, app_loads_df, time_to_emp, vacs_final = run_single_local(**test_params)

                    # Generate plots or metrics (optional step)
                    #ue_vac = plot_records(sim_record, calib_date[0], calib_date[1])
                    sim_grouped['DATE'] = pd.date_range(start=calib_date[0], end= calib_date[1], periods=len(sim_grouped))
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

                    seekers_rec['DATE'] = pd.date_range(start=calib_date[0], end= calib_date[1], periods=len(seekers_rec))
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
                    vacs_finals[name] = vacs_final
                sum_stats_list[name] = sum_stats
                seekers_recs_list[name] = seekers_rec
                avg_wage_off_diffs[name] = avg_wage_off_diff
                app_loads[name] = app_loads_df
                time_to_emps[name] = time_to_emp
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
            else:
                # In multi-simulation mode, these are not populated
                filtered_net_results = {}
                filtered_sim_results = {}
                filtered_sum_stats = {}
                seekers_recs = {}
                filtered_app_loads = {}
                filtered_time_to_emps = {}

            
            # ===== SAVE ALL OBJECTS TO CACHE =====
            print(f"Saving results to {cache_file}...")
            cache_data = {
                'filtered_model_results': filtered_model_results,
                'filtered_net_results': filtered_net_results,
                'filtered_sim_results': filtered_sim_results,
                'filtered_sum_stats': filtered_sum_stats,
                'seekers_recs': seekers_recs,
                'filtered_app_loads': filtered_app_loads,
                'filtered_time_to_emps': filtered_time_to_emps,
                'n_simulations': N_SIMULATIONS,  # Save simulation mode info
                'confidence_level': CONFIDENCE_LEVEL if N_SIMULATIONS else None
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            if N_SIMULATIONS is not None and N_SIMULATIONS > 1:
                print(f"✓ Results saved (Multi-simulation mode: {N_SIMULATIONS} runs per model)")
            else:
                print("✓ Results saved (Single-simulation mode)")



        # palette & colors
        palette = sns.color_palette("Paired", 12)

        blues      = palette[0:2]
        greens     = palette[2:4]
        reds       = palette[4:6]
        oranges    = palette[6:8]
        purples    = palette[8:10]
        yellows    = palette[10:12]

        my_colors = blues + purples + oranges + greens
        plot_colors = dict(zip(name_map.values(), my_colors))

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
            ###############################################################################################################
            #################### UER-Vac w Error ##########################################################################
            ###############################################################################################################
            plot_uer_vac(filtered_model_results, observation, 
                    sep_strings=sep_strings_filtered, 
                        sep=True, save=save_button, path=output_path,
                        free_date_scale=(run == "steady_state"), 
                        calib_date = calib_date)
            
            ###############################################################################################################
            #################### Relative Wages ###########################################################################
            ###############################################################################################################
            plot_rel_wages(filtered_model_results, save = save_button, path = output_path, freq = '3M', colors = plot_colors, unemp_only = True, suffix = "")
            for lowess_fit in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
                plot_rel_wages(filtered_model_results, save = save_button, path = output_path, freq = 'M', colors = plot_colors, unemp_only = True, lowess = True, lowess_frac = lowess_fit, suffix = f"lowess_{lowess_fit}")


            ###############################################################################################################
            #################### LTUER Time Series ########################################################################
            ###############################################################################################################
            plot_ltuer(filtered_model_results, observation, 
                    sep_strings=sep_strings_filtered, 
                    sep=True, save=save_button, path=output_path)
            
            ###############################################################################################################
            #################### Beveridge Curve ##########################################################################
            ###############################################################################################################
            plot_bev_curve_color(filtered_model_results, observation, 
                    sep_strings=sep_strings_filtered, 
                        sep=True, save=save_button, path=output_path, smooth = 3, viridis_color = 'magma')
            

            ###############################################################################################################
            #################### Current vs. Target Demand ################################################################
            ###############################################################################################################
            plot_cd_vs_td(filtered_model_results, save=save_button, path=output_path, colors = plot_colors)

            ###############################################################################################################
            #################### Transition Rates #########################################################################
            ###############################################################################################################
            all_rates_new = pd.read_csv('data/transition_rates_96_24.csv')
            all_rates_new = all_rates_new[(all_rates_new['date'] >= calib_date[0]) & (all_rates_new['date'] <= calib_date[1])]
            all_rates_new['DATE'] = pd.to_datetime(all_rates_new['date'])

            plot_trans_rates(filtered_model_results, observation = all_rates_new, save = save_button, path = output_path, colors = plot_colors)


            ###############################################################################################################
            #################### Hires-Seps Rates #########################################################################
            ###############################################################################################################

            hires_seps_rate(filtered_model_results, jolts = jolts, save = save_button, path = output_path, colors = plot_colors)                    

            ###############################################################################################################
            #################### Seeker Composition - Line ################################################################
            ###############################################################################################################
            plot_seeker_comp_line(filtered_model_results, seekers_comp_obs, save=save_button , path=output_path, colors = plot_colors)

            if N_SIMULATIONS is None or N_SIMULATIONS <= 1:

                ###############################################################################################################
                #################### Applications per unemployed job-seeker ###################################################
                ###############################################################################################################
                

                plt.figure(figsize=(10, 6))
                for i, (name, item) in enumerate(seekers_recs.items()):
                    if name == "Non-behavioural":
                        # Plot every other value starting at the first place (index 0)
                        plt.plot(item['DATE'][3::4], [8] * len(item['DATE'][3::4]), color = plot_colors[name], marker='o', label=f"{name}" , linestyle = "")
                    elif name == "Non-behavioural w. OTJ":
                        # Plot every other value starting at the second place (index 1)
                        plt.plot(item['DATE'][1::4], [8] * len(item['DATE'][1::4]),  color = plot_colors[name], marker='o', label=f"{name}", linestyle = "")
                    else:
                        plt.plot(item['DATE'], item['Application Effort (U)'], marker='o', label=name,  color = plot_colors[name])
                plt.xlabel('Year')
                plt.ylabel('Applications per Unemployed Seeker')
                plt.title('Application Rate Over Time')
                plt.tight_layout()
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), 
                            ncol=len(seekers_recs)//2, frameon=True, fontsize=10)

                # Now save
                plt.savefig(os.path.join(output_path, "applications_per_unemployed_seeker.png"), bbox_inches='tight',  
                pad_inches=0.1,      
                dpi=300)
                plt.close()

                ###############################################################################################################
                #################### Applications per unemployed job-seeker ###################################################
                ###############################################################################################################
                plt.figure(figsize=(10, 6))
                for i, (name, item) in enumerate(seekers_recs.items()):
                    plt.plot(item['DATE'], item['Application Effort (E)'], marker='o', label=name, color = plot_colors[name],)
                plt.xlabel('Year')
                plt.ylabel('Applications per Employed Seeker')
                plt.title('Application Rate Over Time')
                plt.tight_layout()
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), 
                            ncol=len(seekers_recs)//2, frameon=True, fontsize=10)
                            # Now save
                plt.savefig(os.path.join(output_path, "applications_per_employed_seeker.png"), bbox_inches='tight',  
                pad_inches=0.1,      
                dpi=300)
                plt.close()
                
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
                survey_df = pd.read_csv("data/ltuer_cps_data.csv")

                # Use specific year
                plot_ltuer_cdf_with_observed(
                    net_dict=filtered_net_results,
                    names=list(filtered_net_results.keys()),
                    survey_df=survey_df,
                    survey_year=2020,  # Changed from survey_date
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
                                save=save_button , path=output_path, colors = plot_colors)

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
                    plt.savefig(os.path.join(output_path, "behavioural_regimes_time_to_reemp.png"), bbox_inches='tight',  
                pad_inches=0.1,      
                dpi=300)
                    plt.close()
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

                plt.title("Time to Re-Employment (Months Unemployed Prior to Hiring) by Origin Occupation")
                plt.tight_layout()
                if save_button:
                    plt.savefig(os.path.join(output_path, "time_to_reemp_dist.png"), bbox_inches='tight',  
                pad_inches=0.1,      
                dpi=300)
                    plt.close()
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
                fig = plt.subplots(figsize =(12, 8)) 
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
                        plt.savefig(f'{path}gender_wage_gaps.jpg', bbox_inches='tight',  
                pad_inches=0.1,      
                dpi=300)
                    plt.close()


                # Make the plot
                plt.bar(br1, brackets['Full-Time Females'], color ='lightblue', width = barWidth, alpha = 0.8,
                        label ='Women') 
                plt.bar(br2, brackets['Full-Time Males'], color ='orange', width = barWidth, alpha = 0.3,
                        label ='Men') 

                # Adding Xticks 
                plt.xlabel('Income Bracket', fontsize = 15) 
                plt.ylabel('Full-Time Employees per Bracket', fontsize = 15) 
                plt.ticklabel_format(useOffset=False, style='plain')
                plt.xticks([r + barWidth for r in range(len(brackets['Bracket']))], 
                        brackets['Bracket'])

                plt.title('Distribution of Male and Female Wages in US Labour Market', fontsize = 15) 
                plt.legend()
                plt.savefig(f'{output_path}bls_data_gender_wage_gaps.jpg', bbox_inches='tight',  
                pad_inches=0.1,      
                dpi=300)
                plt.close()

                womens_wage = (gender_income['Ceiling'] * gender_income['Full-Time Females']).sum()/(gender_income['Full-Time Females'].sum())
                mens_wage = (gender_income['Ceiling'] * gender_income['Full-Time Males']).sum()/(gender_income['Full-Time Males'].sum())

                plot_gender_gaps(filtered_net_results, save = True, path = output_path)

                
                ###############################################################################################################
                #################### TS Metrics Table #########################################################################
                ###############################################################################################################
                if run == "base":
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

                    def compute_time_series_metrics(res_dict, seekers_info_dict, obs_dict, variables):
                        rows = []
                        for model_name, sim_df in res_dict.items():
                            for var in variables:
                                if var == "Application Effort (U)":
                                    sim_series = seekers_info_dict[model_name][var].values if var in seekers_info_dict[model_name] else None
                                else:
                                    sim_series = sim_df[var].values if var in sim_df else None
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
                                        'Correlation': round(float(np.corrcoef(sim_series, obs_series)[0, 1]), 3)
                                    }
                                else:
                                    # Full set of metrics
                                    row = {
                                        'Model': model_name,
                                        'Variable': var,
                                        'Mean (Sim)': round(float(np.mean(sim_series)), 3),
                                        'Mean (Obs)': round(float(np.mean(obs_series)), 3),
                                        #'Variance (Sim)': round(float(np.var(sim_series)), 3),
                                        #'Variance (Obs)': round(float(np.var(obs_series)), 3),
                                        'SSE': round(float(np.sum((sim_series - obs_series) ** 2)), 3),
                                        'Correlation': round(float(np.corrcoef(sim_series, obs_series)[0, 1]), 3)
                                    }

                                rows.append(row)
                        return pd.DataFrame(rows)

                    BEST_COLOR = "yellow!25"
                    EXCELLENT_COLOR = "red!25"

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
                            "Mean (Sim)", "Mean (Obs)",
                            "Variance (Sim)", "Variance (Obs)",
                            "SSE", "Correlation"
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
                                    df.loc[best_idx, "Mean (Sim)"] = _color_val(raw_val, EXCELLENT_COLOR if is_excellent else BEST_COLOR)
                                
                                    # Color others within threshold (yellow only)
                                    threshold = best_diff * (1 + threshold_pct)
                                    for idx in valid_diffs.index:
                                        if idx != best_idx and diffs.loc[idx] <= threshold:
                                            raw_val = mean_sim.loc[idx]
                                            df.loc[idx, "Mean (Sim)"] = _color_val(raw_val, BEST_COLOR)
                                elif len(valid_diffs) == 1:
                                    # Only one value - make it yellow
                                    idx = valid_diffs.index[0]
                                    raw_val = mean_sim.loc[idx]
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
                                    df.loc[best_idx, "Variance (Sim)"] = _color_val(raw_val, EXCELLENT_COLOR if is_excellent else BEST_COLOR)
                                
                                    threshold = best_diff * (1 + threshold_pct)
                                    for idx in valid_diffs.index:
                                        if idx != best_idx and diffs.loc[idx] <= threshold:
                                            raw_val = var_sim.loc[idx]
                                            df.loc[idx, "Variance (Sim)"] = _color_val(raw_val, BEST_COLOR)
                                elif len(valid_diffs) == 1:
                                    idx = valid_diffs.index[0]
                                    raw_val = var_sim.loc[idx]
                                    df.loc[idx, "Variance (Sim)"] = _color_val(raw_val, BEST_COLOR)

                            # SSE -> lowest wins
                            if "SSE" in sub.columns:
                                sse = pd.to_numeric(sub["SSE"].replace("--", np.nan))
                                valid_sse = sse.dropna()
                            
                                if not valid_sse.empty and len(valid_sse) >= 2:
                                    best_sse = valid_sse.min()
                                    best_idx = valid_sse.idxmin()
                                    second_best = valid_sse.nsmallest(2).iloc[-1]
                                
                                    # Best is "excellent" if second-best is more than threshold% higher
                                    is_excellent = second_best > best_sse * (1 + threshold_pct)
                                
                                    # Color the best
                                    raw_val = float(sse.loc[best_idx])
                                    df.loc[best_idx, "SSE"] = _color_val(raw_val, EXCELLENT_COLOR if is_excellent else BEST_COLOR)
                                
                                    # Color others within threshold (yellow only)
                                    threshold = best_sse * (1 + threshold_pct)
                                    for idx in valid_sse.index:
                                        if idx != best_idx and sse.loc[idx] <= threshold:
                                            raw_val = float(sse.loc[idx])
                                            df.loc[idx, "SSE"] = _color_val(raw_val, BEST_COLOR)
                                elif len(valid_sse) == 1:
                                    idx = valid_sse.index[0]
                                    raw_val = float(sse.loc[idx])
                                    df.loc[idx, "SSE"] = _color_val(raw_val, BEST_COLOR)

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
                                    df.loc[best_idx, "Correlation"] = _color_val(raw_val, EXCELLENT_COLOR if is_excellent else BEST_COLOR)
                                
                                    # Color others within threshold (yellow only)
                                    threshold = best_corr * (1 - threshold_pct)
                                    for idx in valid_corr.index:
                                        if idx != best_idx and corr.loc[idx] >= threshold:
                                            raw_val = float(corr.loc[idx])
                                            df.loc[idx, "Correlation"] = _color_val(raw_val, BEST_COLOR)
                                elif len(valid_corr) == 1:
                                    idx = valid_corr.index[0]
                                    raw_val = float(corr.loc[idx])
                                    df.loc[idx, "Correlation"] = _color_val(raw_val, BEST_COLOR)

                        return df

                    print(filtered_model_results['Non-behavioural']['VACRATE'])
                    # ---- use it ----
                    ts_table = compute_time_series_metrics(
                        filtered_model_results, seekers_recs, comp_series,
                        ['VACRATE','UER','LTUER',"Hires Rate","Separations Rate",
                        'UE_Trans_Rate','EE_Trans_Rate','Application Effort (U)','Seeker Composition']
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

                    # Replace repeated model names with empty string after first occurrence per model
                    ts_table['Model'] = ts_table['Model'].where(~ts_table['Model'].duplicated(), '')

                    highlighted = highlight_best(ts_table)

                    save_button = True

                    if save_button:
                        latex_table = highlighted.to_latex(
                            index=False,
                            escape=False,                 # required so \cellcolor works
                            longtable=False,
                            float_format="%.3f"
                        )
                        with open(f'{output_path}ts_metrics_table.tex', "w") as f:
                            f.write("\\begin{table}[ht]\n")
                            f.write("\\centering\n")
                            f.write("\\begin{adjustbox}{width=\\textwidth}\n")
                            f.write(latex_table)
                            f.write("\\end{adjustbox}\n")
                            f.write("\\end{table}\n")
        
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

