"""
COMPLETE CODE FOR RUNNING MODELS 100 TIMES
===========================================

Insert this code into auto_run_model.py to run each model 100 times and plot means.

STEP 1: Add this section near the top (after imports, around line 45)
"""

# ============================================================================
# MULTI-SIMULATION CONFIGURATION AND FUNCTIONS
# ============================================================================

N_SIMULATIONS = 100  # Number of times to run each model
CONFIDENCE_LEVEL = 0.95  # For confidence intervals

def aggregate_simulation_results(sim_list, metric_cols=None):
    """
    Aggregate multiple simulation runs into mean, std, and confidence intervals.
    
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
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    
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
    z_score = norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
    ci_margin = z_score * (std_vals / np.sqrt(N_SIMULATIONS))
    ci_lower = mean_vals - ci_margin
    ci_upper = mean_vals + ci_margin
    
    # Build result
    result = template[['DATE']].copy() if 'DATE' in template.columns else pd.DataFrame()
    
    for i, col in enumerate(metric_cols):
        result[col] = mean_vals[:, i]
        result[f'{col}_std'] = std_vals[:, i]
        result[f'{col}_ci_lower'] = ci_lower[:, i]
        result[f'{col}_ci_upper'] = ci_upper[:, i]
    
    return result


def run_model_multiple_times(model_name, base_params, calib_params, 
                             calib_date, occ_ids, n_sims=100):
    """
    Run a single model configuration multiple times and aggregate results.
    """
    from copy import deepcopy
    import pandas as pd
    
    print(f"  Running {n_sims} simulations for {model_name}...")
    
    grouped_list = []
    
    for sim_i in range(n_sims):
        if (sim_i + 1) % 10 == 0:
            print(f"    Simulation {sim_i + 1}/{n_sims}")
        
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


"""
STEP 2: REPLACE the model running loop (around lines 335-398)

Delete everything from "# Initialize the results dictionaries" 
to "vacs_finals[name] = vacs_final"

Replace with this:
"""

# ============================================================================
# RUN MODELS MULTIPLE TIMES
# ============================================================================

# Initialize results dictionaries
model_results = {}
model_results_all_sims = {}  # Store individual runs for diagnostics

# Loop through each model configuration
for name, item in calib_list.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    
    # Base parameters
    base_params = deepcopy(params)
    
    # Get calibrated parameters
    if 'cyclical_e' in name:
        calib_params = {
            'gamma_u': param_df.loc[(param_df['model_cat'] == name) & 
                                   (param_df["Parameter"] == "gamma_u"), "Value"].iloc[0],
            'd_u': param_df.loc[(param_df['model_cat'] == name) & 
                               (param_df["Parameter"] == "d_u"), "Value"].iloc[0],
            'theta': param_df.loc[(param_df['model_cat'] == name) & 
                                 (param_df["Parameter"] == "theta"), "Value"].iloc[0],
            'd_v': 0.01,
            'gamma_v': 0.16
        }
    else:
        calib_params = {
            'gamma_u': param_df.loc[(param_df['model_cat'] == name) & 
                                   (param_df["Parameter"] == "gamma_u"), "Value"].iloc[0],
            'd_u': param_df.loc[(param_df['model_cat'] == name) & 
                               (param_df["Parameter"] == "d_u"), "Value"].iloc[0],
            'theta': None,
            'd_v': 0.01,
            'gamma_v': 0.16
        }
    
    calib_params.update(item)
    
    if run == "steady_state":
        calib_params.update({
            "steady_state": True,
            "time_steps": 1000,
            "delay": 25
        })
    else:
        calib_params.update({"steady_state": False})
    
    # RUN MULTIPLE TIMES
    aggregated_results, all_sims = run_model_multiple_times(
        model_name=name,
        base_params=base_params,
        calib_params=calib_params,
        calib_date=calib_date,
        occ_ids=occ_ids,
        n_sims=N_SIMULATIONS
    )
    
    model_results[name] = aggregated_results
    model_results_all_sims[name] = all_sims

print(f"\n{'='*60}")
print(f"Complete! Each model run {N_SIMULATIONS} times")
print(f"Results show mean ± {int(CONFIDENCE_LEVEL*100)}% CI")
print(f"{'='*60}\n")


"""
STEP 3: KEEP the filtering section (around lines 410-415) AS IS:
"""

# def filter_and_relabel(d, mapping, order):
#     return {mapping[k]: d[k] for k in order if k in d}

# filtered_model_results = filter_and_relabel(model_results, name_map, desired_order)


"""
STEP 4: Your plotting code remains THE SAME

All your existing plot calls will work:
    plot_ltuer(filtered_model_results, observation, ...)
    plot_rel_wages(filtered_model_results, ...)
    
They now plot the MEAN across 100 simulations.
"""


"""
OPTIONAL: Add confidence bands to plots

Add these functions to plot_funs.py or insert before plotting:
"""

def plot_ltuer_with_ci(model_results, observation, save=False, path='', colors=None):
    """Plot LTUER with confidence intervals."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model_name, df in model_results.items():
        color = colors.get(model_name, 'black') if colors else None
        
        # Mean line
        ax.plot(df['DATE'], df['LTUER'], 
               label=model_name, color=color, linewidth=2)
        
        # Confidence band
        if 'LTUER_ci_lower' in df.columns:
            ax.fill_between(df['DATE'], 
                           df['LTUER_ci_lower'], 
                           df['LTUER_ci_upper'], 
                           color=color, alpha=0.2)
    
    # Observations
    if observation is not None and 'LTUER' in observation.columns:
        ax.plot(observation['DATE'], observation['LTUER'], 
               label='Observed', color='red', linestyle='--', linewidth=2)
    
    ax.legend(loc='best')
    ax.set_xlabel('Date')
    ax.set_ylabel('Long-term Unemployment Rate')
    ax.set_title(f'Long-term Unemployment Rate (Mean ± 95% CI, N={N_SIMULATIONS})')
    ax.grid(True, alpha=0.3)
    
    if save:
        plt.savefig(f'{path}ltuer_mean_ci.png', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_rel_wages_with_ci(model_results, save=False, path='', 
                           freq='3M', colors=None, unemp_only=True, suffix=""):
    """Plot relative wages with confidence intervals."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model_name, df in model_results.items():
        color = colors.get(model_name, 'black') if colors else None
        
        # Resample if needed
        df_plot = df.set_index('DATE').resample(freq).mean().reset_index()
        
        metric = 'U_REL_WAGE_MEAN' if unemp_only else 'E_REL_WAGE_MEAN'
        
        # Mean line
        ax.plot(df_plot['DATE'], df_plot[metric],
               label=model_name, color=color, linewidth=2)
        
        # Confidence band
        ci_lower = f'{metric}_ci_lower'
        ci_upper = f'{metric}_ci_upper'
        if ci_lower in df_plot.columns:
            ax.fill_between(df_plot['DATE'],
                           df_plot[ci_lower],
                           df_plot[ci_upper],
                           color=color, alpha=0.2)
    
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(loc='best')
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Wage (Post/Pre)')
    title_type = 'U→E' if unemp_only else 'E→E'
    ax.set_title(f'{title_type} Relative Wages (Mean ± 95% CI, N={N_SIMULATIONS})')
    ax.grid(True, alpha=0.3)
    
    if save:
        plt.savefig(f'{path}rel_wages_{suffix}_mean_ci.png', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


# Then use these functions:
# plot_ltuer_with_ci(filtered_model_results, observation, 
#                    save=save_button, path=output_path, colors=plot_colors)
# 
# plot_rel_wages_with_ci(filtered_model_results, save=save_button, 
#                        path=output_path, freq='3M', colors=plot_colors, 
#                        unemp_only=True, suffix="")


"""
TESTING: Start small

Before running 100 simulations for all models, test with:
    N_SIMULATIONS = 3

Once you verify it works, increase to 100.
"""


"""
PERFORMANCE TIP: Cache results

After running, save to avoid re-running:
"""

# After model running
cache_file = f'{cached_output_path}model_results_N{N_SIMULATIONS}.pkl'

if os.path.exists(cache_file):
    print(f"Loading cached results from {cache_file}")
    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)
        model_results = cached['model_results']
        N_SIMULATIONS = cached['n_simulations']
else:
    # ... run models as above ...
    
    # Save
    print(f"Saving results to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'model_results': model_results,
            'n_simulations': N_SIMULATIONS,
            'confidence_level': CONFIDENCE_LEVEL
        }, f)
