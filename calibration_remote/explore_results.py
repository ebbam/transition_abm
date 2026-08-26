"""
Interactive exploration script — loads cached model results and runs plots.
Run in an interactive window (e.g. VS Code "Run in Interactive Window").
"""
import importlib
import pickle
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── project path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from collate_macro_vars import macro_observations, recessions
import plot_funs as pf
importlib.reload(pf)

# ── settings ──────────────────────────────────────────────────────────────────
CALIB_DATE  = ["2001-01-01", "2024-05-01"]
DELAY       = 25
PKL_PATH    = os.path.join(PROJECT_ROOT,
                           "output/calib_final/full_omn/exog_vac/figures/covid_oos/cached_results/model_results.pkl")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/calib_final/full_omn/exog_vac/figures/covid_oos")
SAVE        = True    # set True to write to OUTPUT_PATH instead of displaying

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

# ── load cached results ───────────────────────────────────────────────────────
print(f"Loading cached results from {PKL_PATH}...")
with open(PKL_PATH, 'rb') as f:
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
filtered_ue_stock_dfs = filter_models(cached_data.get('filtered_ue_stock_dfs', {}), exclude=exclude_models)
_desired_order = [
    "Non-behavioural",
    "Non-behavioural w. OTJ",
    "Behavioural w.o. Cyc. OTJ w.o RW",
    "Behavioural w.o. Cyc. OTJ w. RW",
]

def _reorder(d):
    return {k: d[k] for k in _desired_order if k in d}

filtered_model_results   = _reorder(filtered_model_results)
filtered_net_results     = _reorder(filtered_net_results)
filtered_sim_results     = _reorder(filtered_sim_results)
filtered_sum_stats       = _reorder(filtered_sum_stats)
seekers_recs             = _reorder(seekers_recs)
filtered_app_loads       = _reorder(filtered_app_loads)
filtered_time_to_emps    = _reorder(filtered_time_to_emps)
filtered_wage_age_dfs    = _reorder(filtered_wage_age_dfs)
filtered_ue_stock_dfs    = _reorder(filtered_ue_stock_dfs)

print("✓ Results loaded successfully from cache!")

# Trim macro observations to calibration window
observation = macro_observations.loc[
    (macro_observations['DATE'] >= CALIB_DATE[0]) & 
    (macro_observations['DATE'] <= CALIB_DATE[1])].dropna(subset=["UNRATE", "VACRATE"]).reset_index()

rec_window = recessions[
    (recessions["DATE"] >= CALIB_DATE[0]) &
    (recessions["DATE"] <= CALIB_DATE[1])
].copy()

print("Models loaded:", list(filtered_model_results.keys()))
print("Time steps per model:",
      {k: len(v) for k, v in filtered_model_results.items()})

# ── model display colours (extend as needed) ──────────────────────────────────
MODEL_COLORS = {
    name: col for name, col in zip(
        filtered_model_results.keys(),
        ["steelblue", "darkorange", "forestgreen", "crimson",
         "purple", "brown", "teal", "grey"]
    )
}

sep_strings_filtered = [("Non-behavioural", "Non-behavioural w. and w.o OTJ"), 
                            ("w.o RW", "Behavioural w.o Res. Wages"),
                            ("w. RW", "Behavioural w. Res Wages")]
                        # ("w. Strict RW", "Behavioural w. Strict Res Wages")]

                        # Check what years the model data actually covers
first_model = list(filtered_ue_stock_dfs.keys())[0]
stock = filtered_ue_stock_dfs[first_model]

start_date = pd.Timestamp(CALIB_DATE[0])
stock['Year'] = stock['Time Step'].apply(
    lambda ts: (start_date + pd.DateOffset(months=int(ts) - 1)).year
)
print("Max time step:", stock['Time Step'].max())
print("Year range:", stock['Year'].min(), "–", stock['Year'].max())

# ══════════════════════════════════════════════════════════════════════════════
# 8.  Wages by age (cross-sectional profile, expansion vs recession)
# ══════════════════════════════════════════════════════════════════════════════
# recession_years: 0-based simulation year indices overlapping NBER recessions.
# With CALIB_DATE starting 2001-01-01 and DELAY=25:
#   2001 recession ≈ year 0-1, GFC ≈ year 7-9 (adjust as needed).
RECESSION_YEARS = [0, 1, 7, 8, 9]

pf.plot_wages_by_age(
    wage_age_dict   = filtered_wage_age_dfs,
    calib_date      = CALIB_DATE,
    age_bin_width   = 5,
    recession_years = None,
    overlay         = True,
    colors          = MODEL_COLORS,
    save            = SAVE,
    path            = OUTPUT_PATH,
)
# ══════════════════════════════════════════════════════════════════════════════
# 1.  UER / Vacancy rate
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_uer_vac(
    res_dict   = filtered_model_results,
    sep = True,
    sep_strings = sep_strings_filtered,
    macro_obs  = observation,
    calib_date = CALIB_DATE,
    recessions = None,
    free_date_scale= False, 
    save       = SAVE,
    path       = OUTPUT_PATH,
)
print("✓ Plot 1 (UER / vacancy rate) complete.")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Time-to-reemployment ribbon (new)
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = os.path.join(PROJECT_ROOT, "data/")
emp_economy = pd.read_csv(DATA_PATH + "survival_curves_by_phase_economy_wide.csv")
emp_occ     = pd.read_csv(DATA_PATH + "survival_curves_by_phase_occupational_SOC_major_n23.csv")

pf.plot_time_to_reemployment_ribbon(
    tte_dict          = filtered_time_to_emps,
    calib_date        = CALIB_DATE,
    delay             = DELAY,
    max_duration      = 24,
    colors            = MODEL_COLORS,
    empirical_economy = emp_economy,
    empirical_occ     = emp_occ,
    save              = SAVE,
    path              = OUTPUT_PATH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 2b. Mean TTE by layoff date (when were workers laid off vs how long did they search)
# ══════════════════════════════════════════════════════════════════════════════
from collate_macro_vars import recession_spans

pf.plot_tte_by_layoff_date(
    tte_dict        = filtered_time_to_emps,
    calib_date      = CALIB_DATE,
    delay           = DELAY,
    freq            = 'Q',
    benchmark       = 'Non-behavioural w. OTJ',
    recession_spans = recession_spans,
    save            = SAVE,
    path            = OUTPUT_PATH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 2c. TTE by local market tightness at layoff (Option B: slack vs tight)
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_tte_by_tightness(
    tte_dict         = filtered_time_to_emps,
    sim_results_dict = filtered_sim_results,
    calib_date       = CALIB_DATE,
    benchmark        = 'Non-behavioural w. OTJ',
    delay            = DELAY,
    freq             = 'Q',
    recession_spans  = recession_spans,
    save             = SAVE,
    path             = OUTPUT_PATH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 2d. TTE by origin-occupation UER quartile at layoff (Option C: Q1–Q4)
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_tte_by_uer_quantile(
    tte_dict         = filtered_time_to_emps,
    sim_results_dict = filtered_sim_results,
    calib_date       = CALIB_DATE,
    benchmark        = 'Non-behavioural w. OTJ',
    delay            = DELAY,
    freq             = 'Q',
    recession_spans  = recession_spans,
    save             = SAVE,
    path             = OUTPUT_PATH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Time-to-reemployment by SOC major group (one figure per phase)
# ══════════════════════════════════════════════════════════════════════════════
occ_crosswalk = pd.read_csv(DATA_PATH + "occ_uer_ltuer_observed.csv",
                            dtype={'SOC_major_adj': str, 'SOC_minor_adj': str,
                                   'SOC_broad_adj': str, 'SOC2010_adj': str})
occ_crosswalk['SOC_major_adj'] = pd.to_numeric(occ_crosswalk['SOC_major_adj'], errors='coerce')

soc_labs = pd.read_csv(os.path.expanduser(
    "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/"
    "data/occ_macro_vars/OEWS/soc_major_labels.csv"),
    dtype={"soc_code": str})
soc_labs['soc_code'] = pd.to_numeric(soc_labs['soc_code'], errors='coerce')

pf.plot_tte_by_soc_major(
    tte_dict      = filtered_time_to_emps,
    occ_crosswalk = occ_crosswalk,
    soc_labs      = soc_labs,
    empirical_occ = emp_occ,
    max_duration  = 24,
    colors        = MODEL_COLORS,
    save          = SAVE,
    path          = OUTPUT_PATH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Hotspot quadrant: structural vs cyclical UER
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_hotspot_quadrant(
    sim_results_dict = filtered_sim_results,
    occ_crosswalk    = occ_crosswalk,
    soc_labs         = soc_labs,
    metric           = 'UER',
    deviation        = False,
    benchmark        = 'Non-behavioural w. OTJ',
    topk             = 15,
    save             = SAVE,
    path             = OUTPUT_PATH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Simulated vs observed occupational UER bias
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_occ_uer_fit(
    sim_results_dict = filtered_sim_results,
    occ_crosswalk    = occ_crosswalk,
    soc_labs         = soc_labs,
    metric           = 'UER',
    topk             = 15,
    save             = SAVE,
    path             = OUTPUT_PATH,
)

# Notes: this figure could instead have a measure of this difference by year. so that we can see in which periods we are more accurate than others. 

# ══════════════════════════════════════════════════════════════════════════════
# 5b. Re-employment wage heatmap vs benchmark (economy-wide)
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_wage_rehire_heatmap(
    sim_results_dict = filtered_sim_results,
    calib_date       = CALIB_DATE,
    benchmark        = 'Non-behavioural w. OTJ',
    delay            = DELAY,
    freq             = 'Q',
    recession_spans  = recession_spans,
    save             = SAVE,
    path             = OUTPUT_PATH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 5c. Re-employment wage heatmap by UER quartile
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_wage_rehire_by_uer_quantile(
    sim_results_dict = filtered_sim_results,
    calib_date       = CALIB_DATE,
    benchmark        = 'Non-behavioural w. OTJ',
    delay            = DELAY,
    freq             = 'Q',
    recession_spans  = recession_spans,
    save             = SAVE,
    path             = OUTPUT_PATH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Wasserstein distance: simulated vs CPS unemployment duration distribution
# ══════════════════════════════════════════════════════════════════════════════
cps_dur_dist = pd.read_csv(DATA_PATH + "unemp_duration_dist_yearly_SOC_major.csv")

pf.plot_wasserstein_dur_dist(
    ue_stock_dfs  = filtered_ue_stock_dfs,
    cps_dur_dist  = cps_dur_dist,
    occ_crosswalk = occ_crosswalk,
    soc_labs      = soc_labs,
    calib_date    = CALIB_DATE,
    delay         = DELAY,
    year          = 2007,
    colors        = MODEL_COLORS,
    save          = SAVE,
    path          = OUTPUT_PATH,
)
print("✓ Plot 6a (Wasserstein ranked bar — single year) complete.")

_wass_shared_args = dict(
    ue_stock_dfs      = filtered_ue_stock_dfs,
    cps_dur_dist      = cps_dur_dist,
    occ_crosswalk     = occ_crosswalk,
    soc_labs          = soc_labs,
    calib_date        = CALIB_DATE,
    delay             = DELAY,
    colors            = MODEL_COLORS,
    sim_results_dict  = filtered_sim_results,
    save              = SAVE,
    path              = OUTPUT_PATH,
)

# ── 6b. Heatmap: occupation × year ───────────────────────────────────────────
pf.plot_wasserstein_heatmap(**_wass_shared_args)
print("✓ Plot 6b (Wasserstein heatmap — occupation × year) complete.")

pf.plot_wasserstein_heatmap(**_wass_shared_args, mode='binary')   # or 'ranked' or 'raw'

pf.plot_wasserstein_heatmap(**_wass_shared_args, mode='ranked')   # or 'ranked' or 'raw'

pf.plot_wasserstein_heatmap(**_wass_shared_args, mode='top2')   # or 'ranked' or 'raw'

# ── 6c. Box plots: distribution across occupations by year ───────────────────
pf.plot_wasserstein_boxplot(**_wass_shared_args)
print("✓ Plot 6c (Wasserstein boxplot by year) complete.")

# ── 6d. Ribbon: mean ± 1 SD across occupations over time ─────────────────────
pf.plot_wasserstein_ribbon(**_wass_shared_args)
print("✓ Plot 6d (Wasserstein ribbon over time) complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Accepted wages at U→E rehire
# ══════════════════════════════════════════════════════════════════════════════
from collate_macro_vars import recession_spans

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

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Occupation wage trajectories coloured by stress category
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_occ_wage_trajectories(
    sim_results_dict = filtered_sim_results,
    calib_date       = CALIB_DATE,
    benchmark        = 'Non-behavioural w. OTJ',
    topk             = 15,
    delay            = DELAY,
    save             = SAVE,
    path             = OUTPUT_PATH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 8.  Wage scarring by occupation stress category
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_wage_scar_by_stress(
    sim_results_dict = filtered_sim_results,
    calib_date       = CALIB_DATE,
    benchmark        = 'Non-behavioural w. OTJ',
    topk             = 15,
    delay            = DELAY,
    lowess_frac      = 0.15,
    colors           = MODEL_COLORS,
    recession_spans  = recession_spans,
    occ_crosswalk    = occ_crosswalk,
    soc_labs         = soc_labs,
    save             = SAVE,
    path             = OUTPUT_PATH,
)



# ══════════════════════════════════════════════════════════════════════════════
# 8.  Job ladders (individual wage trajectories)
# ══════════════════════════════════════════════════════════════════════════════
pf.plot_job_ladders(
    wage_age_dict      = filtered_wage_age_dfs,
    calib_date         = CALIB_DATE,
    n_trajectories     = 50,
    min_years_observed = 5,
    colors             = MODEL_COLORS,
    save               = SAVE,
    path               = OUTPUT_PATH,
)

# ── placeholder calls for plots we will build this session ────────────────────
# pf.plot_wage_expectation_gap(...)


