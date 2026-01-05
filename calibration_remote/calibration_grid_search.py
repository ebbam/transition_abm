# %% [markdown]
# # Parameter Inference and Calibration: Linking UER to GDP
# 
# View target demand as input to the model (instead of a stylized sinusoidal business cycle). 
# 
# I see two possible options:
# 1. GDP growth in line with Okun's Law specifically for unemployment - a 1% change in GDP: 0.03-0.05% change in UER
# 2. Growth in target demand
# 

# %%
# Import packages
import pandas as pd
from abm_funs import *
from plot_funs import *
from model_fun import *
from collate_macro_vars import *
import numpy as np
import random as random
import matplotlib.pyplot as plt
import tempfile
import pyabc
from scipy.stats import pearsonr, linregress
from pyabc.visualization import plot_kde_matrix, plot_kde_1d
import math as math
from pyabc.transition import MultivariateNormalTransition
import seaborn as sns
from PIL import Image
from pstats import SortKey
import datetime
from collate_macro_vars import *
import csv
from functools import partial

rng = np.random.default_rng()
test_fun()

path: "~/calibration_remote/"

import os
print(os.cpu_count()) 

calib = False
grid_search = True
save = False
parallelise = True

# %%

# Macro observations
observation = macro_observations.loc[
    (macro_observations['DATE'] >= calib_date[0]) & 
    (macro_observations['DATE'] <= calib_date[1])
].dropna(subset=["UNRATE", "VACRATE"]).reset_index()

# Load US_input data
A = pd.read_csv(path + "dRC_Replication/data/occupational_mobility_network.csv", header=None)
employment = round(pd.read_csv(path + "dRC_Replication/data/ipums_employment_2016.csv", header=0).iloc[:, [4]] / 10000)

# Crude approximation using avg unemployment rate of ~5% - should aim for occupation-specific unemployment rates
unemployment = round(employment * (0.05 / 0.95))

# Less crude approximation using avg vacancy rate - should still aim for occupation-specific vacancy rates
vac_rate_base = pd.read_csv(path + "dRC_Replication/data/vacancy_rateDec2000.csv").iloc[:, 2].mean() / 100
vacancies = round(employment * vac_rate_base / (1 - vac_rate_base))

# Needs input data...
demand_target = employment + vacancies
wages = pd.read_csv(path + "dRC_Replication/data/ipums_variables.csv")[['median_earnings']]
occ_ids = pd.read_csv(path + "dRC_Replication/data/ipums_variables.csv")[['id', 'acs_occ_code']]
gend_share = pd.read_csv(path + "data/ipums_variables_w_gender.csv")[['women_pct']]
experience_req = pd.read_csv(path + "dRC_Replication/data/ipums_variables_w_exp.csv")
seps_rates = pd.read_csv(path + "dRC_Replication/data/ipums_variables_w_seps_rate.csv")


mod_data = {
    "A": A,
    "employment": employment,
    'unemployment': unemployment,
    'vacancies': vacancies,
    'demand_target': demand_target,
    'wages': wages,
    'gend_share': gend_share,
    'entry_level': experience_req['entry_level'],
    # 'entry_age': experience_req['entry_age'],
    'experience_age': experience_req['experience_age'],
    'separation_rates': seps_rates['seps_rate']*10
}

###################################
# Initialise the model
##################################
net_temp, vacs = initialise(
    len(mod_data['A']),
    mod_data['employment'].to_numpy(),
    mod_data['unemployment'].to_numpy(),
    mod_data['vacancies'].to_numpy(),
    mod_data['demand_target'].to_numpy(),
    mod_data['A'],
    mod_data['wages'].to_numpy(),
    mod_data['gend_share'].to_numpy(),
    7, 3,
    mod_data['entry_level'],
    mod_data['experience_age'],
    mod_data['separation_rates']
)


# Shorten vac_df to the same length as gdp_dat using a moving average (if needed)
vac_df = observation['VACRATE'].to_numpy()
if len(vac_df) > len(gdp_dat):
    print("smoothing vac_df")
    # Apply moving average with window to smooth and match length
    window = len(vac_df) // len(gdp_dat)
    vac_dat = pd.Series(vac_df).rolling(window=window, min_periods=1).mean()[window-1::window].reset_index(drop=True)
    vac_dat = vac_dat[:len(gdp_dat)]
else:
    vac_dat = vac_df[:len(gdp_dat)]

print(len(vac_dat))
plt.plot(vac_dat, label="Vacancy Rate (smoothed)")


# %%
def harmonise_length(x, y):
    """
    GDP data used to calibrate has lower periodicity than the UER and VACRATE used for calibration. The following linearly interpolates the simulated output to match the frequency of the UER and Vacancy Rate data
    Harmonises the length of the time series to compare to each other. 
        
    Args:
        x (dict): Simulated data with keys "UER" and "VACRATE".
        y (dict): Real-world data with keys "UER" and "VACRATE".
    
    Returns:
        expanded_format: x expanded via linear interpolation - now of same length as y (observed time series of UER and vacancy rate)
    """
    expanded_format = pd.DataFrame({
        col: np.interp(
            np.linspace(0, len(x[col]) - 1, len(y[col])),
            np.linspace(0, len(x[col]) - 1, len(x[col])),
            x[col]
            )
            for col in x.keys()
            })
    return expanded_format

def distance_weighted(x, y): #weight_shape=0, weight_mean=1):
    x_ = harmonise_length(x, y)
    
    # Normalized SSE using variance
    uer_sse = np.sum((x_["UER"] - y["UER"])**2) / np.var(y["UER"])
    #vacrate_sse = np.sum((x_["VACRATE"] - y["VACRATE"])**2) / np.var(y["VACRATE"])

    # Weighted combination
    dist = (np.sqrt(uer_sse))
    return dist

jolts_obs = jolts[(jolts['DATE'] >= calib_date[0]) & (jolts['DATE'] <= calib_date[1])].reset_index()

def distance_uer_sep_rate(x, y): #weight_shape=0, weight_mean=1):
    x_ = harmonise_length(x, y)
    
    # Normalized SSE using variance
    uer_sse = np.sum((x_["UER"] - y["UER"])**2) / np.var(y["UER"])
    seprate_sse = np.sum((x_["SEPSRATE"] - y["SEPSRATE"])**2) / np.var(y["SEPSRATE"])

    # Weighted combination
    dist = (np.sqrt(uer_sse) + np.sqrt(seprate_sse))
    return dist


# %%
params = {'mod_data': mod_data, 
            'net_temp': net_temp,
            'vacs': vacs, 
            'time_steps': len(gdp_dat),
            'gdp_data': gdp_dat,
            "bus_confidence_dat": gdp_dat,
            'app_effort_dat': duration_to_prob_dict,
            "vac_data": vac_dat,
            'occ_shocks_data': occ_shocks_dat,
            'simple_res': True,
            'delay': 5,
            'steady_state': False}

calib_list = {
    "nonbehav": {"otj": False, # has been run
                           "cyc_otj": False, 
                           "cyc_ue": False, 
                           "disc": False},
              "otj_nonbehav": {"otj": True, # has been run
                           "cyc_otj": False, 
                           "cyc_ue": False, 
                           "disc": False},
    #           "otj_cyclical_e": {"otj": True,
    #                        "cyc_otj": True, 
    #                        "cyc_ue": False, 
    #                        "disc": False,
    #                        "bus_confidence_dat": bus_conf_dat},
            #   "otj_cyclical_ue ": {"otj": True,
            #                "cyc_otj": False, 
            #                "cyc_ue": True, 
            #                "disc": False},
            #   "otj_cyclical_e_ue": {"otj": True,
            #                "cyc_otj": True, 
            #                "cyc_ue": True, 
            #                "disc": False},
              "otj_cyclical_e_disc": {"otj": True,
                           "cyc_otj": True, 
                           "cyc_ue": False, 
                           "disc": True},
              # "otj_cyclical_ue_disc": {"otj": True,
              #              "cyc_otj": False, 
              #              "cyc_ue": True, 
              #              "disc": True},
              # "otj_cyclical_e_ue_disc": {"otj": True,
              #              "cyc_otj": True, 
              #              "cyc_ue": True, 
              #              "disc": True}
              "otj_disc": {"otj": True,
                            "cyc_otj": False, 
                            "cyc_ue": False, 
                            "disc": True}
            }


# %%
observation = macro_observations.loc[(macro_observations['DATE'] >= calib_date[0]) & (macro_observations['DATE'] <= calib_date[1])].dropna(subset = ["UNRATE", "VACRATE"]).reset_index()
# %%
# ===== grid_search_joblib.py =====
import os, csv, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from joblib import Parallel, delayed

# ------------------------------------------------------------------
# Avoid BLAS/MKL/OpenBLAS oversubscription inside each worker
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(v, "1")
# ------------------------------------------------------------------

# ---- worker function (top-level) ----
def eval_one(d_u, gamma_u, grid_params, obs_ss):
    """Run one (d_u, gamma_u) combo and return (d_u, gamma_u, distance, error_str)."""
    full_params = dict(grid_params)
    full_params.update({"d_u": d_u, "gamma_u": gamma_u})
    try:
        out = run_single_local(**full_params)  # <-- uses your existing function
        dist = distance_weighted(out, obs_ss) if isinstance(out, dict) else np.inf  # <-- your distance
        return (d_u, gamma_u, float(dist), "")
    except Exception as e:
        return (d_u, gamma_u, float("inf"), f"{type(e).__name__}: {e}")

# ----------------------------------------------------------
if __name__ == "__main__":

    # toggle like in your script
    if grid_search and parallelise:
        # your categories
        model_cats = ["otj_cyclical_e_disc", "otj_disc", "nonbehav", "otj_nonbehav"]

        # your grids (non-behav vs behav)
        nb_d_u_vals     = np.linspace(0.001, 0.2, 10)
        nb_gamma_u_vals = np.linspace(0.001, 0.2, 10)
        b_d_u_vals      = np.linspace(0.001, 0.2, 10)
        b_gamma_u_vals  = np.linspace(0.001, 0.2, 10)

        # cores from SLURM if present, else all cores - 6 (your choice)
        n_procs_env = 100
        n_jobs = max(1, 100)

        for model_cat in model_cats:
            # pick the correct grid
            if "nonbehav" in model_cat:
                d_u_vals, gamma_u_vals = nb_d_u_vals, nb_gamma_u_vals
            else:
                d_u_vals, gamma_u_vals = b_d_u_vals, b_gamma_u_vals

            total = len(d_u_vals) * len(gamma_u_vals)
            print(f"\n Starting grid search for {model_cat} with {total} parameter combinations on {n_jobs} jobs")

            setting = calib_list[model_cat]
            grid_params = {**params, **setting}

            observation_data = {
                "UER": observation["UER"].values,
                "VACRATE": observation["VACRATE"].values
                #'SEPSRATE': jolts_obs["SEPSRATE"].values
            }
            # --- steady-state observation vectors (your code) ---
            # observation_data_steady_state = {
            #     "UER":      np.full_like(observation["UER"].values,      np.nanmean(observation["UER"].values)),
            #     "VACRATE":  np.full_like(observation["VACRATE"].values,  np.nanmean(observation["VACRATE"].values)),
            #     "SEPSRATE": np.full_like(jolts_obs["SEPSRATE"].values,   np.nanmean(jolts_obs["SEPSRATE"].values)),
            # }

            # Build job list
            combos = list(product(d_u_vals, gamma_u_vals))

            # ---- run in parallel with joblib (loky backend) ----
            results = Parallel(
                n_jobs=n_jobs,
                backend="loky",     # robust, process-based
                batch_size="auto",  # lets joblib pick a good chunk size
                verbose=10          # prints progress
            )(
                delayed(eval_one)(d, g, grid_params, observation_data)
                for (d, g) in combos
            )

            # ---------- write the SAME CSV as your code ----------
            out_csv = os.path.expanduser(path + f"output/steady_state/grid_search_results_{model_cat}_test.csv")
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            write_header = not os.path.exists(out_csv)
            with open(out_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["d_u","gamma_u","distance","otj","cyc_otj","cyc_ue","disc","timestamp","model_cat","error"])
                for d_u, gamma_u, dist, err in results:
                    w.writerow([
                        d_u, gamma_u, dist,
                        setting['otj'], setting['cyc_otj'], setting['cyc_ue'], setting['disc'],
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        model_cat,
                        err
                    ])
                    if err:
                        print(f"✗ d_u={d_u:.4f}, gamma_u={gamma_u:.4f} → {err}")
                    else:
                        print(f"✓ d_u={d_u:.4f}, gamma_u={gamma_u:.4f}, dist={dist:.4f}")

            # ---------- pick best ----------
            grid_results = [(d, g, dist) for d, g, dist, _ in results]
            grid_results.sort(key=lambda x: x[2])
            best_d_u, best_gamma_u, best_dist = grid_results[0]
            print(f" Best parameters for {model_cat}: d_u={best_d_u:.4f}, gamma_u={best_gamma_u:.4f}, distance={best_dist:.4f}")

            # ---------- identical plotting as your script ----------
            best_params = dict(grid_params)
            best_params.update({"d_u": best_d_u, "gamma_u": best_gamma_u})
            model_output = run_single_local(**best_params)
            model_output_interp = harmonise_length(model_output, observation_data)
            date_range = pd.date_range(start=observation["DATE"].iloc[0],
                                       end=observation["DATE"].iloc[-1],
                                       periods=len(model_output_interp))

            # Best-Fit UER/SEPSRATE/VACRATE
            plt.figure(figsize=(12, 10))
            plt.subplot(2, 1, 1)
            plt.plot(observation['DATE'], observation["UER"],
                     label="Observed UER", color="black", linewidth=2)
            plt.plot(date_range, model_output_interp["UER"], label="Model UER", color="green", linestyle="--")
            plt.title(f"Best-Fit UER — {model_cat}")
            plt.ylabel("UER"); plt.legend(); plt.grid(True)

            # plt.subplot(3, 1, 2)
            # plt.plot(date_range, jolts_obs['SEPSRATE'], label="Observed SEPSRATE", color="black", linewidth=2)
            # plt.plot(date_range, model_output_interp["SEPSRATE"], label="Model SEPSRATE", color="blue", linestyle="--")
            # plt.title(f"Best-Fit SEPSRATE — {model_cat}")
            # plt.ylabel("SEPSRATE"); plt.xlabel("Date"); plt.legend(); plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(observation['DATE'],
                     observation['VACRATE'],
                     label="Observed VACRATE", color="black", linewidth=2)
            plt.plot(date_range, model_output_interp["VACRATE"], label="Model VACRATE", color="blue", linestyle="--")
            plt.title(f"Best-Fit VACRATE — {model_cat}")
            plt.ylabel("VACRATE"); plt.xlabel("Date"); plt.legend(); plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.expanduser(path + f"output/best_fit_uer_vacrate_{model_cat}_test.png"))
            plt.close()

            # Heatmap (distance surface) — same filename pattern
            du_vals = sorted(set(x[0] for x in grid_results))
            gu_vals = sorted(set(x[1] for x in grid_results))
            du_idx = {val: i for i, val in enumerate(du_vals)}
            gu_idx = {val: i for i, val in enumerate(gu_vals)}
            heatmap = np.full((len(gu_vals), len(du_vals)), np.nan)
            for d_u, gamma_u, dist in grid_results:
                heatmap[gu_idx[gamma_u], du_idx[d_u]] = dist

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                heatmap,
                xticklabels=[f"{v:.3f}" for v in du_vals],
                yticklabels=[f"{v:.3f}" for v in gu_vals],
                cmap="viridis", annot=True, fmt=".2f"
            )
            plt.title(f"Distance Surface — {model_cat}")
            plt.xlabel("d_u"); plt.ylabel("gamma_u")
            plt.tight_layout()
            plt.savefig(os.path.expanduser(path + f"output/grid_search_heatmap_{model_cat}_test.png"))
            plt.close()

# %%
#---------------- Best Parameters ----------------
if grid_search:
    model_cats = ["otj_cyclical_e_disc" , "otj_disc",  "nonbehav", "otj_nonbehav"]
    for model_cat in model_cats:
        import matplotlib.gridspec as gridspec

        grid_results = pd.read_csv(f'{path}output/steady_state/grid_search_results_{model_cat}_test.csv')
        grid_results = grid_results.sort_values(by='distance', ascending=True).reset_index(drop=True)
        best_d_u, best_gamma_u, best_dist = grid_results['d_u'].iloc[0], grid_results['gamma_u'].iloc[0], grid_results['distance'].iloc[0]
        print(f" Best parameters for {model_cat}: d_u={best_d_u:.4f}, gamma_u={best_gamma_u:.4f}, distance={best_dist:.4f}")

        setting = calib_list[model_cat]
        grid_params = {**params, **setting}
        best_params = grid_params.copy()
        best_params.update({"d_u": best_d_u, "gamma_u": best_gamma_u, 'simple_res': True})
        model_output = run_single_local(**best_params)
        date_range = pd.date_range(start=observation["DATE"].iloc[0], end=observation["DATE"].iloc[-1], periods=len(model_output['UER']))

        # Create a figure with 2 rows using gridspec
        fig = plt.figure(figsize=(8, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.5])

        # First plot: Best-Fit Time Series
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(observation['DATE'], observation["UER"], label="Observed UER", color="black", linewidth=2)
        #ax1.plot(observation['DATE'], np.full_like(observation["UER"].values, np.nanmean(observation["UER"].values)), label="Observed UER", color="black", linewidth=2)
        ax1.plot(date_range, model_output["UER"], label="Model UER", color="green", linestyle="--")
        ax1.set_title(f"Best-Fit UER — {model_cat}")
        ax1.set_ylabel("UER")
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(observation['DATE'], observation["VACRATE"], label="Observed VACRATE", color="black", linewidth=2, alpha=0.5)
        #ax2.plot(observation['DATE'], np.full_like(observation["VACRATE"].values, np.nanmean(observation["VACRATE"].values)), label="Observed VACRATE", color="black", linewidth=2)
        ax2.plot(date_range, model_output["VACRATE"], label="Model VACRATE", color="blue", linestyle="--", alpha=0.5)
        ax2.set_ylabel("VACRATE")
        ax2.legend(loc="lower right")
        ax2.grid(True)

        # Second plot: Heatmap
        ax3 = fig.add_subplot(gs[2])
        heatmap_data = grid_results.pivot_table(index='d_u', columns='gamma_u', values='distance')
        sns.heatmap(
            heatmap_data,
            cmap='viridis',
            annot=False,
            cbar_kws={'label': 'Distance'},
            xticklabels=[f"{x:.3f}" for x in heatmap_data.columns],
            yticklabels=[f"{y:.3f}" for y in heatmap_data.index],
            ax=ax3
        )
        ax3.set_xlabel('gamma_u')
        ax3.set_ylabel('d_u')
        ax3.set_title(f'Grid Search Results Heatmap — {model_cat}')

        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.expanduser(path + f"output/steady_state/best_fit_and_heatmap_{model_cat}_test.png"))
        plt.close(fig)



