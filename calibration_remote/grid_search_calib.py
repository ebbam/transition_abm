# Grid search for model parameters
from abm_funs import *
from model_fun import *
from collate_macro_vars import *
import numpy as np
import csv
import datetime
from itertools import product
import os

# -----------------------
# Define parameter grid
# -----------------------
d_u_vals = np.linspace(0.001, 0.05, 6)         # 6 steps between 0.001 and 0.05
gamma_u_vals = np.linspace(0.01, 0.5, 6)       # 6 steps between 0.01 and 0.5

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

params = {'mod_data': mod_data, 
            'net_temp': net_temp,
            'vacs': vacs, 
            'time_steps': len(gdp_dat),
            'delay': 100,
            'gdp_data': gdp_dat,
            "bus_confidence_dat": gdp_dat,
            'app_effort_dat': duration_to_prob_dict,
            "vac_data": vac_dat,
            'simple_res': True,
            'delay': 5}

# -----------------------
# Choose model setup
# -----------------------
model_cat = "otj_nonbehav"
setting = calib_list[model_cat]

# Merge params
d_u_vals = np.linspace(0.001, 0.05, 6)         # 6 steps between 0.001 and 0.05
gamma_u_vals = np.linspace(0.01, 0.5, 6)       # 6 steps between 0.01 and 0.5

# -----------------------
# Choose model setup
# -----------------------
model_cat = "otj_nonbehav"
setting = calib_list[model_cat]

# Merge params
grid_params = {**params, **setting}

# -----------------------
# Prepare output logging
# -----------------------
results_csv = os.path.expanduser(path + f"output/grid_search_results.csv")

if not os.path.exists(results_csv):
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d_u", "gamma_u", "distance", "otj", "cyc_otj", "cyc_ue", "disc", "timestamp", "model_cat"])

# -----------------------
# Run grid search
# -----------------------
grid_results = []

# Get observed data
observation_data = {
    "UER": observation["UER"].values,
    "VACRATE": observation["VACRATE"].values
}

print(f"Starting grid search over {len(d_u_vals) * len(gamma_u_vals)} combinations...")

for d_u, gamma_u in product(d_u_vals, gamma_u_vals):
    full_params = grid_params.copy()
    full_params.update({"d_u": d_u, "gamma_u": gamma_u})

    try:
        model_output = run_single_local(**full_params)

        if isinstance(model_output, dict):
            dist = distance_weighted(model_output, observation_data)
        else:
            dist = np.inf

        grid_results.append((d_u, gamma_u, dist))

        # Write to CSV
        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                d_u, gamma_u, dist,
                setting['otj'], setting['cyc_otj'], setting['cyc_ue'], setting['disc'],
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_cat
            ])

        print(f"✓ d_u={d_u:.4f}, gamma_u={gamma_u:.4f}, dist={dist:.4f}")

    except Exception as e:
        print(f"✗ d_u={d_u:.4f}, gamma_u={gamma_u:.4f} → ERROR: {e}")

# -----------------------
# Find best parameter set
# -----------------------
grid_results.sort(key=lambda x: x[2])
best_d_u, best_gamma_u, best_dist = grid_results[0]
print(f"\n✅ Best parameters: d_u={best_d_u:.4f}, gamma_u={best_gamma_u:.4f}, distance={best_dist:.4f}")