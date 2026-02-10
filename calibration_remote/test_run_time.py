# %%
# CPU LIMITING CONFIGURATION - ADD THESE AT THE VERY TOP
# ============================================================================

import time

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

save_button = False
suffix = ""

complete_nw = False
steady_state_run = False
dropbox = False
calib_source_folder = "cos_calib_smart_hire_3"
if calib_source_folder == "cos_calib_smart_hire_3":
    suffix = "smart_hire"
    dumb_hire_spec = False
elif calib_source_folder == "cos_calib":
    suffix = ""
    dumb_hire_spec = True
else: 
    raise KeyError("Need to specify a dumb_hire_spec with the param source folder!.")

if dropbox:
    output_prefix = f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/{calib_source_folder}/"
    cached_output_path = f'output/{calib_source_folder}/'
else:
    output_prefix = f'output/{calib_source_folder}/'
    cached_output_path = f'output/{calib_source_folder}/'


for nw in ["full_omn"]: 
    print(nw)  
    which_params = f'{calib_source_folder}/{nw}/'
    for run in ["base"]:
        calib_date = ["2000-12-01", "2019-05-01"]
        output_path = os.path.expanduser(f'{output_prefix}{nw}/figures/')

        # Macro observations
        observation = macro_observations.loc[
            (macro_observations['DATE'] >= calib_date[0]) & 
            (macro_observations['DATE'] <= calib_date[1])].dropna(subset=["UNRATE", "VACRATE"]).reset_index()

        mod_data, net_temp, vacs, occ_ids, occ_shocks_dat = network_input_builder(nw, complete_nw, calib_date)

        param_df1 = pd.read_csv(path + f"output/{which_params}calibrated_params_all{suffix}.csv")
        param_df2 = pd.read_csv(path + f"output/{which_params}calibrated_params_all_theta_signal{suffix}.csv")
        param_df = pd.concat([param_df1, param_df2], ignore_index=True)


        # Sort by Timestamp in descending order
        param_df = param_df.sort_values(by='Timestamp', ascending=False)

        params = {
            'mod_data': mod_data,
            'dumb_hire': dumb_hire_spec,
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
            # Apply moving average with window to smooth and match length
            window = len(vac_df) // occ_shocks_dat.shape[1]
            vac_dat = Series(vac_df).rolling(window=window, min_periods=1).mean()[window-1::window].reset_index(drop=True)
            vac_dat = vac_dat[:occ_shocks_dat.shape[1]]
        else:
            vac_dat = vac_df[:occ_shocks_dat.shape[1]]

            
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

            start_time = time.time()

            # Run the model
            run_single_local(**test_params)
            print("--- %s seconds ---" % (time.time() - start_time))

