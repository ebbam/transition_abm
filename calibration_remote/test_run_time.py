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
calib_source_folder = "cos_calib_smart_hire_4"
quick_pickle = False
if calib_source_folder == "cos_calib_smart_hire_4":
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

if not quick_pickle: 
    for nw in ["full_omn"]:
        print(nw)  
        which_params = f'{calib_source_folder}/{nw}/'
        for run in ["base"]:
            calib_date = ["2001-01-01", "2024-05-01"]
            output_path = os.path.expanduser(f'{output_prefix}{nw}/figures/')

            # Macro observations
            observation = macro_observations.loc[
                (macro_observations['DATE'] >= calib_date[0]) & 
                (macro_observations['DATE'] <= calib_date[1])].dropna(subset=["UNRATE", "VACRATE"]).reset_index()

            mod_data, net_temp, vacs, occ_ids, occ_shocks_dat = network_input_builder(nw, complete_nw, calib_date, delay=24)

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
                'delay': 24,
                'gdp_data': gdp_dat,
                'app_effort_dat': duration_to_prob_dict,
                'occ_shocks_data': occ_shocks_dat,
                'vac_method': 'cost_red',
                'simple_res': False,
                'mistake_rate': 0.1
            }

            # Shorten vac_df to the same length as gdp_dat using a moving average (if needed)
            vac_df = observation['VACRATE'].to_numpy()
            if len(vac_df) > observation.shape[0]:
                # Apply moving average with window to smooth and match length
                window = len(vac_df) // occ_shocks_dat.shape[1]
                vac_dat = Series(vac_df).rolling(window=window, min_periods=1).mean()[window-1::window].reset_index(drop=True)
                #vac_dat = vac_dat[:occ_shocks_dat.shape[1]]
                vac_dat = vac_dat[:observation.shape[0]]  # calibration window only
            else:
                #vac_dat = vac_df[:occ_shocks_dat.shape[1]]
                vac_dat = vac_df[:observation.shape[0]]

            print(f"Vac_dat length: {len(vac_dat)}")
            print(f"observation length: {len(observation)}")
                
            # %%
            calib_list = {
                # "nonbehav": {"otj": False, # has been run
                #                     "cyc_otj": False, 
                #                     "cyc_ue": False, 
                #                     "disc": False,
                #                     "delay": 24,
                #                     "bus_confidence_dat": gdp_dat,
                #                     "emp_apps": 1,
                #                     'wage_prefs': False,
                #                     'vac_data': vac_dat,
                #                     'strict_rw': True},
                #         "otj_nonbehav": {"otj": True, # has been run
                #                     "cyc_otj": False, 
                #                     "cyc_ue": False, 
                #                     "disc": False, 
                #                     "delay": 2425,
                #                     'wage_prefs': False,
                #                     'emp_apps': 1,
                #                     "bus_confidence_dat": gdp_dat,
                #                         "vac_data": vac_dat,
                #                         'strict_rw': True},
                        # "otj_cyclical_e_disc": {"otj": True,
                        #             "cyc_otj": True, 
                        #             "cyc_ue": False, 
                        #             "disc": True,
                        #             "delay": 24,
                        #             'emp_apps': 1,
                        #             'wage_prefs': True,
                        #             "bus_confidence_dat": gdp_dat,
                        #             "vac_data": vac_dat,
                        #             'strict_rw': False},
                        # "otj_cyclical_e_disc_strict_rw": {"otj": True,
                        #             "cyc_otj": True, 
                        #             "cyc_ue": False, 
                        #             "disc": True,
                        #             "delay": 24,
                        #             'emp_apps': 1,
                        #             'wage_prefs': True,
                        #             "bus_confidence_dat": gdp_dat,
                        #             "vac_data": vac_dat,
                        #             'strict_rw': True},
                        # "otj_cyclical_e_disc_no_rw": {"otj": True,
                        #             "cyc_otj": True, 
                        #             "cyc_ue": False, 
                        #             "disc": True,
                        #             "delay": 24,
                        #             'wage_prefs': False,
                        #             'emp_apps': 1,
                        #             "bus_confidence_dat": gdp_dat,
                        #             "vac_data": vac_dat,
                        #             'strict_rw': True},
                        # "otj_disc": {"otj": True,
                        #             "cyc_otj": False, 
                        #             "cyc_ue": False, 
                        #             "disc": True,
                        #             "delay": 24,
                        #             'emp_apps': 1,
                        #             'wage_prefs': True,
                        #             "bus_confidence_dat": gdp_dat,
                        #             "vac_data": vac_dat,
                        #             'strict_rw': False},
                        # # "otj_disc_strict_rw": {"otj": True,
                        # #             "cyc_otj": False, 
                        # #             "cyc_ue": False, 
                        # #             "disc": True,
                        # #             "delay": 24,
                        # #             'emp_apps': 1,
                        # #             'wage_prefs': True,
                        # #             "bus_confidence_dat": gdp_dat,
                        # #             "vac_data": vac_dat,
                        # #             'strict_rw': True},
                        "otj_disc_no_rw": {"otj": True,
                                    "cyc_otj": False, 
                                    "cyc_ue": False, 
                                    "disc": True,
                                    "delay": 24,
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
                                'd_v': 0.0001,
                                'gamma_v':0.025}
                else:
                    calib_params = {'gamma_u': 0.22 + param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_u"),"Value"].iloc[0],
                                'd_u': -0.004 + param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_u"),"Value"].iloc[0],
                                'theta': None,
                                'd_v': 0.01,
                                'gamma_v': 0.025}

                print(calib_params)
                test_params.update(calib_params)

                # Update with the values from the calib_list
                test_params.update(item)
                if run == "steady_state":
                    test_params.update({"steady_state": True,
                                        "time_steps": 1000,
                                        "delay": 24})
                else:
                    test_params.update({"steady_state": False})

                start_time = time.time()

                # Run the model
                record_df, grouped, net, data, seekers_rec, avg_wage_offer_diff_df, app_load_df, ue_spell_origin_df, vacs_temp, wage_age_df, ue_stock_df, transition_df = run_single_local(**test_params)

                print("--- %s seconds ---" % (time.time() - start_time))
                # print(grouped.columns)

                plot_wage_comparison({'model_name': record_df}, wages, 
                                        calib_date, 
                                        save = save_button)

                grouped['DATE'] = pd.date_range(
                        start=calib_date[0],
                        periods=len(grouped),
                        freq="ME"
                    )
                with open('save_run_time_test_record_df.pkl', 'wb') as file:
                    pickle.dump(record_df, file)
                with open('save_run_time_test_grouped.pkl', 'wb') as file:
                    pickle.dump(grouped, file)

                with open('save_run_time_test_wage_age_df.pkl', 'wb') as file:
                    pickle.dump(wage_age_df, file)
else:
        # Macro observations
    observation = macro_observations.loc[
        (macro_observations['DATE'] >= calib_date[0]) & 
        (macro_observations['DATE'] <= calib_date[1])].dropna(subset=["UNRATE", "VACRATE"]).reset_index()

    with open('save_run_time_test_grouped.pkl', 'rb') as file:
        grouped = pickle.load(file)
    
    with open('save_run_time_test_wage_age_df.pkl', 'rb') as file:
        wage_age_df = pickle.load(file)

    with open('save_run_time_test_record_df.pkl', 'rb') as file:
        record_df = pickle.load(file)


plot_uer_vac({'model_name': grouped}, observation,
                            sep=False, save=save_button,
                            free_date_scale=False,
                            calib_date = calib_date)

plot_gender_pay_gap_ts({'model_name': grouped}, save=save_button)

###############################################################################################################
#################### Applications per unemployed job-seeker ###################################################
###############################################################################################################

seekers_rec['DATE'] = pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(seekers_rec))
seekers_rec['Application Effort (U)'] = seekers_rec['Applications Sent (Unemployed)'] / seekers_rec['Unemployed Seekers']
seekers_rec['Application Effort (E)'] = seekers_rec['Applications Sent (Employed)'] / seekers_rec['Employed Seekers']

plt.figure(figsize=(10, 6))
plt.plot(seekers_rec['DATE'], seekers_rec['Application Effort (U)'], marker='o')
plt.xlabel('Year')
plt.ylabel('Applications per Unemployed Seeker')
plt.title('Application Rate Over Time')
plt.tight_layout()
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), 
            frameon=True, fontsize=10)

# Now save
#plt.savefig(os.path.join(output_path, "applications_per_unemployed_seeker.png"), bbox_inches='tight',  
#pad_inches=0.1,      
#dpi=300)
plt.show()
plt.close()

# plot_wage_comparison({'model_name': record_df}, wages, 
#                                         calib_date, 
#                                         save = save_button)
                    

# plot_occ_market_tightness({'model_name': record_df},
#                                               calib_date=calib_date,
#                                               save=save_button,
#                                               path=output_path)

# # Retirement rate
# plot_retirement_rate_grid({'model_name': grouped}, calib_date)

# # Wage distributions by year
# plot_wage_dist_by_year({'model_name': wage_age_df})

# # Wage–age correlation
# plot_wage_age_corr_over_time({'model_name': wage_age_df})
# plot_wage_age_growth_scatter({'model_name': wage_age_df})

# plot_occ_wage_growth({'model_name': wage_age_df})
# plot_wage_vs_age_growth_rates({'model_name': wage_age_df})



# print(record_df['Mean Occupational Wage'])
# print(record_df['Employment'])
# result = (record_df
#     .dropna(subset=['Mean Occupational Wage'])
#     .groupby('Time Step')
#     .apply(lambda x: np.average(x['Mean Occupational Wage'], weights=x['Employment']))
#     .to_frame(name='Average Economy-Wide Wage')
#     .reset_index())

# result_vacs = (record_df
#     .dropna(subset=['Mean Vacancy Offer'])
#     .groupby('Time Step')
#     .apply(lambda x: np.average(x['Mean Vacancy Offer'], weights=x['Employment']))
#     .to_frame(name='Average Economy-Wide Vacancy Offer')
#     .reset_index())

# result['DATE'] = pd.date_range(
#                         start=calib_date[0],
#                         periods=len(result),
#                         freq="ME"
#                     )

# result_vacs['DATE'] = pd.date_range(
#                         start=calib_date[0],
#                         periods=len(result_vacs),
#                         freq="ME"
#                     )

# wages = pd.read_csv(path+"data/macro_vars/LES1252881600Q.csv", delimiter=',', decimal='.')
# wages['DATE'] = pd.to_datetime(wages["observation_date"])
# wages.rename(columns={'LES1252881600Q': 'Observed Wages'}, inplace=True)

# wages = wages[(wages['DATE'] >= calib_date[0]) & (wages['DATE'] <= calib_date[1])]

# smoothed_vac_wage = result_vacs['Average Economy-Wide Vacancy Offer'].rolling(window=6, center=True, min_periods=1).mean()

# plt.plot(result['DATE'], result['Average Economy-Wide Wage'], label = 'Average Economy-Wide Wage')
# plt.plot(result_vacs['DATE'], smoothed_vac_wage, label = 'Average Economy-Wide Vacancy Offer (6m MA)')

# #plt.plot(wages['DATE'], wages['Observed Wages']*52*3.15, color = "grey", linestyle = "dashed", label = "Median usual weekly real earnings")
# plt.suptitle("Simulated and Observed Wages")
# plt.title("Comparison of Wages versus Vacancy Wages", fontsize = 12)
# plt.tight_layout()
# plt.legend()
# plt.show()

