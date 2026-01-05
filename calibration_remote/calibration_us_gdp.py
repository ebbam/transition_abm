# Import packages
from abm_funs import *
import numpy as np
import pandas as pd
import random as random
import matplotlib.pyplot as plt
from dask.distributed import Client
import tempfile
import pyabc
from scipy.stats import pearsonr, linregress
from pyabc.visualization import plot_kde_matrix, plot_kde_1d
import math as math
from pyabc.transition import MultivariateNormalTransition
import seaborn as sns
from IPython.display import display
from PIL import Image
from pstats import SortKey
import datetime
from collate_macro_vars import *
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose
import csv
from dtaidistance import dtw
from functools import partial

rng = np.random.default_rng()
test_fun()

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"

import os
print(os.cpu_count()) 

# Run calibration?
calib = True
# Save output files and plots?
save = True
# Run with monthly frequency using interpolated GDP data? If false, quarterly frequency will be used following raw GDP structure.
monthly = False

# %%

#realgdp = macro_observations[["DATE", "REALGDP"]].dropna(subset=["REALGDP"]).reset_index()
realgdp['log_REALGDP'] = np.log2(realgdp['REALGDP'])

# GDP Filter
cycle, trend = hpfilter(realgdp['log_REALGDP'], lamb=129600)
 
# Adding the trend and cycle to the original DataFrame
realgdp['log_Trend'] = trend+1
realgdp['log_Cycle'] = cycle+1
realgdp['Trend'] = np.exp(trend)
realgdp['Cycle'] = np.exp(cycle)

realgdp_no_covid = realgdp[realgdp['DATE'] < "2019-10-1"].copy()
realgdp['scaled_log_Cycle'] = (realgdp['log_Cycle'] - realgdp['log_Cycle'].min()) / (realgdp['log_Cycle'].max() - realgdp['log_Cycle'].min())
realgdp_no_covid['scaled_log_Cycle'] = (realgdp_no_covid['log_Cycle'] - realgdp_no_covid['log_Cycle'].min()) / (realgdp_no_covid['log_Cycle'].max() - realgdp_no_covid['log_Cycle'].min())


# Different calibration windoes
# Full time series: "2024-5-1"
# calib_date = ["2004-12-01", "2019-05-01"]
calib_date = ["2000-12-01", "2019-05-01"]
# calib_date = ["2000-12-01", "2024-05-01"]
# Adapt gdp_dat_pd to have the same dates/frequency as observation
gdp_dat_pd = realgdp[(realgdp['DATE'] >= calib_date[0]) & (realgdp['DATE'] <= calib_date[1])]
observation = macro_observations.loc[(macro_observations['DATE'] >= calib_date[0]) & (macro_observations['DATE'] <= calib_date[1])].dropna(subset=["UNRATE", "VACRATE"]).reset_index()
if monthly:
    gdp_dat_pd = gdp_dat_pd.set_index('DATE').reindex(observation['DATE']).interpolate(method='linear').dropna().reset_index()
print(len(gdp_dat_pd))
print(len(observation))
#gdp_dat = np.array(gdp_dat_pd['scaled_log_Cycle'])# *0.06


###################################
# US MODEL CONDITIONS AND DATA ####
###################################

# Incorporate calibrated parameters
param_base_df = pd.read_csv(path + "output/output_26_03/calibrated_params_all.csv")
# if monthly:
#     param_base_df = pd.read_csv("output_04_02_monthly_freq/calibrated_params_all.csv")
# Sort by Timestamp in descending order
param_base_df = param_base_df.sort_values(by='Timestamp', ascending=False)
# Keep only the latest version of each parameter by removing duplicates
param_base_df = param_base_df.drop_duplicates(subset=['Parameter', 'Behav_Mode'], keep='first')

gdp_dat = np.array(gdp_dat_pd['log_Cycle'])

A = pd.read_csv(path+"dRC_Replication/data/occupational_mobility_network.csv", header=None)
employment = round(pd.read_csv(path+"dRC_Replication/data/ipums_employment_2016.csv", header = 0).iloc[:, [4]]/10000)
# Crude approximation using avg unemployment rate of ~5% - should aim for occupation-specific unemployment rates
unemployment = round(employment*(0.05/0.95))
# Less crude approximation using avg vacancy rate - should still aim for occupation-specific vacancy rates
vac_rate_base = pd.read_csv(path+"dRC_Replication/data/vacancy_rateDec2000.csv").iloc[:, 2].mean()/100
vacancies = round(employment*vac_rate_base/(1-vac_rate_base))
# Needs input data...
demand_target = employment + vacancies
wages = pd.read_csv(path+"dRC_Replication/data/ipums_variables.csv")[['median_earnings']]
gend_share = pd.read_csv(path+"data/ipums_variables_w_gender.csv")[['women_pct']]
mod_data =  {"A": A, "employment": employment, 
             'unemployment':unemployment, 'vacancies':vacancies, 
             'demand_target': demand_target, 'wages': wages, 'gend_share': gend_share}

net_temp, vacs = initialise(len(mod_data['A']), mod_data['employment'].to_numpy(), mod_data['unemployment'].to_numpy(), mod_data['vacancies'].to_numpy(), mod_data['demand_target'].to_numpy(), mod_data['A'], mod_data['wages'].to_numpy(), mod_data['gend_share'].to_numpy(), 0, 7)

params = {'mod_data': mod_data, # mod_data: occupation-level input data (ie. employment/uneployment levels, wages, gender ratio, etc.).
     'net_temp': net_temp, # net_temp: occupational network
     'vacs': vacs, # list of available vacancies in the economy
     'behav_spec': False, # whether or not to enable behavioural element or not (boolean value)
     'time_steps': len(gdp_dat), # number of time steps to iterate the model - for now always exclude ~50 time steps for the model to reach a steady-state
     # 'd_u': del_u, # del_u: spontaneous separation rate
     # 'd_v': del_v, # del_v: spontaneous vacancy rate
     # 'gamma_u': gamma_u, # gamma: "speed" of adjustment to target demand of vacancies and unemployment
     # 'gamma_v': gamma_v, # gamma: "speed" of adjustment to target demand of vacancies and unemployment
     'delay': 0,
     'gdp_data': gdp_dat,
     'simple_res': True} #

params.update(
    param_base_df[param_base_df['Behav_Mode'] == False].set_index('Parameter')['Value'].to_dict()
)


# # %%

# search_effort_dat = pd.read_csv("data/quarterly_search_ts.csv")
# search_effort_dat['DATE'] = pd.to_datetime(search_effort_dat['year'].astype(str) + '-' + (search_effort_dat['quarter'] * 3 - 2).astype(str) + '-01')
# if monthly:
#     search_effort_dat = pd.read_csv("data/monthly_search_ts.csv")
#     search_effort_dat['DATE'] = pd.to_datetime(search_effort_dat['year'].astype(str) + '-' + (search_effort_dat['month']).astype(str) + '-01')
# search_effort_dat = search_effort_dat[(search_effort_dat['DATE'] >= calib_date[0]) & (search_effort_dat['DATE'] <= calib_date[1])]
# search_effort_np = np.array(search_effort_dat['value_smooth'])
# search_effort_np = search_effort_np/search_effort_np.mean()
# #realgdp[(realgdp['DATE'] >= calib_date[0]) & (realgdp['DATE'] <= calib_date[1])]
# # Define a range of bus_cy values  # Generates 100 values from 0 to 1
# search_effort_values = [search_effort(0, 1-b) for b in gdp_dat]  # Apply function
# search_effort_values_dyn = [search_effort(0, (-1)*(1-b)) for b in search_effort_np]

# # Plot the results
# plt.figure(figsize=(8, 5))
# plt.plot(range(len(search_effort_values)), search_effort_values, marker='o', linestyle='-', color='b', label="Search Effort")
# plt.plot(range(len(search_effort_values_dyn)), search_effort_values_dyn, marker='o', linestyle='-', color='r', label="Search Effort TS")
# plt.xlabel("Business Cycle (bus_cy)")
# plt.ylabel("Search Effort (apps)")
# plt.title("Search Effort vs. Business Cycle")
# plt.legend()
# plt.grid(True)
# plt.show()


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

def distance_weighted(x, y, weight_shape=0.7, weight_mean=0.3):
    x_ = harmonise_length(x, y)
    
    # Normalized SSE using variance
    uer_sse = np.sum((x_["UER"] - y["UER"])**2) / np.var(y["UER"])
    vacrate_sse = np.sum((x_["VACRATE"] - y["VACRATE"])**2) / np.var(y["VACRATE"])

    # DTW for shape matching
    dtw_vacrate = dtw.distance(x_["VACRATE"], y["VACRATE"])/np.std(y['VACRATE'])
    dtw_uer = dtw.distance(x_["UER"], y["UER"])/np.std(y['UER'])

    #trend_penalty_uer = trend_penalty(x_["UER"], y["UER"])
    #trend_penalty_vac = trend_penalty(x_["VACRATE"], y["VACRATE"])

    # Weighted combination
    dist = weight_mean * (np.sqrt(uer_sse) + np.sqrt(vacrate_sse)) + weight_shape * (dtw_vacrate + dtw_uer) # + 0.4 * (trend_penalty_uer + trend_penalty_vac)
    return dist


# %%


if calib:
    # my_client = Client(n_workers=2, threads_per_worker=1)
    # Define possible distance functions and behav_spec values
    behav_spec_values = [False, True]
    print("1")
    # CSV filename
    csv_filename = os.path.expanduser(path + "output_18_04/calibrated_params_all.csv")
    print("2")
    # Ensure CSV file starts with headers
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Parameter", "Value", "Behav_Mode", "Timestamp"])
    print("3")

    for behav_spec_val in behav_spec_values:
        print(f"Running calibration with behav_spec={behav_spec_val}")
        if behav_spec_val:
            prior = pyabc.Distribution(d_u=pyabc.RV("uniform", 0.00001, 0.1),
                                        d_v=pyabc.RV("uniform", 0.00001, 0.1),
                                        gamma_u=pyabc.RV("uniform", 0.00001, 1.5),
                                        gamma_v=pyabc.RV("uniform", 0.00001, 0.7))
        else:
            prior = pyabc.Distribution(d_u=pyabc.RV("uniform", 0.00001, 0.1),
                                        d_v=pyabc.RV("uniform", 0.00001, 0.1),
                                        gamma_u=pyabc.RV("uniform", 0.00001, 1.7),
                                        gamma_v=pyabc.RV("uniform", 0.00001, 0.7))

        # Create a new version with different default values
        temp_run = partial(run_single_local, behav_spec=behav_spec_val)

        #########################################
        # Wrapper for pyabc ########
        #########################################
        def pyabc_run_single(parameter):
            res = temp_run(**parameter)
            return res

        # Set up ABC calibration
        calib_sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=2)

        abc = pyabc.ABCSMC(pyabc_run_single, prior, distance_weighted, population_size=2, sampler=calib_sampler)

        db_path = os.path.join(tempfile.gettempdir(), f"test_{behav_spec_val}.db")

        # Ensure observation is properly defined and contains the required columns
        if 'observation' not in locals() or observation.empty:
            raise ValueError("The 'observation' DataFrame is not properly defined or is empty. Ensure 'macro_observations' is correctly initialized and filtered.")
        if 'UER' not in observation.columns or 'VACRATE' not in observation.columns:
            raise ValueError("The 'observation' DataFrame must contain 'UER' and 'VACRATE' columns.")

        data = {'UER': np.array(observation['UER']),
                'VACRATE': np.array(observation['VACRATE'])}

        # Ensure the database path is valid and writable
        if not os.access(os.path.dirname(db_path), os.W_OK):
            raise PermissionError(f"Cannot write to the directory: {os.path.dirname(db_path)}")

        # Add a try-except block to catch and log any errors during the ABC run
        try:
            # Initialize a Dask client with proper resource allocation
            with Client(n_workers=2, threads_per_worker=1, memory_limit="2GB") as client:
                print(f"Dask client initialized: {client}")
                abc.new("sqlite:///" + db_path, data)
                history = abc.run(minimum_epsilon=0.1, max_nr_populations=2)
        except Exception as e:
            raise RuntimeError(f"Error during pyabc run: {e}")

        # Extract parameter estimates
        df, w = history.get_distribution(t=history.max_t)
        final_params = {
            "d_u": np.sum(df["d_u"] * w),
            "d_v": np.sum(df["d_v"] * w),
            "gamma_u": np.sum(df["gamma_u"] * w),
            "gamma_v": np.sum(df["gamma_v"] * w),
        }

        # Save parameter estimates to CSV
        with open(csv_filename, "a", newline="") as file:
            writer = csv.writer(file)
            for param, value in final_params.items():
                writer.writerow([param, value, behav_spec_val, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        # Generate and save plots
        plot_filename_base = f"output_18_04/calibration_behav_{behav_spec_val}"
        
        # KDE Matrix Plot
        plt.figure()
        plot_kde_matrix(
            df, 
            w, 
            limits={"d_u": (0.00001, 0.1), "d_v": (0.00001, 0.1), "gamma_u": (0.00001, 2), "gamma_v": (0.00001, 2)}, 
            refval=final_params, 
            names={"d_u": r"$\delta_u$", "d_v": r"$\delta_v$", "gamma_u": r"$\gamma_u$", "gamma_v": r"$\gamma_v$"}
        )
        plt.savefig(f"{plot_filename_base}_kde_matrix.png", bbox_inches='tight', pad_inches=0.5)
        plt.close()

        # Joint KDE Plot for d_u and d_v
        sns.jointplot(x="d_u", y="d_v", kind="kde", data=df, weights=w, cmap="viridis_r", marginal_kws={'fill': True})
        plt.xlabel(r'$\delta_u$')
        plt.ylabel(r'$\delta_v$')
        plt.axvline(x=np.sum(df["d_u"] * w), color="green", linestyle='dashed')
        plt.axhline(y=np.sum(df["d_v"] * w), color="green", linestyle='dashed')
        plt.suptitle(r"Kernel Density Estimate: $\delta_u$, $\delta_v$", y=1.02)
        plt.savefig(f"{plot_filename_base}_joint_delta.png", bbox_inches='tight', pad_inches=0.5)
        plt.close()

        # Joint KDE Plot for gamma_u and gamma_v
        sns.jointplot(x="gamma_u", y="gamma_v", kind="kde", data=df, weights=w, cmap="viridis_r", marginal_kws={'fill': True, 'color': 'green'})
        plt.xlabel(r'$\gamma_u$')
        plt.ylabel(r'$\gamma_v$')
        plt.axvline(x=np.sum(df["gamma_u"] * w), color="green", linestyle = 'dashed')
        plt.axhline(y=np.sum(df["gamma_v"] * w), color="green", linestyle='dashed')
        plt.suptitle(r"Kernel Density Estimate: $\gamma_u$, $\gamma_u$", y=1.02)
        plt.savefig(f"{plot_filename_base}_joint_gamma.png", bbox_inches='tight', pad_inches=0.5)
        plt.close()

        # Simulation Results Plot
        fig, axes = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches(8, 12)
        n = 15  # Number of samples to plot from each category
        alpha = 0.5

        sum_stats = history.get_weighted_sum_stats_for_model(m=0, t=history.max_t)
        for stored in sum_stats[1][:n]:
            stored_ = harmonise_length(stored, observation)
            axes[0].plot(stored_["UER"], color="green", alpha=alpha)
            axes[1].plot(stored_["VACRATE"], color="green", alpha=alpha)

        #for ax in axes[0]:
        observation.plot(y="UER", ax=axes[0], color="black", linewidth=1.5)
        axes[0].legend().set_visible(False)
        axes[0].set_ylabel("UER")

        #for ax in axes[1]:
        observation.plot(y="VACRATE", ax=axes[1], color="black", linewidth=1.5)
        axes[1].legend().set_visible(False)
        axes[1].set_ylabel("VACANCY RATE")
        ax.yaxis.set_label_position("right")

        fig.suptitle("Simulation Results using Parameters from Posterior (sampled & accepted)")

        fig.legend(
            [
                plt.plot([0], color="green")[0],
                plt.plot([0], color="black")[0],
            ],
            ["Stored, accepted", "Observation"],
            bbox_to_anchor=(0.5, 0.9),
            loc="lower center",
            ncol=4,
        )

        plt.savefig(f"{plot_filename_base}_sim_results.png", bbox_inches='tight', pad_inches=0.5)
        plt.close()


    print("Calibration complete. Results saved to CSV and plots saved as images.")

# %%

# # Load calibrated parameters from CSV
# if not calib and monthly:
#     param_df = pd.read_csv("output_04_02_monthly_freq/calibrated_params_all.csv")

# elif not calib and not monthly:
#     param_df = pd.read_csv("output/output_26_03_pre_OTJ/calibrated_params_all.csv")

elif calib: 
    param_df = pd.read_csv(path + "output_18_04/calibrated_params_all.csv")
# Sort by Timestamp in descending order
param_df = param_df.sort_values(by='Timestamp', ascending=False)

# Keep only the latest version of each parameter by removing duplicates
param_df = param_df.drop_duplicates(subset=['Parameter', 'Behav_Mode'], keep='first')
print(param_df)

final_params = {'mod_data': mod_data, 
            'net_temp': net_temp,
            'vacs': vacs, 
            'time_steps': len(gdp_dat),
            'delay': 0,
            'gdp_data': gdp_dat,
            'simple_res': False}

# Define the plot size
final_params.update({'behav_spec': False})

# Create independent copies for non-behavioral and behavioral parameter sets
non_behav_params = deepcopy(final_params)
non_behav_params.update(
    param_df[param_df['Behav_Mode'] == False].set_index('Parameter')['Value'].to_dict()
)

sim_record_f, sim_net_f, sum_stats_f = run_single_local(**non_behav_params)

behav_params = deepcopy(final_params)
behav_params.update(
    param_df[param_df['Behav_Mode'] == True].set_index('Parameter')['Value'].to_dict()
)
# behav_params.update({'behav_spec': True,
#                      'search_eff_ts': None})  # Ensure correct behavior flag

# # Run the model for behavioral case
# sim_record_t, sim_net_t, sum_stats_t = run_single_local(**behav_params)


# %%

# Summary values for one run 
# sim_record_t = pd.DataFrame(np.transpose(np.hstack(sim_record_t_all)))
sim_record_t = pd.DataFrame(sim_record_t)
sim_record_t.columns =['Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Current Demand', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers']
# sim_record_f = pd.DataFrame(np.transpose(np.hstack(sim_record_f_all)))
sim_record_f = pd.DataFrame(sim_record_f)
sim_record_f.columns =['Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Current Demand', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers']

record1_t = sim_record_t.groupby(['Time Step']).sum().reset_index()  
record1_f = sim_record_f.groupby(['Time Step']).sum().reset_index()

ue_vac_f = record1_f.loc[:,['Workers', 'Unemployment', 'LT Unemployed Persons', 'Current Demand', 'Vacancies', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers']]
ue_vac_f['UE Rate'] = ue_vac_f['Unemployment'] / ue_vac_f['Workers']
ue_vac_f['Vac Rate'] = ue_vac_f['Vacancies'] / ue_vac_f['Target_Demand']
ue_vac_f['LTUE Rate'] = ue_vac_f['LT Unemployed Persons'] / ue_vac_f['Unemployment']
ue_vac_f['PROV DATE'] = pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(sim_record_f))

ue_vac_t = record1_t.loc[:,['Workers', 'Unemployment', 'LT Unemployed Persons', 'Current Demand', 'Vacancies', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers']]
ue_vac_t['UE Rate'] = ue_vac_t['Unemployment'] / ue_vac_t['Workers']
ue_vac_t['Vac Rate'] = ue_vac_t['Vacancies'] / ue_vac_t['Target_Demand']
ue_vac_t['LTUE Rate'] = ue_vac_t['LT Unemployed Persons'] / ue_vac_t['Unemployment']
ue_vac_t['PROV DATE'] = pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(sim_record_t))

# First Figure: Long-term Unemployment Rates Over Time
fig1, ax1 = plt.subplots(figsize=(10, 7))  # Single plot for LTUE Rate
ax1.plot(ue_vac_f['PROV DATE'], ue_vac_f['LTUE Rate'], label="LTUE Rate (Non-behav)", color="red")
ax1.plot(ue_vac_t['PROV DATE'], ue_vac_t['LTUE Rate'], label="LTUE Rate (Behav)", color="lightcoral")
macro_observations.loc[(macro_observations['DATE'] >= calib_date[0]) & (macro_observations['DATE'] <= calib_date[1])].dropna(subset=["UNRATE", "VACRATE"]).reset_index().plot.line(
    ax=ax1, x='DATE', y='LTUER', color="green", linestyle="dotted", label="LTUER (Observed)"
)
ax1.set_xlabel('Date')
ax1.set_ylabel('Long-term Unemployment Rate')
ax1.set_title('Simulated Long-term Unemployment Rate (Simulated vs. Observed)')
ax1.legend()
plt.tight_layout()
if save:
    plt.savefig("output_18_04/long_term_unemployment_rate.png", dpi=300)
    plt.close()
else:
    plt.show()

# Second Figure: Monthly US Unemployment Rate and Vacancy Rates
fig2, ax2 = plt.subplots(figsize=(10, 5))  # Single plot for UER and Vacancy Rates
macro_observations.plot.line(ax=ax2, x='DATE', y='UER', color="blue", linestyle="dotted", label="UER (Observed)")
macro_observations.plot.line(ax=ax2, x='DATE', y='VACRATE', color="red", linestyle="dotted", label="VACRATE (Observed)")
recessions.plot.area(ax=ax2, x='DATE', color="grey", alpha=0.2)
ue_vac_f.plot.line(ax=ax2, x='PROV DATE', y='UE Rate', color="blue", label="UER (Sim. Non-behav.)")
ue_vac_f.plot.line(ax=ax2, x='PROV DATE', y='Vac Rate', color="red", label="VACRATE (Sim. Non-behav)")
ue_vac_t.plot.line(ax=ax2, x='PROV DATE', y='UE Rate', color="skyblue", label="UER (Sim. Behav.)")
ue_vac_t.plot.line(ax=ax2, x='PROV DATE', y='Vac Rate', color="lightcoral", label="VACRATE (Sim. Behav.)")
ax2.set_xlim(calib_date[0], calib_date[1])
ax2.set_ylim(0, 0.12)
ax2.set_title('Unemployment Rate & Vacancy Rates (Simulated vs. Observed)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Monthly UER')
ax2.tick_params(axis='x', rotation=45)
ax2.legend()
plt.tight_layout()
if save:
    plt.savefig("output_18_04/monthly_unemployment_and_vacancy_rates.png", dpi=300)
    plt.close()
else:
    plt.show()

# Second Figure: Beveridge Curve (Non-behavioural vs Behavioural)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

# Beveridge Curve - Non-behavioural
axes2[0].plot(ue_vac_f['UE Rate'], ue_vac_f['Vac Rate'], label="Simulated Values")
axes2[0].scatter(ue_vac_f['UE Rate'], ue_vac_f['Vac Rate'], c=ue_vac_f.index, s=100, lw=0)
axes2[0].plot(observation['UER'], observation['VACRATE'], label="Observed Values (2000-2019)", color="grey")
axes2[0].set_title("Non-behavioural")
axes2[0].set_xlabel("UE Rate")
axes2[0].set_ylabel("Vacancy Rate")
axes2[0].legend()

# Beveridge Curve - Behavioural
axes2[1].plot(ue_vac_t['UE Rate'], ue_vac_t['Vac Rate'], label="Simulated Values")
axes2[1].scatter(ue_vac_t['UE Rate'], ue_vac_t['Vac Rate'], c=ue_vac_t.index, s=100, lw=0)
axes2[1].plot(observation['UER'], observation['VACRATE'], label="Observed Values (2000-2019)", color="grey")
axes2[1].set_title("Behavioural")
axes2[1].set_xlabel("UE Rate")
axes2[1].set_ylabel("Vacancy Rate")
axes2[1].legend()
fig2.suptitle("Beveridge Curve (Simulated vs. Observed)", fontsize=16)

# Adjust layout and show/save
plt.tight_layout()
if save:
    plt.savefig("output_18_04/beveridge_curve_comparison.png", dpi=300)
    plt.close()
else:
    plt.show()
    

# %%

# Non-recession period
fig, ax = plt.subplots(2,2, figsize = (14,8), sharex = True)

ax[0,0].stackplot(pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(sim_record_f)), ue_vac_f['Employed Seekers'], ue_vac_f['Unemployed Seekers'], labels=['Employed Seekers (Non-behav.)', 'Unemployed Seekers (Non-behav.)'], colors=['blue', 'red'])
ax[0,1].stackplot(pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(sim_record_t)), ue_vac_t['Employed Seekers'], ue_vac_t['Unemployed Seekers'], labels=['Employed Seekers (Behav.)', 'Unemployed Seekers (Behav.)'], colors=['skyblue', 'lightcoral'])

ax[1,0].stackplot(
    pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(ue_vac_f)),
    ue_vac_f["Employed Seekers"] / (ue_vac_f["Employed Seekers"] + ue_vac_f["Unemployed Seekers"]),
    ue_vac_f["Unemployed Seekers"] / (ue_vac_f["Employed Seekers"] + ue_vac_f["Unemployed Seekers"]),
    labels=["Emp Seekers (Non-behav.)", "Unemp Seekers (Non-behav.)"],
    colors=["blue", "red"]
)

ax[1,1].stackplot(
    pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(ue_vac_t)),
    ue_vac_t["Employed Seekers"] / (ue_vac_t["Employed Seekers"] + ue_vac_t["Unemployed Seekers"]),
    ue_vac_t["Unemployed Seekers"] / (ue_vac_t["Employed Seekers"] + ue_vac_t["Unemployed Seekers"]),
    labels=["Emp Seekers (Behav.)", "Unemp Seekers (Behav.)"],
    colors=["skyblue", "lightcoral"]
)

# Set individual titles
ax[0,0].set_ylim(0,4000)
ax[0,1].set_ylim(0,4000)
ax[0,0].set_title("Total - Non-Behavioral Model")
ax[0,1].set_title("Total - Behavioral Model")

ax[1,0].set_title("Share - Non-Behavioral Model")
ax[1,1].set_title("Share - Behavioral Model")

# Set common axis labels
fig.suptitle("Monthly Composition of Job Seekers", fontsize=14)  # Figure-wide title
fig.supxlabel("Date")  # Shared x-axis label
fig.supylabel("Composition of Job Seekers")  # Shared y-axis label

# Rotate x-ticks properly
for a in ax.flatten():
    #a.set_xticklabels(a.get_xticklabels(), rotation=45)
    a.legend(loc="upper right")  # Apply legend to each subplot
    #recessions.plot.area(ax = a,  x= 'DATE', color = "grey", alpha = 0.2)

# Show plot
if save:
    plt.savefig(f"output_18_04/seeker_composition.png", dpi = 300)
    plt.close()
else: 
    plt.show()

# %%

# behav_params.update({'search_eff_ts': search_effort_np}) 
# # Run the model for behavioral case
# sim_record_t_ts, sim_net_t_ts, sum_stats_t_ts = run_single_local(**behav_params)

# # %%

# # Summary values for one run 
# # sim_record_t = pd.DataFrame(np.transpose(np.hstack(sim_record_t_all)))
# sim_record_t_ts = pd.DataFrame(sim_record_t_ts)
# sim_record_t_ts.columns =['Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Current Demand', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers']

# record1_t_ts = sim_record_t_ts.groupby(['Time Step']).sum().reset_index()  

# ue_vac_t_ts = record1_t_ts.loc[:,['Workers', 'Unemployment', 'LT Unemployed Persons', 'Vacancies', 'Target_Demand']]
# ue_vac_t_ts['UE Rate'] = ue_vac_t_ts['Unemployment'] / ue_vac_t_ts['Workers']
# ue_vac_t_ts['Vac Rate'] = ue_vac_t_ts['Vacancies'] / ue_vac_t_ts['Target_Demand']
# ue_vac_t_ts['LTUE Rate'] = ue_vac_t_ts['LT Unemployed Persons'] / ue_vac_t_ts['Unemployment']
# ue_vac_t_ts['PROV DATE'] = pd.date_range(start=calib_date[0], end=calib_date[1], periods=len(sim_record_t_ts))

# Create a figure with a 1-row, 3-column layout
fig, axes = plt.subplots(1,2,figsize=(17, 10))  # 3 rows, 1 column

### **Third Plot: Monthly US Unemployment Rate (Seasonally Adjusted)**
macro_observations.plot.line(ax=axes[0], x='DATE', y='UER', color="blue", linestyle="dotted", label="UER (Observed)")
macro_observations.plot.line(ax=axes[1], x='DATE', y='UER', color="blue", linestyle="dotted", label="UER (Observed)")
#macro_observations.plot.line(ax=axes[2], x='DATE', y='UER', color="blue", linestyle="dotted", label="UER (Observed)")
macro_observations.plot.line(ax=axes[0], x='DATE', y='VACRATE', color="red", linestyle="dotted", label="VACRATE (Observed)")
macro_observations.plot.line(ax=axes[1], x='DATE', y='VACRATE', color="red", linestyle="dotted", label="VACRATE (Observed)")
#macro_observations.plot.line(ax=axes[2], x='DATE', y='VACRATE', color="red", linestyle="dotted", label="VACRATE (Observed)")
recessions.plot.area(ax=axes[0], x='DATE', color="grey", alpha=0.2)
recessions.plot.area(ax=axes[1], x='DATE', color="grey", alpha=0.2)
#recessions.plot.area(ax=axes[2], x='DATE', color="grey", alpha=0.2)
ue_vac_f.plot.line(ax=axes[0], x='PROV DATE', y='UE Rate', color="skyblue", label="UER (Sim. Non-behav.)")
ue_vac_f.plot.line(ax=axes[0], x='PROV DATE', y='Vac Rate', color="lightcoral", label="VACRATE (Sim. Non-behav)")
ue_vac_t.plot.line(ax=axes[1], x='PROV DATE', y='UE Rate', color="skyblue", label="UER (Sim. Behav.)")
ue_vac_t.plot.line(ax=axes[1], x='PROV DATE', y='Vac Rate', color="lightcoral", label="VACRATE (Sim. Behav.)")
#ue_vac_t_ts.plot.line(ax=axes[2], x='PROV DATE', y='UE Rate', color="skyblue", label="UER (Sim. Behav. w SE TS)")
#ue_vac_t_ts.plot.line(ax=axes[2], x='PROV DATE', y='Vac Rate', color="lightcoral", label="VACRATE (Sim. Behav. w SE TS)")

axes[0].set_xlim(calib_date[0], calib_date[1])
axes[1].set_xlim(calib_date[0], calib_date[1])
axes[2].set_xlim(calib_date[0], calib_date[1])
axes[0].set_ylim(0, 0.2)
axes[1].set_ylim(0, 0.2)
axes[2].set_ylim(0, 0.2)
axes[0].set_title('Monthly US Unemployment and Vacancy Rate (Seasonally Adjusted)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Monthly UER & Vac Rate')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()
axes[1].set_xlabel('Time')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()
axes[2].set_xlabel('Time')
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend()


# Adjust layout to prevent overlap
plt.tight_layout()

if save:
    plt.savefig(f"output_18_04/results_comp.png", dpi = 300)
    plt.close()
# Show the final figure with all 3 subplots
else: 
    plt.show()

# %%

# --- Second Figure: Bottom Two Subplots ---
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

# Third Plot (Bottom-Left: Non-behavioural Beveridge Curve)
#axes2[0].plot(ue_vac_f['PROV DATE'], ue_vac_f['Employment'], label="Emp Non-behavioural")
axes2[0].plot(ue_vac_f['PROV DATE'], ue_vac_f['Unemployment'], label="Unemp Non-behavioural")
axes2[0].plot(ue_vac_f['PROV DATE'], ue_vac_f['Vacancies'], label="Vacs Non-behavioural")
axes2[0].set_title("Non-behavioural")
# axes2[0].set_ylim(0.015, 0.055)
# axes2[0].set_xlim(0.03, 0.125)
axes2[0].legend()

# Fourth Plot (Bottom-Right: Behavioural Beveridge Curve)
# Third Plot (Bottom-Left: Non-behavioural Beveridge Curve)
#axes2[1].plot(ue_vac_t['PROV DATE'], ue_vac_t['Employment'], label="Emp Behavioural")
axes2[1].plot(ue_vac_t['PROV DATE'], ue_vac_t['Unemployment'], label="Unemp Behavioural")
axes2[1].plot(ue_vac_t['PROV DATE'], ue_vac_t['Vacancies'], label="Vacs Behavioural")
axes2[1].set_title("Behavioural")
# axes2[1].set_ylim(0.015, 0.055)
# axes2[1].set_xlim(0.03, 0.125)
axes2[1].legend()

plt.tight_layout()
# Save the figure to the output folder
if save:
    plt.savefig(f"output_18_04/bev_curves.png", dpi=300)
    plt.close()  # Close figure to free memory
else:
    plt.show()

# --- Second Figure: Bottom Two Subplots ---
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

# Third Plot (Bottom-Left: Non-behavioural Beveridge Curve)
#axes2[0].plot(ue_vac_f['PROV DATE'], ue_vac_f['Employment'], label="Emp Non-behavioural")
axes2[0].plot(ue_vac_f['PROV DATE'], ue_vac_f['Current Demand'], label="CD Non-behavioural")
axes2[0].plot(ue_vac_f['PROV DATE'], ue_vac_f['Target_Demand'], label="TD Non-behavioural")
axes2[0].set_title("Non-behavioural")
# axes2[0].set_ylim(0.015, 0.055)
# axes2[0].set_xlim(0.03, 0.125)
axes2[0].legend()

# Fourth Plot (Bottom-Right: Behavioural Beveridge Curve)
# Third Plot (Bottom-Left: Non-behavioural Beveridge Curve)
#axes2[1].plot(ue_vac_t['PROV DATE'], ue_vac_t['Employment'], label="Emp Behavioural")
axes2[1].plot(ue_vac_t['PROV DATE'], ue_vac_t['Current Demand'], label="CD Behavioural")
axes2[1].plot(ue_vac_t['PROV DATE'], ue_vac_t['Target_Demand'], label="TD Behavioural")
axes2[1].set_title("Behavioural")
# axes2[1].set_ylim(0.015, 0.055)
# axes2[1].set_xlim(0.03, 0.125)
axes2[1].legend()

plt.tight_layout()
# Save the figure to the output folder
if save:
    plt.savefig(f"output_18_04/demand_curves.png", dpi=300)
    plt.close()  # Close figure to free memory
else:
    plt.show()

# %%

fig, ax = plt.subplots(1, 3, figsize=(20, 7), sharey = True)
print("--------------------")
for i, k in enumerate([net_temp, sim_net_f, sim_net_t]):
    emp_counter = 0
    women = 0
    men = 0

    w_wages = []
    m_wages = []

    w_wage = 0
    m_wage = 0

    for occ in k:
        emp_counter += len(occ.list_of_employed)
        women += len([wrkr for wrkr in occ.list_of_employed if wrkr.female])
        men += len([wrkr for wrkr in occ.list_of_employed if not(wrkr.female)])
        w_wages.extend([wrkr.wage for wrkr in occ.list_of_employed if wrkr.female])
        m_wages.extend([wrkr.wage for wrkr in occ.list_of_employed if not(wrkr.female)])
        w_wage += sum([wrkr.wage for wrkr in occ.list_of_employed if wrkr.female])
        m_wage += sum([wrkr.wage for wrkr in occ.list_of_employed if not(wrkr.female)])
    
        
    t= " \n" + " \n" + "Female share of employed: " + str(round((women/emp_counter)*100)) + "% \n" + "Mean Female Wage: $" + str(round(w_wage/women)) + "\n" + "Mean Male Wage: $" + str(round(m_wage/men)) + "\n" + "Gender wage gap: " + str(round(100*(1 - (w_wage/women)/(m_wage/men)))) + "%" + "\n" + "--------------------"

    n_bins = 10
    women = np.array(w_wages)
    men = np.array(m_wages)

    # We can set the number of bins with the *bins* keyword argument.
    ax[i].hist(women, bins=n_bins, alpha = 0.3, color = 'purple', label = 'Women', fill = True, hatch = '.')
    ax[i].hist(men, bins=n_bins, alpha = 0.3, label = 'Men', color = 'green', fill = True, hatch = '.')  
    ax[i].axvline(women.mean(), color='purple', linestyle='dashed', linewidth=1, label = 'Women Avg.')
    ax[i].axvline(men.mean(), color='green', linestyle='dashed', linewidth=1, label = 'Men Avg.')
    ax[i].legend(loc='upper right') 
    ax[i].annotate(t, xy = (0, -0.3), xycoords='axes fraction')
    ax[0].set_title('Initialised State')
    ax[1].set_title('Non-behavioural')
    ax[2].set_title('Behavioural')  

    ax[0].set_xlabel('Wage')
    ax[1].set_xlabel('Wage')
    ax[2].set_xlabel('Wage')  
    for axis in ax:
        axis.set_ylabel('Number of Workers')

fig.suptitle('Distribution of Male and Female Wages', fontsize = 15) 
fig.subplots_adjust(bottom=0.1)

if save:
    plt.savefig(f'output_18_04/gender_wage_gaps.jpg', dpi = 300)
    plt.close()
else:
    plt.show()



# %%

fig, ax = plt.subplots(figsize=(10, 7), sharex = True)
names = ['Non-behavioural', 'Behavioural']
colors = ['blue', 'orange']
for i, k in enumerate([sim_net_f, sim_net_t]):
    w_time_unemp = []
    m_time_unemp = []

    for occ in k:
        w_time_unemp.extend([wrkr.time_unemployed for wrkr in occ.list_of_unemployed if wrkr.female])
        m_time_unemp.extend([wrkr.time_unemployed for wrkr in occ.list_of_unemployed if not(wrkr.female)])

    women = np.array(w_time_unemp)
    men = np.array(m_time_unemp)
    total = np.array(w_time_unemp)
    n_bins = 10

    # We can set the number of bins with the *bins* keyword argument.
    ax.hist(total, bins=n_bins, alpha = 0.5, label = names[i])
    ax.axvline(total.mean(), linestyle='dashed', color = colors[i], linewidth=1, label = f'{names[i]} (mean)')
    #ax.axvline(total.max(), color='grey', linestyle='dashed', linewidth=1, label = 'Max Total')
    ax.legend(loc='upper right') 
    ax.set_title('Non-behavioural')
    ax.set_title('Behavioural')  
    
    #ax[i,1].hist(women, bins=n_bins, alpha = 0.5, label = 'Women')
    #ax[i,1].hist(men, bins=n_bins, alpha = 0.5, label = 'Men')  
    #ax[i,1].axvline(women.mean(), color='blue', linestyle='dashed', linewidth=1, label = 'Women Avg.')
    #ax[i,1].axvline(men.mean(), color='orange', linestyle='dashed', linewidth=1, label = 'Men Avg.')
    #ax[i,1].axvline(women.max(), color='blue', alpha = 0.5, linestyle='dotted', linewidth=1, label = 'Women Max')
    #ax[i,1].axvline(men.max(), color='orange', alpha = 0.5, linestyle='dotted', linewidth=1, label = 'Men Max')
    ax.legend(loc='upper right')
    ax.set_title('Initialised State: Total')
    #ax[0,1].set_title('Initialised State: by Gender')
    ax.set_title('Non-behavioural: Total')
    #ax[1,1].set_title('Non-behavioural: by Gender')
    ax.set_title('Behavioural: Total')  
    #ax[2,1].set_title('Behavioural: by Gender')
    ax.set_xlabel("Time Unemployed (simulated months)")
    ax.set_ylabel("Number of Workers")
    #ax.set_yscale("log")

fig.suptitle('Distribution of Time Spent Unemployed in Model') 
if save:
    plt.savefig(f'output_18_04/ltuer_distributions.jpg', dpi = 300)
    plt.close()
else:
    plt.show()