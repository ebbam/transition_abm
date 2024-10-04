# # Parameter Inference and Calibration
#import sys
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, 'Z:/transition_abm/calibration/')

# Import packages
from abm_funs import *
import os
import pyabc
import csv
import tempfile
import statistics
import numpy as np
import pandas as pd
import random as random
import matplotlib.pyplot as plt
from pyabc.visualization import plot_kde_matrix
import math as math
from statistics import mode
from pyabc.transition import MultivariateNormalTransition
import seaborn as sns
from IPython import display
from pstats import SortKey
print("Packages installed.")

rng = np.random.default_rng()
path = "calibration/"
###################################
# US MODEL CONDITIONS AND DATA ####
###################################
# Set preliminary parameters for delta_u, delta_v, and gamma 
# - these reproduce nice Beveridge curves but were arrived at non-systematically
del_u = 0.015
del_v = 0.009
behav_spec = False
gamma_u = gamma_v = gamma = 0.1

A = pd.read_csv(path + "dRC_Replication/data/occupational_mobility_network.csv", header=None)
employment = round(pd.read_csv(path + "dRC_Replication/data/ipums_employment_2016.csv", header = 0).iloc[:, [4]]/10000)
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

print("Data imported.")

net_temp, vacs = initialise(len(mod_data['A']), mod_data['employment'].to_numpy(), mod_data['unemployment'].to_numpy(), mod_data['vacancies'].to_numpy(), mod_data['demand_target'].to_numpy(), mod_data['A'], mod_data['wages'].to_numpy(), mod_data['gend_share'].to_numpy(), 0, 3)
print("Network and vacs initialised.")

# The following are the input data to the function that runs the model. Perhaps not an appropriately named dictionary
# as only a few of the dictionary elements are parameters. 

parameter = {'mod_data': mod_data, # mod_data: occupation-level input data
    # (ie. employment/uneployment levels, wages, gender ratio, etc.).
     # net_temp: occupational network
     'net_temp': net_temp,
     # list of available vacancies in the economy
     'vacs': vacs, 
     # whether or not to enable behavioural element or not (boolean value)
     'behav_spec': behav_spec,
     # number of time steps to iterate the model - for now always exclude ~50 time steps for the model to reach a steady-state
     'time_steps': 300,
     # del_u: spontaneous separation rate
     'd_u': del_u,
     # del_v: spontaneous vacancy rate
     'd_v': del_v,
     # gamma: "speed" of adjustment to target demand of vacancies and unemployment
     'gamma': gamma,
     # bus_cycle_len: length of typical business cycle (160 months as explained above)
     'bus_cycle_len': 160,
     # delay: number of time steps to exclude from calibration sample to allow model 
     # to reach steady state and expansion phase of business cycle - this is certainly inefficient and should be changed.
     'delay': 120}


# The following line runs one base example of the model itself with the parameters outlined above. 
# This demonstrates the issue I mention above about needing time to achieve a "steady-state" from the initialised state of about 50 time steps.
# Run model without behavioural spec
rec = run_single(**parameter)
print("Single run executed.")

# ## Parameter inference
# 
# ### Observed Values/Data
# 
# Reference values for the various calibration steps are loaded and plotted below for reference. Thus far, I have included variables from the JOLTS (total nonfarm job openings rate, separation rate, quits rate, hires rate - all in seasonally adjusted monthly values). Additionally, I include the seasonally adjusted monthly unemployment rate and quarterly real GDP (although the latter has not been used yet in this script). 
# 
# Recession dates are downloaded from FRED but sourced from NBER business cycle indicators mentioned above. For each time series, I include the source URL. 

# Observed unemployment rate
# Monthly, seasonally adjusted
# Source: https://fred.stlouisfed.org/series/UNRATE

unrate = pd.read_csv(path+"data/macro_vars/UNRATE.csv", delimiter=',', decimal='.')
unrate["DATE"] = pd.to_datetime(unrate["DATE"])
unrate["UER"] = unrate['UNRATE']/100
unrate['FD_UNRATE'] = pd.Series(unrate['UER']).diff()

# Monthly, seasonally adjusted job openings rate (total nonfarm)
# Source: https://fred.stlouisfed.org/series/JTSJOR

jorate = pd.read_csv(path+"data/macro_vars/JTSJOR.csv", delimiter=',', decimal='.')
jorate["DATE"] = pd.to_datetime(jorate["DATE"])
jorate["VACRATE"] = jorate['JTSJOR']/100
jorate['FD_VACRATE'] = pd.Series(jorate['VACRATE']).diff()

macro_observations = pd.merge(unrate, jorate, how = 'outer', on = 'DATE')

# Recession dates
# Source: https://fred.stlouisfed.org/series/USREC#:%7E:text=For%20daily%20data%2C%20the%20recession,the%20month%20of%20the%20trough

recessions = pd.read_csv(path+"data/macro_vars/USREC.csv", delimiter=',', decimal='.')
recessions["DATE"] = pd.to_datetime(recessions["DATE"])

# Real GDP
# Source: https://fred.stlouisfed.org/series/GDPC1
realgdp = pd.read_csv(path+"data/macro_vars/GDPC1.csv", delimiter=',', decimal='.')
realgdp["DATE"] = pd.to_datetime(realgdp["DATE"])
realgdp["REALGDP"] = realgdp['GDPC1']
realgdp['FD_REALGDP'] = pd.Series(realgdp['REALGDP']).diff()

macro_observations = pd.merge(macro_observations, realgdp, how = 'outer', on = 'DATE')

## JOLTS SURVEY: https://www.bls.gov/charts/job-openings-and-labor-turnover/hire-seps-rates.htm

# Separation rates (Total nonfarm): JOLTS Survey - monthly, seasonally adjusted
# Source: https://fred.stlouisfed.org/series/JTSTSR
seps = pd.read_csv(path+"data/macro_vars/JTSTSR.csv", delimiter=',', decimal='.')
seps["DATE"] = pd.to_datetime(seps["DATE"])
seps["SEPSRATE"] = seps['JTSTSR']/100
seps['FD_SEPSRATE'] = pd.Series(seps['SEPSRATE']).diff()

# Quits rate (Total nonfarm): JOLTS Survey - monthly, seasonally adjusted
# Source: https://fred.stlouisfed.org/series/JTSQUR
quits = pd.read_csv(path+"data/macro_vars/JTSQUR.csv", delimiter=',', decimal='.')
quits["DATE"] = pd.to_datetime(quits["DATE"])
quits["QUITSRATE"] = quits['JTSQUR']/100
quits['FD_QUITSRATE'] = pd.Series(quits['QUITSRATE']).diff()

jolts = pd.merge(quits, seps, how = 'left', on = 'DATE')

# Hires rate (Total nonfarm): JOLTS Survey - monthly, seasonally adjusted
# Source: https://fred.stlouisfed.org/series/JTSHIR
hires = pd.read_csv(path+"data/macro_vars/JTSHIR.csv", delimiter=',', decimal='.')
hires["DATE"] = pd.to_datetime(hires["DATE"])
hires["HIRESRATE"] = hires['JTSHIR']/100
hires['FD_HIRESRATE'] = pd.Series(hires['HIRESRATE']).diff()

jolts = pd.merge(jolts, hires, how = 'left', on = 'DATE')

# Incorporating one set of simulated data
sim_data = pd.DataFrame(rec['data'])
sim_data['PROV DATE'] = pd.date_range(start = "2010-01-01", periods = len(sim_data), freq = "ME")
sim_data['FD_SIMUER'] = pd.Series(sim_data['UER']).diff()
sim_data['FD_SIMVACRATE'] = pd.Series(sim_data['VACRATE']).diff()

# Non-recession period
fig, ax = plt.subplots()
macro_observations.plot.line(ax = ax, figsize = (8,5), x= 'DATE', y = 'UER', color = "blue", linestyle = "dotted")
macro_observations.plot.line(ax = ax, figsize = (8,5), x= 'DATE', y = 'VACRATE', color = "red", linestyle = "dotted")
recessions.plot.area(ax = ax, figsize = (8,5), x= 'DATE', color = "grey", alpha = 0.2)
sim_data.plot.line(ax = ax, x = 'PROV DATE', y = 'UER', color = "purple", label = "UER (sim.)")
sim_data.plot.line(ax = ax, x = 'PROV DATE', y = 'VACRATE', color = "pink", label = "VACRATE (sim.)")

sim_data['VACRATE'].mean()

plt.xlim("2010-01-01", "2019-12-01")
plt.ylim(0.01, 0.11)

# Add title and axis labels
plt.title('Fig. 6: Monthly US Unemployment Rate (Seasonally Adjusted)')
plt.xlabel('Time')
plt.ylabel('Monthly UER')
plt.xticks(rotation=45)

# Save the plot
plt.savefig(path + 'output/uer_vac_descriptive.jpg', dpi = 300)

print("Macro variables loaded and abbreviated.")

def pyabc_run_single(paramater):     
    res = run_single(**parameter)
    return res 

parameter = {'mod_data': mod_data,
          'net_temp': net_temp,
          'vacs': vacs, 
          'behav_spec': behav_spec,
          'time_steps': 220,
          'd_u': del_u,
          'd_v': del_v,
          'gamma': gamma,
          'bus_cycle_len': 160,
          'delay': 120}

# Proposed priors for d_u and d_v taken from the separations and 
# job openings rates modelled in the first few plots of this notebook
# These priors can of course be more carefully selected calculating directly from those rates....next steps
prior = pyabc.Distribution(d_u = pyabc.RV("uniform", 0.001, 0.1),
                          d_v = pyabc.RV("uniform", 0.001, 0.1),
                          gamma = pyabc.RV("uniform", 0.05, 0.2))

# distance function jointly minimises distance between simulated 
# mean of UER and vacancy rates to real-world UER and vacancy rates
# def distance_mean(x, y):
#     diff_uer = (x["UER"].mean(axis = 0)) - ((y["data"]["UER"]).mean(axis=0))
#     diff_vac = (x["VACRATE"].mean(axis = 0)) - ((y["data"]["VACRATE"]).mean(axis=0))
#     dist_mean = np.sqrt(diff_uer**2 + diff_vac**2)
#     return dist_mean

def distance(x, y):
    diff_uer = np.sum((x["UER"][0:100] - y["UER"])**2)
    diff_vac = np.sum((x["VACRATE"][0:100] - y["VACRATE"])**2)
    dist = np.sqrt(diff_uer + diff_vac)
    return dist

calib_sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs = 50)
abc = pyabc.ABCSMC(pyabc_run_single, prior, distance, population_size = 500, sampler = calib_sampler)

db_path = os.path.join(tempfile.gettempdir(), "test.db")

# The following creates the "reference" values from the observed data - I pull the non-recession or expansion period from 2010-2019.
observation = macro_observations.loc[(macro_observations['DATE'] >= '2010-01-01') & (macro_observations['DATE'] <= "2019-12-01")].reset_index()
buffer = int((len(observation) - parameter['time_steps'])/2)
obs_abbrev = (observation[buffer + (int(parameter['delay']/2)):(buffer + parameter['time_steps']) - int((parameter['delay']/2))]).reset_index()
data = {'UER': np.array(obs_abbrev['UER']),
        'VACRATE': np.array(obs_abbrev['VACRATE'])}

abc.new("sqlite:///" + db_path, data)

history = abc.run(minimum_epsilon=0.2, max_nr_populations=10)

print("Pyabc run finished.")



gt = {"d_u": jolts['SEPSRATE'].mean(axis = 0), "d_v": jolts['QUITSRATE'].mean(axis = 0), "gamma": 0.2}

df, w = history.get_distribution()
# scipy.describe(df['d_u'])

plot_kde_matrix(
    df,
    w,
    limits={"d_u": (0.001, 0.08), "d_v": (0.001, 0.08), "gamma": (0.04, 0.7)},
    refval=gt,
    refval_color='k',
)

sns.kdeplot(x = "d_u", y = "d_v", data = df, weights = w, cmap = "viridis", fill = True)
#plt.axvline(x = sum(df['d_u']*w))
#plt.axhline(y = sum(df['d_v']*w))
plt.savefig(path + 'output/kde_plot.jpg', dpi = 300)

sns.jointplot(x = "d_u", y = "d_v", kind = "kde", data = df, weights = w, cmap = "viridis_r")

plt.title("KDE Plot")
plt.savefig(path + 'output/joint_plot.jpg', dpi = 300)

# The following graphs shows simulation results using parameter combinations sampled from the original prior (worst fit), final posterior (better fit), and accepted parameter combinations from the final posterior distribution which gives the best fit. It seems the prior set is likelly too restrictive as the algorithm has a difficult time arriving at an adequate vacancy rate! To be explored further...The left (right) column shows the results for the UER (Vacancy rate) and the black line in each plot demonstrates the observed data from BLS and JOLTS.

####################################################################################
#### Prior and Posterior Distribution outputs versus Observed UER and Vacancy Rates
fig, axes = plt.subplots(3, 2, sharex=True)
fig.set_size_inches(8, 12)
n = 5  # Number of samples to plot from each category
#Plot samples from the prior
alpha = 0.5
for _ in range(n):
    parameter.update(prior.rvs())
    prior_sample = run_single(**parameter)
    #print(prior_sample)
    axes[0,0].plot(prior_sample["UER"], color="red", alpha=alpha)
    axes[0,1].plot(prior_sample["VACRATE"], color="red", alpha=alpha)

# Fit a posterior KDE and plot samples form it
posterior = MultivariateNormalTransition()
posterior.fit(*history.get_distribution(m=0))

for _ in range(n):
    parameter.update(posterior.rvs())
    posterior_sample = run_single(**parameter)
    axes[1,0].plot(posterior_sample["UER"], color="blue", alpha=alpha)
    axes[1,1].plot(posterior_sample["VACRATE"], color="blue", alpha=alpha)

# Plot the stored summary statistics
sum_stats = history.get_weighted_sum_stats_for_model(m=0, t=history.max_t)
for stored in sum_stats[1][:n]:
    axes[2,0].plot(stored["UER"], color="green", alpha=alpha)
    axes[2,1].plot(stored["VACRATE"], color="green", alpha=alpha)

# Plot the observed UER from BLS
for ax in axes[:,0]:
    obs_abbrev.plot(y="UER", ax=ax, color="black", linewidth=1.5)
    ax.legend().set_visible(False)
    ax.set_ylabel("UER")
    
# Plot the observed VACRATE from JOLTS
for ax in axes[:,1]:
    obs_abbrev.plot(y="VACRATE", ax=ax, color="black", linewidth=1.5)
    ax.legend().set_visible(False)
    ax.set_ylabel("VACANCY RATE")
    ax.yaxis.set_label_position("right")

fig.suptitle("Simulation Results using Parameters from Prior (sampled), Posterior (sampled), and Posterior (sampled & accepted)")
# Add a legend with pseudo artists to first plot
fig.legend(
    [
        plt.plot([0], color="red")[0],
        plt.plot([0], color="blue")[0],
        plt.plot([0], color="green")[0],
        plt.plot([0], color="black")[0],
    ],
    ["Prior", "Posterior", "Stored, accepted", "Observation"],
    bbox_to_anchor=(0.5, 0.9),
    loc="lower center",
    ncol=4,
)

plt.savefig(path + 'output/prior_post_selected_distributions_plot.jpg', dpi = 300)

# ## Testing Selected Parameters
# 
# Below I pull the weighted mean of the posterior. Not sure if this is the correct way to pull the triangulated parameter estimate...? Indeed, the model run with these parameters does not look good and both look lower than represented in the heat/contour maps above. The model results with these parameters look bad both with respect to replicating a Beveridge curve as well as we did earlier with hand-selected estimates (and you'll see by the warnings that the delta_u is likely too high....again, I think that this is becuause of poor choice of arguments to the SMCABC algorithm above. In other words, not quite there...to be improved...but getting closer :) 

d_u_hat = sum(df['d_u']*w)
print("d_u_hat: ", d_u_hat)

d_v_hat = sum(df['d_v']*w)
print("d_v_hat: ", d_v_hat)

gamma_hat = sum(df['gamma']*w)
print("gam_hat: ", gamma_hat)


parameters = {'mod_data': mod_data, 
             'net_temp': net_temp,
              'vacs': vacs, 
              'behav_spec': False,
              'time_steps': 220,
              'runs': 2,
              'd_u': d_u_hat,
              'd_v': d_v_hat,
              'gamma': gamma_hat,
              'bus_cycle_len': 160,
              'bus_amp': 0.02}


sim_record_f_all, net_nonbehav, net_nonbehav_list = run_sim(**parameters)

parameters.update({'behav_spec': True})
sim_record_t_all, net_behav, net_behav_list = run_sim(**parameters)


# Summary values for one run 
sim_record_t = pd.DataFrame(np.transpose(np.hstack(sim_record_t_all)))
sim_record_t.columns =['Sim', 'Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Target_Demand']
sim_record_f = pd.DataFrame(np.transpose(np.hstack(sim_record_f_all)))
sim_record_f.columns =['Sim', 'Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Target_Demand']

record1_t = sim_record_t[(sim_record_t['Sim'] == 0)].groupby(['Sim', 'Time Step']).sum().reset_index()
record1_f = sim_record_f[(sim_record_f['Sim'] == 0)].groupby(['Sim', 'Time Step']).sum().reset_index()

end_t = record1_t[(record1_t['Time Step'] == 220)]
end_f = record1_f[(record1_f['Time Step'] == 220)]

ue_vac_f = record1_f.loc[:,['Workers', 'Unemployment', 'Vacancies', 'Target_Demand']]
ue_vac_f['UE Rate'] = ue_vac_f['Unemployment'] / ue_vac_f['Workers']
ue_vac_f['Vac Rate'] = ue_vac_f['Vacancies'] / ue_vac_f['Target_Demand']
ue_vac_f = ue_vac_f[46:]

ue_vac_t = record1_t.loc[:,['Workers', 'Unemployment', 'Vacancies', 'Target_Demand']]
ue_vac_t['UE Rate'] = ue_vac_t['Unemployment'] / ue_vac_t['Workers']
ue_vac_t['Vac Rate'] = ue_vac_t['Vacancies'] / ue_vac_t['Target_Demand']
ue_vac_t = ue_vac_t[46:]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ue_vac_f = ue_vac_f[100:300]
ue_vac_t = ue_vac_t[100:300]

ax1.plot(ue_vac_f['UE Rate'], ue_vac_f['Vac Rate'])
ax1.scatter(ue_vac_f['UE Rate'], ue_vac_f['Vac Rate'], c=ue_vac_f.index, s=100, lw=0)
ax1.set_title("Non-behavioural")
ax1.set_xlabel("UE Rate")
ax1.set_ylabel("Vacancy Rate")

ax2.plot(ue_vac_t['UE Rate'], ue_vac_t['Vac Rate'])
ax2.set_title("Behavioural")
ax2.scatter(ue_vac_t['UE Rate'], ue_vac_t['Vac Rate'], c=ue_vac_t.index, s=100, lw=0) 
ax2.set_xlabel("UE Rate")
ax2.set_ylabel("Vacancy Rate")

    
fig.suptitle("USA Model Beveridge Curve", fontweight = 'bold')
fig.tight_layout()

plt.savefig(path + 'output/run_w_calib_params.jpg', dpi = 300)

# ## Save results for import into model run

calib_params = [{"Parameter": "d_u", 'Value': d_u_hat},
                {"Parameter": "d_v", "Value": d_v_hat},
                {"Parameter": "gamma", "Value": gamma_hat},]
print(calib_params)

with open('../data/calibrated_params.csv', 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames = ['Parameter', 'Value']) 
    writer.writeheader() 
    writer.writerows(calib_params) 

import csv
calib_params = {"d_u": [d_u_hat],
                "d_v": [d_v_hat],
                "gamma": [gamma_hat]}

with open('../data/calibrated_params.csv', 'w') as csvfile: 
   # pass the csv file to csv.writer.
    writer = csv.writer(csvfile)
     
    # convert the dictionary keys to a list
    key_list = list(calib_params.keys())
     
    # the length of the keys corresponds to
    # no. of. columns.
    writer.writerow(calib_params.keys())

    # corresponding values to the column
    writer.writerow([calib_params[x][0] for x in key_list])

# # IGNORE FROM HERE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ### Parameter Inference on a Single Parameter
# 
# #### Calibrating to observed UER
# The following section performs a first attempt at parameter inference to triangulate just one parameter (delta_u or separation rate) in relation to observed unemployment rate (mean only - will be expanded to additional moment(s)) in subsequent steps.

# def pyabc_run_single(paramater):
#     res = run_single(**parameter)
#     return res 

# parameter = {'mod_data': mod_data,
#           'net_temp': net_temp,
#           'vacs': vacs, 
#           'behav_spec': behav_spec,
#           'time_steps': 200,
#           'd_u': del_u,
#           'd_v': del_v,
#           'gamma': gamma,
#           'bus_cycle_len': 160,
#           'delay': 120}

# # distribution of del_u
# prior = pyabc.Distribution(d_u = pyabc.RV("uniform", 0.005, 0.07))

# # distance function on full time series
# # def distance(x, y):
# #     diff = x["data"]["UER"] - y["data"]["UER"]
# #     dist = np.sqrt(np.sum(diff**2))
# #     return dist

# # distance function on just the mean of UER
# def distance(x, y):
#     diff = ((x["data"]["UER"]).mean(axis = 0)) - ((y["data"]["UER"]).mean(axis=0))
#     dist = abs(diff)
#     return dist

# abc = pyabc.ABCSMC(pyabc_run_single, prior, distance, population_size=15)

# db_path = os.path.join(tempfile.gettempdir(), "test.db")

# # Non-recession period unemployment rate
# observation = macro_observations.loc[(macro_observations['DATE'] >= '2010-01-01') & (macro_observations['DATE'] <= "2019-12-01")].reset_index()
# buffer = int((len(observation) - parameter['time_steps'])/2)
# obs_abbrev = (observation[buffer + 25:(buffer + parameter['time_steps']) - 25]).reset_index()

# abc.new("sqlite:///" + db_path, {"data": obs_abbrev})

# history = abc.run(minimum_epsilon = 0.01, max_nr_populations=10)


# fig, ax = plt.subplots()
# for t in range(history.max_t + 1):
#     df, w = history.get_distribution(m=0, t=t)
#     pyabc.visualization.plot_kde_1d(
#         df,
#         w,
#         xmin=0,
#         xmax=0.1,
#         x="d_u",
#         xname=r"$\d_u$",
#         ax=ax,
#         label=f"PDF t={t}",
#     )
# ax.axvline(jolts['SEPSRATE'].mean(axis = 0), color="k", linestyle="dashed", label= "Separation Rate")
# ax.axvline(jolts['QUITSRATE'].mean(axis = 0),color="k", linestyle="dashed", label = "Quit Rate")
# ax.legend()

# fig, arr_ax = plt.subplots(1, 3, figsize=(12, 4))

# pyabc.visualization.plot_sample_numbers(history, ax=arr_ax[0])
# pyabc.visualization.plot_epsilons(history, ax=arr_ax[1])
# pyabc.visualization.plot_effective_sample_sizes(history, ax=arr_ax[2])

# fig.tight_layout()

# df, w = history.get_distribution()
# print(mode(df['d_u']))
# df.mean()

# from pyabc.transition import MultivariateNormalTransition

# fig, axes = plt.subplots(nrows=3, sharex=True)
# fig.set_size_inches(8, 12)
# n = 5  # Number of samples to plot from each category
# # Plot samples from the prior
# alpha = 0.5
# for _ in range(n):
#     prior_sample = run_single(**prior.rvs())
#     #print(prior_sample)
#     prior_sample['data'].plot.line(y="UER", ax=axes[0], color="C1", alpha=alpha)


# # Fit a posterior KDE and plot samples form it
# posterior = MultivariateNormalTransition()
# posterior.fit(*history.get_distribution(m=0))

# for _ in range(n):
#     posterior_sample = run_single(**posterior.rvs())
#     posterior_sample['data'].plot.line(y="UER", ax=axes[1], color="C0", alpha=alpha)


# # Plot the stored summary statistics
# sum_stats = history.get_weighted_sum_stats_for_model(m=0, t=history.max_t)
# for stored in sum_stats[1][:n]:
#     stored["data"].plot(y="UER", ax=axes[2], color="C2", alpha=alpha)


# # Plot the observation
# for ax in axes:
#     obs_abbrev.plot(y="UER", ax=ax, color="k", linewidth=1.5)
#     ax.legend().set_visible(False)
#     ax.set_ylabel("UER")

# # Add a legend with pseudo artists to first plot
# axes[0].legend(
#     [
#         plt.plot([0], color="C1")[0],
#         plt.plot([0], color="C0")[0],
#         plt.plot([0], color="C2")[0],
#         plt.plot([0], color="k")[0],
#     ],
#     ["Prior", "Posterior", "Stored, accepted", "Observation"],
#     bbox_to_anchor=(0.5, 1),
#     loc="lower center",
#     ncol=4,
# )


# #### Calibrating to observed UER & observed Vacancy rate
# The following section performs a first attempt at parameter inference to triangulate just one parameter (delta_u or separation rate) in relation to both observed unemployment rate and vacancy rate (mean only - will be expanded to additional moment(s) in subsequent steps).
# 

# # distance function on both mean of UER and vacancy rates
# def distance(x, y):
#     diff_uer = ((x["data"]["UER"]).mean(axis = 0)) - ((y["data"]["UER"]).mean(axis=0))
#     diff_vac = ((x["data"]["VACRATE"]).mean(axis = 0)) - ((y["data"]["VACRATE"]).mean(axis=0))
#     dist = np.sqrt(diff_uer**2 + diff_vac**2)
#     return dist

# abc = pyabc.ABCSMC(pyabc_run_single, prior, distance, population_size=15)

# db_path = os.path.join(tempfile.gettempdir(), "test.db")

# # Non-recession period unemployment rate
# observation = macro_observations.loc[(macro_observations['DATE'] >= '2010-01-01') & (macro_observations['DATE'] <= "2019-12-01")].reset_index()
# buffer = int((len(observation) - parameter['time_steps'])/2)
# obs_abbrev = (observation[buffer + 25:(buffer + parameter['time_steps']) - 25]).reset_index()

# abc.new("sqlite:///" + db_path, {"data": obs_abbrev})

# history = abc.run(minimum_epsilon = 0.015, max_nr_populations=10)


# fig, ax = plt.subplots()
# for t in range(history.max_t + 1):
#     df, w = history.get_distribution(m=0, t=t)
#     pyabc.visualization.plot_kde_1d(
#         df,
#         w,
#         xmin=0,
#         xmax=0.1,
#         x="d_u",
#         xname=r"$\d_u$",
#         ax=ax,
#         label=f"PDF t={t}",
#     )
# ax.axvline(jolts['SEPSRATE'].mean(axis = 0), color="k", linestyle="dashed", label= "Separation Rate")
# ax.axvline(jolts['QUITSRATE'].mean(axis = 0),color="k", linestyle="dashed", label = "Quit Rate")
# ax.legend()

# fig, arr_ax = plt.subplots(1, 3, figsize=(12, 4))

# pyabc.visualization.plot_sample_numbers(history, ax=arr_ax[0])
# pyabc.visualization.plot_epsilons(history, ax=arr_ax[1])
# pyabc.visualization.plot_effective_sample_sizes(history, ax=arr_ax[2])

# fig.tight_layout()

# ####################################################################################
# #### Prior and Posterior Distribution outputs versus Observed UER and Vacancy Rates
# fig, axes = plt.subplots(3, 2, sharex=True)
# fig.set_size_inches(8, 12)
# n = 5  # Number of samples to plot from each category
# # Plot samples from the prior
# alpha = 0.5
# for _ in range(n):
#     prior_sample = run_single(**prior.rvs())
#     #print(prior_sample)
#     prior_sample['data'].plot.line(y="UER", ax=axes[0,0], color="C1", alpha=alpha)
#     prior_sample['data'].plot.line(y="VACRATE", ax=axes[0,1], color="C1", alpha=alpha)


# # Fit a posterior KDE and plot samples form it
# posterior = MultivariateNormalTransition()
# posterior.fit(*history.get_distribution(m=0))

# for _ in range(n):
#     posterior_sample = run_single(**posterior.rvs())
#     posterior_sample['data'].plot.line(y="UER", ax=axes[1,0], color="C0", alpha=alpha)
#     posterior_sample['data'].plot.line(y="VACRATE", ax=axes[1,1], color="C0", alpha=alpha)


# # Plot the stored summary statistics
# sum_stats = history.get_weighted_sum_stats_for_model(m=0, t=history.max_t)
# for stored in sum_stats[1][:n]:
#     stored["data"].plot(y="UER", ax=axes[2,0], color="C2", alpha=alpha)
#     stored["data"].plot(y="VACRATE", ax=axes[2,1], color="C2", alpha=alpha)


# # Plot the observation
# for ax in axes:
#     obs_abbrev.plot(y="UER", ax=ax, color="k", linewidth=1.5)
#     ax.legend().set_visible(False)
#     ax.set_ylabel("K")

# # Add a legend with pseudo artists to first plot
# axes[0].legend(
#     [
#         plt.plot([0], color="C1")[0],
#         plt.plot([0], color="C0")[0],
#         plt.plot([0], color="C2")[0],
#         plt.plot([0], color="k")[0],
#     ],
#     ["Prior", "Posterior", "Stored, accepted", "Observation"],
#     bbox_to_anchor=(0.5, 1),
#     loc="lower center",
#     ncol=4,
# )


