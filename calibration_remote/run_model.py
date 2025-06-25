# %% [markdown]
# # Running Model
# 
# Calibrated params from calibration_us_gdp.ipynb

# %%
# Import packages
from abm_funs import *
from plot_funs import*
# from us_input import *
import numpy as np
from copy import deepcopy 
import pandas as pd
import random as random
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import math as math
import importlib
from copy import deepcopy
from pandas import Series
import os
import seaborn as sns
import datetime
from collate_macro_vars import *
from statsmodels.tsa.filters import hp_filter, bk_filter, cf_filter
from quantecon import hamilton_filter
from statsmodels.tsa.seasonal import seasonal_decompose
import csv

rng = np.random.default_rng()
test_fun()

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"

import os
print(os.cpu_count()) 

save = False

# %%
# Search Effort Time Series
search_effort_dat = pd.read_csv("data/quarterly_search_ts.csv")
search_effort_dat['DATE'] = pd.to_datetime(search_effort_dat['year'].astype(str) + '-' + (search_effort_dat['quarter'] * 3 - 2).astype(str) + '-01')
search_effort_np = np.array(search_effort_dat['value_smooth'])
search_effort_np = search_effort_np/search_effort_np.mean()
#realgdp[(realgdp['DATE'] >= calib_date[0]) & (realgdp['DATE'] <= calib_date[1])]

# Define a range of bus_cy values  # Generates 100 values from 0 to 1
search_effort_values = [search_effort(0, b, False, 0.1) for b in gdp_dat]  # Apply function
search_effort_bk_values = [search_effort(0, b, False, 0.1) for b in gdp_dat_bk]  # Apply function
search_effort_hamilton_values = [search_effort(0, b, False, 0.1) for b in gdp_dat_hamilton]  # Apply function
# def search_effort_ts(t_unemp, se):
#     apps = max(0, round(10 - 100*(1-se)))
#     # if discouraged:
#     #     apps = round(a_stable/((t_unemp)**2 + 1)) + 1
#     return apps
search_effort_values_dyn = [search_effort_ts(0, b) for b in search_effort_np]

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(range(len(search_effort_values)), search_effort_values, marker='o', linestyle='-', color='b', label="Search Effort")
plt.plot(range(len(search_effort_bk_values)), search_effort_bk_values, marker='o', linestyle='-', color='g', label="Search Effort BK")
plt.plot(range(len(search_effort_hamilton_values)), search_effort_hamilton_values, marker='o', linestyle='-', color='purple', label="Search Effort Hamilton")

plt.plot(range(len(search_effort_values_dyn)), search_effort_values_dyn, marker='o', linestyle='-', color='r', label="Search Effort TS")
plt.xlabel("Business Cycle (bus_cy)")
plt.ylabel("Search Effort (apps)")
plt.title("Search Effort vs. Business Cycle")
plt.legend()
plt.grid(True)
plt.show()

# %%
series_dict = {}
for i, a in enumerate([0.3, 0.2, 0.1, 0.05, 0.01]):
    efforts = []
    for t_unemp in range(15):
        apps = search_effort(t_unemp, 1, True, a)
        efforts.append(apps)

    series_dict[f"Alpha: {a}"] = {
        "x": list(range(0, len(efforts))),
        "y": efforts
        }

for i, a in series_dict.items():
    print(a)
    if i == 'Alpha: 0.1':
        plt.plot(a['x'], a['y'],marker='o', linestyle='-', label=i)
    else:
        plt.plot(a['x'], a['y'],  linestyle='-', label=i)
    plt.xlabel("Unemployment Duration (quarters)")
    plt.ylabel("Applications Sent")

   
plt.axvline(x=8, color='black', linestyle='--', linewidth=1, label='2 years')
plt.axvline(x=12, color='grey', linestyle='--', linewidth=1, label='3 years')
plt.legend(loc='upper right')
plt.grid(True)
plt.suptitle("Applications Sent by Unemployment Duration")
plt.tight_layout()
plt.show()

print(series_dict)
def utility_fun(lambda_param, utility):
    if lambda_param == 1:
        return(np.log(utility))
    else:
        return((utility**(1-lambda_param))/(1-lambda_param))

lambdas = [-0.5, -0.3, 0, 0.3, 0.5]
utilities = [0, 0.2, 0.5, 1, 2]

plt.figure(figsize=(10, 6))

for lambda_p in lambdas:
    utility_outputs = []
    for utility in utilities:
        utility_outputs.append(utility_fun(lambda_p, utility))
    plt.plot(utilities, utility_outputs, marker='o', label=f'λ={lambda_p}')

plt.xlabel('Utility Input')
plt.ylabel('Utility Output') 
plt.title('Utility Function Output for Different λ Values')
plt.grid(True)
plt.legend()
plt.show()



# %% [markdown]
# ### Unemployed Search Effort Time Series

# %%


# Macro observations
observation = macro_observations.loc[(macro_observations['DATE'] >= calib_date[0]) & (macro_observations['DATE'] <= calib_date[1])].dropna(subset=["UNRATE", "VACRATE"]).reset_index()
# Load US_input data
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
occ_ids = pd.read_csv(path+"dRC_Replication/data/ipums_variables.csv")[['id', 'acs_occ_code']]
gend_share = pd.read_csv(path+"data/ipums_variables_w_gender.csv")[['women_pct']]
mod_data =  {"A": A, "employment": employment, 
             'unemployment':unemployment, 'vacancies':vacancies, 
             'demand_target': demand_target, 'wages': wages, 'gend_share': gend_share}

###################################
# Initialise the model
##################################
net_temp, vacs = initialise(len(mod_data['A']), mod_data['employment'].to_numpy(), mod_data['unemployment'].to_numpy(), mod_data['vacancies'].to_numpy(), mod_data['demand_target'].to_numpy(), mod_data['A'], mod_data['wages'].to_numpy(), mod_data['gend_share'].to_numpy(), 0, 0)

# observation = macro_observations.loc[(macro_observations['DATE'] >= calib_date[0]) & (macro_observations['DATE'] <= calib_date[1])].dropna(subset = ["UNRATE", "VACRATE"]).reset_index()
# Load calibrated parameters from CSV
param_df = pd.read_csv("output_06_06/calibrated_params_all.csv")
# Sort by Timestamp in descending order
param_df = param_df.sort_values(by='Timestamp', ascending=False)

params = {'mod_data': mod_data, 
            'net_temp': net_temp,
            'vacs': vacs, 
            'time_steps': len(gdp_dat),
            'delay': 100,
            'gdp_data': gdp_dat}

# Shorten vac_df to the same length as gdp_dat using a moving average (if needed)   
vac_df = observation['VACRATE'].to_numpy()
if len(vac_df) > len(gdp_dat):
    print("smoothing vac_df")
    # Apply moving average with window to smooth and match length
    window = len(vac_df) // len(gdp_dat)
    vac_dat = Series(vac_df).rolling(window=window, min_periods=1).mean()[window-1::window].reset_index(drop=True)
    vac_dat = vac_dat[:len(gdp_dat)]
else:
    vac_dat = vac_df[:len(gdp_dat)]
    
print(len(vac_dat))
plt.plot(vac_dat, label="Vacancy Rate (smoothed)")


# %%
####################
# Model Run ########
####################
def run_single_local( #behav_spec, 
                    d_u, 
                    #d_v,
                    gamma_u,
                    #gamma_v,
                    otj,
                    cyc_otj, 
                    cyc_ue, 
                    disc,
                    mod_data = mod_data, 
                    net_temp = net_temp, 
                    vacs = vacs, 
                    time_steps = len(gdp_dat), # set equal to length of gdp_data
                    delay = 100,
                    gdp_data = gdp_dat,
                    bus_confidence_dat = gdp_dat,
                    simple_res = False, 
                    vac_data = vac_dat):
    
    """ Runs the model once
    Argsuments:
       behav_spec: whether or not to run the behavioural model
       data: data required of initialise function  
       time_steps: Number of time steps for single model run
       d_u: parameter input to separation probability
       d_v: parameter input to vacancy opening probability

    Returns:
       dataframe of model run results
    """
    # Records variables of interest for plotting
    # Initialise deepcopy occupational mobility network
    #print(behav_spec)
    record = [] 
    #print(parameter['vacs'])
    vacs_temp = deepcopy(vacs)
    net = deepcopy(net_temp)
    seekers_rec = []
    time_steps = time_steps + delay

    for t in range(time_steps):
        #if t == 1:
            #print(behav_spec)
        if t <= delay:
            curr_bus_cy = 1
            bus_conf = 1
            ue_bc = 1
            vr_t = 0.03
        if t > delay:
            curr_bus_cy = gdp_data[t-delay]
            bus_conf = bus_confidence_dat[t-delay]
            ue_bc = curr_bus_cy
            vr_t = vac_data[t-delay]
        if not cyc_ue:
            ue_bc = 1
        # search_eff_curr = search_eff_ts[t]
        # Ensure number of workers in economy has not changed
        #tic = time.process_time()
        emp_seekers = 0
        unemp_seekers = 0
        u_apps = 0
        u_searchers = 0
        for occ in net:
            ### APPLICATIONS
            # Questions to verify:
            # - CANNOT be fired and apply in same time step ie. time_unemployed > 0
            # - CAN be rejected and apply in the same time step - no protected attribute
            # isolate list of vacancies in economy that are relevant to the occupation
            # - avoids selecting in each search_and_apply application
            r_vacs = [vac for vac in vacs_temp if occ.list_of_neigh_bool[vac.occupation_id]]          
    
            for u in occ.list_of_unemployed:
                unemp_seekers += 1
                # this one if only using simple scaling factor for the search effort
                u.search_and_apply(net, r_vacs, disc, ue_bc, 0.1)
                # use the following if we wish to incorporate the entire TS of search effort
                #u.search_and_apply(net, r_vacs, behav_spec, search_eff_curr)
            
            if otj:
                # For both models, a mean of 40% of employed workers are searching for new jobs
                # This fluctuates with the business cycle in the behavioural model in line with gdp
                if cyc_otj:
                    search_scaling = curr_bus_cy*0.07
                # Static mean in the non-behavioural model
                else:
                    search_scaling = 0.07
                for e in random.sample(occ.list_of_employed, int(search_scaling*len(occ.list_of_employed))):
                    emp_seekers += 1
                    e.emp_search_and_apply(net, r_vacs, disc)

            u_apps += sum(wrkr.apps_sent for wrkr in occ.list_of_unemployed if  wrkr.apps_sent is not None)
            u_searchers += len(occ.list_of_unemployed)

            ### SEPARATIONS
            try:
                occ.separate_workers(d_u, gamma_u, curr_bus_cy)
            except Exception as e:
                return np.inf


        ### HIRING
        # Ordering of hiring randomised to ensure list order does not matter in filling vacancies...
        # Possibly still introduces some bias...this seems to be where the "multiple offer" challenge Maria mentioned comes from
        # ....might be better to do this using an unordered set?
        for v_open in sorted(vacs_temp,key=lambda _: random.random()):
            # Removes any applicants that have already been hired in another vacancy
            v_open.applicants[:] = [app for app in v_open.applicants if not(app.hired)]
            v_open.time_open += 1
            if len(v_open.applicants) > 0:
                v_open.hire(net)
                v_open.filled = True
                #vacs.remove(v_open)
                assert(len(v_open.applicants) == 0)
            else:
                pass

        vacs_temp = [v for v in vacs_temp if not(v.filled) and v.time_open <= 1] 

        # # Reset counters for record in time t
        # empl = 0 
        # unemp = 0
        # n_ltue = 0
        # curr_demand = 0
        # t_demand = 0

        ### OPEN VACANCIES
        # Update vacancies after all shifts have taken place
        # Could consider making this a function of the class itself?
        for occ in net:
            u_rel_wage = sum(wrkr.ue_rel_wage for wrkr in occ.list_of_employed if wrkr.hired and wrkr.ue_rel_wage is not None)
            e_rel_wage = sum(wrkr.ee_rel_wage for wrkr in occ.list_of_employed if wrkr.hired and wrkr.ee_rel_wage is not None)
            ue = len([w for w in occ.list_of_employed if w.hired and w.ue_rel_wage is not None])
            ee = len([w for w in occ.list_of_employed if w.hired and w.ee_rel_wage is not None])
            # Update time_unemployed and long-term unemployed status of unemployed workers
            # Remove protected "hired" attribute of employed workers
            occ.update_workers()
            # Assert that all unemployed people have spent 1 or more time periods unemployed
            assert(sum([worker.time_unemployed <= 0 for worker in occ.list_of_unemployed]) == 0)
            # Assert that all employed people have spent 0 time periods unemployed
            assert(sum([worker.time_unemployed <= 0 for worker in occ.list_of_employed]) == len(occ.list_of_employed))
            emp = len(occ.list_of_employed)
            curr_vacs = len([v_open for v_open in vacs_temp if v_open.occupation_id == occ.occupation_id])
            occ.current_demand = (curr_vacs + emp)
            # If real-world vacancy rate is greater than the current vacancy rate, then we create new vacancies 
            vac_prob = max(0, vr_t - (curr_vacs/(occ.current_demand + 1)))
            # vac_prob = d_v + ((gamma_v * max(0, occ.target_demand*(bus_conf) - occ.current_demand)) / (emp + 1))
            vacs_create = int(np.random.binomial(emp, vac_prob))

            #vacs_create = emp*int(vac_prob) + int(np.random.binomial(emp, vac_prob%1))
            for v in range(vacs_create):
                vacs_temp.append(vac(occ.occupation_id, [], occ.wage, False, 0))
                
            empl = len(occ.list_of_employed) 
            unemp = len(occ.list_of_unemployed)
            n_ltue = sum(wrkr.longterm_unemp for wrkr in occ.list_of_unemployed)
            curr_demand = occ.current_demand
            t_demand = occ.target_demand*bus_conf
            vacs_occ = len([v for v in vacs_temp if v.occupation_id == occ.occupation_id])
            wages_occ = sum(wrkr.wage for wrkr in occ.list_of_employed)
            # Calculate average relative wage for unemployed and employed workers

            ### UPDATE INDICATOR RECORD
            record.append([t+1, occ.occupation_id, empl, unemp, empl + unemp, vacs_occ, n_ltue, curr_demand, t_demand, emp_seekers, unemp_seekers, wages_occ, u_rel_wage, e_rel_wage, ue, ee])
            # record = np.append(record, 
            #                         np.array([[t+1, occ.occupation_id, empl, unemp, empl + unemp, len(vacs_temp), n_ltue, curr_demand, t_demand, emp_seekers, unemp_seekers]]), 
            #                         axis = 0)
        seekers_rec.append([t+1, unemp_seekers, u_apps])


    record_temp_df = pd.DataFrame(record, columns=['Time Step', 'Occupation', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Current_Demand', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers', 'Total_Wages', 'U_Rel_Wage', 'E_Rel_Wage', 'UE_Transitions', 'EE_Transitions'])
    record_df = record_temp_df[record_temp_df['Time Step'] > delay]
    grouped = record_df.groupby('Time Step').sum().reset_index()

    grouped['UER'] = grouped['Unemployment'] / grouped['Workers']
    grouped['U_REL_WAGE_MEAN'] = grouped['U_Rel_Wage'] / grouped['UE_Transitions']
    grouped['E_REL_WAGE_MEAN'] = grouped['E_Rel_Wage'] / grouped['EE_Transitions']
    grouped['UE_Trans_Rate'] = grouped['UE_Transitions'] / grouped['Workers']
    grouped['EE_Trans_Rate'] = grouped['EE_Transitions'] / grouped['Workers']
    grouped['VACRATE'] = grouped['Vacancies'] / (grouped['Vacancies'] + grouped['Employment'])

    data = {'UER': grouped['UER'], 'VACRATE': grouped['VACRATE']}

    seekers_rec = pd.DataFrame(seekers_rec, columns=['Time Step', 'Unemployed Seekers', 'Applications Sent'])
    seekers_rec = seekers_rec[seekers_rec['Time Step'] > delay]

    if simple_res:
        return data
    else:
        return record_df, grouped, net, data, seekers_rec

#########################################
# Wrapper for pyabc ########
#########################################
def pyabc_run_single(parameter):     
    res = run_single_local(**parameter)
    return res 
    
run_single_local(
    d_u = 0.01, 
    #d_v = 0.1,
    gamma_u = 0.01,
    #gamma_v = 0.1,
    otj = True,
    cyc_otj = True, 
    cyc_ue = True, 
    disc = False,
    mod_data = mod_data, 
    net_temp = net_temp, 
    vacs = vacs, 
    time_steps = len(gdp_dat), # set equal to length of gdp_data
    delay = 100,
    gdp_data = gdp_dat,
    bus_confidence_dat = gdp_dat,
    simple_res = False, 
    vac_data = vac_dat
)

# %%
calib_list = {
    "nonbehav": {"otj": False, # has been run
                           "cyc_otj": False, 
                           "cyc_ue": False, 
                           "disc": False,
                           "vac_data": vac_dat,
                           "delay": 70,
                           "bus_confidence_dat": gdp_dat,
                           'vac_data': vac_dat},
              "otj_nonbehav": {"otj": True, # has been run
                           "cyc_otj": False, 
                           "cyc_ue": False, 
                           "disc": False, 
                           "delay": 70,
                           "bus_confidence_dat": gdp_dat,
                           "vac_data": vac_dat},
            #   "otj_cyclical_e": {"otj": True,
            #                "cyc_otj": True, 
            #                "cyc_ue": False, 
            #                "disc": False,
            #                "bus_confidence_dat": gdp_dat},
            #   "otj_cyclical_ue": {"otj": True,
            #                "cyc_otj": False, 
            #                "cyc_ue": True, 
            #                "disc": False,
            #                "bus_confidence_dat": gdp_dat},
            #   "otj_cyclical_e_ue": {"otj": True,
            #                "cyc_otj": True, 
            #                "cyc_ue": True, 
            #                "disc": False,
            #                "bus_confidence_dat": gdp_dat},
              "otj_cyclical_e_disc": {"otj": True,
                           "cyc_otj": True, 
                           "cyc_ue": False, 
                           "disc": True,
                           "delay": 70,
                           "bus_confidence_dat": gdp_dat,
                           "vac_data": vac_dat},
            #   "otj_cyclical_ue_disc": {"otj": True,
            #                "cyc_otj": False, 
            #                "cyc_ue": True, 
            #                "disc": True,
            #                "bus_confidence_dat": gdp_dat},
              # "otj_cyclical_e_ue_disc": {"otj": True,
              #              "cyc_otj": True, 
              #              "cyc_ue": True, 
              #              "disc": True,
              #              "bus_confidence_dat": gdp_dat},
            "otj_disc": {"otj": True,
                          "cyc_otj": False, 
                          "cyc_ue": False, 
                          "disc": True,
                            "delay": 70,
                          "bus_confidence_dat": gdp_dat,
                          "vac_data": vac_dat}
            }

# %%
# Initialize the results dictionaries
model_results = {}
net_results = {}
sim_results = {}
sum_stats_list = {}
seekers_recs = {}

print(param_df)
# Loop through each model configuration
for name, item in calib_list.items():
    print(name)
    # Create a deep copy of the base parameters
    test_params = deepcopy(params)

    # Update with values from param_df specific to the current model name
    if name in param_df['model_cat'].values:
        param_updates = param_df[param_df['model_cat'] == name].set_index('Parameter')['Value'].to_dict()
        print(param_updates)
        test_params.update(param_updates)

    # Update with the values from the calib_list
    test_params.update(item)
    
    # Run the model
    sim_record, sim_grouped, sim_net, sum_stats, seekers_rec = run_single_local(**test_params)

    # Generate plots or metrics (optional step)
    #ue_vac = plot_records(sim_record, calib_date[0], calib_date[1])
    sim_grouped['DATE'] = pd.date_range(start=calib_date[0], end= calib_date[1], periods=len(sim_grouped))
    sim_grouped['LTUE Rate'] = sim_grouped['LT Unemployed Persons'] / sim_grouped['Unemployment']
    sim_grouped['AVGWAGE'] = sim_grouped['Total_Wages'] / sim_grouped['Employment']
    sim_record['LTUE Rate'] = sim_record['LT Unemployed Persons'] / sim_record['Unemployment']
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

    # Store the results
    model_results[name] = sim_grouped
    sim_results[name] = sim_record
    net_results[name] = sim_net
    sum_stats_list[name] = sum_stats
    seekers_recs[name] = seekers_rec

# %%
filtered_model_results = {key: model_results[key] for key in ["nonbehav","otj_nonbehav", "otj_cyclical_e_disc", 'otj_disc'] if key in model_results}
filtered_net_results = {key: net_results[key] for key in ["nonbehav","otj_nonbehav", "otj_cyclical_e_disc", 'otj_disc'] if key in net_results}
filtered_sim_results = {key: sim_results[key] for key in ["nonbehav","otj_nonbehav", "otj_cyclical_e_disc", 'otj_disc'] if key in sim_results}
filtered_sum_stats = {key: sum_stats_list[key] for key in ["nonbehav","otj_nonbehav", "otj_cyclical_e_disc", 'otj_disc'] if key in sum_stats_list}

plt.figure(figsize=(10, 6))
for i, (name, item) in enumerate(seekers_recs.items()):
    plt.plot(item['DATE'], item['Applications Sent'] / item['Unemployed Seekers'], marker='o', label=name)
plt.xlabel('Year')
plt.ylabel('Applications per Unemployed Seeker')
plt.title('Application Rate Over Time')
plt.tight_layout()
plt.legend()
plt.show()


plot_ltuer(filtered_model_results, observation, sep_strings = [("nonbehav", "Non-behavioural w. and w.o OTJ"), ("disc", "Behavioural")], sep = True)
plot_ltuer_dist(filtered_net_results, gender = False, names = ["Nonbehav", "Nonbehav w. OTJ", "Behavioural w. Cyc. OTJ","Behavioural w.o. Cyc. OTJ"])
plot_bev_curve(filtered_model_results, observation, sep_strings = [("nonbehav", "Non-behavioural w. and w.o OTJ"), ("disc", "Behavioural")], sep = True)
plot_uer_vac(filtered_model_results, observation, sep_strings = [("nonbehav", "Non-behavioural w. and w.o OTJ"), ("disc", "Behavioural")], sep = True)
plot_seeker_comp(filtered_model_results, share = True)
plot_cd_vs_td(filtered_model_results)


# %%
# moved to macro-vars folder from  data/behav_params/Eeckhout_Replication/cps_data/transition_rates_96_24.csv' - created in grouped_transition_rates.R script
all_rates_new = pd.read_csv('data/transition_rates_96_24.csv')
all_rates_new = all_rates_new[(all_rates_new['date'] >= calib_date[0]) & (all_rates_new['date'] <= calib_date[1])]
all_rates_new['DATE'] = pd.to_datetime(all_rates_new['date'])

import plot_funs
importlib.reload(plot_funs)
from plot_funs import *

plot_trans_rates(filtered_model_results, observation = all_rates_new, names = ["Nonbehav", "Nonbehav w. OTJ", "Behavioural w. Cyc. OTJ","Behavioural w.o. Cyc. OTJ"])
plot_rel_wages(filtered_model_results, names = ["Nonbehav", "Nonbehav w. OTJ", "Behavioural w. Cyc. OTJ","Behavioural w.o. Cyc. OTJ"])


# %%
# Example usage:
# occ_ltuers = pd.read_csv(path+"data/highlev_occ_ltuers.csv")
# occ_ltuers = occ_ltuers.groupby('occupation')['ltuer'].mean().reset_index()
# Pasted from data/occ_macro_vars/CPS_LTUER/occ_ltuer_observed.csv
occ_ltuer_obs = pd.read_csv(path + "data/occ_uer_ltuer_observed.csv")

import plot_funs
importlib.reload(plot_funs)
from plot_funs import *

plot_occupation_uer(filtered_sim_results, occ_ltuer_obs)
# plot_ltuer_difference_heatmap(filtered_sim_results, 
#                               occ_ltuer_obs, 
#                               difference_type='absolute', save=False, path=path)


# %%

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
plt.show() 

womens_wage = (gender_income['Ceiling'] * gender_income['Full-Time Females']).sum()/(gender_income['Full-Time Females'].sum())
mens_wage = (gender_income['Ceiling'] * gender_income['Full-Time Males']).sum()/(gender_income['Full-Time Males'].sum())

plot_gender_gaps(filtered_net_results, names = ["Nonbehav", "Nonbehav w. OTJ", "Behavioural w. Cyc. OTJ","Behavioural w.o. Cyc. OTJ"])



# %%
from PIL import Image, ImageDraw, ImageFont

def combine_model_images(image_dir, calib_list, sep_strings, output_path="combined_calibration.png"):
    # Load images into groups
    grouped = {match: [] for match, _ in sep_strings}
    titles = {match: title for match, title in sep_strings}
    unmatched = []

    for model_name in calib_list.keys():
        filename = f"calibration_{model_name}_sim_results.png"
        filepath = os.path.join(image_dir, filename)
        if not os.path.exists(filepath):
            continue

        img = Image.open(filepath)
        matched = False
        for match, _ in sep_strings:
            if match in model_name:
                grouped[match].append((model_name, img))
                matched = True
                break
        if not matched:
            unmatched.append((model_name, img))

    if unmatched:
        grouped["__unmatched__"] = unmatched
        titles["__unmatched__"] = "Other Models"

    # === Font and Layout Configuration ===
    title_font_size = 80
    name_font_size = 60

    title_height = title_font_size + 20
    name_height = name_font_size + 10
    spacing = 20

    try:
        font = ImageFont.truetype("arial.ttf", name_font_size)
        title_font = ImageFont.truetype("arial.ttf", title_font_size)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Safely get size of the first available image
    first_img = None
    for models in grouped.values():
        if models:
            first_img = models[0][1]
            break

    if first_img is None:
        raise ValueError("No images found in any group.")

    img_width, img_height = first_img.size

    # === Canvas Size Calculation ===
    num_cols = len(grouped)
    max_rows = max(len(models) for models in grouped.values())

    total_width = num_cols * (img_width + spacing)
    total_height = max_rows * (img_height + name_height + spacing) + title_height

    combined_img = Image.new("RGB", (total_width, total_height), color="white")
    draw = ImageDraw.Draw(combined_img)

    # === Drawing images and text ===
    for col, (match, models) in enumerate(grouped.items()):
        x_offset = col * (img_width + spacing)

        # Column title
        title_text = titles[match]
        text_width = draw.textlength(title_text, font=title_font)
        draw.text((x_offset + (img_width - text_width) // 2, 5), title_text, fill="black", font=title_font)

        for row, (name, img) in enumerate(models):
            y_offset = title_height + row * (img_height + name_height + spacing)

            # Model name
            name_text_width = draw.textlength(name, font=font)
            draw.text((x_offset + (img_width - name_text_width) // 2, y_offset), name, fill="black", font=font)

            # Paste image
            combined_img.paste(img, (x_offset, y_offset + name_height))

    # === Save the result ===
    combined_img.save(output_path)
    print(f"Combined image saved to {output_path}")

combine_model_images(
    image_dir="output_06_06/",
    calib_list=calib_list,
    sep_strings=[
        ("nonbehav", "Non-behavioural w. and w.o OTJ"),
        ("ue", "Cyclical UE SE")
    ],
    output_path="all_calibrations_combined.png"
)


