import numpy as np
import pandas as pd
import labornet as lbn
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
# paths
path_data = "../data/"
path_exp_sim = "../results/csv/"

# simulation conditions
shock = "beveridgeCurve"

# importing adjacency matrices
file_occmobnet = "occupational_mobility_network.csv"
A_omn = np.genfromtxt(path_data + file_occmobnet, delimiter=',')
n = A_omn.shape[0]
A_kn = np.ones([n,n])/n

# employment information#ipums_employment2016 = "ipums_employment_2016.csv"
ipums_employment2016 = "ipums_employment_2016.csv"
df_labs_emp = pd.read_csv(path_data+ipums_employment2016)
employment = np.array(df_labs_emp["IPUMS_CPS_av_monthly_employment_whole_period"])
ipums_lab_file = "ipums_variables.csv"
df_labs = pd.read_csv(path_data + ipums_lab_file)

δ_u = 0.016 + 0.00000001 # adding zeros since useful for defining names
δ_v = 0.012 + 0.00000001
γ_u = 10*δ_u
γ_v = γ_u
τ = 3 # time steps after which worker is long-term unemployed
r = 0.5502916755953751
# fraction of labor force with which to run solution (to align with simulation)
diminishing_factor = 1.0


# shock parameters (cycle amplitude and frequency)
cycle_amp = 0.065
cycle_period = lbn.bussiness_cycle_period(τ)
cycle_period

parameter_names = "_deltau" + str(δ_u)[3:6] + "v" + str(δ_v)[3:6] + \
    "gamma" + str(γ_u)[2:5] + "_tau" + str(round(τ)) + "_amplitude" + \
    str(cycle_amp)[2:] + "_period" + str(cycle_period) + "_dimfact" + str(diminishing_factor)[2:]

parameter_names

def save_result(U, V, E, U_all, D, τ, matrix, params=parameter_names, shock=shock):
    """Function that saves unemployment, vacnacies, employment, longterm unep,
    and demand into csv files
    """
    names = ["u_per_occ_num","v_per_occ_num", "e_per_occ_num", \
            "ddagger_per_occ_num"]
    for i, array in enumerate([U, V, E, D]):
        df = pd.DataFrame()
        df["id"] = np.arange(0, 464)
        df["label"] = df_labs["label"]
        for t in range(t_simulation):
            df["t" + str(t)] = array[t, :]
        df.to_csv(path_exp_sim + names[i] + matrix + shock + params+ ".csv" )
    print("saving file " + path_exp_sim + names[0] + matrix + shock + params + ".csv")

############
# Shock with sin wave
############
# shock and time conditions
# NOTE one time step ~ 6.75 weeks
t_shock = 103 # time at which shock starts
t_simulation = 600
t_shock = int(78+2) # time at which shock starts
t_simulation = 600

employment_0 = employment[:]
unemployment_0 = δ_u * employment_0
vacancies_0 = δ_v * employment_0
# labor force is all workers, employed + unemployed
L = np.sum(employment_0 + unemployment_0)

# initial demand and target demand
D_0 = employment_0 + unemployment_0
# set final demand equal to initial demand
D_f = employment_0 + unemployment_0

# run model for kn
U_kn, V_kn, E_kn, U_all_kn, D_kn = lbn.run_numerical_solution(\
    lbn.fire_and_hire_workers, t_simulation, δ_u, δ_v, \
    γ_u, γ_v, employment_0, unemployment_0, vacancies_0, \
    lbn.target_demand_cycle, D_0, D_f, t_shock, cycle_amp, cycle_period, \
    lbn.matching_probability, A_kn, τ)

# run model for OMN
U_omn, V_omn, E_omn, U_all_omn, D_omn = lbn.run_numerical_solution(\
    lbn.fire_and_hire_workers, t_simulation, δ_u, δ_v, \
    γ_u, γ_v, employment_0, unemployment_0, vacancies_0, \
    lbn.target_demand_cycle, D_0, D_f, t_shock, cycle_amp, cycle_period, \
    lbn.matching_probability, A_omn, τ)
save_result(U_omn, V_omn, E_omn, U_all_omn, D_omn, τ, "OMN")
