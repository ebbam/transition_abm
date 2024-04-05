import numpy as np
import random
import pandas as pd
import labornet as lbn
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
# paths
path_data = "data/"
path_local = "../"
path_exp_sim = "../results/csv/"
path_exp_fig = "../results/fig/"

# variable that says if results are saved in csv or fig
save_csv =  True
save_fig = False
run_random = False
run_random_ret = True
# setting seed for reproducibility
np.random.seed(12345)
random.seed(12345)
# paths
path_data = "../data/"
path_local = "../"
path_exp_sim = "../results/csv/"

# simulation conditions
shock = "FO_automation"
# shock = "SMLautomation"
network_and_employment = "new_emp"
network_and_employment = "new_emp_selfloops"

file_occmobnet_hetero = "occupational_mobility_network_heteroloops.csv"
file_occmobnet = "occupational_mobility_network.csv"
file_occmobnet_resh_edges = "occupational_mobility_network_reshuffle_edges"
file_occmobnet_resh_weights = "occupational_mobility_network_reshuffle_weights"
file_occmobnet_retrained_random = "occupational_mobility_network_retrained_random"
file_occmobnet_retrained = "occupational_mobility_network_retrained.csv"

ipums_lab_file = "ipums_variables.csv"
ipums_mSML = "ipums_labs_mSML_manual.csv"
df_labs = pd.read_csv(path_data + ipums_lab_file)
wage = np.array(df_labs["log_median_earnings"])
p = np.array(df_labs["auto_prob_average"])
df_sml = pd.read_csv(path_data + ipums_mSML)


ipums_employment2016 = "ipums_employment_2016.csv"
df_labs_emp = pd.read_csv(path_data+ipums_employment2016)
employment = np.array(df_labs_emp["IPUMS_CPS_av_monthly_employment_whole_period"])

# employment = np.array(df_labs_emp["IPUMS_CPS_av_monthly_employment_2016"])
if shock == "FO_automation":
    p = np.array(df_labs["auto_prob_average"])
elif shock == "SMLautomation":
    p = np.array(df_sml['mSML'])/5

# zeros are used since it is useful for defining file names
δ_u = 0.016 + 0.00000001
δ_v = 0.012 + 0.00000001
γ_u = 10*δ_u
γ_v = γ_u
τ = int(4 - 1 )# the -1 is due to python starting to count until 0
r = 0.5502916755953751
# fraction of labor force with which to run solution (to align with simulation)
# use 0.01 to compare with simulations, 1.0 otherwise
diminishing_factor = 0.01#1.0
n_shuffle = 10

# occupational mobility network
A_omn = np.genfromtxt(path_data + file_occmobnet, delimiter=',')
A_omn_hetero = np.genfromtxt(path_data + file_occmobnet_hetero, delimiter=',')
A_omn_retrained = np.genfromtxt(path_data + file_occmobnet_retrained, delimiter=',')
n = A_omn.shape[0]
# complete network
A_kn = np.ones([n,n])/n

# shock and time conditions
t_shock = 100 # time at which shock starts
t_simulation = 600
shock_duration_years = 30
shock_duration = shock_duration_years * 52/6.75 # NOTE one time step ~6.75 weeks
time_array = [t*6.75/52 for t in range(t_simulation)]
t_steady_start = 25
t_steady_end = 75
t_transition_start = int(t_shock +0*shock_duration)
t_transition_end = int(t_shock + 1*shock_duration)

employment_0 = employment[:]
unemployment_0 = δ_u * employment_0
vacancies_0 = δ_v * employment_0
# labor force is all workers, employed + unemployed
L = np.sum(employment_0 + unemployment_0)

# initial demand and target demand
D_0 = employment_0 + unemployment_0
# set random automation probabilities
D_f = lbn.labor_restructure(D_0, p)

# get demand in sigmoid
sigmoid_half_life, k = lbn.calibrate_sigmoid(shock_duration)

parameter_names = "_deltau" + str(δ_u)[3:6] + "v" + str(δ_v)[3:6] + \
    "gamma" + str(γ_u)[2:5] + "_tau" + str(round(τ)) + "_shockduration" + \
    str(shock_duration_years) + "_dimfact" + str(diminishing_factor)[2:]


def lt_unemployment(U_all, τ):
    """ takes the array of all unemployment spells and with tau gives
    array with number of long term unemployed as defined by τ threshold
    """
    U_lt = u_longterm_from_jobspell(U_all, τ)
    # the -1 in tau is due to python counting starting on 0
    lt_unemployment = np.sum(U_all[:, τ:, :], axis=1)
    return lt_unemployment

def u_longterm_from_jobspell(U_ltm, τ):
    # NOTE -1 since python starts counting on 1
    return np.sum(U_ltm[:, τ:, :], axis=1)

def save_result(U, V, E, U_all, D, τ, matrix, params=parameter_names, shock=shock):
    """Function that saves unemployment, vacnacies, employment, longterm unep,
    and demand into csv files
    """
    names = ["u_per_occ_num","v_per_occ_num", "e_per_occ_num", \
            "ddagger_per_occ_num", "ltu_per_occ_num"]
    U_longterm = lt_unemployment(U_all, τ)
    for i, array in enumerate([U, V, E, D, U_longterm]):
        df = pd.DataFrame()
        df["id"] = np.arange(0, 464)
        df["label"] = df_labs["label"]
        for t in range(t_simulation):
            df["t" + str(t)] = array[t, :]
        df.to_csv(path_exp_sim + names[i] + matrix + shock + params+ ".csv" )
    print("saving file " + path_exp_sim + names[0] + matrix + shock + params + ".csv")

def save_percentage_change(E, U, U_all, τ, t_steady_start, t_steady_end, \
    t_transition_start, t_transition_end, matrix, params=parameter_names, \
        shock=shock):
    """Function that computes percentage change in unemployment and longterm
    unemployment. For steady state averages u and ltu from steady start to
    steady end
    """
    U_lt = u_longterm_from_jobspell(U_all, τ)
    u_perc_change_num = lbn.percentage_change_u(E, U, t_steady_start, \
                t_steady_end, t_transition_start, t_transition_end)
    ltu_perc_change_num = lbn.percentage_change_ltu(E, U, U_lt, \
                t_steady_start, t_steady_end, t_transition_start, t_transition_end)
    df = pd.DataFrame()
    df["id"] = np.arange(0, 464)
    df["label"] = df_labs["label"]
    df["u_perc_change"] = u_perc_change_num
    df["ltu_perc_change"] = ltu_perc_change_num
    df.to_csv(path_exp_sim + "u_ltu_perc_change" + matrix + shock + params+ ".csv" )
    print("saving file "+ path_exp_sim + "u_ltu_perc_change" + matrix + shock + params+ ".csv")


# run and save model for kn
U_kn, V_kn, E_kn, U_all_kn, D_kn = lbn.run_numerical_solution(\
    lbn.fire_and_hire_workers, t_simulation, δ_u, δ_v, \
    γ_u, γ_v, employment_0, unemployment_0, vacancies_0, \
    lbn.target_demand_automation, D_0, D_f, t_shock, k, sigmoid_half_life, \
    lbn.matching_probability, A_kn, τ)
save_result(U_kn, V_kn, E_kn, U_all_kn, D_kn, τ, "kn")
save_percentage_change(E_kn, U_kn, U_all_kn, τ, t_steady_start, t_steady_end, \
    t_transition_start, t_transition_end, "kn")


# run and save model for OMN
U_omn, V_omn, E_omn, U_all_omn, D_omn = lbn.run_numerical_solution(\
    lbn.fire_and_hire_workers, t_simulation, δ_u, δ_v, \
    γ_u, γ_v, employment_0, unemployment_0, vacancies_0, \
    lbn.target_demand_automation, D_0, D_f, t_shock, k, sigmoid_half_life, \
    lbn.matching_probability, A_omn, τ)
save_result(U_omn, V_omn, E_omn, U_all_omn, D_omn, τ, "OMN")
save_percentage_change(E_omn, U_omn, U_all_omn, τ, t_steady_start, t_steady_end, \
    t_transition_start, t_transition_end, "OMN")

# run and save model for OMN heterogenous loops
U_omn_hetero, V_omn_hetero, E_omn_hetero, U_all_omn_hetero, D_omn_hetero = \
    lbn.run_numerical_solution(\
    lbn.fire_and_hire_workers, t_simulation, δ_u, δ_v, \
    γ_u, γ_v, employment_0, unemployment_0, vacancies_0, \
    lbn.target_demand_automation, D_0, D_f, t_shock, k, sigmoid_half_life, \
    lbn.matching_probability, A_omn_hetero, τ)
save_result(U_omn_hetero, V_omn_hetero, E_omn_hetero, U_all_omn_hetero, D_omn_hetero, τ, \
            "OMNhetero")
save_percentage_change(E_omn_hetero, U_omn_hetero, U_all_omn_hetero, τ, t_steady_start, t_steady_end, \
    t_transition_start, t_transition_end, "OMNhetero")


U_retrained, V_retrained, E_retrained, U_all_retrained, D_retrained = lbn.run_numerical_solution(\
    lbn.fire_and_hire_workers, t_simulation, δ_u, δ_v, \
    γ_u, γ_v, employment_0, unemployment_0, vacancies_0, \
    lbn.target_demand_automation, D_0, D_f, t_shock, k, sigmoid_half_life, \
    lbn.matching_probability, A_omn_retrained, τ)

save_result(U_retrained, V_retrained, E_retrained, U_all_retrained, D_retrained, \
            τ, "OMNretrained", params=parameter_names)
save_percentage_change(E_retrained, U_retrained, U_all_retrained, τ, t_steady_start, t_steady_end, \
    t_transition_start, t_transition_end, "OMNretrained", params=parameter_names)


if run_random_ret:
    for i in range(n_shuffle):
        A_omn_retrained_random = np.genfromtxt(path_data + file_occmobnet_retrained_random + str(i) + ".csv", delimiter=',')
        U_retrained_random, V_retrained_random, E_retrained_random, U_all_retrained_random, D_retrained_random = lbn.run_numerical_solution(\
            lbn.fire_and_hire_workers, t_simulation, δ_u, δ_v, \
            γ_u, γ_v, employment_0, unemployment_0, vacancies_0, \
            lbn.target_demand_automation, D_0, D_f, t_shock, k, sigmoid_half_life, \
            lbn.matching_probability, A_omn_retrained_random, τ)

        save_result(U_retrained_random, V_retrained_random, E_retrained_random, U_all_retrained_random, D_retrained_random, \
                    τ, "OMNretrainedrandom"+str(i), params=parameter_names)
        save_percentage_change(E_retrained_random, U_retrained_random, U_all_retrained_random, τ, t_steady_start, t_steady_end, \
            t_transition_start, t_transition_end, "OMNretrainedrandom"+str(i), params=parameter_names)


if run_random:
    for i in range(n_shuffle):
        A_omn_reshuffle_edges = np.genfromtxt(path_data + file_occmobnet_resh_edges + str(i)+ ".csv", delimiter=',')
        A_omn_reshuffle_weights = np.genfromtxt(path_data + file_occmobnet_resh_weights + str(i)+".csv", delimiter=',')

        U_resh_edges, V_resh_edges, E_resh_edges, U_all_resh_edges, D_resh_edges = lbn.run_numerical_solution(\
            lbn.fire_and_hire_workers, t_simulation, δ_u, δ_v, \
            γ_u, γ_v, employment_0, unemployment_0, vacancies_0, \
            lbn.target_demand_automation, D_0, D_f, t_shock, k, sigmoid_half_life, \
            lbn.matching_probability, A_omn_reshuffle_edges, τ)

        save_result(U_resh_edges, V_resh_edges, E_resh_edges, U_all_resh_edges, D_resh_edges, \
                    τ, "OMNreshuffleedges" + str(i), params=parameter_names)
        save_percentage_change(E_resh_edges, U_resh_edges, U_all_resh_edges, τ, t_steady_start, t_steady_end, \
            t_transition_start, t_transition_end, "OMNreshuffleedges"+ str(i), params=parameter_names)

        U_resh_weights, V_resh_weights, E_resh_weights, U_all_resh_weights, D_resh_weights = lbn.run_numerical_solution(\
            lbn.fire_and_hire_workers, t_simulation, δ_u, δ_v, \
            γ_u, γ_v, employment_0, unemployment_0, vacancies_0, \
            lbn.target_demand_automation, D_0, D_f, t_shock, k, sigmoid_half_life, \
            lbn.matching_probability, A_omn_reshuffle_weights, τ)

        save_result(U_resh_weights, V_resh_weights, E_resh_weights, U_all_resh_weights, D_resh_weights, \
                    τ, "OMNreshuffleweights" + str(i), params=parameter_names)
        save_percentage_change(E_resh_weights, U_resh_weights, U_all_resh_weights, τ, t_steady_start, t_steady_end, \
            t_transition_start, t_transition_end, "OMNreshuffleweights"+ str(i), params=parameter_names)
