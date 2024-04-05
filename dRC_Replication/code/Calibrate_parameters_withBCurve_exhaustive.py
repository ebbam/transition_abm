import numpy as np
import pandas as pd
import labornet as lbn
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from sys import argv
import copy


# paths
path_data = "data/"
path_local = "../"
path_exp_sim = "../results/csv/"

# simulation conditions
shock = "beveridgeCurve"
# data
ipums_adj_file = "transition_probability_matrix_weights.csv"
ipums_employment2016 = "ipums_employment_2016.csv"
df_labs_emp = pd.read_csv(path_local+path_data+ipums_employment2016)
employment = np.array(df_labs_emp["IPUMS_CPS_av_monthly_employment_whole_period"])
A_omn = np.genfromtxt("../" + path_data + ipums_adj_file,delimiter=',')
n = A_omn.shape[0]
r = 0.5502916755953751
A_omn = lbn.add_self_loops(A_omn, r)
ipums_lab_file = "ipums_variables.csv"
ipums_mSML = "ipums_labs_mSML_manual.csv"
df_labs = pd.read_csv(path_local + path_data + ipums_lab_file)
df_labs = pd.read_csv(path_local + path_data + ipums_lab_file)
wage = np.array(df_labs["log_median_earnings"])

# Importing Beveridge curve data
df_u = pd.read_csv(path_local + path_data + "Total_unemployment_BLS1950-2018.csv")
df_v = pd.read_csv(path_local + path_data + "vacancy_rateDec2000.csv")
# Start vacancy rate and unemployment rate since Jan 2001
df_v = df_v.iloc[1:]
df_u = df_u.loc[(df_u['Year']>=2001)]
#putting all values of unemployment in a list
list_unemployment = []
#56 since we excluded previous data points
for i in df_u.index:
    list_unemployment = list_unemployment + list(df_u.loc[i][1:].values)
list_vacancies = list(df_v["JobOpenings"])
# Until Sept 2018
list_unemployment =  list_unemployment[:-3]
list_vacancies = list_vacancies[:-2]
assert(len(list_unemployment) == len(list_vacancies))

# set rela bev curve polygon
real_bev_poly =  lbn.get_polygon(list_unemployment , list_vacancies)

#### setting number of simulations and time steps duration
t_sim = 400#  total simulation time
t_shock = 100#100

# list of parameters to explore
δ_u_list = [0.0150001 + i * 0.00025 for i in range(1,9)]#[0:2]
δ_v_list = [0.009001 + i * 0.00025 for i in range(1,18)]#[0:2]
amplitude_list = [0.06 + 0.0025*i for i in range(9)]#[0:2]
τ_list = [i for i in range(2,8)]#[0:2]

# δ_u_list = [0.0160001]#[0:2]
# δ_v_list = [0.012001]#[0:2]
# amplitude_list = [0.065] #[0:2]
# τ_list = [3]

# Calibration of the Beveridge curve with sin wage
df_err = lbn.find_best_parameters(δ_u_list, δ_v_list, τ_list, amplitude_list, \
        employment, real_bev_poly, t_shock, t_sim, A_omn)
df_err.to_csv(path_exp_sim + "calibration_" + network_and_employment + ".csv")
