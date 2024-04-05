### Plots in python of the BeveridgeCurve using the csv files.
import numpy as np
import pandas as pd
import labornet as lbn
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec


figsize_ = (12,8)
# figsize_ = (18,10)
fontsize_ticks = 32#16
fontsize_axis = 40#26
fontsize_title = 28
fontsize_legend = 20
linewidth_ = 3

path_data = "../data/"
path_exp_sim = "../results/csv/"
path_exp_fig = "../results/fig/"
path_exp_sim = "../results/simulations/"
path_exp_numerics = "../results/csv/"

# details of the simulation, so far hand copied
matrix_omn = "OMN"#"kn"#"JS"#"OMN"#"OMN" #"kn"#"OMN"
matrix_kn = "kn"#"kn"#"JS"#"OMN"#"OMN" #"kn"#"OMN"
matrix = matrix_omn#matrix_omn


shock = "beveridgeCurve"

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
cycle_freq = cycle_period

parameter_names = "_deltau" + str(δ_u)[3:6] + "v" + str(δ_v)[3:6] + \
    "gamma" + str(γ_u)[2:5] + "_tau" + str(round(τ)) + "_amplitude" + \
    str(cycle_amp)[2:] + "_period" + str(cycle_period) + "_dimfact" + str(diminishing_factor)[2:]


def make_df_num_array(matrix, shock, parameter_names):
    """ returns data frames of numerical solution of unemp, emp,
    longterm unemp and target demand
    """
    df_num_u = pd.read_csv(path_exp_numerics + "u_per_occ_num" + matrix\
                    + shock + parameter_names +  ".csv")
    df_num_e = pd.read_csv(path_exp_numerics + "e_per_occ_num" + matrix\
                    + shock + parameter_names +  ".csv")
    df_num_v = pd.read_csv(path_exp_numerics + "v_per_occ_num" + matrix\
                    + shock + parameter_names +  ".csv")
    df_num_ddagger = pd.read_csv(path_exp_numerics + "ddagger_per_occ_num" + \
                    matrix + shock + parameter_names +  ".csv")
    return df_num_u, df_num_e, df_num_v, df_num_ddagger

def make_u_v_rates(df_num_u, df_num_e, df_num_v):
    un = np.array(df_num_u.iloc[:,3:])
    en = np.array(df_num_e.iloc[:,3:])
    vn= np.array(df_num_v.iloc[:,3:])
    u_total_num = un.sum(axis=0)
    e_total_num = en.sum(axis=0)
    v_total_num = vn.sum(axis=0)
    n_agents_num = un.sum(axis=0)[0] + en.sum(axis=0)[0]

    u_rate = 100*u_total_num/ n_agents_num
    v_rate = 100*v_total_num/ (v_total_num  + e_total_num )
    return u_rate, v_rate


df_num_u_omn, df_num_e_omn, df_num_v_omn, df_num_ddagger_omn = \
        make_df_num_array("OMN", shock, parameter_names)
u_omn, v_omn = make_u_v_rates(df_num_u_omn, df_num_e_omn, df_num_v_omn)


######
# importing historical bev curve data
######
df_u = pd.read_csv(path_data + "Total_unemployment_BLS1950-2018.csv")
df_v = pd.read_csv(path_data + "vacancy_rateDec2000.csv")
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
11, 2.5
list_vacancies[101]
df_v[:20]
df_v[100:120]
# first period is beginning of data set
period0_start = 0
# second period starts in Dec 2001 when first recession ends
period1_start = 11
# third period starts in Dec 2007 when second recession starts
period2_start = 82
# fourth period starts in June 2009 when second recession starts
period3_start = 101
# start of recession due to financial crisis
u_start = list_unemployment[period2_start]
v_start = list_vacancies[period2_start]

######
# Plotting Beveridge Curve in one axis
######
f = plt.figure(figsize=figsize_)
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])
ax1.tick_params(axis='both',labelsize=fontsize_ticks )
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
# #plot numerical

ax1.quiver(u_omn[210:210 + cycle_freq-1], v_omn[210:210 + cycle_freq-1], u_omn[210 + 1:210 + cycle_freq ]- \
u_omn[210:210 + cycle_freq-1], v_omn[210 + 1:210 + cycle_freq]-v_omn[210:210 + cycle_freq-1], \
scale_units='xy', angles='xy', scale=1, alpha=0.8, width=0.006, label="Model's Beveridge Curve")

# #plot empirical
ax1.plot(list_unemployment[period2_start:period3_start], \
list_vacancies[period2_start:period3_start], "*-", \
label="Empirical Beveridge Curve", alpha = 0.5, color="grey", linewidth=linewidth_)
ax1.plot(list_unemployment[period3_start:], \
list_vacancies[period3_start:], "*-", alpha = 0.5, color="grey", linewidth=linewidth_)
if matrix == "OMN":
    ax1.set_ylim([1.5, 6])
    # ax2.set_xlim([2.5, 10.5])
else:
    ax1.set_ylim([1.5, 7])
    # ax2.set_xlim([2.5, 11])
ax1.set_xlabel("unemployment rate (%)", fontsize=fontsize_axis)
ax1.set_ylabel("vacancy rate (%)", fontsize=fontsize_axis)
ax1.set_ylim([1.5, 6])
ax1.set_xlim([1.5, 10.5])
plt.legend(fontsize=fontsize_legend)
#plt.savefig(path_exp_fig+"BeveridgeCurveMatchingArrows_"+ shock + matrix + parameter_names + ".svg", bbox_inches="tight")
plt.show()
