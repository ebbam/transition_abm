import pandas as pd
import numpy as np
import seaborn as sns
import labornet as lbn
from matplotlib import pylab as plt
from matplotlib import ticker
import matplotlib.cm as cm
from matplotlib import colors
import scipy.stats
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
import random

import matplotlib.colors
import scipy
import seaborn as sns
from matplotlib import pyplot as plt

#### Plotting details
figsize_ = (10,10)
figsize_ = (18,10)
fontsize_ticks = 32#16
fontsize_axis = 40#26
fontsize_title = 28
fontsize_legend = 16
linewidth_ = 4
plot_kn = True
plot_score = True

path_data = "../data/"
path_labels = "data/processed/labelled_files/"
path_exp_num = "../results/csv/"
path_exp_fig = "../results/fig/"

matrix_omn = "OMN"
matrix_kn = "kn"
n_occ = 464

n_dif_sim = 10#2
t_sim = 600#600 # total simulation time
t_shock = 100#100

δ_u = 0.016 + 0.00000001 # adding zeros since useful for defining names
δ_v = 0.012 + 0.00000001
γ_u = 10*δ_u
γ_v = γ_u
τ = 3 # the one is due to python starting to count until 0
r = 0.5502916755953751
# fraction of labor force with which to run solution (to align with simulation)
diminishing_factor = 0.01#1.0

color_omn = "#004D40"
color_kn = "#D81B60"
color_retrained = "#513e5c"
color_reshedges = "#FFC107"
color_reshweight = "#1E88E5"

#defining the shock
shock = "FO_automation"
# shock = "SMLautomation"

ipums_lab_file = "ipums_variables.csv"
ipums_mSML = "ipums_labs_mSML_manual.csv"
df_labs = pd.read_csv(path_data + ipums_lab_file)
df_labs = pd.read_csv(path_data + ipums_lab_file)
df_sml = pd.read_csv(path_data + ipums_mSML)

median_wage = np.array(df_labs["median_earnings"])
wage = np.array(df_labs["log_median_earnings"])
if shock == "FO_automation":
    p = np.array(df_labs["auto_prob_average"])
elif shock == "SMLautomation":
    p = np.array(df_sml['mSML'])/5

ipums_employment2016 = "ipums_employment_2016.csv"
df_labs_emp = pd.read_csv(path_data+ipums_employment2016)
employment = np.array(df_labs_emp["IPUMS_CPS_av_monthly_employment_whole_period"])


###
# adding information on employment 2016
###
t_shock = 100 # time at which shock starts
t_simulation = 600
shock_duration_years = 30
diminishing_factor = 1.0

size_emp = [35 + 0.0002*(employment[i]) for i in range(len(employment))]

parameter_names = "_deltau" + str(δ_u)[3:6] + "v" + str(δ_v)[3:6] + \
    "gamma" + str(γ_u)[2:5] + "_tau" + str(round(τ)) + "_shockduration" + \
    str(shock_duration_years) + "_dimfact" + str(diminishing_factor)[2:]


df_omn = pd.read_csv(path_exp_num + "u_ltu_perc_change" + "OMN" + shock\
                + parameter_names+ ".csv")
#
# df_retrained = pd.read_csv(path_exp_num + "u_ltu_perc_change" + "OMNretrained" + shock\
#                 + parameter_names+ ".csv")

df_kn = pd.read_csv(path_exp_num + "u_ltu_perc_change" + "kn" + shock\
                + parameter_names+ ".csv")


##################
# KN vs OMN Plotting both lt u
##################

df_omn.loc[183]
df_omn.loc[282]
# getting array for OMN to plot
array_of_plot_u = np.array(df_omn["u_perc_change"])
array_of_plot_ltu = np.array(df_omn["ltu_perc_change"])
array_of_plot_u_kn = np.array(df_kn["u_perc_change"])
array_of_plot_ltu_kn = np.array(df_kn["ltu_perc_change"])

size_emp = [35 + 0.0002*(employment[i]) for i in range(len(employment))]

#######
# wage plots
########

len(median_wage[p <= 0.5])

# get index of an occupation with ~0.5 automation probability
idx_05 = np.abs(p-0.5).argmin()
ltu_less05 = array_of_plot_ltu_kn[idx_05 ]
ltu_less05
mean_medianwage_noauto = np.mean(median_wage[p <= 0.5])
mean_medianwage_noauto

np.mean(median_wage[np.logical_and(p<=0.5, array_of_plot_ltu < ltu_less05)])
np.mean(median_wage[np.logical_and(p<=0.5, array_of_plot_ltu > ltu_less05)])


scipy.stats.pearsonr(median_wage, array_of_plot_ltu -  array_of_plot_ltu_kn)
f = plt.figure(figsize=(15,22))
f.subplots_adjust(hspace=0.2)
gs = gridspec.GridSpec(2, 1,height_ratios=[1, 1])
ax2 = plt.subplot(gs[1])
ax1 = plt.subplot(gs[0], sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labelsize=fontsize_ticks)
ax2.tick_params(labelsize=fontsize_ticks)
ax1.scatter(median_wage/1000, array_of_plot_u -  array_of_plot_u_kn, s=size_emp, color=color_omn, alpha=0.8)
ax2.scatter(median_wage/1000, array_of_plot_ltu -  array_of_plot_ltu_kn, s=size_emp, color=color_omn, alpha=0.8)
ax1.axhline(y=0, linestyle=":", linewidth=linewidth_,color="k", alpha=0.8)
ax2.axhline(y=0, linestyle=":", linewidth=linewidth_,color="k", alpha=0.8)
ax2.set_xlabel("Median wage (USD thousands)", fontsize=fontsize_axis)
ax1.set_ylabel("Network effects on\nunemployment", fontsize=fontsize_axis)
ax2.set_ylabel("Network effects on\nlong term unemployment", fontsize=fontsize_axis)
plt.show()
