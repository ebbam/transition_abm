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
df_retrained = pd.read_csv(path_exp_num + "u_ltu_perc_change" + "OMNretrained" + shock\
                + parameter_names+ ".csv")

df_kn = pd.read_csv(path_exp_num + "u_ltu_perc_change" + "kn" + shock\
                + parameter_names+ ".csv")

occ_to_show = [319, 234, 301, 183, 442, 58, 96, 98, 271, 282, 325, 433]
edgecolors= ["k" for i in range(464)]
linewidth_scatter = [1 for i in range(464)]
for occ in occ_to_show:
        edgecolors[occ] = "brown"
        linewidth_scatter[occ] = 3

labels = df_omn["label"]

if shock[0:3] == "FO_":
    array_x_axis = p
    ticks = [str(i/10) for i in range(-1,11)]
    n_ticks = 22
    xlabel_title = "Probability of automatability"
    x_ref = "auto_prob_average_manual"
elif shock[0:3] == "SML":
    n_ticks = 45
    array_x_axis = p
    if plot_score:
            array_x_axis = p
            split = max(array_x_axis) - min(array_x_axis)
            ticks = [str(round( min(array_x_axis) + i*split/10, 2)) for i in range(-1,11)]
            xlabel_title = "Suitability for Machine Learning"
            x_ref = "mSML"

##################
# KN vs OMN Plotting both lt u
##################
array_of_plot_u = np.array(df_omn["u_perc_change"])
array_of_plot_ltu = np.array(df_omn["ltu_perc_change"])
array_of_plot_u_kn = np.array(df_kn["u_perc_change"])
array_of_plot_ltu_kn = np.array(df_kn["ltu_perc_change"])

color_kn = "sandybrown"#"darkorange"

f = plt.figure(figsize=(20,20))
f.subplots_adjust(hspace=0.4)
gs = gridspec.GridSpec(2, 1,height_ratios=[1, 1])
ax2 = plt.subplot(gs[1])
ax1 = plt.subplot(gs[0], sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labelsize=fontsize_ticks)
ax2.tick_params(labelsize=fontsize_ticks)
if shock[:3] == "FO_":
        pass
else:
        ax1.set_ylim([-15,15])
        ax2.set_ylim([-40,40])
        ax2.set_xlim([3.1, 3.9])
        if plot_score:
                ax2.set_xlim([3.1/5, 3.9/5])
# plotting unemployment change stuff
u_plot =ax1.scatter(array_x_axis, array_of_plot_u, s=size_emp, edgecolors=edgecolors, linewidth=linewidth_scatter, \
        alpha=0.99, label="Occupational mobility network", c=median_wage,    cmap="viridis", norm=matplotlib.colors.LogNorm())
for i, txt in enumerate(labels):
        if i in occ_to_show:
                ax1.annotate(txt, (array_x_axis[i], array_of_plot_u[i]), fontsize=25, verticalalignment="top")
arr1inds = array_x_axis.argsort()
sorted_arr1 = array_x_axis[arr1inds[::-1]]
sorted_arr2 = array_of_plot_u_kn[arr1inds[::-1]]
ax1.plot(sorted_arr1, sorted_arr2, ".-", alpha=0.6, linewidth=4, markersize=15, color=color_kn, label="Complete network")
ax1.axhline(y=0, linestyle=":", linewidth=linewidth_,color="royalblue", alpha=0.8)
ax1.set_ylim([-25, 125])
# plotting longterm unemployment change
# plotting OMN stuff

ltu_plot =ax2.scatter(array_x_axis, array_of_plot_ltu, s=size_emp, edgecolors=edgecolors, linewidth=linewidth_scatter, \
        alpha=0.99, label="Network structure", c=median_wage,    cmap="viridis", norm=matplotlib.colors.LogNorm())

for i, txt in enumerate(labels):
        if i in occ_to_show:
                ax2.annotate(txt, (array_x_axis[i], array_of_plot_ltu[i]), fontsize=25, verticalalignment="top")

arr1inds = array_x_axis.argsort()
sorted_arr1 = array_x_axis[arr1inds[::-1]]
sorted_arr2 = array_of_plot_ltu_kn[arr1inds[::-1]]
ax2.plot(sorted_arr1, sorted_arr2, ".-", alpha=0.6, linewidth=4, markersize=15, color=color_kn, label="No network \nstructure")
ax2.axhline(y=0, linestyle=":", linewidth=linewidth_,color="royalblue", alpha=0.8)
ax2.set_ylim([-100, 260])
plt.colorbar(u_plot,ax=ax1)
plt.colorbar(ltu_plot,ax=ax2)
if shock[:3] == "FO_":
        ax1.legend(loc=2, fontsize=32)
        ax2.set_xlabel("Probability of Computerization\n(automation level)", fontsize=fontsize_axis)
        ax1.set_ylabel("unemployment rate", fontsize=fontsize_axis)
        ax2.set_ylabel("long-term\nunemployment rate", fontsize=fontsize_axis)
else:
        ax2.set_xlabel("Suitability for Machine Learning score\n(automation level)", fontsize=fontsize_axis)
plt.show()


##################
# Highlight occupations
##################


array_of_plot_u = np.array(df_omn["u_perc_change"])
array_of_plot_ltu = np.array(df_omn["ltu_perc_change"])
array_of_plot_u_kn = np.array(df_kn["u_perc_change"])
array_of_plot_ltu_kn = np.array(df_kn["ltu_perc_change"])

color_kn = "sandybrown"#"darkorange"

f = plt.figure(figsize=(20,20))
f.subplots_adjust(hspace=0.4)
gs = gridspec.GridSpec(2, 1,height_ratios=[1, 1])
ax2 = plt.subplot(gs[1])
ax1 = plt.subplot(gs[0], sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labelsize=fontsize_ticks)
ax2.tick_params(labelsize=fontsize_ticks)
if shock[:3] == "FO_":
        pass
else:
        ax1.set_ylim([-15,15])
        ax2.set_ylim([-40,40])
        ax2.set_xlim([3.1, 3.9])
        if plot_score:
                ax2.set_xlim([3.1/5, 3.9/5])
# plotting unemployment change stuff
u_plot =ax1.scatter(array_x_axis, array_of_plot_u, s=size_emp, edgecolors='k', \
        alpha=0.99, label="Occupational mobility network", c=median_wage,    cmap="viridis", norm=matplotlib.colors.LogNorm())

u_plot =ax1.scatter(array_x_axis, array_of_plot_u, s=size_emp, edgecolors='k', \
        alpha=0.99, label="Occupational mobility network", c=median_wage,    cmap="viridis", norm=matplotlib.colors.LogNorm())

for i in occ_to_show:
        ax1.scatter(array_x_axis[i], array_of_plot_u[i], c="red", s=50)

arr1inds = array_x_axis.argsort()
sorted_arr1 = array_x_axis[arr1inds[::-1]]
sorted_arr2 = array_of_plot_u_kn[arr1inds[::-1]]
ax1.plot(sorted_arr1, sorted_arr2, ".-", alpha=0.6, linewidth=4, markersize=15, color=color_kn, label="Complete network")
ax1.axhline(y=0, linestyle=":", linewidth=linewidth_,color="royalblue", alpha=0.8)
ax1.set_ylim([-25, 125])
# plotting longterm unemployment change
# plotting OMN stuff

ltu_plot =ax2.scatter(array_x_axis, array_of_plot_ltu, s=size_emp, edgecolors='k', \
        alpha=0.99, label="Network structure", c=median_wage,    cmap="viridis", norm=matplotlib.colors.LogNorm())

for i, txt in enumerate(labels):
        if i in occ_to_show:
                ax2.annotate(txt, (array_x_axis[i], array_of_plot_ltu[i]), fontsize=25, verticalalignment="top")

for i in occ_to_show:
        ax2.scatter(array_x_axis[i], array_of_plot_ltu[i], c="red", s=50)


arr1inds = array_x_axis.argsort()
sorted_arr1 = array_x_axis[arr1inds[::-1]]
sorted_arr2 = array_of_plot_ltu_kn[arr1inds[::-1]]
ax2.plot(sorted_arr1, sorted_arr2, ".-", alpha=0.6, linewidth=4, markersize=15, color=color_kn, label="No network \nstructure")
ax2.axhline(y=0, linestyle=":", linewidth=linewidth_,color="royalblue", alpha=0.8)
ax2.set_ylim([-100, 260])
plt.colorbar(u_plot,ax=ax1)
plt.colorbar(ltu_plot,ax=ax2)
if shock[:3] == "FO_":
        ax1.legend(loc=2, fontsize=32)
        ax2.set_xlabel("Probability of Computerization\n(automation level)", fontsize=fontsize_axis)
        ax1.set_ylabel("unemployment rate", fontsize=fontsize_axis)
        ax2.set_ylabel("long-term\nunemployment rate", fontsize=fontsize_axis)
else:
        ax2.set_xlabel("Suitability for Machine Learning score\n(automation level)", fontsize=fontsize_axis)
plt.show()
