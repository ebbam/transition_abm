import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pylab as plt
from matplotlib import ticker
import matplotlib.cm as cm
from matplotlib import colors
import scipy.stats
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
import random


#### Plotting details
figsize_ = (10,10)
fontsize_ticks = 26#16
fontsize_axis = 30#34#26
fontsize_title = 28
fontsize_legend = 20
linewidth_ = 3.5
linewidth_thin = 2.5
plot_simulations = True#False
plot_sharp_transition =False#True

path_exp_sim = "../results/simulations/"
path_exp_numerics = "../results/csv/"
path_label = "../data/"
path_exp_fig = "../results/fig/"

# details of the simulation, so far hand copied
matrix_omn = "OMN"
matrix_kn = "kn"

color_omn = "#004D40"
color_kn = "#D81B60"
color_retrained = "#513e5c"
color_reshedges = "#FFC107"
color_reshweight = "#1E88E5"
color_retrained_random = "#1E88E5"

n_dif_sim = 10
t_sim = 600#600 # total simulation time
t_simulation = 600
t_shock = 100#100

# parameters
δ_u = 0.0160001
δ_v = 0.012001
# δ_v = δ_u
γ_u = 10*δ_u
γ_v = γ_u
τ = 3#7#6#5
# fraction of labor force with which sim and num are run
diminishing_factor = 0.01#1.0


#defining the shock
shock = "FO_automation"
# shock = "SMLautomation"

shock_duration_years = 30
timestep_duration = 27/τ
shock_duration_timesteps = round(shock_duration_years*52 /timestep_duration)
# defining number of agents used 0.001 ~ 150k agents, 0.01 ~ 1.5M agents

diminishing_factor = 0.01#0.01#1.0
diminishing_factor_kn = 0.1#0.1#1.0#0.1
diminishing_factor_sim = 0.01

### variables exclusive for dim
# period to measure changes in
t_transition_start = int(t_shock + 0*shock_duration_timesteps)
t_transition_end = int(t_shock + 1*shock_duration_timesteps)

# name csv with parameter names
parameter_names = "_deltau" + str(δ_u)[3:6] + "v" + str(δ_v)[3:6] + \
    "gamma" + str(γ_u)[2:5] + "_tau" + str(round(τ)) + "_shockduration" + \
    str(shock_duration_years) + "_dimfact" + str(diminishing_factor)[2:]

parameter_names_sim = "_deltau" + str(δ_u)[3:6] + "v" + str(δ_v)[3:6] + \
        "gamma" + str(γ_u)[2:5] + "_tau" + str(round(τ +1)) + "_shockduration" + \
        str(shock_duration_years) + "_dimfact" + str(diminishing_factor_sim)[2:]


# getting file name for omn and kn
import_name_num = matrix_omn + shock + parameter_names +  ".csv"
import_name_sim = matrix_omn + shock + parameter_names_sim +  ".csv"

# same but for complete network
import_name_num_kn = matrix_kn + shock + parameter_names + ".csv"
import_name_sim_kn = matrix_omn + shock + parameter_names_sim +  ".csv"

def make_df_num_array(matrix, shock, parameter_names):
    """ returns data frames of numerical solution of unemp, emp,
    longterm unemp and target demand
    """
    df_num_u = pd.read_csv(path_exp_numerics + "u_per_occ_num" + matrix\
                    + shock + parameter_names +  ".csv")
    df_num_e = pd.read_csv(path_exp_numerics + "e_per_occ_num" + matrix\
                    + shock + parameter_names +  ".csv")
    df_num_ltu = pd.read_csv(path_exp_numerics + "ltu_per_occ_num" + matrix\
                    + shock + parameter_names +  ".csv")
    df_num_ddagger = pd.read_csv(path_exp_numerics + "ddagger_per_occ_num" + \
                    matrix + shock + parameter_names +  ".csv")
    return df_num_u, df_num_e, df_num_ltu, df_num_ddagger

def make_u_ltu_rates(df_num_u, df_num_e, df_num_ltu):
    un = np.array(df_num_u.iloc[:,3:])
    en = np.array(df_num_e.iloc[:,3:])
    ltun= np.array(df_num_ltu.iloc[:,3:])
    u_total_num = un.sum(axis=0)
    e_total_num = en.sum(axis=0)
    ltu_total_num = ltun.sum(axis=0)
    n_agents_num = un.sum(axis=0)[0] + en.sum(axis=0)[0]

    u_rate = 100*u_total_num/ n_agents_num
    ltu_rate = 100*ltu_total_num/ n_agents_num
    return u_rate, ltu_rate

def make_u_ltu_rates_occ(occ, df_num_u, df_num_e, df_num_ltu):
    un = np.array(df_num_u.iloc[occ,3:])
    en = np.array(df_num_e.iloc[occ,3:])
    ltun = np.array(df_num_ltu.iloc[occ,3:])
    n_agents_num = un + en
    u_rate = 100*un/ n_agents_num
    ltu_rate = 100*ltun/ n_agents_num
    return u_rate, ltu_rate


def make_simulations_df(n_dif_sim, matrix, shock, parameter_names):
    """making list of data frames of simulations
    """
    # list of dataframes
    df_sim_u_list = []
    df_sim_v_list = []
    df_sim_e_list = []
    df_sim_ltu_list = []

    import_sim_name = matrix + shock + parameter_names +  ".csv"

    for s in range(1,n_dif_sim + 1):
        u_sim_occ_name = "u_per_occ_sim" + str(s) + import_sim_name
        v_sim_occ_name = "v_per_occ_sim" + str(s) + import_sim_name
        e_sim_occ_name = "e_per_occ_sim" + str(s)+ import_sim_name
        ltu_sim_occ_name = "ltu_per_occ_sim"+ str(s) + import_sim_name

        df_sim_u = pd.read_csv(path_exp_sim + u_sim_occ_name)
        df_sim_v = pd.read_csv(path_exp_sim + v_sim_occ_name)
        df_sim_e = pd.read_csv(path_exp_sim + e_sim_occ_name)
        df_sim_ltu = pd.read_csv(path_exp_sim + ltu_sim_occ_name)

        df_sim_u_list.append(df_sim_u)
        df_sim_v_list.append(df_sim_v)
        df_sim_e_list.append(df_sim_e)
        df_sim_ltu_list.append(df_sim_ltu)

    return df_sim_u_list, df_sim_v_list, df_sim_e_list, df_sim_ltu_list



df_num_u_omn, df_num_e_omn, df_num_ltu_omn, df_num_ddagger_omn = \
        make_df_num_array("OMN", shock, parameter_names)
u_omn, ltu_omn = make_u_ltu_rates(df_num_u_omn, df_num_e_omn, df_num_ltu_omn)

df_num_u_kn, df_num_e_kn, df_num_ltu_kn, df_num_ddagger_kn = \
        make_df_num_array("kn", shock, parameter_names)
u_kn, ltu_kn = make_u_ltu_rates(df_num_u_kn, df_num_e_kn, df_num_ltu_kn)


if plot_simulations:
    df_sim_u_list_omn, df_sim_v_list_omn, df_sim_e_list_omn, \
        df_sim_ltu_list_omn = make_simulations_df(n_dif_sim, "OMN", shock, \
                                parameter_names_sim)

    df_sim_u_list_kn, df_sim_v_list_kn, df_sim_e_list_kn, \
        df_sim_ltu_list_kn = make_simulations_df(n_dif_sim, "kn", shock, \
                                parameter_names_sim)


# just to choose some initial occupations
random.seed(1) # for reproducibility though might not be needed

occupations_to_plot = [183, 234] # [0, 230, 154, 301, 234]
occupations_to_plot_ind = [0, 230, 154, 301, 234] # [0, 230, 154, 301, 234]
timestep_duration = 27/(τ + 1)
shock_duration_timesteps = round(shock_duration_years*52 /timestep_duration)

time_scale_year = (27/(τ + 1))/52 #5.4 week over 52 weeks that are a year
starting_time = 2005
time_array = starting_time  + time_scale_year * np.array([i for i in range(t_simulation)])

# t_transition_start = int(t_shock +0*shock_duration_timesteps)
# t_transition_end = int(t_shock + 1*shock_duration_timesteps)
t_transition_start = starting_time  + int(t_shock*time_scale_year + 0*shock_duration_timesteps*time_scale_year)
t_transition_end = starting_time  +  int(t_shock*time_scale_year + 1*shock_duration_timesteps*time_scale_year)


if shock[0:13] == "FO_automation":
    occupations_to_plot = [298, 234]
    occupations_to_plot = [282, 319]
    occupations_to_plot = [183, 234]
elif shock[0:13] == "SMLautomation":
    occupations_to_plot = [298, 218]

##### Plot aggregate rates and demand

f = plt.figure(figsize=(10,15))
# f = plt.figure(figsize=(14,10))
f.subplots_adjust(hspace=0.5)
gs = gridspec.GridSpec(3, 1,height_ratios=[1,1,1])
gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.2)
ax0 = plt.subplot(gs00[0])
ax1 = plt.subplot(gs00[1], sharex=ax0)
ax2 = plt.subplot(gs[1], sharex=ax0)
ax3 = plt.subplot(gs[2], sharex=ax0)

ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax1.tick_params(labelsize=fontsize_ticks)
ax2.tick_params(labelsize=fontsize_ticks)
ax0.tick_params(labelsize=fontsize_ticks)
ax3.tick_params(labelsize=fontsize_ticks)
#### Plotting demand
ax0.set_xlim([2010,2060])
ax0.axvline(x=int(starting_time  + time_scale_year * t_shock), linestyle=":", linewidth=linewidth_,color="brown", alpha=0.5, label="Automation starts")
if plot_sharp_transition:
        ax0.axvspan(t_transition_start + 0.25*shock_duration_years, t_transition_end - 0.25*shock_duration_years, facecolor='coral', alpha=0.5)

ax0.axvspan(t_transition_start, t_transition_end, facecolor='grey', alpha=0.4)
colors = ["indianred", "cornflowerblue", "b"]
ddagn = np.array(df_num_ddagger_omn.iloc[occupations_to_plot[1],3:])
# divide over diminishing_factor*million to have totalmillion of agents
ax0.plot(time_array, ddagn / (diminishing_factor*1e6), color=colors[1], linewidth=5 )
# ax0.set_ylabel(r'$d^\dagger_{i,t}$'+"\n(Million\nworkers)",fontsize=fontsize_axis)#, rotation=0)
ax0.set_xlim([2005,2050])
ax0.axes.get_xaxis().set_visible(False)

ax1.set_xlim([2010,2060])
ax1.set_ylabel("target demand \n(M of workers)", fontsize=fontsize_axis)
ax1.axvline(x=int(starting_time  + time_scale_year * t_shock), linestyle=":", linewidth=linewidth_,color="brown", alpha=0.5, label="Automation starts")
if plot_sharp_transition:
        ax0.axvspan(t_transition_start + 0.25*shock_duration_years, t_transition_end - 0.25*shock_duration_years, facecolor='coral', alpha=0.5)
ax1.axvspan(t_transition_start, t_transition_end, facecolor='grey', alpha=0.4)

ddagn = np.array(df_num_ddagger_omn.iloc[occupations_to_plot[0],3:])
ax1.plot(time_array, ddagn / (diminishing_factor*1e6), color=colors[0], linewidth=5 )
ax1.set_ylabel(r'$d^\dagger_{i,t}$'+"\n(Million\nworkers)",fontsize=fontsize_axis)#, rotation=0)
ax1.set_xlim([2005,2050])
ax1.set_title(" ",fontsize=fontsize_title)

### Plotting unemployment
average_u = np.zeros(t_sim)
average_u_kn = np.zeros(t_sim)
if plot_simulations:
    for s in range(n_dif_sim):
        df_us = df_sim_u_list_omn[s]
        df_es = df_sim_e_list_omn[s]
        us = np.array(df_us.iloc[:,3:])
        es = np.array(df_es.iloc[:,3:])
        u_total_sim = us.sum(axis=0)
        e_total_sim = es.sum(axis=0)
        n_agents = es.sum(axis=0)[0] + us.sum(axis=0)[0]
        average_u = average_u + u_total_sim / n_agents
        ax2.plot(time_array, 100*u_total_sim  /  n_agents,linewidth=linewidth_, color=color_omn, alpha=0.2)

        df_us_kn = df_sim_u_list_kn[s]
        df_es_kn = df_sim_u_list_kn[s]

        us_kn = np.array(df_us_kn.iloc[:,3:])
        es_kn = np.array(df_es_kn.iloc[:,3:])
        u_total_sim_kn = us_kn.sum(axis=0)
        e_total_sim_kn = es_kn.sum(axis=0)
        n_agents = es.sum(axis=0)[0] + us.sum(axis=0)[0]
        average_u_kn = average_u_kn + u_total_sim_kn / n_agents
        ax2.set_xlim([2005,2035])
        ax2.plot(time_array, 100*u_total_sim_kn  /  n_agents,linewidth=linewidth_, color =color_kn, alpha=0.2)
    print("L = " , n_agents)
    # average_u_kn = average_u_kn / n_dif_sim
    # ax1.plot(time_array, 100*average_u_kn, "-",color="red",linewidth=linewidth_, label="No network, simulations\naverage")

# average_u = average_u / n_dif_sim
# ax1.plot(time_array, 100*average_u, "-",color="limegreen",linewidth=linewidth_, label="Network, simulations")

######### Numerical
ax2.plot(time_array, u_omn , "--", color="k", linewidth=linewidth_thin,label="Network, analytical")
ax2.plot(time_array, u_kn , "-.",linewidth=linewidth_thin,  color="k", label="Complete network, numerical\nsolution")

#ax1.set_ylim([4, 5])
ax2.axvline(x=int(starting_time  + time_scale_year * t_shock), linestyle=":", linewidth=linewidth_,color="brown", alpha=0.5, label="Automation starts")
if plot_sharp_transition:
        ax1.axvspan(t_transition_start + 0.25*shock_duration_years, t_transition_end - 0.25*shock_duration_years, facecolor='coral', alpha=0.5)
ax2.axvspan(t_transition_start, t_transition_end, facecolor='grey', alpha=0.4)
ax2.set_xlim([2010,2060])
ax2.set_ylabel("unemployment \n rate (%)", fontsize=fontsize_axis)
ax2.set_ylim([4.5, 7.5])
ax2.set_ylim([3.5, 7.5])
if shock[13:] == "_095":
    ax2.set_ylim([4.5, 10])
elif shock[13:] == "_105":
    ax2.set_ylim([2, 7.1])

# Plot longterm unemployment
average_ltu = np.zeros(t_sim)
average_ltu_kn = np.zeros(t_sim)
# simulations
if plot_simulations:
    for s in range(n_dif_sim):
        df_us = df_sim_u_list_omn[s]
        df_ltus = df_sim_ltu_list_omn[s]
        df_es = df_sim_e_list_omn[s]
        us = np.array(df_us.iloc[:,3:])
        es = np.array(df_es.iloc[:,3:])
        ltus = np.array(df_ltus.iloc[:,3:])
        u_total_sim = us.sum(axis=0)
        e_total_sim = es.sum(axis=0)
        ltu_total_sim = ltus.sum(axis=0)
        n_agents = es.sum(axis=0)[0] + us.sum(axis=0)[0]
        average_ltu = average_ltu + ltu_total_sim / n_agents
        ax3.plot(time_array, 100*ltu_total_sim /  n_agents,linewidth=linewidth_, color = "#004D40", alpha=0.3)

        df_us_kn = df_sim_u_list_kn[s]
        df_ltus_kn = df_sim_ltu_list_kn[s]
        df_es_kn = df_sim_e_list_kn[s]
        us_kn = np.array(df_us_kn.iloc[:,3:])
        es_kn = np.array(df_es_kn.iloc[:,3:])
        ltus_kn = np.array(df_ltus_kn.iloc[:,3:])
        u_total_sim_kn = us_kn.sum(axis=0)
        e_total_sim_kn = es_kn.sum(axis=0)
        ltu_total_sim_kn = ltus_kn.sum(axis=0)
        n_agents = es.sum(axis=0)[0] + us.sum(axis=0)[0]
        average_ltu_kn = average_ltu_kn + ltu_total_sim_kn / n_agents
        ax3.plot(time_array, 100*ltu_total_sim_kn /  n_agents,linewidth=linewidth_, color = "#D81B60", alpha=0.3)
    average_ltu_kn = average_ltu_kn / n_dif_sim
    # ax2.plot(time_array, 100*average_ltu_kn, "-",color="red" ,linewidth=linewidth_,label="No network, simulations\naverage")
    #
    # average_ltu = average_ltu / n_dif_sim
    # ax2.plot(time_array, 100*average_ltu, "-",color="limegreen" , linewidth=linewidth_, label="Network, simulations")

###### Numerical

ax3.plot(time_array, ltu_omn, "--", linewidth=linewidth_thin, color="k",label="Network, analytical")
ax3.plot(time_array, ltu_kn, "-.", linewidth=linewidth_thin,color="k", label="No Network, numerical\nsolution")
ax3.axvline(x=int(starting_time  + time_scale_year * t_shock), linestyle=":", linewidth=linewidth_,color="brown", alpha=0.5, label="Automation starts")
if plot_sharp_transition:
        ax2.axvspan(t_transition_start + 0.25*shock_duration_years, t_transition_end - 0.25*shock_duration_years, facecolor='coral', alpha=0.5)
ax3.axvspan(t_transition_start, t_transition_end, facecolor='grey', alpha=0.4)
ax3.set_ylim([0.6, 3.0])
ax3.set_xlim([2010,2060])
if shock[13:] == "_095":
    ax3.set_ylim([0.6, 6.0])
elif shock[13:] == "_105":
    ax3.set_ylim([0.0, 3.0])
ax3.set_ylabel("long-term \nunemployment\nrate (%)", fontsize=fontsize_axis)
if plot_simulations:
        plt.savefig(path_exp_fig+"Unemployment_demand_simvsnum"+ shock + "duration" +str(shock_duration_years) +"params"+ parameter_names +"OMN.svg", bbox_inches="tight")
        plt.savefig(path_exp_fig+"Unemployment_demand_simvsnum"+ shock + "duration" +str(shock_duration_years) +"params"+ parameter_names +"OMN.png", bbox_inches="tight")
plt.show()

print("pre shock steady state unemp OMN ", round(u_omn[t_shock], 1))
print("pre shock steady state unemp kn ", round(u_kn[t_shock], 1))
print("rel diff between OMN and KN = ", (u_omn[t_shock] - u_kn[t_shock])/u_kn[t_shock])
print("rel diff between OMN and KN = ", (5.3 - 4.1)/4.1)
print("spike unemp OMN ", round(max(u_omn), 1))
print("spike unemp kn ", round(max(u_kn), 1))
print("relative dif = ", (max(u_omn) -u_omn[t_shock])/u_omn[t_shock]  )
print("pre shock steady longterm unemp state OMN ", round(ltu_omn[t_shock], 1))
print("pre shock steady longterm unemp state kn ", round(ltu_kn[t_shock], 1))
print("spike longterm unemp OMN ", round(max(ltu_omn), 1))
print("spike longterm unemp kn ", round(max(ltu_kn), 1))
print("relative dif = ", (max(ltu_omn) -ltu_omn[t_shock])/ltu_omn[t_shock]  )
print("post shock steady state unemp omn ",  u_omn[-1])
print("pre psot shock unemp difference unemp omn " , (u_omn[t_shock] -  u_omn[-1]))

# # # ####################################################
# # # # Plot it occupation wise
# # # ####################################################
# #


for occ in [284, 26, 319, 374, 249, 112, 374, 183]:#[224, 374, 386, 265, 81, 112, 98, 95, 0, 298, 234, 248, 218, 216, 319, 304, 301, 283, 183]:#occupations_to_plot[0:1]:#range(464):
    f = plt.figure(figsize=figsize_)
    f.subplots_adjust(hspace=0.2)
    gs = gridspec.GridSpec(2, 1,height_ratios=[1, 1])
    ax2 = plt.subplot(gs[1])
    ax1 = plt.subplot(gs[0], sharex=ax2)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.tick_params(labelsize=fontsize_ticks)
    ax1.tick_params(labelsize=fontsize_ticks)

    # get numerical u and ltu
    u_omn, ltu_omn = make_u_ltu_rates_occ(occ, df_num_u_omn, df_num_e_omn, df_num_ltu_omn)
    print(occ)
    print(df_us["label"][occ])
    # # ###Plotting unemployment
    if plot_simulations:
        average_u = np.zeros(t_sim)
        for s in range(n_dif_sim):
            df_us = df_sim_u_list_omn[s]
            df_es = df_sim_e_list_omn[s]

            us = np.array(df_us.iloc[occ,3:])
            es = np.array(df_es.iloc[occ,3:])
            average_u = average_u + us / (us + es)
            ax1.plot(time_array, 100*us / (us + es), color = "forestgreen", alpha=0.2)

        average_u = average_u / n_dif_sim
        ax1.plot(time_array, 100*average_u, "-", color="brown", alpha=0.8, linewidth=linewidth_, label="Network, simulations\naverage")
    ######## Numerical
    ax1.plot(time_array, u_omn, "k--",linewidth=linewidth_, label="Network, numerical\nsolution")
    ax1.axvline(x=int(starting_time  + time_scale_year * t_shock), linestyle=":", linewidth=linewidth_,color="brown", alpha=0.5, label="Automation starts")
    ax1.axvspan(t_transition_start, t_transition_end, facecolor='grey', alpha=0.4)

    ax1.set_ylabel("unemployment \n rate (%)", fontsize=fontsize_axis)
    # ax1.set_title("Frey and Osborne automation shock\n"+df_us["label"][occ] , fontsize=fontsize_title)
    #ax1.set_title("Suitability for Machine Learning automation shock", fontsize=20)
    #ax1.set_ylim([2, 6.3])
    #ax1.set_ylim([4, 5])

    # ax1.legend(fontsize=fontsize_legend)
    # Plot longterm unemployment
    if plot_simulations:
        average_ltu = np.zeros(t_sim)
        for s in range(n_dif_sim):
            df_us = df_sim_u_list_omn[s]
            df_es = df_sim_e_list_omn[s]
            df_ltus = df_sim_ltu_list_omn[s]
            us = np.array(df_us.iloc[occ,3:])
            es = np.array(df_es.iloc[occ,3:])
            ltus = np.array(df_ltus.iloc[occ,3:])
            average_ltu = average_ltu + ltus / (us + es)
            ax2.plot(time_array, 100*ltus / (us + es), color = "forestgreen", alpha=0.2)

        average_ltu = average_ltu / n_dif_sim
        ax2.plot(time_array, 100*average_ltu, "-", alpha=0.8, color="brown",linewidth=linewidth_, label="Network, simulations\naverage")


    ax2.plot(time_array, ltu_omn, "k--", linewidth=linewidth_,label="Network, numerical\nsolution")
    ax2.axvline(x=int(starting_time  + time_scale_year * t_shock), linestyle=":", linewidth=linewidth_,color="brown", alpha=0.5, label="Automation starts")
    ax2.axvspan(t_transition_start, t_transition_end, facecolor='grey', alpha=0.4)
    #ax2.set_xlim([starting_time + 5, starting_time + t_sim*time_scale_year ])
    ax2.set_xlim([starting_time+5,2060 ])
    # NOTE add xlabel here
    # ax2.set_xlabel("time (years)", fontsize=fontsize_axis)
    ax2.set_ylabel("long-term \nunemployment\nrate (%)", fontsize=fontsize_axis)
    # ax2.legend(fontsize=fontsize_legend)
    plt.show()
