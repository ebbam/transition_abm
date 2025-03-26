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
from scipy.stats import pearsonr
print("Packages installed.")

rng = np.random.default_rng()
path = ""
###################################
# US MODEL CONDITIONS AND DATA ####
###################################
# Set preliminary parameters for delta_u, delta_v, and gamma 
# - these reproduce nice Beveridge curves but were arrived at non-systematically
del_u = 0.015
del_v = 0.009
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

 
# Recording outside of loop for global access
sim_record_t_all, net_behav, net_behav_list = 0, 0, 0
sim_record_f_all, net_nonbehav, net_nonbehav_list = 0, 0, 0
for behav_spec in [True, False]:

# The following are the input data to the function that runs the model.
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
        'delay': 120, 
        'bus_amp': 0.02}

    ####################
    # Model Run ########
    ####################
    def run_single(mod_data = mod_data, 
                net_temp = net_temp, 
                vacs = vacs, 
                behav_spec = behav_spec, 
                time_steps = 200, 
                d_u = del_u, 
                d_v = del_v,
                gamma = gamma,
                bus_cycle_len = 160,
                delay = 80, 
                bus_amp = 0.02):
        #net_temp, vacs = initialise(len(mod_data['A']), mod_data['employment'].to_numpy(), mod_data['unemployment'].to_numpy(), mod_data['vacancies'].to_numpy(), mod_data['demand_target'].to_numpy(), mod_data['A'], mod_data['wages'].to_numpy(), mod_data['gend_share'].to_numpy())
        #behav_spec = False
        #time_steps = 30
        #gamma = 0.1
        #d_v = 0.009
        
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
        record = [np.sum(np.concatenate((np.zeros((464, 1)), 
                                        mod_data['employment'].to_numpy(), 
                                        mod_data['unemployment'].to_numpy(), 
                                        mod_data['employment'].to_numpy() + mod_data['unemployment'].to_numpy(),
                                        mod_data['vacancies'].to_numpy(), 
                                        np.zeros((464, 1)),
                                        mod_data['demand_target'].to_numpy()), axis = 1), 
                                        axis = 0)]
        
        #print(parameter['vacs'])
        vacs_temp = deepcopy(vacs)
        net = deepcopy(net_temp)
        for t in range(time_steps):
            # Ensure number of workers in economy has not changed
            #tic = time.process_time()
            for occ in net:
                ### APPLICATIONS
                # Questions to verify:
                # - CANNOT be fired and apply in same time step ie. time_unemployed > 0
                # - CAN be rejected and apply in the same time step - no protected attribute
                # isolate list of vacancies in economy that are relevant to the occupation
                # - avoids selecting in each search_and_apply application
                r_vacs = [vac for vac in vacs_temp if occ.list_of_neigh_bool[vac.occupation_id]]                
                for u in occ.list_of_unemployed:
                    u.search_and_apply(net, r_vacs, behav_spec)

                ### SEPARATIONS
                occ.separate_workers(d_u, gamma)

            ### HIRING
            # Ordering of hiring randomised to ensure list order does not matter in filling vacancies...
            # Possibly still introduces some bias...this seems to be where the "multiple offer" challenge Maria mentioned comes from
            # ....might be better to do this using an unordered set?
            for v_open in sorted(vacs_temp,key=lambda _: random.random()):
                # Removes any applicants that have already been hired in another vacancy
                v_open.applicants[:] = [app for app in v_open.applicants if not(app.hired)]
                if len(v_open.applicants) > 0:
                    v_open.hire(net)
                    v_open.filled = True
                    #vacs.remove(v_open)
                    assert(len(v_open.applicants) == 0)
                else:
                    pass

            vacs_temp = [v for v in vacs_temp if not(v.filled)] 

            # Reset counters for record in time t
            empl = 0 
            unemp = 0
            n_ltue = 0
            t_demand = 0

            ### OPEN VACANCIES
            # Update vacancies after all shifts have taken place
            # Could consider making this a function of the class itself?
            for occ in net:
                # Update time_unemployed and long-term unemployed status of unemployed workers
                # Remove protected "hired" attribute of employed workers
                occ.update_workers()
                emp = len(occ.list_of_employed)
                occ.current_demand = bus_cycle_demand(len([v_open for v_open in vacs_temp if v_open.occupation_id == occ.occupation_id]) + emp, t, bus_amp, bus_cycle_len)
                vac_prob = d_v + ((1 - d_v) * (gamma * max(0, occ.target_demand - occ.current_demand))) / (emp + 1)
                vacs_create = emp*int(vac_prob) + int(np.random.binomial(emp, vac_prob%1))
                for v in range(vacs_create):
                    vacs_temp.append(vac(occ.occupation_id, [], occ.wage, False))

                empl += len(occ.list_of_employed) 
                unemp += len(occ.list_of_unemployed)
                n_ltue += sum(wrkr.longterm_unemp for wrkr in occ.list_of_unemployed)
                t_demand += occ.target_demand

            ### UPDATE INDICATOR RECORD
            record = np.append(record, 
                                np.array([[t+1, empl, unemp, empl + unemp, len(vacs_temp), n_ltue, t_demand]]), 
                                axis = 0)

        print("Done after ", t + 1, " time steps.")

        # clean_record = pd.DataFrame(record[delay:])
        # clean_record.columns =['Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Target_Demand']
        # clean_record['UER'] = clean_record['Unemployment']/clean_record['Workers']
        # clean_record['VACRATE'] = clean_record['Vacancies']/clean_record['Target_Demand']
        #data = clean_record[['Time Step', 'UER', 'VACRATE']]
        data = {'UER': record[delay:,2]/record[delay:,3], 
                'VACRATE': record[delay:,4]/record[delay:,6]}

        #ltuer = (clean_record['LT Unemployed Persons']/clean_record['Workers']).mean(axis = 0)
        #vac_rate = (clean_record['Vacancies']/clean_record['Target_Demand']).mean(axis = 0)
        return data

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
    sim_data = pd.DataFrame(rec)
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
    if behav_spec:
        plt.savefig(path + 'output/uer_vac_descriptive_behav.jpg', dpi = 300)
    else:
        plt.savefig(path + 'output/uer_vac_descriptive.jpg', dpi = 300)

    print("Macro variables loaded and abbreviated.")

    def pyabc_run_single(parameter):     
        res = run_single(**parameter)
        return res 

    # Proposed priors for d_u and d_v taken from the separations and 
    # job openings rates modelled in the first few plots of this notebook
    # These priors can of course be more carefully selected calculating directly from those rates....next steps
    prior = pyabc.Distribution(d_u = pyabc.RV("uniform", 0.001, 0.05),
                            d_v = pyabc.RV("uniform", 0.001, 0.05),
                            gamma = pyabc.RV("uniform", 0.05, 0.3),
                            # TASK: INCREASE UPPER PRIOR BOUND HERE!!! Previously 0.04... now testing up to 0.1
                            bus_amp = pyabc.RV("uniform", 0.005, 0.1))

    # distance function jointly minimises distance between simulated 
    # mean of UER and vacancy rates to real-world UER and vacancy rates
    # Now also matches the shape/oscillation of each variable
    def distance_weighted(x, y, weight_shape=0.5, weight_mean=0.5):
        """
        Weighted distance function combining mean difference and correlation.

        Args:
            x (dict): Simulated data with keys "UER" and "VACRATE".
            y (dict): Real-world data with keys "UER" and "VACRATE".
            weight_shape (float): Weight for shape matching (correlation).
            weight_mean (float): Weight for mean matching (SSE).

        Returns:
            dist (float): Combined distance measure (UER and VACRATE).
        """
        # Calculate sum of squared errors (SSE) for UER and VACRATE
        uer_sse = np.sum(((x["UER"][0:120] - y["UER"]) / np.mean(y["UER"]))**2)
        vacrate_sse = np.sum(((x["VACRATE"][0:120] - y["VACRATE"]) / np.mean(y["VACRATE"]))**2)
        
        # Correlation between simulated and real-world VACRATE (shape matching)
        corr_vacrate = 1 - pearsonr(x["VACRATE"][0:120], y["VACRATE"])[0]
        
        # Correlation between simulated and real-world UER (shape matching)
        corr_uer = 1- pearsonr(x["UER"][0:120], y["UER"])[0]
        
        # Define total distance as weighted sum of mean and shape components
        dist = weight_mean * (np.sqrt(uer_sse) + np.sqrt(vacrate_sse)) + weight_shape * (corr_vacrate + corr_uer)
        return dist

    calib_sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs = 100)
    abc = pyabc.ABCSMC(pyabc_run_single, prior, distance_weighted, population_size = 500, sampler = calib_sampler)

    db_path = os.path.join(tempfile.gettempdir(), "test.db")

    # The following creates the "reference" values from the observed data - I pull the non-recession or expansion period from 2010-2019.
    observation = macro_observations.loc[(macro_observations['DATE'] >= '2006-12-01') & (macro_observations['DATE'] <= "2016-11-01")].reset_index()
    #buffer = int((len(observation) - parameter['time_steps'])/2)
    #obs_abbrev = (observation[buffer + (int(parameter['delay']/2)):(buffer + parameter['time_steps']) - int((parameter['delay']/2))]).reset_index()
    #
    data = {'UER': np.array(observation['UER']),
            'VACRATE': np.array(observation['VACRATE'])}

    abc.new("sqlite:///" + db_path, data)

    history = abc.run(minimum_epsilon=0.1, max_nr_populations=5)

    print("Pyabc run finished.")


    gt = {"d_u": jolts['SEPSRATE'].mean(axis = 0), "d_v": jolts['QUITSRATE'].mean(axis = 0), "gamma": 0.2, "bus_amp": 0.02}

    df, w = history.get_distribution()
    gt = {"d_u": df['d_u'].mean(axis = 0), "d_v": np.sum(df['d_v']*w), "gamma": np.sum(df['gamma']*w), "bus_amp": np.sum(df['bus_amp']*w)}

    plot_kde_matrix(
        df,
        w,
        limits={"d_u": (0.001, 0.05), "d_v": (0.001, 0.05), "gamma": (0.05, 0.3), 'bus_amp': (0.005, 0.1)},
        refval=gt,
        refval_color='k',
    )
    # Not sure if the value to be extracted is the weighted mean of the outcome or simply the mean?
    sns.kdeplot(x = "d_u", y = "d_v", data = df, weights = w, cmap = "viridis", fill = True)
    plt.axvline(x = df['d_u'].mean(axis = 0))
    plt.axvline(x = np.sum(df['d_u']*w), color = 'red')
    plt.axhline(y = df['d_v'].mean(axis = 0))
    plt.axhline(y = np.sum(df['d_v']*w), color = 'red')
    if behav_spec:
        plt.savefig(path + 'output/kde_plot_behav.jpg', dpi = 300)
    else:
        plt.savefig(path + 'output/kde_plot.jpg', dpi = 300)

    sns.jointplot(x = "d_u", y = "d_v", kind = "kde", data = df, weights = w, cmap = "viridis_r")
    plt.axvline(x = df['d_u'].mean(axis = 0))
    plt.axvline(x = np.sum(df['d_u']*w), color = 'red')
    plt.axhline(y = df['d_v'].mean(axis = 0))
    plt.axhline(y = np.sum(df['d_v']*w), color = 'red')
    plt.title("KDE Plot")
    if behav_spec:
        plt.savefig(path + 'output/joint_plot_behav.jpg', dpi = 300)
    else:
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
        axes[1,0].set_xlim([0,120])
        axes[1,1].set_xlim([0,120])

    # Plot the stored summary statistics
    sum_stats = history.get_weighted_sum_stats(t=history.max_t)
    for stored in sum_stats[1][:n]:
        axes[2,0].plot(stored["UER"], color="green", alpha=alpha)
        axes[2,1].plot(stored["VACRATE"], color="green", alpha=alpha)

    # Plot the observed UER from BLS
    for ax in axes[:,0]:
        observation.plot(y="UER", ax=ax, color="black", linewidth=1.5)
        ax.legend().set_visible(False)
        ax.set_ylabel("UER")
        
    # Plot the observed VACRATE from JOLTS
    for ax in axes[:,1]:
        observation.plot(y="VACRATE", ax=ax, color="black", linewidth=1.5)
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

    if behav_spec:
        plt.savefig(path + 'output/prior_post_selected_distributions_plot_behav.jpg', dpi = 300)
    else:
        plt.savefig(path + 'output/prior_post_selected_distributions_plot.jpg', dpi = 300)
    

    # ## Testing Selected Parameters
    # 
    # Below I pull the weighted mean of the posterior. Not sure if this is the correct way to pull the triangulated parameter estimate...? Indeed, the model run with these parameters does not look good and both look lower than represented in the heat/contour maps above. The model results with these parameters look bad both with respect to replicating a Beveridge curve as well as we did earlier with hand-selected estimates (and you'll see by the warnings that the delta_u is likely too high....again, I think that this is becuause of poor choice of arguments to the SMCABC algorithm above. In other words, not quite there...to be improved...but getting closer :) 

    d_u_hat = np.sum(df['d_u']*w)
    print("d_u_hat: ", d_u_hat)

    d_v_hat = np.sum(df['d_v']*w)
    print("d_v_hat: ", d_v_hat)

    gamma_hat = np.sum(df['gamma']*w)
    print("gamma_hat: ", gamma_hat)

    bus_amp_hat = np.sum(df['bus_amp']*w)
    print('bus_amp_hat: ', bus_amp_hat)


    parameter.pop('delay', None)
    parameter.update({'runs': 2,
                    'd_u': d_u_hat,
                    'd_v': d_v_hat,
                    'gamma': gamma_hat,
                    'bus_amp': bus_amp_hat})

    if behav_spec:
        sim_record_t_all, net_behav, net_behav_list = run_sim(**parameter)

        calib_params = {"d_u": [d_u_hat],
                    "d_v": [d_v_hat],
                    "gamma": [gamma_hat],
                    "bus_hat": [bus_amp_hat]}

        # Writing the parameter values to CSV
        with open(path + 'data/calibrated_params_behav.csv', 'w', newline='') as csvfile:
            # Initialize the writer object
            writer = csv.writer(csvfile)
            
            # Write the header (the keys of the dictionary)
            writer.writerow(calib_params.keys())
            
            # Write the values (taking the first element of each list)
            writer.writerow([calib_params[key][0] for key in calib_params])
    else:
        sim_record_f_all, net_nonbehav, net_nonbehav_list = run_sim(**parameter)
        calib_params = {"d_u": [d_u_hat],
                    "d_v": [d_v_hat],
                    "gamma": [gamma_hat],
                    "bus_hat": [bus_amp_hat]}

        # Writing the parameter values to CSV
        with open(path + 'data/calibrated_params.csv', 'w', newline='') as csvfile:
            # Initialize the writer object
            writer = csv.writer(csvfile)
            
            # Write the header (the keys of the dictionary)
            writer.writerow(calib_params.keys())
            
            # Write the values (taking the first element of each list)
            writer.writerow([calib_params[key][0] for key in calib_params])
    


# Summary values for one run 
sim_record_t = pd.DataFrame(np.transpose(np.hstack(sim_record_t_all)))
sim_record_t.columns =['Sim', 'Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Target_Demand']
sim_record_f = pd.DataFrame(np.transpose(np.hstack(sim_record_f_all)))
sim_record_f.columns =['Sim', 'Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Target_Demand']

record1_t = sim_record_t[(sim_record_t['Sim'] == 0)].groupby(['Sim', 'Time Step']).sum().reset_index() #  
record1_f = sim_record_f[(sim_record_f['Sim'] == 0)].groupby(['Sim', 'Time Step']).sum().reset_index() #  & (sim_record_t['Time Step'] >= 80)

end_t = record1_t[(record1_t['Time Step'] == 280)]
end_f = record1_f[(record1_f['Time Step'] == 280)]

ue_vac_f = record1_f.loc[:,['Workers', 'Unemployment', 'Vacancies', 'Target_Demand']]
ue_vac_f['UE Rate'] = ue_vac_f['Unemployment'] / ue_vac_f['Workers']
ue_vac_f['Vac Rate'] = ue_vac_f['Vacancies'] / ue_vac_f['Target_Demand']
ue_vac_f = ue_vac_f[46:]

ue_vac_t = record1_t.loc[:,['Workers', 'Unemployment', 'Vacancies', 'Target_Demand']]
ue_vac_t['UE Rate'] = ue_vac_t['Unemployment'] / ue_vac_t['Workers']
ue_vac_t['Vac Rate'] = ue_vac_t['Vacancies'] / ue_vac_t['Target_Demand']
ue_vac_t = ue_vac_t[46:]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
#ue_vac_f = ue_vac_f[40:]
#ue_vac_t = ue_vac_t[40:]

ax1.plot(ue_vac_f['UE Rate'], ue_vac_f['Vac Rate'])
ax1.scatter(ue_vac_f['UE Rate'], ue_vac_f['Vac Rate'], c=ue_vac_f.index, s=100, lw=0)
ax1.plot(observation['UER'], observation['VACRATE'], c = "grey", alpha = 0.5)
ax1.set_title("Non-behavioural")
ax1.set_xlabel("UE Rate")
ax1.set_ylabel("Vacancy Rate")

ax2.plot(ue_vac_t['UE Rate'], ue_vac_t['Vac Rate'])
ax2.set_title("Behavioural")
ax2.scatter(ue_vac_t['UE Rate'], ue_vac_t['Vac Rate'], c=ue_vac_t.index, s=100, lw=0) 
ax2.plot(observation['UER'], observation['VACRATE'], c="grey", alpha = 0.5)
ax2.set_xlabel("UE Rate")
ax2.set_ylabel("Vacancy Rate")
    
fig.suptitle("USA Model Beveridge Curve", fontweight = 'bold')
fig.tight_layout()

plt.savefig(path + 'output/run_w_calib_params.jpg', dpi = 300)


# Incorporating one set of simulated data
ue_vac_f['PROV DATE'] = pd.date_range(start = "2005-12-01", periods = len(ue_vac_f), freq = "ME")
ue_vac_f['FD_SIMUER'] = pd.Series(ue_vac_f['UE Rate']).diff()
ue_vac_f['FD_SIMVACRATE'] = pd.Series(ue_vac_f['Vac Rate']).diff()

ue_vac_t['PROV DATE'] = pd.date_range(start = "2005-12-01", periods = len(ue_vac_f), freq = "ME")
ue_vac_t['FD_SIMUER'] = pd.Series(ue_vac_t['UE Rate']).diff()
ue_vac_t['FD_SIMVACRATE'] = pd.Series(ue_vac_t['Vac Rate']).diff()


# Non-recession period
fig, ax = plt.subplots()
macro_observations.plot.line(ax = ax, figsize = (8,5), x= 'DATE', y = 'UER', color = "blue", linestyle = "dotted")
macro_observations.plot.line(ax = ax, figsize = (8,5), x= 'DATE', y = 'VACRATE', color = "red", linestyle = "dotted")
recessions.plot.area(ax = ax, figsize = (8,5), x= 'DATE', color = "grey", alpha = 0.2)
ue_vac_f.plot.line(ax = ax, x = 'PROV DATE', y = 'UE Rate', color = "blue", label = "UER (Sim. Non-behav.)")
ue_vac_f.plot.line(ax = ax, x = 'PROV DATE', y = 'Vac Rate', color = "red", label = "VACRATE (Sim. Non-behav)")
ue_vac_t.plot.line(ax = ax, x = 'PROV DATE', y = 'UE Rate', color = "skyblue", label = "UER (Sim. Behav.)")
ue_vac_t.plot.line(ax = ax, x = 'PROV DATE', y = 'Vac Rate', color = "lightcoral", label = "VACRATE (Sim. Behav.)")
plt.xlim('2005-12-01', "2020-01-01")
plt.ylim(0, 0.2)

# Add title and axis labels
plt.title('Monthly US Unemployment Rate (Seasonally Adjusted)')
plt.xlabel('Time')
plt.ylabel('Monthly UER')
plt.xticks(rotation=45)

plt.savefig(path + 'output/run_w_calib_params_real_rates.jpg', dpi = 300)



