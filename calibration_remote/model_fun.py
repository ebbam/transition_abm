######## Central Run Function #############

import numpy as np
import pandas as pd
from abm_funs import *
import random as random
from copy import deepcopy 
import math as math
from scipy.stats import norm

####################
# Model Run ########
####################
def run_single_local(d_u, 
                    d_v,
                    gamma_v,
                    gamma_u,
                    otj,
                    cyc_otj, 
                    cyc_ue, 
                    disc,
                    mod_data, 
                    net_temp, 
                    vacs, 
                    time_steps, # set equal to length of gdp_data
                    delay,
                    gdp_data,
                    occ_shocks_data,
                    bus_confidence_dat,
                    app_effort_dat,
                    vac_data,
                    steady_state,
                    wage_prefs,
                    emp_apps,
                    theta,
                    mistake_rate,
                    strict_rw,
                    dumb_hire,
                    simple_res = False):

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
    if simple_res:
        record = np.empty((0, 10))
    else:
        record = [] 

    vacs_temp = deepcopy(vacs)
    net = deepcopy(net_temp)
    seekers_rec = []
    time_steps = time_steps + delay
    avg_wage_offer_diff_records = []
    app_load_records = []
    retired_all = 0
    ue_spell_records = []

    for t in range(time_steps-1):
        # Steady State option
        if t < delay | steady_state:
            curr_bus_cy = 1
            vr_t = 0.03
        if t >= delay and not steady_state:
            if occ_shocks_data is not None:
                curr_bus_cy = np.take(occ_shocks_data, indices=t-delay, axis=1)
            # else:
            #     curr_bus_cy = gdp_data[t-delay]

            vr_t = vac_data[t-delay]

        # Ensure number of workers in economy has not changed
        emp_seekers = 0
        unemp_seekers = 0
        u_apps = 0
        e_apps = 0
        
        vacancies_by_occ = defaultdict(list)
        for v in vacs_temp:
            vacancies_by_occ[v.occupation_id].append(v)

        # allocate new entrants across entry-level occupations exactly - I was originally running into a rounding error
        entry_occs = [o for o in net if o.entry_level]
        entry_td = np.sum([o.target_demand for o in entry_occs])

        # Build an allocation dict: occupation_id -> number of new entrants
        if retired_all > 0 and entry_td > 0 and len(entry_occs) > 0:
            probs = np.array([o.target_demand for o in entry_occs], dtype=float)
            probs = probs / probs.sum()  # relative shares
            draws = np.random.multinomial(retired_all, probs)  # sums exactly to retired_all
            entry_alloc = {o.occupation_id: n for o, n in zip(entry_occs, draws)}
        else:
            entry_alloc = {}

        for occ in net:
            occ.separated = 0
            occ.hired = 0
            if occ_shocks_data is not None and t >= delay and not steady_state:
                occ_shock = curr_bus_cy[occ.occupation_id]
            else:
                occ_shock = curr_bus_cy

            # Entry 0.1%
            # occ.entry_rate(0.01)
            # Entry by partitioning retired demand according to target demand across entry level occupations
            if occ.entry_level:
                occ.entry(entry_alloc.get(occ.occupation_id, 0))
            # Remove the top 2% of earners in an occupation's employed list
            # occ.entry_and_exit_fixed(0.02)

            ### APPLICATIONS
            # isolate list of vacancies in economy that are relevant to the occupation
            # - avoids selecting in each search_and_apply application
            r_vacs = vacancies_by_occ  
    
            for u in occ.list_of_unemployed:
                unemp_seekers += 1
                u.search_and_apply(net, r_vacs, disc, app_effort_dat, wage_prefs, mistake_rate, unique_random = False, global_pool_override = vacs_temp, strict_reservation_wage = strict_rw)

            if otj:
                # # For both models, a mean of 7% of employed workers are searching for new jobs
                # # This fluctuates with the business cycle in the behavioural model in line with gdp
                # if cyc_otj:
                #     search_scaling = occ_shock*0.07
                # # Static mean in the non-behavioural model
                # else:
                #     search_scaling = 0.07
                # for e in random.sample(occ.list_of_employed, int(search_scaling*len(occ.list_of_employed))):
                #     emp_seekers += 1
                #     e_apps += e.emp_search_and_apply(net, r_vacs, disc)
                if cyc_otj:
                    #print(f'Competition: {occ.competition_last}')
                    for e in occ.list_of_employed:
                        prob = p_search_logit(
                            age=e.age,
                            # Taken from calculated competition metric for occupation
                            comp=occ.competition_last,
                            alpha=-1.58,     # tune baseline
                            beta_A=-0.05,  # age effect
                            beta_C=theta,    # competition effect
                            A0=40.0
                        )
                        #print(f'Search Probability: {prob}')
                        if np.random.random() < prob:
                            emp_seekers += 1
                            e_apps += e.emp_search_and_apply(net, r_vacs, disc, emp_apps, wage_prefs, mistake_rate, unique_random = False, global_pool_override = vacs_temp)
                else:
                    for e in random.sample(occ.list_of_employed, int(0.07*len(occ.list_of_employed))):
                        emp_seekers += 1
                        e_apps += e.emp_search_and_apply(net, r_vacs, disc, emp_apps, wage_prefs, mistake_rate, unique_random = False, global_pool_override = vacs_temp)

            u_apps += sum(wrkr.apps_sent for wrkr in occ.list_of_unemployed if wrkr.apps_sent is not None)

            d_wage_offers = [w.d_wage_offer for w in occ.list_of_unemployed if hasattr(w, 'd_wage_offer')]
            avg_diff = np.mean(d_wage_offers) if d_wage_offers else np.nan
            avg_wage_offer_diff_records.append({
                'Time Step': t+1,
                'Occupation': occ.occupation_id,
                'Avg_Wage_Offer_Diff': avg_diff
            })

            ### SEPARATIONS
            try:
                occ.separate_workers(d_u, gamma_u, occ_shock)
            except Exception as e:
                print("Error at separation step")
                return np.inf
            
                    
        seekers_rec.append([t+1, unemp_seekers, u_apps, emp_seekers, e_apps])

        # Collect application flow per occupation
        app_load_rows = [
            (t+1, v.occupation_id, 
             # All applicants
             len(v.applicants), 
             # Only unemployed applicants
            len([app for app in v.applicants if app.time_unemployed != 0]), 
            # Only employed applicants
            len([app for app in v.applicants if app.time_unemployed == 0]))
            for v in vacs_temp
        ]
        if app_load_rows:  # only if there are vacancies
            app_load_step = pd.DataFrame(app_load_rows, columns=['Time Step', 'Occupation', 'ApplicantsPerVac', 'UnempApplicantsPerVac', 'EmpApplicantsPerVac'])
            app_load_agg = app_load_step.groupby(['Time Step', 'Occupation']).agg(
                OpenVacs=('ApplicantsPerVac', 'size'),
                TotalApplicants=('ApplicantsPerVac', 'sum'),
                MeanAppsPerVac=('ApplicantsPerVac', 'mean'),
                MedianAppsPerVac=('ApplicantsPerVac', 'median'),
                TotalUnempAppsPerVac=('UnempApplicantsPerVac', 'sum')
            ).reset_index()
            app_load_agg['MeanUnempAppsPerVac'] = app_load_agg['TotalUnempAppsPerVac'] / app_load_agg['TotalApplicants']

            # Find level of competition in neighboring occupations in the network
            app_load_agg['AppsPerVac'] = app_load_agg.apply(
                lambda r: (r['TotalApplicants'] / r['OpenVacs'])
                        if r['OpenVacs'] > 0 else np.nan,
                axis=1
            )
            comp_map = dict(zip(app_load_agg['Occupation'], app_load_agg['AppsPerVac']))

            # Assign to occupations for use in next step
            for occ in net:
                val = comp_map.get(occ.occupation_id, np.nan)
                if np.isnan(val):
                    # fallback if no vacancies: keep last known
                    val = getattr(occ, "competition_last", 0.0)
                occ.competition_last = val

            app_load_records.append(app_load_agg)
    
        vacancies_by_occ = defaultdict(list)
        for v in vacs_temp:
            vacancies_by_occ[v.occupation_id].append(v)
        
        ### HIRING
        # Ordering of hiring randomised to ensure list order does not matter in filling vacancies...
        for v_open in sorted(vacs_temp,key=lambda _: random.random()):
            if dumb_hire:
                # Removes any applicants that have already been hired in another vacancy
                v_open.applicants[:] = [app for app in v_open.applicants if not(app.hired)]
            
            v_open.time_open += 1
            if len(v_open.applicants) > 0:
                ret = v_open.hire(net, dumb_hire)
                # If hire() succeeded (didn't return None on first line)
                if ret is not None:
                    # Hire was successful
                    v_open.filled = True
                    
                    # Record UE spell only if this was a UE→E hire (not EE→E)
                    origin_occ, dest_occ, ue_dur = ret
                    if origin_occ is not None and dest_occ is not None and ue_dur is not None:
                        ue_spell_records.append((t+1, origin_occ, dest_occ, ue_dur))
            assert(len(v_open.applicants) == 0)


        # Close a vacancy after it's been open for 6 months
        vacs_temp = [v for v in vacs_temp if not(v.filled) and v.time_open < 6] 

        # # Reset counters for record in time t
        if simple_res:
            empl = 0 
            unemp = 0
            n_ltue = 0
            t_demand = 0
            seps = 0

        ### OPEN VACANCIES
        # Update vacancies after all shifts have taken place
        # Could consider making this a function of the class itself?
        retired_all = 0
        for occ in net:
            if occ_shocks_data is not None and t >= delay and not steady_state:
                occ_shock = curr_bus_cy[occ.occupation_id]
            else:
                occ_shock = curr_bus_cy
            # Only consider unemployed workers who have d_wage_offer attribute
            u_rel_wage = sum(wrkr.ue_rel_wage for wrkr in occ.list_of_employed if wrkr.hired and wrkr.ue_rel_wage is not None)
            e_rel_wage = sum(wrkr.ee_rel_wage for wrkr in occ.list_of_employed if wrkr.hired and wrkr.ee_rel_wage is not None)
            ue = len([w for w in occ.list_of_employed if w.hired and w.ue_rel_wage is not None])
            ee = len([w for w in occ.list_of_employed if w.hired and w.ee_rel_wage is not None])
            # Update time_unemployed and long-term unemployed status of unemployed workers
            # Remove protected "hired" attribute of employed workers
            occ.update_workers()
            retired = occ.retire_workers()
            retired_all += retired
            # Assert that all unemployed people have spent 1 or more time periods unemployed
            assert(sum([worker.time_unemployed <= 0 for worker in occ.list_of_unemployed]) == 0)
            # Assert that all employed people have spent 0 time periods unemployed
            assert(sum([worker.time_unemployed <= 0 for worker in occ.list_of_employed]) == len(occ.list_of_employed))
            emp = len(occ.list_of_employed)
            curr_vacs = len([v_open for v_open in vacs_temp if v_open.occupation_id == occ.occupation_id])
            occ.current_demand = (curr_vacs + emp)
            # If real-world vacancy rate is greater than the current vacancy rate, then we create new vacancies 
            vac_prob= max(0, vr_t - (curr_vacs/(occ.target_demand + 1)))
            #vac_prob = d_v + (1-d_v)*((gamma_v * max(0, occ.target_demand*(occ_shock) - occ.current_demand)) / (emp + 1))
            vacs_create = int(np.random.binomial(occ.target_demand, np.clip(vac_prob, 0, 1)))

            #vacs_create = emp*int(vac_prob) + int(np.random.binomial(emp, vac_prob%1))
            for v in range(vacs_create):
                                # Draw a random wage
                w = np.random.lognormal(occ.wage_mu, occ.wage_sigma)

                # Compute 25th and 75th percentiles
                z25 = norm.ppf(0.25)
                z75 = norm.ppf(0.75)
                w25 = np.exp(occ.wage_mu + occ.wage_sigma * z25)
                w75 = np.exp(occ.wage_mu + occ.wage_sigma * z75)

                # Clip to IQR (25th–75th)
                w_clipped = w #np.clip(w, w25, w75)

                # Create vacancy object with clipped wage
                vacs_temp.append(
                    vac(
                        occ.occupation_id,
                        [],
                        w_clipped,
                        False,
                        0
                    )
                )
                #vacs_temp.append(vac(occ.occupation_id, [], np.clip(np.random.lognormal(occ.wage_mu, occ.wage_sigma), 15080, 250000), False, 0)) #np.random.normal(occ.wage, 0.05*occ.wage)
                #vacs_temp.append(vac(occ.occupation_id, [], np.random.normal(occ.wage, 0.05*occ.wage), False, 0)) #

            if simple_res:
                empl += len(occ.list_of_employed) 
                unemp += len(occ.list_of_unemployed)
                n_ltue += sum(wrkr.longterm_unemp for wrkr in occ.list_of_unemployed)
                t_demand += occ.target_demand*occ_shock
                seps += occ.separated + retired
                #e_seekers += emp_seekers
                #u_seekers += unemp_seekers

            
            else:
                empl = len(occ.list_of_employed) 
                unemp = len(occ.list_of_unemployed)
                n_ltue = sum(wrkr.longterm_unemp for wrkr in occ.list_of_unemployed)
                curr_demand = occ.current_demand
                t_demand = occ.target_demand*occ_shock
                vacs_occ = len([v for v in vacs_temp if v.occupation_id == occ.occupation_id])
                vacs_wage = np.mean([v.wage for v in vacs_temp if v.occupation_id == occ.occupation_id]) if vacs_occ > 0 else np.nan
                wage_occ = np.mean([wrkr.wage for wrkr in occ.list_of_employed]) if emp > 0 else np.nan
                wages_occ = sum(wrkr.wage for wrkr in occ.list_of_employed)
                seps = occ.separated + retired
                retirees = retired
                hires = occ.hired
                entry_level_entrants = entry_alloc.get(occ.occupation_id, 0) if occ.entry_level else 0

                ### UPDATE INDICATOR RECORD
                record.append([t+1, occ.occupation_id, empl, unemp, empl + unemp, vacs_occ, n_ltue, curr_demand, t_demand, emp_seekers, unemp_seekers, wages_occ, u_rel_wage, e_rel_wage, ue, ee, seps, hires, vacs_wage, wage_occ, retirees, entry_level_entrants])

        if simple_res:
            ### UPDATE INDICATOR RECORD
            record = np.append(record, 
                        np.array([[t+1, empl, unemp, empl + unemp, len(vacs_temp), n_ltue, t_demand, seps, emp_seekers, unemp_seekers]]), axis=0)

        else:
            record_temp_df = pd.DataFrame(record, columns=['Time Step', 'Occupation', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Current_Demand', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers', 'Total_Wages', 'U_Rel_Wage', 'E_Rel_Wage', 'UE_Transitions', 'EE_Transitions', "Separations", "Hires", "Mean Vacancy Offer", "Mean Occupational Wage", "Retirees", "Entry_Level_Hires"])
            record_df = record_temp_df[record_temp_df['Time Step'] >= delay]
            grouped = record_df.groupby('Time Step').sum().reset_index()

            grouped['UER'] = grouped['Unemployment'] / grouped['Workers']
            grouped['U_REL_WAGE_MEAN'] = grouped['U_Rel_Wage'] / grouped['UE_Transitions']
            grouped['E_REL_WAGE_MEAN'] = grouped['E_Rel_Wage'] / grouped['EE_Transitions']
            grouped['UE_Trans_Rate'] = grouped['UE_Transitions'] / grouped['Workers']
            grouped['EE_Trans_Rate'] = grouped['EE_Transitions'] / grouped['Workers']
            grouped['VACRATE'] = grouped['Vacancies'] / (grouped['Vacancies'] + grouped['Employment'])
            grouped['Hires Rate'] = grouped['Hires'] / grouped['Employment']
            grouped['Separations Rate'] = (grouped['Separations'] + grouped['EE_Transitions']) / grouped['Employment']
            grouped['E-U Rate'] = grouped['Separations'] / grouped['Employment']

            data = {'UER': grouped['UER'], 'VACRATE': grouped['VACRATE'], }#, 'SEPSRATE': grouped['Separations Rate']}
            avg_wage_offer_diff_df = pd.DataFrame(avg_wage_offer_diff_records)

            if app_load_records:
                app_load_df = pd.concat(app_load_records, ignore_index=True)
            else:
                app_load_df = pd.DataFrame(columns=['Time Step','Occupation','OpenVacs','TotalApplicants','MeanAppsPerVac','MedianAppsPerVac'])

            ue_spell_origin_df = pd.DataFrame(ue_spell_records, columns=['Time Step', 'OriginOccupation', 'DestinationOccupation', 'UEDuration'])

    if simple_res:
        if theta is not None:
            data = {'UER': np.array(record[delay:,2]/record[delay:,3]),
                'VACRATE': np.array(record[delay:,4]/(record[delay:,4] + record[delay:,1])),
                'Seeker Composition': np.array(record[delay:,8]/(record[delay:,8] + record[delay:,9]))}
        else:
            data = {'UER': np.array(record[delay:,2]/record[delay:,3]),
                'VACRATE': np.array(record[delay:,4]/(record[delay:,4] + record[delay:,1]))}
           # 'SEPSRATE': np.array(record[delay:,7]/(record[delay:,1]+1))}
        return data
    else:

        seekers_rec = pd.DataFrame(seekers_rec, columns=['Time Step', 'Unemployed Seekers', 'Applications Sent (Unemployed)', 'Employed Seekers', 'Applications Sent (Employed)'])
        seekers_rec = seekers_rec[seekers_rec['Time Step'] >= delay]
        return record_df, grouped, net, data, seekers_rec, avg_wage_offer_diff_df, app_load_df, ue_spell_origin_df, vacs_temp

#########################################
# # Wrapper for pyabc ########
# #########################################
# def pyabc_run_single(parameter):     
#     res = run_single_local(**parameter)
#     return res 