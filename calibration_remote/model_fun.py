######## Central Run Function #############

import numpy as np
import pandas as pd
from abm_funs import *
import random as random
from copy import deepcopy 
import math as math

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
                    mod_data, 
                    net_temp, 
                    vacs, 
                    time_steps, # set equal to length of gdp_data
                    delay,
                    gdp_data,
                    bus_confidence_dat,
                    app_effort_dat,
                    vac_data,
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
    #print(behav_spec)
    if simple_res:
        record = np.empty((0, 7))
    else:
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
        #u_searchers = 0
        
        for occ in net:
            # Exit and entry
            # Remove the top 2% of earners in an occupation's employed list
            occ.entry_and_exit(0.02)

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
                u.search_and_apply(net, r_vacs, disc, ue_bc, 0.1, app_effort_dat)
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
            #u_searchers += len(occ.list_of_unemployed)

            ### SEPARATIONS
            try:
                occ.separate_workers(d_u, gamma_u, curr_bus_cy)
            except Exception as e:
                return np.inf
                    
        seekers_rec.append([t+1, unemp_seekers, u_apps])


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

        vacs_temp = [v for v in vacs_temp if not(v.filled)] 

        # # Reset counters for record in time t
        if simple_res:
            empl = 0 
            unemp = 0
            n_ltue = 0
            # curr_demand = 0
            t_demand = 0

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
                vacs_temp.append(vac(occ.occupation_id, [], np.random.normal(occ.wage, 0.05*occ.wage), False, 0))
            if simple_res:
                empl += len(occ.list_of_employed) 
                unemp += len(occ.list_of_unemployed)
                n_ltue += sum(wrkr.longterm_unemp for wrkr in occ.list_of_unemployed)
                t_demand += occ.target_demand*bus_conf
            
            else:
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


        if simple_res:
            ### UPDATE INDICATOR RECORD
            record = np.append(record, 
                        np.array([[t+1, empl, unemp, empl + unemp, len(vacs_temp), n_ltue, t_demand]]), axis=0)

        else:
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

    if simple_res:
        data = {'UER': np.array(record[delay:,2]/record[delay:,3]), 
            'VACRATE': np.array(record[delay:,4]/(record[delay:,4] + record[delay:,1]))}
        return data
    else:

        seekers_rec = pd.DataFrame(seekers_rec, columns=['Time Step', 'Unemployed Seekers', 'Applications Sent'])
        seekers_rec = seekers_rec[seekers_rec['Time Step'] > delay]
        return record_df, grouped, net, data, seekers_rec

#########################################
# # Wrapper for pyabc ########
# #########################################
# def pyabc_run_single(parameter):     
#     res = run_single_local(**parameter)
#     return res 