# Import packages
import os
import numpy as np
import pandas as pd
import random as random
from copy import deepcopy
import math as math
import cProfile, pstats, io
rng = np.random.default_rng()

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/"

# Testing import works
def test_fun():
    print('hello - import works!')
    
    
    
## Defining functions
# Ranking utility/decision-making function
def util(w_current, w_offered, skill_sim):
    return 1/(1+(math.exp(-((w_offered - w_current)/10000))))

# Simple quadratic for now in which a worker increases search effort for a period of 6 time steps (ie. months) 
# unemployed after which a worker begins to become discouraged. 
# This follows definition from the US BLS and Pew Research Centre
def search_effort(t_unemp):
    return round(20/((t_unemp-6)**2 + 1)) + 1

## Defining classes
# Potentially redundant use of IDs in the below classes...to check
class worker:
    def __init__(wrkr, occupation_id, 
                 # employed, 
                 longterm_unemp, 
                 # time_employed,
                 time_unemployed, wage, hired, female, risk_av_score):
        # State-specific attributes:
        # Occupation id
        wrkr.occupation_id = occupation_id
        # Binary variable for whether long-term unemployed
        wrkr.longterm_unemp = longterm_unemp
        # Number of time steps unemployed (perhaps redundant with above)
        # Used as criteria for impatience
        wrkr.time_unemployed = time_unemployed
        # Worker wage
        wrkr.wage = wage
        # Whether the worker has been hired in this time step - reset to zero at the end of every time step
        # Used as protective attribute in hiring process (ie. cannot be hired twice in same time step)
        wrkr.hired = hired
        # Binary for whether the worker is female
        wrkr.female = female
        # Risk aversion: Stefi suggested to use number of 
        # occupations previously held as proxy ie. len(emp_history)
        # Currently takes a value 0-9 indicating at which index of utility ranked vacancies to start sampling/slicing
        wrkr.risk_aversion = risk_av_score
    
    def search_and_apply(wrkr, net, vac_list, beh):
        # A sample of relevant vacancies are found that are in neighboring occupations
        # Will need to add a qualifier in case sample is greater than available relevant vacancies
        # ^^ have added qualifier...bad form to reassign list?
        # Select different random sample of "relevant" vacancies found by each worker
        found_vacs = random.sample(vac_list, min(len(vac_list), 30))
        if beh:
            # Sort found relevant vacancies by utility-function defined above and apply to amount dictated by impatience
            for v in sorted(found_vacs, key = lambda v: util(wrkr.wage, v.wage,net[wrkr.occupation_id].list_of_neigh_weights[v.occupation_id]), reverse = True)[slice(wrkr.risk_aversion, wrkr.risk_aversion + search_effort(wrkr.time_unemployed))]:
                # Introduce randomness here...?
                v.applicants.append(wrkr)
        else:
            vs = random.sample(found_vacs, min(len(found_vacs), 10))
            for r in vs:
                r.applicants.append(wrkr)
            
class occupation:
    def __init__(occ, occupation_id, list_of_employed, list_of_unemployed, 
                 list_of_neigh_bool, list_of_neigh_weights, current_demand, 
                 target_demand, wage):
        occ.occupation_id = occupation_id
        occ.list_of_employed = list_of_employed
        occ.list_of_unemployed = list_of_unemployed
        occ.list_of_neigh_bool = list_of_neigh_bool
        occ.list_of_neigh_weights = list_of_neigh_weights
        occ.current_demand = current_demand
        occ.target_demand = target_demand
        occ.wage = wage
    
    def separate_workers(occ, delta_u, gam):
        if(len(occ.list_of_employed) != 0):
            sep_prob = delta_u + gam * max(0, occ.current_demand - occ.target_demand)/(len(occ.list_of_employed) + 1)
            # Included as a warning - particularly relevant in parameter inference when sampling from prior
            if sep_prob > 1:
                sep_prob = 0.9
                print("sep_prob above 1 - reset")
            elif sep_prob < 0:
                sep_prob = 0.1
                print("sep_prob below 0 - reset")
            w = np.random.binomial(len(occ.list_of_employed), sep_prob)
            separated_workers = random.sample(occ.list_of_employed, w)
            occ.list_of_unemployed = occ.list_of_unemployed + separated_workers
            occ.list_of_employed = list(set(occ.list_of_employed) - set(separated_workers))
    
    def update_workers(occ):
        # Possible for loop to replace
        for w in occ.list_of_unemployed:
            w.time_unemployed += 1
            # Chosen 12 months - can be modified
            w.longterm_unemp = True if w.time_unemployed >= 12 else False
            # Possible for loop to replace
        for e in occ.list_of_employed:
            e.hired = False
        
class vac:
    def __init__(v, occupation_id, applicants, wage, filled):
        v.occupation_id = occupation_id
        v.applicants = applicants
        v.wage = wage
        v.filled = filled
        
    # Function to hire a worker from pool of vacancies    
    def hire(v, net):
        a = random.choice(v.applicants)
        assert(not(a.hired))
        net[v.occupation_id].list_of_employed.append(net[a.occupation_id].list_of_unemployed.pop(net[a.occupation_id].list_of_unemployed.index(a)))
        a.occupation_id = v.occupation_id
        a.time_unemployed = 0
        a.wage = v.wage
        #a.emp_history.append(v.occupation_id)
        a.hired = True
        v.applicants.clear()

        
def bus_cycle_demand(d_0, time, amp, period):
    """Business cycle demand equation
    Args:
        d_0: current_demand (emp + vacancies)
        amplitude: amplitude of business cycle # not quite sure what this should be....need to look at the literature
        period: period for full business cycle # I believe this should be between 2-10 years....?
    Returns
        target demand influenced by business cycle
    """
    d_target =  d_0 * (1 - amp * np.sin((2*np.pi / period) * time))
    return d_target




### Function and condition to initialise network
def initialise(n_occ, employment, unemployment, vacancies, demand_target, A, wages, gend_share):
    """ Makes a list of occupations with initial conditions
       Args:
           n_occ: number of occupations initialised (464)
           employment: vector with employment of each occupation
           unemployment: vector with unemployment of each occupation
           vacancies: vector with vacancies of each occupation
           demand_target: vector with (initial) target_demand for each occupation (never updated)
           A: adjacency matrix of network (not including auto-transition probability)
           wages: vector of wages of each occupation

       Returns:
            occupations: list of occupations with above attributes
            vacancies: list of vacancies with occupation id, wage, and list of applicants
       """
    occs = []
    vac_list = []
    ids = 0
    for i in range(0, n_occ):
        # appending relevant number of vacancies to economy-wide vacancy list
        for v in range(round(vacancies[i,0])):
            vac_list.append(vac(i, [], wages[i,0], False))
            
        occ = occupation(i, [], [], list(A[i] > 0), list(A[i]),
                         (employment[i,0] + vacancies[i,0]), 
                         demand_target[i,0], wages[i,0])
        # creating the workers of occupation i and attaching to occupation
        ## adding employed workers
        g_share = gend_share[i,0]
        for e in range(round(employment[i,0])):
            # Assume they have all at least 1 t.s. of employment
            if np.random.rand() <= g_share:
                occ.list_of_employed.append(worker(occ.occupation_id, False, 1, wages[i,0], False, True,
                                               abs(int(np.random.normal(7,2)))))
            else:
                occ.list_of_employed.append(worker(occ.occupation_id, False, 1, wages[i,0], False, False,
                                               abs(int(np.random.normal(3,2)))))
            ## adding unemployed workers
        for u in range(round(unemployment[i,0])):
            if np.random.rand() <= g_share:
                # Assigns time unemployed from absolute value of normal distribution....
                occ.list_of_unemployed.append(worker(occ.occupation_id, False, max(1,(int(np.random.normal(2,2)))), 
                                                     wages[i,0], False, True, abs(int(np.random.normal(7,2)))))
            else:
                # Assigns time unemployed from absolute value of normal distribution....
                occ.list_of_unemployed.append(worker(occ.occupation_id, False, max(1,(int(np.random.normal(2,2)))), 
                                                     wages[i,0], False, False, abs(int(np.random.normal(3,2)))))
                
                
                
        occs.append(occ)
        ids += 1
    return occs, vac_list
    
    
    
    ####################
# Model Run ########
####################
def run_single(mod_data, 
               net_temp, 
               vacs, 
               behav_spec, 
               time_steps, 
               d_u, 
               d_v,
               gamma,
               bus_cycle_len,
               delay):
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
    record = np.concatenate((np.zeros((464, 1)), 
                                 mod_data['employment'].to_numpy(), 
                                 mod_data['unemployment'].to_numpy(), 
                                 mod_data['employment'].to_numpy() + mod_data['unemployment'].to_numpy(),
                                 mod_data['vacancies'].to_numpy(), 
                                 np.zeros((464, 1)),
                                 mod_data['demand_target'].to_numpy()), axis = 1)
    
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
            occ.current_demand = bus_cycle_demand(len([v_open for v_open in vacs_temp if v_open.occupation_id == occ.occupation_id]) + emp, t, 0.02, bus_cycle_len)
            vac_prob = d_v + ((1 - d_v) * (gamma * max(0, occ.target_demand - occ.current_demand))) / (emp + 1)
            # Included as a warning - particularly relevant in parameter inference when sampling from prior
            if vac_prob > 1:
                vac_prob = 0.9
                print("vac_prob above 1 - reset")
            elif vac_prob < 0:
                vac_prob = 0.1
                print("vac_prob below 0 - reset")
                
            for v in range(int(np.random.binomial(emp, vac_prob))):
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
    clean_record = pd.DataFrame(record)
    clean_record.columns =['Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Target_Demand']
    clean_record = clean_record[clean_record["Time Step"] > delay].groupby(['Time Step']).sum().reset_index()
    clean_record['UER'] = clean_record['Unemployment']/clean_record['Workers']
    clean_record['VACRATE'] = clean_record['Vacancies']/clean_record['Target_Demand']
    data = clean_record[['Time Step', 'UER', 'VACRATE']]
    #print(data)
    #uer = uer[(uer['Time Step'] >= 10)]

    #ltuer = (clean_record['LT Unemployed Persons']/clean_record['Workers']).mean(axis = 0)
    #vac_rate = (clean_record['Vacancies']/clean_record['Target_Demand']).mean(axis = 0)
    return {"data": data}
            #'ltuer': ltuer,
            #'vac_rate': vac_rate}



#########################################
# Model Run with Simulation Spec ########
#########################################
def run_sim(mod_data, net_temp, vacs, behav_spec, time_steps, runs, d_u, d_v, gamma, bus_cycle_len):
    """ Runs the model through designated time_steps "runs" times
    Argsuments:
       behav_spec: whether or not to run the behavioural model
       data: data required of initialise function  
       time_steps: Number of time steps for single model run
       runs: Number of simulation runs ie. how many times to run the model
       d_u: parameter input to separation probability
       d_v: parameter input to vacancy opening probability

    Returns:
       dataframe of model run results
    """
    # Records variables of interest for plotting
    #record_all = np.empty(shape=(0,8))
    record_single = np.concatenate((np.zeros((464, 1)), 
                                 np.zeros((464, 1)), 
                                 mod_data['employment'].to_numpy(), 
                                 mod_data['unemployment'].to_numpy(), 
                                 mod_data['employment'].to_numpy() + mod_data['unemployment'].to_numpy(),
                                 mod_data['vacancies'].to_numpy(), 
                                 np.zeros((464, 1)),
                                 mod_data['demand_target'].to_numpy()), axis = 1)
    for run in range(runs):
        print("Run: ", run)
        # Initialise deepcopy occupational mobility network
        record = deepcopy(record_single)
        net = deepcopy(net_temp)
        for t in range(time_steps):
            #print("Time: ", t)
#             # Shock incorporation....to be changed to dynamic from past applications
#             if t == 400 and shock:
#                 print("initiatied shock!")
#                 net_temp[0].target_demand += 25
#                 net_temp[1].target_demand += 50
#                 net_temp[2].target_demand += 50
#                 net_temp[3].target_demand += 50
#                 net_temp[4].target_demand = 100

            # Ensure number of workers in economy has not changed
            #tic = time.process_time()
            for occ in net:
                ### APPLICATIONS
                # Questions to verify:
                # - CANNOT be fired and apply in same time step ie. time_unemployed > 0
                # - CAN be rejected and apply in the same time step - no protected attribute
                # isolate list of vacancies in economy that are relevant to the occupation
                # - avoids selecting in each search_and_apply application
                r_vacs = [vac for vac in vacs if occ.list_of_neigh_bool[vac.occupation_id]]                
                for u in occ.list_of_unemployed:
                    u.search_and_apply(net, r_vacs, behav_spec)
                
                ### SEPARATIONS
                occ.separate_workers(d_u, gamma)
        
            ### HIRING
            # Ordering of hiring randomised to ensure list order does not matter in filling vacancies...
            # Possibly still introduces some bias...this seems to be where the "multiple offer" challenge Maria mentioned comes from
            # ....might be better to do this using an unordered set?
            for v_open in sorted(vacs,key=lambda _: random.random()):
                # Removes any applicants that have already been hired in another vacancy
                v_open.applicants[:] = [app for app in v_open.applicants if not(app.hired)]
                if len(v_open.applicants) > 0:
                    v_open.hire(net)
                    v_open.filled = True
                    #vacs.remove(v_open)
                    assert(len(v_open.applicants) == 0)
                else:
                    pass
                
            vacs = [v for v in vacs if not(v.filled)] 

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
                occ.current_demand = bus_cycle_demand(len([v_open for v_open in vacs if v_open.occupation_id == occ.occupation_id]) + emp, t, 0.02, bus_cycle_len)
                vac_prob = d_v + ((1 - d_v) * (gamma * max(0, occ.target_demand - occ.current_demand))) / (emp + 1)
                for v in range(int(np.random.binomial(emp, vac_prob))):
                    vacs.append(vac(occ.occupation_id, [], occ.wage, False))
                    
                empl += len(occ.list_of_employed) 
                unemp += len(occ.list_of_unemployed)
                n_ltue += sum(wrkr.longterm_unemp for wrkr in occ.list_of_unemployed)
                t_demand += occ.target_demand
            
                ### UPDATE INDICATOR RECORD
                record = np.append(record, 
                                   np.array([[run, t+1, empl, unemp, empl + unemp, len(vacs), n_ltue, t_demand]]), 
                                   axis = 0)

        print("Done after ", t + 1, " time steps.")
        if run == 0:
            record_all = record
            net_list = [net]
        else:
            record_all = np.dstack((record_all, record))
            net_list.append(net)
        print(record_all.shape)
        
    print("Done after ", run + 1, " runs.")
    # Cannot seem to return modified variable....same net_temp as went in goes out
    return record_all, net, net_list


    