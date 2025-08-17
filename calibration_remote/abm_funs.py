# Import packages
import numpy as np
import pandas as pd
import random as random
from copy import deepcopy 
import math as math
from collections import defaultdict
import cProfile, pstats, io
rng = np.random.default_rng()

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/data/"

#########################################

# Testing import works
def test_fun():
    print('NEW Function import successful')
    
## Defining functions
# Ranking utility/decision-making function
def util(w_current, w_offered, skill_sim):
    return (w_offered - w_current)*skill_sim
    #return 1/(1+(math.exp(-((w_offered - w_current)/10000))))

#### Dynamic Search Effort ######
# Load the data

app_effort = pd.read_csv(path + "app_probs_long.csv")

# Helper function: Sample an integer from the bin's internal distribution
def sample_within_bin(bin_label, expectation=False):
    if bin_label == "0":
        return 1
    elif bin_label == "1 to 10":
        return 5.5 if expectation else random.randint(1, 10)
    elif bin_label == "11 to 20":
        return 15.5 if expectation else random.randint(11, 20)
    elif bin_label == "21 to 80":
        return 50.5 if expectation else random.randint(21, 80)
    elif bin_label == "81 or more":
        return 90.5 if expectation else random.randint(81, 100)
    else:
        raise ValueError(f"Unknown bin label: {bin_label}")

# Main function: given duration and bin probabilities, return application count
def applications_sent(duration_months,
                             duration_to_prob_dict,
                             expectation=False):
    """
    If expectation=True return the expected value (float);
    otherwise return a random draw (int).
    """
    if duration_months not in duration_to_prob_dict:
        #raise ValueError(f"No probability distribution for duration {duration_months}")
        return 1

    # dict: bin_label -> probability
    bin_probs = duration_to_prob_dict[duration_months]

    if expectation:                       # new deterministic branch
        exp_val = sum(
            p * sample_within_bin(b, expectation=True)
            for b, p in bin_probs.items()
        )
        return exp_val                  # float (can be fractional)

    # ------- stochastic branch (unchanged) -------
    bin_labels     = list(bin_probs.keys())
    probabilities  = list(bin_probs.values())
    chosen_bin     = random.choices(bin_labels, weights=probabilities, k=1)[0]
    return max(1, sample_within_bin(chosen_bin, expectation=False))

# Convert duration to int (you can round or floor as needed)
app_effort["PRUNEDUR_MO"] = app_effort["PRUNEDUR_MO"].round().astype(int)

# Create mapping: duration -> {bin_label: probability}
duration_to_prob_dict = defaultdict(dict)

for _, row in app_effort.iterrows():
    duration = row["PRUNEDUR_MO"]
    category = row["Category"]
    prob = row["Probability"]
    duration_to_prob_dict[duration][category] = prob

# Convert defaultdict to plain dict (optional)
duration_to_prob_dict = dict(duration_to_prob_dict)

apps = applications_sent(duration_months=1, duration_to_prob_dict=duration_to_prob_dict, expectation = True)

if __name__ == "__main__":
    print(f"Applications sent: {apps}")

def search_effort_alpha(t_unemp, bus_cy, disc, alpha):
    apps = 10
    if t_unemp > 1 and disc:
        apps = round(apps * (1 + alpha*(t_unemp-1)))
        if t_unemp >= 7 and disc:
            apps = apps * (1 - alpha*(t_unemp-7))
    return round(max(1, apps))

# Alternative search effort function that is dictated by the time series of a business cycle
def search_effort_ts(t_unemp, se):
    apps = max(0, round(10 - 100*(1-se)))
    return apps

#### Reservation Wage Adjustment Rate ######
# Load the data
res_wage_dat = pd.read_csv(path + "res_wage_dists.csv")

# Main function: given duration and bin probabilities, return application count
def reservation_wage(duration_months,
                             res_wage_data,
                             expectation=False,
                             # Either lm (no sample balancing), glm, eb (entropy balancing)
                             balancing_method='lm'):
    """
    If expectation=True return the expected value (float);
    otherwise return a random draw (int).
    """
    if duration_months not in res_wage_data['dur_unemp'].values:
        #raise ValueError(f"No probability distribution for duration {duration_months}")
        res_wage = np.random.normal(loc=0.75, scale=np.max(res_wage_data[f'{balancing_method}_se']), size=1)[0]
        if expectation:
            res_wage = 0.75
    else:
        mean_val = res_wage_data[f'{balancing_method}_preds_fit'][res_wage_data['dur_unemp'] == duration_months]
        se_val = res_wage_data[f'{balancing_method}_se'][res_wage_data['dur_unemp'] == duration_months]

        if expectation:
            res_wage = mean_val.iloc[0]  # Extract scalar from Series
        else:
            res_wage = np.random.normal(loc=mean_val.iloc[0], scale=se_val.iloc[0], size=1)[0]
    
    return(res_wage)

## Defining classes
# Potentially redundant use of IDs in the below classes...to check
class worker:
    def __init__(wrkr, occupation_id, 
                 # employed, 
                 longterm_unemp, 
                 # time_employed,
                 time_unemployed, wage, hired, female, age, risk_av_score, ee_rel_wage, ue_rel_wage, applicants_sent, d_wage_offer):
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
        wrkr.age = age
        # Risk aversion: Stefi suggested to use number of 
        # occupations previously held as proxy ie. len(emp_history)
        # Currently takes a value 0-9 indicating at which index of utility ranked vacancies to start sampling/slicing
        wrkr.risk_aversion = risk_av_score
        wrkr.ee_rel_wage = ee_rel_wage
        wrkr.ue_rel_wage = ue_rel_wage
        wrkr.apps_sent = applicants_sent
        wrkr.d_wage_offer = d_wage_offer


    def search_and_apply(wrkr, net, vacancies_by_occ, disc, bus_cy, alpha, app_effort):
        MAX_VACS = 30
        wrkr_occ = wrkr.occupation_id
        neigh_probs = net[wrkr_occ].list_of_neigh_weights  # Already normalized
        occ_ids = list(range(len(neigh_probs)))

        # Build a flat list of vacancies and their weights
        all_vacs = []
        weights = []

        for occ_id, occ_prob in zip(occ_ids, neigh_probs):
            vacs = vacancies_by_occ.get(occ_id, [])
            for v in vacs:
                all_vacs.append(v)
                weights.append(occ_prob)

        if not all_vacs:
            wrkr.apps_sent = 0
            return

        # Sample up to MAX_VACS vacancies directly
        found_vacs = random.choices(all_vacs, weights=weights, k=min(MAX_VACS, len(all_vacs)))
        mean_wage = np.mean([v.wage for v in found_vacs]) if found_vacs else 0

        vsent = 0

        if disc:
            # Step 3: Apply reservation wage filter
            if wrkr.time_unemployed > 3:
                res_wage = wrkr.wage * (1 - 0.1 * (wrkr.time_unemployed - 3))
            else:
                res_wage = wrkr.wage

            found_vacs = [v for v in found_vacs if v.wage >= res_wage]

            # Step 4: Rank by utility
            sorted_vacs = sorted(
                found_vacs,
                key=lambda v: util(
                    wrkr.wage, v.wage,
                    net[wrkr_occ].list_of_neigh_weights[v.occupation_id]
                ),
                reverse=True
            )

            n_apps = applications_sent(wrkr.time_unemployed, app_effort, expectation=False)
            chosen_vacs = sorted_vacs[wrkr.risk_aversion: wrkr.risk_aversion + n_apps]

            for v in chosen_vacs:
                v.applicants.append(wrkr)
                vsent += 1
        else:
            for v in random.sample(found_vacs, min(len(found_vacs), 10)):
                v.applicants.append(wrkr)
                vsent += 1

        wrkr.d_wage_offer = mean_wage-res_wage if disc else np.nan
        wrkr.apps_sent = vsent

    # def search_and_apply(wrkr, net, vacancies_by_occ, disc, bus_cy, alpha, app_effort):
    #     MAX_VACS = 30
    #     wrkr_occ = wrkr.occupation_id
    #     neigh_probs = net[wrkr_occ].list_of_neigh_weights
    #     occ_ids = list(range(len(neigh_probs)))

    #     found_vacs = []
    #     seen_occ_ids = set()

    #     while len(found_vacs) < MAX_VACS:
    #         sampled_occ = random.choices(occ_ids, weights=neigh_probs, k=1)[0]
    #         if sampled_occ in seen_occ_ids:
    #             continue

    #         occ_vacs = vacancies_by_occ.get(sampled_occ, [])
    #         random.shuffle(occ_vacs)

    #         for v in occ_vacs:
    #             found_vacs.append(v)
    #             if len(found_vacs) >= MAX_VACS:
    #                 break

    #         seen_occ_ids.add(sampled_occ)

    #     found_vacs = found_vacs[:MAX_VACS]
    #     vsent = 0

    #     if disc:
    #         if wrkr.time_unemployed > 3:
    #             res_wage = wrkr.wage * (1 - 0.1 * (wrkr.time_unemployed - 3))
    #         else:
    #             res_wage = wrkr.wage

    #         found_vacs = [v for v in found_vacs if v.wage >= res_wage]

    #         sorted_vacs = sorted(
    #             found_vacs,
    #             key=lambda v: util(
    #                 wrkr.wage, v.wage,
    #                 net[wrkr_occ].list_of_neigh_weights[v.occupation_id]
    #             ),
    #             reverse=True
    #         )

    #         n_apps = applications_sent(wrkr.time_unemployed, app_effort, expectation=False)
    #         chosen_vacs = sorted_vacs[wrkr.risk_aversion: wrkr.risk_aversion + n_apps]

    #         for v in chosen_vacs:
    #             v.applicants.append(wrkr)
    #             vsent += 1
    #     else:
    #         for v in random.sample(found_vacs, min(len(found_vacs), 10)):
    #             v.applicants.append(wrkr)
    #             vsent += 1

    #     wrkr.apps_sent = vsent
    
    # def search_and_apply(wrkr, net, vac_list, disc, bus_cy, alpha, app_effort):
    #     # A sample of relevant vacancies are found that are in neighboring occupations
    #     # Select different random sample of "relevant" vacancies found by each worker
    #     found_vacs = random.sample(vac_list, min(len(vac_list), 30))
    #     vsent = 0
    #     if disc:
    #         # res_wage = wrkr.wage * reservation_wage(wrkr.time_unemployed, res_wage_dat, expectation = False, balancing_method = 'lm')
    #         if wrkr.time_unemployed > 3:
    #             res_wage = wrkr.wage * (1-(0.1*(wrkr.time_unemployed-3)))
    #         else:
    #             res_wage = wrkr.wage

    #         found_vacs = [v for v in found_vacs if v.wage >= res_wage]
    #         # Sort found relevant vacancies by utility-function defined above and apply to amount dictated by impatience
    #         for v in sorted(found_vacs, key = lambda v: util(wrkr.wage, v.wage, net[wrkr.occupation_id].list_of_neigh_weights[v.occupation_id]), 
    #                         reverse = True)[slice(wrkr.risk_aversion, wrkr.risk_aversion + applications_sent(wrkr.time_unemployed, app_effort, expectation = False))]:
    #             # Introduce randomness here...?
    #             vsent += 1
    #             v.applicants.append(wrkr)
    #     else:
    #         vs = random.sample(found_vacs, min(len(found_vacs), 10))
    #         for r in vs:
    #             vsent += 1
    #             r.applicants.append(wrkr)
    #     wrkr.apps_sent = vsent
    #     #print(f'Applications sent: {vsent}')

    def emp_search_and_apply(wrkr, net, vac_list, disc):
        # A sample of relevant vacancies are found that are in neighboring occupations
        # Will need to add a qualifier in case sample is greater than available relevant vacancies
        # ^^ have added qualifier...bad form to reassign list?
        # Select different random sample of "relevant" vacancies found by each worker
        # found_vacs = random.sample(vac_list, min(len(vac_list), 30))
        MAX_VACS = 30
        wrkr_occ = wrkr.occupation_id
        neigh_probs = net[wrkr_occ].list_of_neigh_weights  # Already normalized
        occ_ids = list(range(len(neigh_probs)))

        # Step 1: Build a flat list of vacancies and their weights
        all_vacs = []
        weights = []

        for occ_id, occ_prob in zip(occ_ids, neigh_probs):
            vacs = vac_list.get(occ_id, [])
            for v in vacs:
                all_vacs.append(v)
                weights.append(occ_prob)

        if not all_vacs:
            wrkr.apps_sent = 0
            return

        # Step 2: Sample up to MAX_VACS vacancies directly
        found_vacs = random.choices(all_vacs, weights=weights, k=min(MAX_VACS, len(all_vacs)))
        mean_wage = np.mean([v.wage for v in found_vacs]) if found_vacs else 0

        if disc:
            found_vacs = [v for v in found_vacs if v.wage >= wrkr.wage*1.05]
        # Filter found_vacs to keep only elements where util(el) > 0
        # We assume that employed workers will only apply to vacancies for which there is a wage gain. 
        filtered_vacs = [el for el in found_vacs if util(wrkr.wage, el.wage, net[wrkr.occupation_id].list_of_neigh_weights[el.occupation_id]) > 0]
        vs = random.sample(filtered_vacs, min(len(filtered_vacs), 5))
        for r in vs:
            r.applicants.append(wrkr)
        wrkr.apps_sent = 0
        wrkr.d_wage_offer = np.nan
            
class occupation:
    def __init__(occ, occupation_id, list_of_employed, list_of_unemployed, 
                 list_of_neigh_bool, list_of_neigh_weights, current_demand, 
                 target_demand, wage, separated, hired, entry_level_bool, experience_age):
        occ.occupation_id = occupation_id
        occ.list_of_employed = list_of_employed
        occ.list_of_unemployed = list_of_unemployed
        occ.list_of_neigh_bool = list_of_neigh_bool
        occ.list_of_neigh_weights = list_of_neigh_weights
        occ.current_demand = current_demand
        occ.target_demand = target_demand
        occ.wage = wage
        occ.separated = separated
        occ.hired = hired
        occ.entry_level = entry_level_bool
        occ.experience_age = experience_age

    def separate_workers(occ, delta_u, gam, bus_cy):
        if(len(occ.list_of_employed) != 0):
            sep_prob = delta_u + (1-delta_u)*((gam * max(0, len(occ.list_of_employed) - (occ.target_demand*bus_cy)))/(len(occ.list_of_employed) + 1))
            w = np.random.binomial(len(occ.list_of_employed), sep_prob)
            occ.separated = w
            separated_workers = random.sample(occ.list_of_employed, w)
            #print(f'Separated workers: {len(separated_workers)}')
            occ.list_of_unemployed = occ.list_of_unemployed + separated_workers
            occ.list_of_employed = list(set(occ.list_of_employed) - set(separated_workers))

    
    def update_workers(occ):
        # Possible for loop to replace
        for w in occ.list_of_unemployed:
            w.time_unemployed += 1
            # Chosen 6 months - can be modified - 27 weeks (~ 6 mos) according to BLS.
            # We call update_workers before we tally LTUE people so time spent unemployed will be 1 greater than actual time spent "applying"
            # Given unemployed individuals are given one time period of unemployment from the moment they are fired (ie. they have not navigated the market as unemployed yet)
            # we set the threshold for LTUER at 7 (not 6)
            w.longterm_unemp = True if w.time_unemployed >= 7 else False
            w.ue_rel_wage = None
            w.ee_rel_wage = None
            w.hired = False
            w.apps_sent = 0
            w.age += 0.083
        for e in occ.list_of_employed:
            e.hired = False
            e.time_unemployed = 0
            e.ue_rel_wage = None
            e.ee_rel_wage = None
            e.apps_sent = 0
            e.age += 0.083

    def retire_workers(occ):
        """ Function to retire workers over the age of 65 """
        occ.list_of_unemployed = [w for w in occ.list_of_unemployed if w.age <= 65]
        occ.list_of_employed = [e for e in occ.list_of_employed if e.age <= 65]
    
    def entry_and_exit_fixed(occ, rate):
        """ Function to handle entry and exit of workers in the economy """
        # Take the top 2% of earners and assume they are new entrants
        # Remove them from list_of_employed and move them to list_of_unemployed in the same occupation with the lowest wage in that occupation
        occ.list_of_employed.sort(key=lambda x: x.wage, reverse=True)
        emp_no = len(occ.list_of_employed)
        if emp_no == 0:
            new_wage = occ.wage
        else:
            bottom_10 = max(1, int(emp_no * 0.05))
            new_wage = np.mean([wrkr.wage for wrkr in occ.list_of_employed[-bottom_10:]])
        n_new = int(emp_no * rate)
        new_workers = occ.list_of_employed[:n_new]
        occ.list_of_employed = occ.list_of_employed[n_new:]

        # Add new workers to the unemployed list with a wage of 0 and time unemployed of 0
        for nw in new_workers:
            nw.longterm_unemp = False
            nw.time_unemployed = 0
            nw.wage = new_wage
            nw.hired = False
            nw.ue_rel_wage = None
            nw.ee_rel_wage = None
            nw.apps_sent = 0
            occ.list_of_unemployed.append(nw)

    def entry(occ, rate):
        """ Function to handle entry of workers in the economy """
        # If an entry-level occupation, add new workers to the employed list at a particular rate
        occ.list_of_employed.sort(key=lambda x: x.wage, reverse=True)
        if occ.entry_level:
            emp_no = len(occ.list_of_employed)
            if emp_no == 0:
                new_wage = occ.wage*0.95
                fem_share = 0.5
            else:
                bottom_10 = max(1, int(emp_no * 0.05))
                new_wage = np.mean([wrkr.wage for wrkr in occ.list_of_employed[-bottom_10:]])
                fem_share = np.mean([wrkr.female for wrkr in occ.list_of_employed])

            # number of new entrants
            for i in range(int(emp_no*rate)):
                occ.list_of_employed.append(worker(occ.occupation_id, False, 0, new_wage, 
                                                   False, 
                                                   random.random() < fem_share, 
                                                   occ.experience_age,
                                                   abs(int(np.random.normal(7, 2))), 1, 1, 0, 0))


class vac:
    def __init__(v, occupation_id, applicants, wage, filled, time_open):
        v.occupation_id = occupation_id
        v.applicants = applicants
        v.wage = wage
        v.filled = filled
        v.time_open = time_open
        
    # Function to hire a worker from pool of vacancies    
    def hire(v, net):
        a = random.choice(v.applicants)
        assert(not(a.hired))
        try:
            net[v.occupation_id].list_of_employed.append(net[a.occupation_id].list_of_employed.pop(net[a.occupation_id].list_of_employed.index(a)))
            a.ee_rel_wage = v.wage/a.wage
            #net[v.occupation_id].list_of_employed.append(a)
            #net[a.occupation_id].list_of_employed.remove(a)
        except ValueError:
            try:
                # Second attempt (fallback)
                net[v.occupation_id].list_of_employed.append(net[a.occupation_id].list_of_unemployed.pop(net[a.occupation_id].list_of_unemployed.index(a)))
                a.ue_rel_wage = v.wage/a.wage   
            except ValueError:
                print("Indexing failed - worker not found in either employed or unemployed list")
        net[v.occupation_id].hired += 1
        a.occupation_id = v.occupation_id
        a.time_unemployed = 0
        # Their new wage is now the vacancy's wage - the relative wage will be updated in the update_workers function
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
def initialise(n_occ, employment, unemployment, vacancies, demand_target, A, wages, gend_share, fem_ra, male_ra, entry_level, experience_age):
    """ Makes a list of occupations with initial conditions
       Args:
           n_occ: number of occupations initialised (464)
           employment: vector with employment of each occupation
           unemployment: vector with unemployment of each occupation
           vacancies: vector with vacancies of each occupation
           demand_target: vector with (initial) target_demand for each occupation (never updated)
           A: adjacency matrix of network (not including auto-transition probability)
           wages: vector of wages of each occupation
           entry_leve: boolean of whether it is an entry-level occupation
           experience_age: minimum wage of entry (including education and minimum experience)

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
            vac_list.append(vac(i, [], wages[i,0], False, 0))
            
        occ = occupation(i, [], [], list(A[i] > 0), list(A[i]),
                         (employment[i,0] + vacancies[i,0]), 
                         demand_target[i,0], wages[i,0], 0, 0, entry_level[i], experience_age[i])
        # creating the workers of occupation i and attaching to occupation
        ## adding employed workers
        g_share = gend_share[i,0]
        for e in range(round(employment[i,0])):
            # Assume they have all at least 1 t.s. of employment
            if np.random.rand() <= g_share:
                occ.list_of_employed.append(worker(occ.occupation_id, False, 1, wages[i,0], False, True, np.random.uniform(low=experience_age[i], high = 65),
                                               abs(int(np.random.normal(fem_ra,2))), 1, 1,0,0))
            else:
                occ.list_of_employed.append(worker(occ.occupation_id, False, 1, wages[i,0], False, False, np.random.uniform(experience_age[i], high = 65),
                                               abs(int(np.random.normal(male_ra,2))), 1, 1,0,0))
            ## adding unemployed workers
        for u in range(round(unemployment[i,0])):
            if np.random.rand() <= g_share:
                # Assigns time unemployed from absolute value of normal distribution....
                occ.list_of_unemployed.append(worker(occ.occupation_id, False, max(1,(int(np.random.normal(2,2)))), 
                                                     np.random.normal(wages[i,0], 0.05* wages[i,0]), False, True, 
                                                     np.random.uniform(experience_age[i], high = 65), 
                                                     abs(int(np.random.normal(fem_ra,0.1))), 1, 1, 1, 0))
            else:
                # Assigns time unemployed from absolute value of normal distribution....
                occ.list_of_unemployed.append(worker(occ.occupation_id, False, max(1,(int(np.random.normal(2,2)))), 
                                                     np.random.normal(wages[i,0], 0.05* wages[i,0]), False, False, 
                                                     np.random.uniform(experience_age[i], high = 65), 
                                                     abs(int(np.random.normal(male_ra,0.1))), 1, 1, 1, 0))


        occs.append(occ)
        ids += 1
    return occs, vac_list


