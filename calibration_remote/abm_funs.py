# Import packages
import numpy as np
import pandas as pd
import random as random
from copy import deepcopy 
import math as math
import heapq
from collections import defaultdict
from scipy.stats import norm
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
        res_wage = np.random.normal(loc=0.5, scale=np.max(res_wage_data[f'{balancing_method}_se']), size=1)[0]
        if expectation:
            res_wage = 0.5
    else:
        mean_val = res_wage_data[f'{balancing_method}_preds_fit'][res_wage_data['dur_unemp'] == duration_months]
        se_val = res_wage_data[f'{balancing_method}_se'][res_wage_data['dur_unemp'] == duration_months]

        if expectation:
            res_wage = mean_val.iloc[0]  # Extract scalar from Series
        else:
            res_wage = np.random.normal(loc=mean_val.iloc[0], scale=se_val.iloc[0], size=1)[0]
    
    return res_wage

## Defining classes
# Potentially redundant use of IDs in the below classes...to check
class worker:
    def __init__(wrkr, occupation_id, 
                 # employed, 
                 longterm_unemp, 
                 # time_employed,
                 time_unemployed, wage, hired, female, age, risk_av_score, ee_rel_wage, ue_rel_wage, applicants_sent, d_wage_offer, last_ue_duration):
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
        wrkr.last_ue_duration = last_ue_duration

    # def search_and_apply(
    #     wrkr,
    #     net,
    #     vacancies_by_occ,
    #     disc,
    #     app_effort,
    #     wage_prefs,
    #     mis_rate: float = 0.10,              # PROBABILITY TO REPLACE A SAMPLED VAC WITH A GLOBAL RANDOM VAC
    #     unique_random: bool = False,         # TRY TO AVOID DUPLICATE RANDOM PICKS IF TRUE
    #     global_pool_override=None  # OPTIONAL PRECOMPUTED GLOBAL POOL (FOR PERFORMANCE)
    # ):

    #     MAX_VACS = 30
    #     wrkr_occ = wrkr.occupation_id
    #     neigh_probs = net[wrkr_occ].list_of_neigh_weights  # ALREADY NORMALIZED
    #     occ_ids = list(range(len(neigh_probs)))

    #     # BUILD NEIGHBOURHOOD POOL (UNCHANGED)
    #     all_vacs = []
    #     weights = []
    #     for occ_id, occ_prob in zip(occ_ids, neigh_probs):
    #         vacs = vacancies_by_occ.get(occ_id, [])
    #         for v in vacs:
    #             all_vacs.append(v)
    #             weights.append(occ_prob)

    #     if not all_vacs:
    #         wrkr.apps_sent = 0
    #         wrkr.d_wage_offer = np.nan
    #         return 0

    #     # SAMPLE NEIGHBOURHOOD VACANCIES (UNCHANGED)
    #     try:
    #         found_vacs = random.choices(all_vacs, weights=weights, k=min(MAX_VACS, len(all_vacs)))
    #     except ValueError:
    #         wrkr.apps_sent = 0
    #         wrkr.d_wage_offer = np.nan
    #         return 0

    #     # PREPARE GLOBAL POOL (ECONOMY-WIDE). ALLOW OVERRIDE FOR PERFORMANCE (CHANGE)
    #     if global_pool_override is not None:
    #         global_pool = list(global_pool_override)
    #     else:
    #         global_pool = [v for vacs in vacancies_by_occ.values() for v in vacs]

    #     global_pool_size = len(global_pool)

    #     # CREATE PARALLEL FLAGS TO MARK WHICH SAMPLED VACS WERE REPLACED BY GLOBAL PICKS (CHANGE)
    #     random_flags = [False] * len(found_vacs)

    #     # PERFORM REPLACEMENTS: EACH SAMPLED VAC HAS INDEPENDENT mis_rate PROBABILITY (UNCHANGED)
    #     if mis_rate > 0 and global_pool_size > 0:
    #         if unique_random and len(found_vacs) <= global_pool_size:
    #             # IF UNIQUE AND POOL BIG ENOUGH: SAMPLE UNIQUE GLOBAL PICKS FOR ALL REPLACEMENTS
    #             replace_idxs = [i for i in range(len(found_vacs)) if random.random() < mis_rate]
    #             if replace_idxs:
    #                 chosen_idxs = random.sample(range(global_pool_size), k=len(replace_idxs))
    #                 for idx_pos, chosen_idx in zip(replace_idxs, chosen_idxs):
    #                     found_vacs[idx_pos] = global_pool[chosen_idx]
    #                     random_flags[idx_pos] = True
    #         else:
    #             # FALLBACK: POSSIBLE DUPLICATES (SIMPLE AND FAST)
    #             for i in range(len(found_vacs)):
    #                 rand_added = 0
    #                 if random.random() < mis_rate:
    #                     rand_added += 1
    #                     found_vacs[i] = random.choice(global_pool)
    #                     random_flags[i] = True

    #     # MEAN WAGE OF THE SAMPLED POOL (INCLUDES GLOBAL PICKS)
    #     mean_wage_sampled = np.mean([v.wage for v in found_vacs]) if found_vacs else 0

    #     vsent = 0

    #     # APPLY RESERVATION WAGE FILTER, BUT ALWAYS KEEP GLOBAL RANDOM PICKS (CHANGE)
    #     if disc:
    #         if wage_prefs:
    #             # RESERVATION WAGE AS YOU DEFINED
    #             if wrkr.time_unemployed > 3:
    #                 res_wage = wrkr.wage * (1 - 0.1 * (wrkr.time_unemployed - 3))
    #             else:
    #                 res_wage = wrkr.wage

    #             # KEEP VACANCIES THAT (A) ARE GLOBAL RANDOM PICKS OR (B) MEET RES_WAGE
    #             found_vacs_res = []
    #             for v, is_rand in zip(found_vacs, random_flags):
    #                 if is_rand:
    #                     # GLOBAL PICK: BYPASS WAGE FILTER (CHANGE)
    #                     found_vacs_res.append(v)
    #                 else:
    #                     if v.wage >= res_wage:
    #                         found_vacs_res.append(v)

    #         else:
    #             # NO WAGE PREFS: keep all sampled (including global picks)
    #             found_vacs_res = found_vacs
    #             res_wage = np.nan

    #         # RANK BY UTILITY (UNCHANGED)
    #         sorted_vacs = sorted(
    #             found_vacs_res,
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

    #         # MEAN WAGE OF ACTUALLY APPLIED-TO VACANCIES (USED FOR d_wage_offer) (CHANGE)
    #         mean_wage_applied = np.mean([v.wage for v in chosen_vacs]) if chosen_vacs else np.nan

    #     else:
    #         # NON-DISCRIMINATING CASE: random sample as before
    #         sampled_for_emp = random.sample(found_vacs, min(len(found_vacs), 7))
    #         for v in sampled_for_emp:
    #             v.applicants.append(wrkr)
    #             vsent += 1
    #         res_wage = np.nan
    #         mean_wage_applied = np.mean([v.wage for v in sampled_for_emp]) if sampled_for_emp else np.nan

    #     # SET OUTPUTS: USE mean_wage_applied WHERE AVAILABLE (CHANGE)
    #     wrkr.d_wage_offer = (mean_wage_applied - res_wage) if disc else np.nan
    #     wrkr.apps_sent = vsent

    #     return vsent

    # def search_and_apply(
    #     wrkr,
    #     net,
    #     vacancies_by_occ,
    #     disc,
    #     app_effort,
    #     wage_prefs,
    #     mis_rate: float = 0.10,              # PROBABILITY TO REPLACE A SAMPLED VAC WITH A GLOBAL RANDOM VAC
    #     unique_random: bool = False,         # TRY TO AVOID DUPLICATE RANDOM PICKS IF TRUE
    #     global_pool_override=None  # OPTIONAL PRECOMPUTED GLOBAL POOL (FOR PERFORMANCE)
    # ):

    #     MAX_VACS = 30
    #     wrkr_occ = wrkr.occupation_id
    #     neigh_probs = net[wrkr_occ].list_of_neigh_weights  # ALREADY NORMALIZED
    #     occ_ids = list(range(len(neigh_probs)))

    #     # BUILD NEIGHBOURHOOD POOL (UNCHANGED)
    #     all_vacs_ = []
    #     weights = []
    #     for occ_id, occ_prob in zip(occ_ids, neigh_probs):
    #         vacs = vacancies_by_occ.get(occ_id, [])
    #         for v in vacs:
    #             all_vacs_.append(v)
    #             weights.append(occ_prob)

    #     if not all_vacs_:
    #         wrkr.apps_sent = 0
    #         wrkr.d_wage_offer = np.nan
    #         return 0
        
    #     if wage_prefs:
    #         res_wage = wrkr.wage*reservation_wage(wrkr.time_unemployed, res_wage_dat)
    #         all_vacs = [v for v in all_vacs_ if v.wage >= res_wage]
    #     else:
    #         all_vacs = all_vacs_
    #         res_wage = np.nan

    #     # SAMPLE NEIGHBOURHOOD VACANCIES (UNCHANGED)
    #     try:
    #         found_vacs = random.choices(all_vacs, weights=weights, k=min(MAX_VACS, len(all_vacs)))
    #         len_f_v = len(found_vacs)
    #         if wage_prefs:
    #             print(len_f_v)
    #     except ValueError:
    #         wrkr.apps_sent = 0
    #         wrkr.d_wage_offer = np.nan
    #         return 0

    #     # MEAN WAGE OF THE SAMPLED POOL (INCLUDES GLOBAL PICKS)
    #     mean_wage_sampled = np.mean([v.wage for v in found_vacs]) if found_vacs else 0

    #     vsent = 0

    #     # APPLY RESERVATION WAGE FILTER, BUT ALWAYS KEEP GLOBAL RANDOM PICKS (CHANGE)
    #     if disc:
    #         # if wage_prefs:
    #         #     #res_wage = wrkr.wage*reservation_wage(wrkr.time_unemployed, res_wage_dat)
    #         #     # # RESERVATION WAGE AS YOU DEFINED
    #         #     # if wrkr.time_unemployed > 3:
    #         #     #     res_wage = wrkr.wage * (1 - 0.1 * (wrkr.time_unemployed - 3))
    #         #     # else:
    #         #     #     res_wage = wrkr.wage

    #         #     found_vacs_res = [v for v in found_vacs if v.wage >= res_wage]
    #         #     len_f_v_res = len(found_vacs_res)

    #         #     print(f'Found vacs: {len_f_v}; Found vacs above RW: {len_f_v_res}')
    #         # else:
    #         #     # NO WAGE PREFS: keep all sampled (including global picks)
    #         #     found_vacs_res = found_vacs
    #         #     res_wage = np.nan

    #         # RANK BY UTILITY (UNCHANGED)
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
    #             if mis_rate > 0 and random.random() < mis_rate:
    #                 random_vac = random.choice(global_pool_override)
    #                 # APPLY MISSING RATE: REPLACE WITH RANDOM GLOBAL PICK
    #                 random_vac.applicants.append(wrkr)
    #             else:
    #                 v.applicants.append(wrkr)
    #             vsent += 1

    #         # # MEAN WAGE OF ACTUALLY APPLIED-TO VACANCIES (USED FOR d_wage_offer) (CHANGE)
    #         # mean_wage_applied = np.mean([v.wage for v in chosen_vacs]) if chosen_vacs else np.nan

    #     else:
    #         for v in random.sample(found_vacs, min(len(found_vacs), 7)):
    #             if mis_rate > 0 and random.random() < mis_rate:
    #                 random_vac = random.choice(global_pool_override)
    #                 # APPLY MISSING RATE: REPLACE WITH RANDOM GLOBAL PICK
    #                 random_vac.applicants.append(wrkr)
    #             else:
    #                 v.applicants.append(wrkr)
    #             vsent += 1
    #         res_wage = np.nan

    #     # SET OUTPUTS: USE mean_wage_applied WHERE AVAILABLE (CHANGE)
    #     wrkr.d_wage_offer = (mean_wage_sampled - res_wage) if disc else np.nan
    #     wrkr.apps_sent = vsent

    #     return vsent


    # def search_and_apply(
    #     wrkr,
    #     net,
    #     vacancies_by_occ,
    #     disc,
    #     app_effort,
    #     wage_prefs,
    #     mis_rate: float = 0.10,
    #     unique_random: bool = False,
    #     global_pool_override=None
    # ):
    #     MAX_VACS = 30
    #     wrkr_occ = wrkr.occupation_id

    #     # local-bindings for speed
    #     rand = random.random
    #     r_choice = random.choice
    #     r_sample = random.sample
    #     neigh_probs = net[wrkr_occ].list_of_neigh_weights  # already normalized
    #     occ_count = len(neigh_probs)

    #     # Build (vacancy, weight) pairs in one pass and optionally apply reservation wage
    #     all_vacs = []
    #     weights = []
    #     # If wage_prefs, compute reservation wage once
    #     if wage_prefs:
    #         res_wage = wrkr.wage * reservation_wage(wrkr.time_unemployed, res_wage_dat)
    #     else:
    #         res_wage = np.nan

    #     for occ_id, occ_prob in enumerate(neigh_probs):
    #         if occ_prob == 0:
    #             continue
    #         vacs = vacancies_by_occ.get(occ_id)
    #         if not vacs:
    #             continue
    #         # iterate once, keep only those that meet reservation wage if requested
    #         if wage_prefs:
    #             for v in vacs:
    #                 w = getattr(v, "wage", None)
    #                 if w is not None and w >= res_wage:
    #                     all_vacs.append(v)
    #                     weights.append(occ_prob)
    #         else:
    #             # fast extend when no wage filtering
    #             all_vacs.extend(vacs)
    #             weights.extend([occ_prob] * len(vacs))

    #     if not all_vacs:
    #         wrkr.apps_sent = 0
    #         wrkr.d_wage_offer = np.nan
    #         return 0

    #     n_to_sample = min(MAX_VACS, len(all_vacs))

    #     # If weights are all equal (or not provided), we can use random.sample (faster)
    #     use_weights = any(w != weights[0] for w in weights) if weights else False

    #     if use_weights:
    #         # random.choices supports weights but returns with replacement.
    #         # To emulate original behaviour (k may be <= population), we accept replacement.
    #         found_vacs = random.choices(all_vacs, weights=weights, k=n_to_sample)
    #     else:
    #         # if all weights equal -> sampling without replacement is faster and simpler
    #         if n_to_sample == len(all_vacs):
    #             found_vacs = list(all_vacs)
    #         else:
    #             # sample without replacement
    #             found_vacs = r_sample(all_vacs, k=n_to_sample)

    #     # mean wage of sampled pool
    #     # use numpy.fromiter for a tiny speed advantage on larger lists
    #     wages_iter = (getattr(v, "wage", 0.0) for v in found_vacs)
    #     try:
    #         mean_wage_sampled = float(np.mean(list(wages_iter))) if found_vacs else 0.0
    #     except Exception:
    #         mean_wage_sampled = 0.0

    #     vsent = 0

    #     if mean_wage_sampled < res_wage:
    #         print(f'Reservation wage: {res_wage}; Sampled Wage: {mean_wage_sampled}')

    #     # Helper: compute utility quickly; local-binding of net neighbor weights lookup
    #     neigh_weights = net[wrkr_occ].list_of_neigh_weights
    #     util_fn = util  # local bind (function)

    #     if disc:
    #         # # compute number of applications
    #         # n_apps = applications_sent(wrkr.time_unemployed, app_effort, expectation=False)
    #         # # we will pick top (risk_aversion + n_apps) by utility, then slice to chosen range
    #         # want_k = wrkr.risk_aversion + n_apps

    #         # Use heapq.nlargest to avoid sorting entire found_vacs if found_vacs is large.
    #         # Create tuples (utility, vacancy) and get top want_k
    #         # utility calculation uses neighbor weight lookup; bind local to avoid repeated global lookups.
    #         def _util_for_v(v):
    #             # inline attribute lookups and neighbor weight
    #             v_wage = getattr(v, "wage", 0.0)
    #             occ_w = neigh_weights[v.occupation_id] if v.occupation_id < len(neigh_weights) else 0.0
    #             return util_fn(wrkr.wage, v_wage, occ_w)

    #         # # if want_k is small relative to found_vacs, nlargest is faster than full sort
    #         # top_k = heapq.nlargest(want_k, found_vacs, key=_util_for_v) if want_k > 0 else []

    #         # # chosen_vacs are slice from risk_aversion to risk_aversion + n_apps
    #         # start = wrkr.risk_aversion
    #         # end = start + n_apps

    #         n_apps = applications_sent(wrkr.time_unemployed, app_effort, expectation=False)
    #         available = len(found_vacs)

    #         # Check if we have enough vacancies to support both risk aversion AND desired applications
    #         if available < wrkr.risk_aversion + n_apps:
    #             # Not enough vacancies - need to compromise
    #             if available <= wrkr.risk_aversion:
    #                 # Can't afford to skip any
    #                 adjusted_start = 0
    #                 adjusted_n_apps = min(n_apps, available)
    #             else:
    #                 # Can skip some, but need to reduce either start or n_apps
    #                 # Option A: Reduce risk aversion to ensure we get n_apps
    #                 adjusted_start = max(0, available - n_apps)
    #                 adjusted_n_apps = min(n_apps, available - adjusted_start)
                    
    #                 # Option B: Keep full risk aversion, accept fewer apps
    #                 # adjusted_start = wrkr.risk_aversion
    #                 # adjusted_n_apps = available - wrkr.risk_aversion
    #         else:
    #             # Plenty of vacancies - use normal behavior
    #             adjusted_start = wrkr.risk_aversion
    #             adjusted_n_apps = n_apps

    #         want_k = adjusted_start + adjusted_n_apps

    #         top_k = heapq.nlargest(want_k, found_vacs, key=_util_for_v) if want_k > 0 else []
    #         chosen_vacs = top_k[adjusted_start:adjusted_start + adjusted_n_apps]
    #         # apply to chosen vacancies
    #         # use local references for speed
    #         gpool = global_pool_override
    #         append_to_applicants = lambda vac: vac.applicants.append(wrkr)

    #         for v in chosen_vacs:
    #             if mis_rate > 0 and rand() < mis_rate and gpool:
    #                 random_vac = r_choice(gpool)
    #                 append_to_applicants(random_vac)
    #             else:
    #                 append_to_applicants(v)
    #             vsent += 1

    #     else:
    #         # when not disc, choose up to 7 random vacancies from found_vacs without replacement if possible
    #         n_choice = min(len(found_vacs), 7)
    #         # If n_choice == len(found_vacs), use the whole list
    #         chosen_sample = found_vacs if n_choice == len(found_vacs) else r_sample(found_vacs, k=n_choice)
    #         gpool = global_pool_override
    #         append_to_applicants = lambda vac: vac.applicants.append(wrkr)

    #         for v in chosen_sample:
    #             if mis_rate > 0 and rand() < mis_rate and gpool:
    #                 random_vac = r_choice(gpool)
    #                 append_to_applicants(random_vac)
    #             else:
    #                 append_to_applicants(v)
    #             vsent += 1

    #     # Set wrkr outputs (note: original used mean_wage_sampled - res_wage if disc)
    #     wrkr.d_wage_offer = (mean_wage_sampled - res_wage) if disc else np.nan
    #     wrkr.apps_sent = vsent

    #     return vsent
    
    # def search_and_apply(
    #     wrkr,
    #     net,
    #     vacancies_by_occ,
    #     disc,
    #     app_effort,
    #     wage_prefs,
    #     mis_rate: float = 0.10,
    #     unique_random: bool = False,
    #     global_pool_override=None
    # ):
    #     MAX_VACS = 30
    #     wrkr_occ = wrkr.occupation_id

    #     # local-bindings for speed
    #     rand = random.random
    #     r_choice = random.choice
    #     r_sample = random.sample
    #     neigh_probs = net[wrkr_occ].list_of_neigh_weights
    #     occ_count = len(neigh_probs)

    #     # Build (vacancy, weight) pairs in one pass
    #     # *** REMOVE reservation wage filtering here ***
    #     all_vacs = []
    #     weights = []
        
    #     # Compute reservation wage once if needed
    #     if wage_prefs:
    #         res_wage = wrkr.wage * reservation_wage(wrkr.time_unemployed, res_wage_dat)
    #     else:
    #         res_wage = np.nan

    #     for occ_id, occ_prob in enumerate(neigh_probs):
    #         if occ_prob == 0:
    #             continue
    #         vacs = vacancies_by_occ.get(occ_id)
    #         if not vacs:
    #             continue
            
    #         # *** DON'T filter by wage here - include all vacancies ***
    #         all_vacs.extend(vacs)
    #         weights.extend([occ_prob] * len(vacs))

    #     if not all_vacs:
    #         wrkr.apps_sent = 0
    #         wrkr.d_wage_offer = np.nan
    #         #print(f"{wrkr_occ} all_vacs empty")
    #         return 0

    #     n_to_sample = min(MAX_VACS, len(all_vacs))

    #     # Sampling logic (unchanged)
    #     use_weights = any(w != weights[0] for w in weights) if weights else False

    #     if use_weights:
    #         found_vacs = random.choices(all_vacs, weights=weights, k=n_to_sample)
    #     else:
    #         if n_to_sample == len(all_vacs):
    #             found_vacs = list(all_vacs)
    #         else:
    #             found_vacs = r_sample(all_vacs, k=n_to_sample)

    #     # mean wage of sampled pool
    #     wages_iter = (getattr(v, "wage", 0.0) for v in found_vacs)
    #     try:
    #         mean_wage_sampled = float(np.mean(list(wages_iter))) if found_vacs else 0.0
    #     except Exception:
    #         mean_wage_sampled = 0.0

    #     vsent = 0

    #     # Helper: compute utility quickly; local-binding of net neighbor weights lookup
    #     neigh_weights = net[wrkr_occ].list_of_neigh_weights
    #     util_fn = util

    #     if disc:
    #         # *** MODIFY utility function to include reservation wage penalty ***
    #         def _util_for_v(v):
    #             v_wage = getattr(v, "wage", 0.0)
    #             occ_w = neigh_weights[v.occupation_id] if v.occupation_id < len(neigh_weights) else 0.0
    #             base_util = util_fn(wrkr.wage, v_wage, occ_w)
                
    #             # Apply reservation wage as utility modifier (not hard filter)
    #             if wage_prefs and not np.isnan(res_wage):
    #                 if v_wage < res_wage:
    #                     # Penalize vacancies below reservation wage
    #                     # The further below, the stronger the penalty
    #                     shortfall_ratio = (res_wage - v_wage) / res_wage
    #                     # Exponential decay: 50% penalty at res_wage, 90% at 50% below res_wage
    #                     penalty = np.exp(-3 * shortfall_ratio)
    #                     return base_util * penalty
                
    #             return base_util

    #         n_apps = applications_sent(wrkr.time_unemployed, app_effort, expectation=False)
    #         available = len(found_vacs)

    #         # Your risk aversion adjustment (UNCHANGED - this is correct!)
    #         if available < wrkr.risk_aversion + n_apps:
    #             if available <= wrkr.risk_aversion:
    #                 adjusted_start = 0
    #                 adjusted_n_apps = min(n_apps, available)
    #             else:
    #                 adjusted_start = max(0, available - n_apps)
    #                 adjusted_n_apps = min(n_apps, available - adjusted_start)
    #         else:
    #             adjusted_start = wrkr.risk_aversion
    #             adjusted_n_apps = n_apps

    #         want_k = adjusted_start + adjusted_n_apps

    #         top_k = heapq.nlargest(want_k, found_vacs, key=_util_for_v) if want_k > 0 else []
    #         chosen_vacs = top_k[adjusted_start:adjusted_start + adjusted_n_apps]
            
    #         # apply to chosen vacancies
    #         gpool = global_pool_override
    #         append_to_applicants = lambda vac: vac.applicants.append(wrkr)

    #         for v in chosen_vacs:
    #             if mis_rate > 0 and rand() < mis_rate and gpool:
    #                 random_vac = r_choice(gpool)
    #                 append_to_applicants(random_vac)
    #             else:
    #                 append_to_applicants(v)
    #             vsent += 1

    #     else:
    #         # when not disc, choose up to 7 random vacancies
    #         n_choice = min(len(found_vacs), 7)
    #         chosen_sample = found_vacs if n_choice == len(found_vacs) else r_sample(found_vacs, k=n_choice)
    #         # if n_choice < 7:
    #         #     print(f'Occ_id = {wrkr_occ}; n_vacs = {n_choice}; Len of chosen_sample = {len(chosen_sample)}')
    #         gpool = global_pool_override
    #         append_to_applicants = lambda vac: vac.applicants.append(wrkr)

    #         for v in chosen_sample:
    #             if mis_rate > 0 and rand() < mis_rate and gpool:
    #                 random_vac = r_choice(gpool)
    #                 append_to_applicants(random_vac)
    #             else:
    #                 append_to_applicants(v)
    #             vsent += 1

    #     # Set wrkr outputs
    #     wrkr.d_wage_offer = (mean_wage_sampled - res_wage) if disc else np.nan
    #     wrkr.apps_sent = vsent

    #     return vsent
### MOST RECENT 19 DECEMBER 
        # def search_and_apply(
        #     wrkr,
        #     net,
        #     vacancies_by_occ,
        #     disc,
        #     app_effort,
        #     wage_prefs,
        #     mis_rate: float = 0.10,
        #     unique_random: bool = False,
        #     global_pool_override=None,
        #     strict_reservation_wage: bool = False  # NEW PARAMETER
        # ):
        #     MAX_VACS = 30
        #     wrkr_occ = wrkr.occupation_id

        #     # local-bindings for speed
        #     rand = random.random
        #     r_choice = random.choice
        #     r_sample = random.sample
        #     neigh_probs = net[wrkr_occ].list_of_neigh_weights
        #     occ_count = len(neigh_probs)

        #     # Build (vacancy, weight) pairs in one pass
        #     all_vacs = []
        #     weights = []
            
        #     # Compute reservation wage once if needed (only when disc=True AND wage_prefs=True)
        #     if wage_prefs:
        #         res_wage = wrkr.wage * reservation_wage(wrkr.time_unemployed, res_wage_dat)
        #     else:
        #         res_wage = np.nan

        #     for occ_id, occ_prob in enumerate(neigh_probs):
        #         if occ_prob == 0:
        #             continue
        #         vacs = vacancies_by_occ.get(occ_id)
        #         if not vacs:
        #             continue
                
        #         # Include all vacancies (no filtering here yet)
        #         all_vacs.extend(vacs)
        #         weights.extend([occ_prob] * len(vacs))

        #     if not all_vacs:
        #         wrkr.apps_sent = 0
        #         wrkr.d_wage_offer = np.nan
        #         return 0

        #     n_to_sample = min(MAX_VACS, len(all_vacs))

        #     # Sampling logic
        #     use_weights = any(w != weights[0] for w in weights) if weights else False

        #     if use_weights:
        #         found_vacs = random.choices(all_vacs, weights=weights, k=n_to_sample)
        #     else:
        #         if n_to_sample == len(all_vacs):
        #             found_vacs = list(all_vacs)
        #         else:
        #             found_vacs = r_sample(all_vacs, k=n_to_sample)

        #     # *** APPLY STRICT RESERVATION WAGE FILTER IF REQUESTED ***
        #     # Only applies when disc=True AND wage_prefs=True AND strict_reservation_wage=True
        #     if disc and wage_prefs and not np.isnan(res_wage) and strict_reservation_wage:
        #         found_vacs = [v for v in found_vacs if getattr(v, "wage", 0.0) >= res_wage]
                
        #         # If no vacancies pass the filter, return early
        #         if not found_vacs:
        #             wrkr.apps_sent = 0
        #             wrkr.d_wage_offer = np.nan
        #             return 0

        #     # mean wage of sampled pool (after potential filtering)
        #     wages_iter = (getattr(v, "wage", 0.0) for v in found_vacs)
        #     try:
        #         mean_wage_sampled = float(np.mean(list(wages_iter))) if found_vacs else 0.0
        #     except Exception:
        #         mean_wage_sampled = 0.0

        #     vsent = 0

        #     # Helper: compute utility quickly
        #     neigh_weights = net[wrkr_occ].list_of_neigh_weights
        #     util_fn = util

        #     if disc:
        #         # Define utility function
        #         def _util_for_v(v):
        #             v_wage = getattr(v, "wage", 0.0)
        #             occ_w = neigh_weights[v.occupation_id] if v.occupation_id < len(neigh_weights) else 0.0
        #             base_util = util_fn(wrkr.wage, v_wage, occ_w)
                    
        #             # *** ONLY APPLY PENALTY IF wage_prefs=True AND NOT USING STRICT FILTER ***
        #             if wage_prefs and not np.isnan(res_wage) and not strict_reservation_wage:
        #                 if v_wage < res_wage:
        #                     # Penalize vacancies below reservation wage
        #                     shortfall_ratio = (res_wage - v_wage) / res_wage
        #                     penalty = np.exp(-3 * shortfall_ratio)
        #                     return base_util * penalty
                    
        #             return base_util

        #         n_apps = applications_sent(wrkr.time_unemployed, app_effort, expectation=False)
        #         available = len(found_vacs)

        #         # Risk aversion adjustment
        #         if available < wrkr.risk_aversion + n_apps:
        #             if available <= wrkr.risk_aversion:
        #                 adjusted_start = 0
        #                 adjusted_n_apps = min(n_apps, available)
        #             else:
        #                 adjusted_start = max(0, available - n_apps)
        #                 adjusted_n_apps = min(n_apps, available - adjusted_start)
        #         else:
        #             adjusted_start = wrkr.risk_aversion
        #             adjusted_n_apps = n_apps

        #         want_k = adjusted_start + adjusted_n_apps

        #         top_k = heapq.nlargest(want_k, found_vacs, key=_util_for_v) if want_k > 0 else []
        #         chosen_vacs = top_k[adjusted_start:adjusted_start + adjusted_n_apps]
                
        #         # apply to chosen vacancies
        #         gpool = global_pool_override
        #         append_to_applicants = lambda vac: vac.applicants.append(wrkr)

        #         for v in chosen_vacs:
        #             if mis_rate > 0 and rand() < mis_rate and gpool:
        #                 random_vac = r_choice(gpool)
        #                 append_to_applicants(random_vac)
        #             else:
        #                 append_to_applicants(v)
        #             vsent += 1

        #     else:
        #         # when not disc, choose up to 9 random vacancies (NO RESERVATION WAGE LOGIC)
        #         n_choice = min(len(found_vacs), 9)
        #         chosen_sample = found_vacs if n_choice == len(found_vacs) else r_sample(found_vacs, k=n_choice)
                
        #         gpool = global_pool_override
        #         append_to_applicants = lambda vac: vac.applicants.append(wrkr)

        #         for v in chosen_sample:
        #             if mis_rate > 0 and rand() < mis_rate and gpool:
        #                 random_vac = r_choice(gpool)
        #                 append_to_applicants(random_vac)
        #             else:
        #                 append_to_applicants(v)
        #             vsent += 1

        #     # Set wrkr outputs (only meaningful when disc=True)
        #     wrkr.d_wage_offer = (mean_wage_sampled - res_wage) if disc else np.nan
        #     wrkr.apps_sent = vsent

        #     return vsent

    def search_and_apply(
        wrkr,
        net,
        vacancies_by_occ,
        disc,
        app_effort,
        wage_prefs,
        mis_rate: float = 0.10,
        unique_random: bool = False,
        global_pool_override=None,
        strict_reservation_wage: bool = False,
        below_resw_prob: float = 0.15
    ):
        MAX_VACS = 30
        wrkr_occ = wrkr.occupation_id

        # local-bindings for speed
        rand = random.random
        r_choice = random.choice
        r_sample = random.sample
        neigh_probs = net[wrkr_occ].list_of_neigh_weights
        occ_count = len(neigh_probs)

        # Build (vacancy, weight) pairs in one pass
        all_vacs = []
        weights = []
        
        # Compute reservation wage once if needed
        if wage_prefs:
            res_wage = wrkr.wage * reservation_wage(wrkr.time_unemployed, res_wage_dat)
        else:
            res_wage = np.nan

        for occ_id, occ_prob in enumerate(neigh_probs):
            if occ_prob == 0:
                continue
            vacs = vacancies_by_occ.get(occ_id)
            if not vacs:
                continue
            
            # Include all vacancies (no filtering here yet)
            all_vacs.extend(vacs)
            weights.extend([occ_prob] * len(vacs))

        if not all_vacs:
            wrkr.apps_sent = 0
            wrkr.d_wage_offer = np.nan
            return 0

        n_to_sample = min(MAX_VACS, len(all_vacs))

        # Sampling logic
        use_weights = any(w != weights[0] for w in weights) if weights else False

        if use_weights:
            found_vacs = random.choices(all_vacs, weights=weights, k=n_to_sample)
        else:
            if n_to_sample == len(all_vacs):
                found_vacs = list(all_vacs)
            else:
                found_vacs = r_sample(all_vacs, k=n_to_sample)

        # *** APPLY STRICT RESERVATION WAGE FILTER IF REQUESTED ***
        if wage_prefs and not np.isnan(res_wage) and strict_reservation_wage:
            found_vacs = [v for v in found_vacs if getattr(v, "wage", 0.0) >= res_wage]
            
            # If no vacancies pass the filter, return early
            if not found_vacs:
                wrkr.apps_sent = 0
                wrkr.d_wage_offer = np.nan
                return 0

        # mean wage of sampled pool (after potential filtering)
        wages_iter = (getattr(v, "wage", 0.0) for v in found_vacs)
        try:
            mean_wage_sampled = float(np.mean(list(wages_iter))) if found_vacs else 0.0
        except Exception:
            mean_wage_sampled = 0.0

        vsent = 0

        if disc:
            # *** Split vacancies into above/below reservation wage ***
            if wage_prefs and not np.isnan(res_wage) and not strict_reservation_wage:
                above_resw = []
                below_resw = []
                for v in found_vacs:
                    v_wage = getattr(v, "wage", 0.0)
                    if v_wage >= res_wage:
                        above_resw.append(v)
                    else:
                        below_resw.append(v)
            else:
                above_resw = found_vacs
                below_resw = []
            
            # Get neighbor weights for utility calculation
            neigh_weights = net[wrkr_occ].list_of_neigh_weights

            n_apps = applications_sent(wrkr.time_unemployed, app_effort, expectation=False)
            
            # *** MAIN APPLICATIONS: Only consider above-reservation vacancies ***
            available = len(above_resw)

            # Risk aversion adjustment
            if available < wrkr.risk_aversion + n_apps:
                if available <= wrkr.risk_aversion:
                    adjusted_start = 0
                    adjusted_n_apps = min(n_apps, available)
                else:
                    adjusted_start = max(0, available - n_apps)
                    adjusted_n_apps = min(n_apps, available - adjusted_start)
            else:
                adjusted_start = wrkr.risk_aversion
                adjusted_n_apps = n_apps

            want_k = adjusted_start + adjusted_n_apps

            # Rank and select from ABOVE reservation wage vacancies using util()
            # Use lambda to call util() with the right parameters
            def utility_key(v):
                v_wage = getattr(v, "wage", 0.0)
                occ_w = neigh_weights[v.occupation_id] if v.occupation_id < len(neigh_weights) else 0.0
                return util(wrkr.wage, v_wage, occ_w)
            
            top_k = heapq.nlargest(want_k, above_resw, key=utility_key) if want_k > 0 else []
            chosen_vacs = top_k[adjusted_start:adjusted_start + adjusted_n_apps]
            
            # *** Probabilistically add some below-reservation vacancies ***
            if below_resw and wage_prefs and not np.isnan(res_wage) and not strict_reservation_wage:
                # Each below-reservation vacancy has below_resw_prob chance of being applied to
                for v in below_resw:
                    if rand() < below_resw_prob:
                        chosen_vacs.append(v)
            
            # apply to chosen vacancies
            gpool = global_pool_override
            append_to_applicants = lambda vac: vac.applicants.append(wrkr)

            for v in chosen_vacs:
                if mis_rate > 0 and rand() < mis_rate and gpool:
                    random_vac = r_choice(gpool)
                    append_to_applicants(random_vac)
                else:
                    append_to_applicants(v)
                vsent += 1

        else:
            # when not disc, choose up to 8 random vacancies
            n_choice = min(len(found_vacs), 8)
            chosen_sample = found_vacs if n_choice == len(found_vacs) else r_sample(found_vacs, k=n_choice)
            
            gpool = global_pool_override
            append_to_applicants = lambda vac: vac.applicants.append(wrkr)

            for v in chosen_sample:
                if mis_rate > 0 and rand() < mis_rate and gpool:
                    random_vac = r_choice(gpool)
                    append_to_applicants(random_vac)
                else:
                    append_to_applicants(v)
                vsent += 1

        # Set wrkr outputs
        wrkr.d_wage_offer = (mean_wage_sampled - res_wage) if disc else np.nan
        wrkr.apps_sent = vsent

        return vsent

    def emp_search_and_apply(wrkr, net, vac_list, disc, emp_apps, wage_prefs,   
                          mis_rate: float = 0.10,                 # PROBABILITY TO REPLACE A SAMPLED VAC WITH A GLOBAL RANDOM VAC
        unique_random: bool = False,            # TRY TO AVOID DUPLICATE RANDOM PICKS IF TRUE
        global_pool_override = None ):
        # A sample of relevant vacancies are found that are in neighboring occupations
        # Select different random sample of "relevant" vacancies found by each worker
        # found_vacs = random.sample(vac_list, min(len(vac_list), 30))
        MAX_VACS = 30
        wrkr_occ = wrkr.occupation_id
        neigh_probs = net[wrkr_occ].list_of_neigh_weights  # Already normalized
        occ_ids = list(range(len(neigh_probs)))

        # Build a flat list of vacancies and their weights
        all_vacs = []
        weights = []

        for occ_id, occ_prob in zip(occ_ids, neigh_probs):
            vacs = vac_list.get(occ_id, [])
            for v in vacs:
                all_vacs.append(v)
                weights.append(occ_prob)

        if not all_vacs:
            wrkr.apps_sent = 0
            wrkr.d_wage_offer = np.nan
            # ALL-CAPS CHANGE: INSTRUMENTATION ATTRIBUTE WHEN NOTHING SENT
            wrkr.random_apps_sent = 0
            return 0

        # Sample up to MAX_VACS vacancies directly
        try:
            found_vacs = random.choices(all_vacs, weights=weights, k=min(MAX_VACS, len(all_vacs)))
        
        except ValueError:
            wrkr.apps_sent = 0
            wrkr.d_wage_offer = np.nan
            wrkr.random_apps_sent = 0  # ALL-CAPS CHANGE: INSTRUMENTATION
            return 0
            
        mean_wage = np.mean([v.wage for v in found_vacs]) if found_vacs else 0

        if disc and wage_prefs:
            found_vacs = [v for v in found_vacs if v.wage >= np.random.normal(wrkr.wage*1.05, 0.05*wrkr.wage*1.05)]

        elif wage_prefs and not disc:
            found_vacs = [v for v in found_vacs if v.wage >= np.random.normal(wrkr.wage, 0.05*wrkr.wage)]

        else:
            found_vacs = found_vacs
        # Filter found_vacs to keep only elements where util(el) > 0
        # We assume that employed workers will only apply to vacancies for which there is a wage gain. 
        #filtered_vacs = [el for el in found_vacs if util(wrkr.wage, el.wage, net[wrkr.occupation_id].list_of_neigh_weights[el.occupation_id]) > 0]
        vs = random.sample(found_vacs, min(len(found_vacs), emp_apps))
        sent_apps = len(vs)
        for r in vs:
            if mis_rate > 0 and random.random() < mis_rate:
                random_vac = random.choice(global_pool_override)
                    # APPLY MISSING RATE: REPLACE WITH RANDOM GLOBAL PICK
                random_vac.applicants.append(wrkr)
            else:
                r.applicants.append(wrkr)

        wrkr.apps_sent = 0
        wrkr.d_wage_offer = np.nan

        return sent_apps

    # def emp_search_and_apply(wrkr, net, vac_list, disc, emp_apps, wage_prefs):
    #     # A sample of relevant vacancies are found that are in neighboring occupations
    #     # Select different random sample of "relevant" vacancies found by each worker
    #     # found_vacs = random.sample(vac_list, min(len(vac_list), 30))
    #     MAX_VACS = 30
    #     wrkr_occ = wrkr.occupation_id
    #     neigh_probs = net[wrkr_occ].list_of_neigh_weights  # Already normalized
    #     occ_ids = list(range(len(neigh_probs)))

    #     # Build a flat list of vacancies and their weights
    #     all_vacs = []
    #     weights = []

    #     for occ_id, occ_prob in zip(occ_ids, neigh_probs):
    #         vacs = vac_list.get(occ_id, [])
    #         for v in vacs:
    #             all_vacs.append(v)
    #             weights.append(occ_prob)

    #     if not all_vacs:
    #         wrkr.apps_sent = 0
    #         return

    #     # Sample up to MAX_VACS vacancies directly
    #     try:
    #         found_vacs = random.choices(all_vacs, weights=weights, k=min(MAX_VACS, len(all_vacs)))
        
    #     except ValueError:
    #         wrkr.apps_sent = 0
    #         wrkr.d_wage_offer = np.nan
    #         return 0
        
    #     mean_wage = np.mean([v.wage for v in found_vacs]) if found_vacs else 0

    #     if disc and wage_prefs:
    #         found_vacs = [v for v in found_vacs if v.wage >= np.random.normal(wrkr.wage*1.05, 0.05*wrkr.wage*1.05)]
    #     elif wage_prefs and not disc:
    #         found_vacs = [v for v in found_vacs if v.wage >= np.random.normal(wrkr.wage, 0.05*wrkr.wage)]
    #     else:
    #         found_vacs = found_vacs
    #     # Filter found_vacs to keep only elements where util(el) > 0
    #     # We assume that employed workers will only apply to vacancies for which there is a wage gain. 
    #     #filtered_vacs = [el for el in found_vacs if util(wrkr.wage, el.wage, net[wrkr.occupation_id].list_of_neigh_weights[el.occupation_id]) > 0]
    #     vs = random.sample(found_vacs, min(len(found_vacs), emp_apps))
    #     sent_apps = len(vs)
    #     for r in vs:
    #         r.applicants.append(wrkr)
    #     wrkr.apps_sent = 0
    #     wrkr.d_wage_offer = np.nan

    #     return sent_apps
            
class occupation:
    def __init__(occ, occupation_id, list_of_employed, list_of_unemployed, 
                 list_of_neigh_bool, list_of_neigh_weights, current_demand, 
                 target_demand, wage, wage_mu, wage_sigma, separated, hired, entry_level_bool, experience_age, seps_rate, competition_last):
        occ.occupation_id = occupation_id
        occ.list_of_employed = list_of_employed
        occ.list_of_unemployed = list_of_unemployed
        occ.list_of_neigh_bool = list_of_neigh_bool
        occ.list_of_neigh_weights = list_of_neigh_weights
        occ.current_demand = current_demand
        occ.target_demand = target_demand
        occ.wage = wage
        occ.wage_mu = wage_mu
        occ.wage_sigma = wage_sigma
        occ.separated = separated
        occ.hired = hired
        occ.entry_level = entry_level_bool
        occ.experience_age = experience_age
        occ.seps_rate = seps_rate
        occ.competition_last = competition_last

    def separate_workers(occ, delta_u, gam, bus_cy):
        if(len(occ.list_of_employed) != 0):
            sep_prob_base = delta_u + (1-delta_u)*((gam * max(0, len(occ.list_of_employed) - (occ.target_demand*bus_cy)))/(len(occ.list_of_employed) + 1))
            sep_prob_gamma = delta_u + occ.seps_rate*(1-delta_u)*((gam * max(0, len(occ.list_of_employed) - (occ.target_demand*bus_cy)))/(len(occ.list_of_employed) + 1))
            sep_prob_delta = occ.seps_rate*delta_u + (1-delta_u)*((gam * max(0, len(occ.list_of_employed) - (occ.target_demand*bus_cy)))/(len(occ.list_of_employed) + 1))
            w = np.random.binomial(len(occ.list_of_employed), sep_prob_base) #sep_prob_delta) #sep_prob_base) #sep_prob_delta) # sep_prob_base
            occ.separated = int(w)
            separated_workers = random.sample(occ.list_of_employed, int(w))
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
        workers_pre = len(occ.list_of_unemployed) + len(occ.list_of_employed)
        occ.list_of_unemployed = [w for w in occ.list_of_unemployed if w.age <= 65]
        occ.list_of_employed = [e for e in occ.list_of_employed if e.age <= 65]
        workers_sep = workers_pre - (len(occ.list_of_unemployed) + len(occ.list_of_employed))
        return(workers_sep)

    # def entry_and_exit_fixed(occ, rate):
    #     """ Function to handle entry and exit of workers in the economy """
    #     # Take the top 2% of earners and assume they are new entrants
    #     # Remove them from list_of_employed and move them to list_of_unemployed in the same occupation with the lowest wage in that occupation
    #     occ.list_of_employed.sort(key=lambda x: x.wage, reverse=True)
    #     emp_no = len(occ.list_of_employed)
    #     if emp_no == 0:
    #         new_wage = occ.wage
    #     else:
    #         bottom_10 = max(1, int(emp_no * 0.05))
    #         new_wage = np.mean([wrkr.wage for wrkr in occ.list_of_employed[-bottom_10:]])
    #     n_new = int(emp_no * rate)
    #     new_workers = occ.list_of_employed[:n_new]
    #     occ.list_of_employed = occ.list_of_employed[n_new:]

    #     # Add new workers to the unemployed list with a wage of 0 and time unemployed of 0
    #     for nw in new_workers:
    #         nw.longterm_unemp = False
    #         nw.time_unemployed = 0
    #         nw.wage = new_wage
    #         nw.hired = False
    #         nw.ue_rel_wage = None
    #         nw.ee_rel_wage = None
    #         nw.apps_sent = 0
    #         occ.list_of_unemployed.append(nw)

    # def entry_rate(occ, rate):
    #     """ Function to handle entry of workers in the economy """
    #     # If an entry-level occupation, add new workers to the employed list at a particular rate
    #     occ.list_of_employed.sort(key=lambda x: x.wage, reverse=True)
    #     if occ.entry_level:
    #         emp_no = len(occ.list_of_employed)
    #         if emp_no == 0:
    #             new_wage = occ.wage*0.95
    #             fem_share = 0.5
    #         else:
    #             bottom_10 = max(1, int(emp_no * 0.05))
    #             new_wage = np.mean([wrkr.wage for wrkr in occ.list_of_employed[-bottom_10:]])
    #             fem = random.random() < np.mean([wrkr.female for wrkr in occ.list_of_employed])
    #             ra = 3 if fem else 7

    #         for i in range(int(emp_no*rate)):
    #                                                 # occupation_id,     
    #             occ.list_of_employed.append(worker(occ.occupation_id, 
    #                                                  # longterm_unemp, 
    #                                                  False, 
    #                                                 # time_unemployed, 
    #                                                  0, 
    #                                                 # wage, 
    #                                                 new_wage, 
    #                                                 # hired, 
    #                                                 False, 
    #                                                 # female, 
    #                                                 fem, 
    #                                                 # age, 
    #                                                 occ.experience_age,
    #                                                 # risk_av_score, 
    #                                                 abs(int(np.random.normal(ra, 2))), 
    #                                                 # ee_rel_wage, 
    #                                                 1, 
    #                                                 # ue_rel_wage, 
    #                                                 1, 
    #                                                 # applicants_sent, 
    #                                                 0, 
    #                                                 # d_wage_offer)
    #                                                 0))

    def entry(occ, entry_tot):
        """Add exactly `entry_tot` new workers to this occupation (if entry-level)."""
        if entry_tot <= 0 or not occ.entry_level:
            return

        occ.list_of_employed.sort(key=lambda x: x.wage, reverse=True)
        emp_no = len(occ.list_of_employed)

        # Wage offered to new entrants: avg of bottom 5% employed (or 95% of mean wage if empty)
        if emp_no == 0:
            new_wage = occ.wage * 0.95
            fem_prob = 0.5
        else:
            bottom_5 = max(1, int(emp_no * 0.05))
            new_wage = np.mean([wrkr.wage for wrkr in occ.list_of_employed[-bottom_5:]])
            fem_prob = float(np.mean([wrkr.female for wrkr in occ.list_of_employed]))

        for _ in range(entry_tot):
            fem = (random.random() < fem_prob)
            ra = 3 if fem else 7

            occ.list_of_employed.append(worker(
                occ.occupation_id,         # occupation_id
                False,                     # longterm_unemp
                0,                         # time_unemployed
                new_wage,                  # wage
                False,                     # hired
                fem,                       # female
                occ.experience_age,        # age
                abs(int(np.random.normal(ra, 2))),  # risk_av_score
                1,                         # ee_rel_wage
                1,                         # ue_rel_wage
                0,                         # applicants_sent
                0,                         # d_wage_offer
                None                       # last_ue_duration
            ))

    def update_competition_metric(
            occ,
            net,
            vacancies_by_occ,
            use_weights: bool = True,
            include_self: bool = True,
            metric: str = "u_per_v",
        ):
            """
            Compute neighborhood competition once and store it in occ.comp_last.
            metric:
            - "u_per_v": unemployed / max(1, vacancies) (default)
            - "apps_per_v": total applicants / max(1, vacancies)  (requires applicants populated)
            """
            def vac_count(j: int) -> int:
                return len(vacancies_by_occ.get(j, []))

            def unemp_count(j: int) -> int:
                return len(net[j].list_of_unemployed)

            def apps_count(j: int) -> int:
                vacs = vacancies_by_occ.get(j, [])
                return sum(len(v.applicants) for v in vacs)

            numer = 0.0
            denom = 0.0
            for j, is_neigh in enumerate(occ.list_of_neigh_bool):
                if not is_neigh:
                    continue
                if not include_self and j == occ.occupation_id:
                    continue
                w = 1.0 if not use_weights or occ.list_of_neigh_weights is None else float(occ.list_of_neigh_weights[j])
                v = vac_count(j)
                if metric == "u_per_v":
                    comp_j = unemp_count(j) / max(1, v)
                elif metric == "apps_per_v":
                    comp_j = apps_count(j) / max(1, v)
                else:
                    raise ValueError(f"Unknown metric '{metric}'")
                numer += w * comp_j
                denom += w

            occ.comp_last = (numer / denom) if denom > 0 else 0.0
            return occ.comp_last

class vac:
    def __init__(v, occupation_id, applicants, wage, filled, time_open):
        v.occupation_id = occupation_id
        v.applicants = applicants
        v.wage = wage
        v.filled = filled
        v.time_open = time_open
        
    # # Function to hire a worker from pool of vacancies    
    # def hire(v, net):
    #     a = random.choice(v.applicants)
    #     assert(not(a.hired))
    #     try:
    #         net[v.occupation_id].list_of_employed.append(net[a.occupation_id].list_of_employed.pop(net[a.occupation_id].list_of_employed.index(a)))
    #         a.ee_rel_wage = v.wage/a.wage
    #     except ValueError:
    #         try:
    #             # Second attempt (fallback)
    #             net[v.occupation_id].list_of_employed.append(net[a.occupation_id].list_of_unemployed.pop(net[a.occupation_id].list_of_unemployed.index(a)))
    #             a.ue_rel_wage = v.wage/a.wage   
    #         except ValueError:
    #             print("Indexing failed - worker not found in either employed or unemployed list")
    #     net[v.occupation_id].hired += 1
    #     a.occupation_id = v.occupation_id
    #     a.last_ue_duration = a.time_unemployed
    #     a.time_unemployed = 0
    #     # Their new wage is now the vacancy's wage - the relative wage will be updated in the update_workers function
    #     a.wage = v.wage
    #     a.hired = True
    #     v.applicants.clear()

        # Function to hire a worker from pool of vacancies    
    def hire(v, net):
        a = random.choice(v.applicants)
        assert not a.hired

        origin_occ = None
        ue_dur = None

        try:
            # EE → E (no unemployment spell)
            net[v.occupation_id].list_of_employed.append(
                net[a.occupation_id].list_of_employed.pop(
                    net[a.occupation_id].list_of_employed.index(a)
                )
            )
            a.ee_rel_wage = v.wage / a.wage

        except ValueError:
            try:
                # UE → E (capture origin + unemployment duration BEFORE moving)
                src_list = net[a.occupation_id].list_of_unemployed
                idx = src_list.index(a)

                origin_occ = a.occupation_id
                ue_dur = src_list[idx].time_unemployed  # spell length right now

                net[v.occupation_id].list_of_employed.append(src_list.pop(idx))
                a.ue_rel_wage = v.wage / a.wage

            except ValueError:
                print("Indexing failed - worker not found in either employed or unemployed list")

        net[v.occupation_id].hired += 1
        a.occupation_id = v.occupation_id
        a.time_unemployed = 0          # reset after hire
        a.wage = v.wage
        a.hired = True
        v.applicants.clear()

        # Return origin & spell length only for UE→E hires; EE→E returns None
        if origin_occ is not None and ue_dur is not None:
            return (origin_occ, a.occupation_id, int(ue_dur))
        return None

        
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


# ### Function and condition to initialise network
# def initialise(n_occ, employment, unemployment, vacancies, demand_target, A, wages, wage_mu, wage_sigma, gend_share, fem_ra, male_ra, entry_level, experience_age, sep_rates):
#     """ Makes a list of occupations with initial conditions
#        Args:
#            n_occ: number of occupations initialised (464)
#            employment: vector with employment of each occupation
#            unemployment: vector with unemployment of each occupation
#            vacancies: vector with vacancies of each occupation
#            demand_target: vector with (initial) target_demand for each occupation (never updated)
#            A: adjacency matrix of network (not including auto-transition probability)
#            wages: vector of wages of each occupation
#            entry_leve: boolean of whether it is an entry-level occupation
#            experience_age: minimum wage of entry (including education and minimum experience)

#        Returns:
#             occupations: list of occupations with above attributes
#             vacancies: list of vacancies with occupation id, wage, and list of applicants
#        """
#     occs = []
#     vac_list = []
#     ids = 0
#     for i in range(0, n_occ):
#         # appending relevant number of vacancies to economy-wide vacancy list
#         for v in range(round(vacancies[i,0])):
#             vac_list.append(vac(i, [], #wages[i]
#                                 np.clip(np.random.lognormal(wage_mu[i], wage_sigma[i]), 15080, 250000), False, np.random.uniform(0, 5)))
            
#         occ = occupation(i, [], [], list(A[i] > 0), list(A[i]),
#                          (employment[i,0] + vacancies[i,0]), 
#                          demand_target[i,0], wages[i], wage_mu[i], wage_sigma[i], 0, 0, entry_level[i], experience_age[i], sep_rates[i], np.nan)
        
    
#         # creating the workers of occupation i and attaching to occupation
#         ## adding employed workers
#         g_share = gend_share[i,0]
#         for e in range(round(employment[i,0])):
#             # Assume they have all at least 1 t.s. of employment
#             if np.random.rand() <= g_share:
#                 occ.list_of_employed.append(worker(occ.occupation_id, False, 1, 
#                                                    np.clip(np.random.lognormal(wage_mu[i], wage_sigma[i]), 15080, 250000),
#                                                    #wages[i], 
#                                                    False, True, np.random.uniform(low=experience_age[i], high = 65),
#                                                abs(int(np.random.normal(fem_ra,2))), 1, 1,0,0, None))
#             else:
#                 occ.list_of_employed.append(worker(occ.occupation_id, False, 1, 
#                                                    np.clip(np.random.lognormal(wage_mu[i], wage_sigma[i]), 15080, 250000),
#                                                    #wages[i], 
#                                                    False, False, np.random.uniform(experience_age[i], high = 65),
#                                                abs(int(np.random.normal(male_ra,2))), 1, 1,0,0,None))
#             ## adding unemployed workers
#         for u in range(round(unemployment[i,0])):
#             if np.random.rand() <= g_share:
#                 # Assigns time unemployed from absolute value of normal distribution....
#                 occ.list_of_unemployed.append(worker(occ.occupation_id, False, max(1,(int(np.random.normal(2,2)))), 
#                                                      np.clip(np.random.lognormal(wage_mu[i], wage_sigma[i]), 15080, 250000),
#                                                      #np.random.normal(wages[i], 0.05* wages[i]), 
#                                                      False, True, 
#                                                      np.random.uniform(experience_age[i], high = 65), 
#                                                      abs(int(np.random.normal(fem_ra,0.1))), 1, 1, 1, 0, None))
#             else:
#                 # Assigns time unemployed from absolute value of normal distribution....
#                 occ.list_of_unemployed.append(worker(occ.occupation_id, False, max(1,(int(np.random.normal(2,2)))), 
#                                                      np.clip(np.random.lognormal(wage_mu[i], wage_sigma[i]), 15080, 250000),
#                                                     # np.random.normal(wages[i], 0.05* wages[i]), 
#                                                      False, False, 
#                                                      np.random.uniform(experience_age[i], high = 65), 
#                                                      abs(int(np.random.normal(male_ra,0.1))), 1, 1, 1, 0, None))


#         occs.append(occ)
#         ids += 1
#     return occs, vac_list

### Function and condition to initialise network
def initialise(n_occ, employment, unemployment, vacancies, demand_target, A, wages, wage_mu, wage_sigma, gend_share, fem_ra, male_ra, entry_level, experience_age, sep_rates):
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
            # Draw a random wage
            w = np.random.lognormal(wage_mu[i], wage_sigma[i])

            # Compute 25th and 75th percentiles
            z25 = norm.ppf(0.25)
            z75 = norm.ppf(0.75)
            w25 = np.exp(wage_mu[i] + wage_sigma[i] * z25)
            w75 = np.exp(wage_mu[i] + wage_sigma[i] * z75)

            # Clip to IQR (25th–75th)
            w_clipped = w#np.clip(w, w25, w75)
            vac_list.append(vac(i, [], #wages[i]
                                w_clipped, False, np.random.uniform(0, 5)))
            
        occ = occupation(i, [], [], list(A[i] > 0), list(A[i]),
                         (employment[i,0] + vacancies[i,0]), 
                         demand_target[i,0], wages[i], wage_mu[i], wage_sigma[i], 0, 0, entry_level[i], experience_age[i], sep_rates[i], np.nan)
        
    
        # creating the workers of occupation i and attaching to occupation
        ## adding employed workers
        g_share = gend_share[i,0]
        for e in range(round(employment[i,0])):
            # Draw a random wage
            w = np.random.lognormal(wage_mu[i], wage_sigma[i])

            # Compute 25th and 75th percentiles
            z25 = norm.ppf(0.25)
            z75 = norm.ppf(0.75)
            w25 = np.exp(wage_mu[i] + wage_sigma[i] * z25)
            w75 = np.exp(wage_mu[i] + wage_sigma[i] * z75)

            # Clip to IQR (25th–75th)
            w_clipped = w #np.clip(w, w25, w75)
            # Assume they have all at least 1 t.s. of employment
            if np.random.rand() <= g_share:
                occ.list_of_employed.append(worker(occ.occupation_id, False, 1, 
                                                   w_clipped,
                                                   #np.clip(np.random.lognormal(wage_mu[i], wage_sigma[i]), 15080, 250000),
                                                   #wages[i], 
                                                   False, True, np.random.uniform(low=experience_age[i], high = 65),
                                               abs(int(np.random.normal(fem_ra,2))), 1, 1,0,0, None))
            else:
                occ.list_of_employed.append(worker(occ.occupation_id, False, 1, 
                                                   w_clipped,
                                                   # np.clip(np.random.lognormal(wage_mu[i], wage_sigma[i]), 15080, 250000).
                                                   #wages[i], 
                                                   False, False, np.random.uniform(experience_age[i], high = 65),
                                               abs(int(np.random.normal(male_ra,2))), 1, 1,0,0,None))
            ## adding unemployed workers
        for u in range(round(unemployment[i,0])):
            # Draw a random wage
            w = np.random.lognormal(wage_mu[i], wage_sigma[i])

            # Compute 25th and 75th percentiles
            z25 = norm.ppf(0.25)
            z75 = norm.ppf(0.75)
            w25 = np.exp(wage_mu[i] + wage_sigma[i] * z25)
            w75 = np.exp(wage_mu[i] + wage_sigma[i] * z75)

            # Clip to IQR (25th–75th)
            w_clipped = w #np.clip(w, w25, w75)
            if np.random.rand() <= g_share:
                # Assigns time unemployed from absolute value of normal distribution....
                occ.list_of_unemployed.append(worker(occ.occupation_id, False, max(1,(int(np.random.normal(2,2)))), 
                                                     w_clipped,
                                                     #np.clip(np.random.lognormal(wage_mu[i], wage_sigma[i]), 15080, 250000),
                                                     #np.random.normal(wages[i], 0.05* wages[i]), 
                                                     False, True, 
                                                     np.random.uniform(experience_age[i], high = 65), 
                                                     abs(int(np.random.normal(fem_ra,0.1))), 1, 1, 1, 0, None))
            else:
                # Assigns time unemployed from absolute value of normal distribution....
                occ.list_of_unemployed.append(worker(occ.occupation_id, False, max(1,(int(np.random.normal(2,2)))), 
                                                     w_clipped,
                                                     #np.clip(np.random.lognormal(wage_mu[i], wage_sigma[i]), 15080, 250000),
                                                    # np.random.normal(wages[i], 0.05* wages[i]), 
                                                     False, False, 
                                                     np.random.uniform(experience_age[i], high = 65), 
                                                     abs(int(np.random.normal(male_ra,0.1))), 1, 1, 1, 0, None))


        occs.append(occ)
        ids += 1
    return occs, vac_list

def _sigmoid(x):
    if x < -709:  # Prevent overflow in exp for large negative x
        return 0.0
    elif x > 709:  # Prevent overflow in exp for large positive x
        return 1.0
    else:
        return 1.0 / (1.0 + np.exp(-x))

def p_search_logit(age, comp, *, alpha=0.0, beta_A=0.05, beta_C=-0.8, beta_CA=0.0, A0=40.0):
    """
    Logit p = alpha + beta_A*(age-A0) + beta_C*comp + beta_CA*comp*(age-A0)
    No cycle term, no occ_shock.
    """
    logit_val = (alpha
                 + beta_A * (age - A0)
                 - beta_C * comp)
    return _sigmoid(logit_val)
