######## Central Run Function #############

import numpy as np
import pandas as pd
from abm_funs import *
import random as random
from copy import deepcopy 
import math as math
from scipy.stats import norm
from collections import deque

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
                    simple_res = False,
                    vac_method = "g_prime_norm",
                    vac_wage_update = True,
                    entry_alloc_scalar = 1.0,
                    jg_frac = 0.0,
                    jg_wage = "entry_anchor",
                    ltu_hire_bias = 0.0,
                    entry_hire_bias = 0.0):

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
    if not simple_res:
        _next_wid = 0
        for _occ in net:
            for _w in _occ.list_of_employed + _occ.list_of_unemployed:
                _w.worker_id = _next_wid
                _next_wid += 1
    # occ_shocks_data has shape (n_occs, 24 + delay + calib_length).
    # Cols 0..23 are used to pre-fill the deque for the new vacancy rule.
    # The loop runs over the remaining delay+calib_length columns, so we
    # subtract 24 here and index occ_shocks_data as 24+t throughout.
    # Results are recorded from t >= delay in both vacancy rule modes.
    time_steps = time_steps - 24
    avg_wage_offer_diff_records = []
    app_load_records = []
    retired_all = 0
    ue_spell_records = []
    transition_records = []   # rich per-event log: UE + EE, with wages + demographics
    ue_stock_records = []
    wage_age_records = []
    gender_median_records = []
    d_initial = 0.032

    endog_vac_prob = (vac_method != "exog")
    if not endog_vac_prob:
        gamma_v = np.nan
        d_v = np.nan
    # Initialise 24-month demand history for integrated YoY vacancy signal.
    # For new vacancy rule: pre-fill from cols 0..23 of occ_shocks_data so
    # g_prime has a genuine historical baseline from the first loop step.
    # For old vacancy rule: flat initialisation (deque is unused).
    for occ in net:
        td_base = float(occ.target_demand)
        if endog_vac_prob and occ_shocks_data is not None and occ_shocks_data.shape[1] >= 24:
            history = [td_base * float(occ_shocks_data[occ.occupation_id, k])
                       for k in range(24)]
        else:
            history = [td_base] * 24
        occ.td_history = deque(history, maxlen=24)
        occ.target_vr = d_initial if endog_vac_prob else np.nan
        occ.g_norm_ema = 0.0  # AR(1) running mean of g_prime_norm, used for demeaning

    for t in range(time_steps-1):
        # occ_shocks_data is indexed at 24+t to skip the deque pre-fill columns.
        # This is consistent for both endog_vac_prob=True and endog_vac_prob=False.
        if steady_state:
            curr_bus_cy = 1
            vr_t = 0.04
        else:
            if occ_shocks_data is not None:
                curr_bus_cy = np.take(occ_shocks_data, indices=24 + t, axis=1)
            vr_t = vac_data[t - delay] if t >= delay else 0.04

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
            occ.failed_hires = 0
            if isinstance(curr_bus_cy, np.ndarray):
                occ_shock = curr_bus_cy[occ.occupation_id]
            else:
                occ_shock = curr_bus_cy

            # Entry 0.1%
            # occ.entry_rate(0.01)
            # Entry by partitioning retired demand according to target demand across entry level occupations
            if occ.entry_level:
                n_total = entry_alloc.get(occ.occupation_id, 0)
                if jg_frac > 0.0 and n_total > 0:
                    n_jg  = round(jg_frac * n_total)
                    n_reg = n_total - n_jg
                    if n_jg  > 0:
                        occ.entry(n_jg,  enter_emp=True,  use_wage_dist=True, jg_wage=jg_wage, entry_time_step=t+1)
                    if n_reg > 0:
                        occ.entry(n_reg, enter_emp=False, use_wage_dist=True, entry_time_step=t+1)
                else:
                    occ.entry(n_total, enter_emp=False, use_wage_dist=True, entry_time_step=t+1)
            # Remove the top 2% of earners in an occupation's employed list
            # occ.entry_and_exit_fixed(0.02)

            ### APPLICATIONS
            # isolate list of vacancies in economy that are relevant to the occupation
            # - avoids selecting in each search_and_apply application
            r_vacs = vacancies_by_occ  
            
            unemp_seekers += len(occ.list_of_unemployed)
            for u in occ.list_of_unemployed:
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
                    if occ.competition_last > 1:
                        raise KeyError("competition value > 1")
                    # beta_A=0 so age has no effect; s_prob is identical for every worker
                    s_prob = p_search_logit(
                            #age=0,
                            comp=occ.competition_last,
                            alpha=-4,
                            #beta_A=0,
                            beta_C=theta
                            #A0=40.0
                        )
                    # Draw all Bernoulli outcomes at once instead of one random call per worker
                    n_emp = len(occ.list_of_employed)
                    if n_emp > 0:
                        searching_idx = np.where(np.random.random(n_emp) < s_prob)[0]
                        emp_seekers += len(searching_idx)
                        for idx in searching_idx:
                            e_apps += occ.list_of_employed[idx].emp_search_and_apply(net, r_vacs, disc, emp_apps, wage_prefs, mistake_rate, unique_random = False, global_pool_override = vacs_temp)
                else:
                    for e in random.sample(occ.list_of_employed, int(0.03*len(occ.list_of_employed))):
                        emp_seekers += 1
                        e_apps += e.emp_search_and_apply(net, r_vacs, disc, emp_apps, wage_prefs, mistake_rate, unique_random = False, global_pool_override = vacs_temp)

            if not simple_res:
                u_apps += sum(wrkr.apps_sent for wrkr in occ.list_of_unemployed if wrkr.apps_sent is not None)
            if not simple_res:
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
            
                    
        if not simple_res:
            seekers_rec.append([t+1, unemp_seekers, u_apps, emp_seekers, e_apps])
        # Compute competition metric directly from vacancies_by_occ (no DataFrame needed)
        if vacs_temp:
            comp_map = {
                occ_id: np.mean([1.0 / len(v.applicants) if v.applicants else 1.0 for v in vacs])
                for occ_id, vacs in vacancies_by_occ.items()
            }
            for occ in net:
                val = comp_map.get(occ.occupation_id, np.nan)
                if np.isnan(val):
                    val = getattr(occ, "competition_last", 0.0)
                occ.competition_last = val

            if not simple_res:
                app_load_rows = [
                    (t+1, v.occupation_id,
                     len(v.applicants),
                     1.0 / len(v.applicants) if len(v.applicants) > 0 else 1.0,
                     len([app for app in v.applicants if app.time_unemployed != 0]),
                     len([app for app in v.applicants if app.time_unemployed == 0]))
                    for v in vacs_temp
                ]
                app_load_step = pd.DataFrame(app_load_rows, columns=['Time Step', 'Occupation', 'ApplicantsPerVac', 'VacsPerApp', 'UnempApplicantsPerVac', 'EmpApplicantsPerVac'])
                app_load_agg = app_load_step.groupby(['Time Step', 'Occupation']).agg(
                    OpenVacs=('ApplicantsPerVac', 'size'),
                    TotalApplicants=('ApplicantsPerVac', 'sum'),
                    MeanAppsPerVac=('ApplicantsPerVac', 'mean'),
                    MedianAppsPerVac=('ApplicantsPerVac', 'median'),
                    TotalUnempAppsPerVac=('UnempApplicantsPerVac', 'sum'),
                    MeanVacsPerApp=('VacsPerApp', 'mean')
                ).reset_index()
                app_load_records.append(app_load_agg)

        # # Collect application flow per occupation
        # app_load_rows = [
        #     (t+1, v.occupation_id, 
        #      # All applicants
        #      len(v.applicants), 
        #     1.0 / len(v.applicants) if len(v.applicants) > 0 else 1.0,  # ADD THIS: VacsPerApp for each vacancy
        #      # Only unemployed applicants
        #     len([app for app in v.applicants if app.time_unemployed != 0]), 
        #     # Only employed applicants
        #     len([app for app in v.applicants if app.time_unemployed == 0]))
        #     for v in vacs_temp
        # ]
        # if app_load_rows:  # only if there are vacancies
        #     app_load_step = pd.DataFrame(app_load_rows, columns=['Time Step', 'Occupation', 'ApplicantsPerVac', 'VacsPerApp', 'UnempApplicantsPerVac', 'EmpApplicantsPerVac'])
        #     app_load_agg = app_load_step.groupby(['Time Step', 'Occupation']).agg(
        #         OpenVacs=('ApplicantsPerVac', 'size'),
        #         TotalApplicants=('ApplicantsPerVac', 'sum'),
        #         MeanAppsPerVac=('ApplicantsPerVac', 'mean'),
        #         MedianAppsPerVac=('ApplicantsPerVac', 'median'),
        #         TotalUnempAppsPerVac=('UnempApplicantsPerVac', 'sum'),
        #         MeanVacsPerApp=('VacsPerApp', 'mean')  # ADD THIS: Mean of 1/applicants

        #     ).reset_index()
        #     # app_load_agg['MeanUnempAppsPerVac'] = app_load_agg['TotalUnempAppsPerVac'] / app_load_agg['TotalApplicants']

        #     # # Find level of competition in neighboring occupations in the network
        #     # app_load_agg['AppsPerVac'] = app_load_agg.apply(
        #     #     lambda r: (r['TotalApplicants'] / r['OpenVacs'])
        #     #             if r['OpenVacs'] > 0 else np.nan,
        #     #     axis=1
        #     # )
        #     # app_load_agg['VacsPerApp'] = app_load_agg.apply(
        #     #     lambda r: (r['OpenVacs'] / r['TotalApplicants'])
        #     #             if r['TotalApplicants'] > 0 else 0.99,
        #     #     axis=1)
            
        #     comp_map = dict(zip(app_load_agg['Occupation'], app_load_agg['MeanVacsPerApp']))

        #     # Assign to occupations for use in next step
        #     for occ in net:
        #         val = comp_map.get(occ.occupation_id, np.nan)
        #         if np.isnan(val):
        #             # fallback if no vacancies: keep last known
        #             val = getattr(occ, "competition_last", 0.0)
        #         occ.competition_last = val

        #     app_load_records.append(app_load_agg)
    
        ### HIRING
        # Ordering of hiring randomised to ensure list order does not matter in filling vacancies...
        random.shuffle(vacs_temp)
        for v_open in vacs_temp:
            if dumb_hire:
                # Removes any applicants that have already been hired in another vacancy
                v_open.applicants[:] = [app for app in v_open.applicants if not(app.hired)]
            
            v_open.time_open += 1
            if len(v_open.applicants) > 0:
                ret = v_open.hire(net, dumb_hire, ltu_hire_bias, entry_hire_bias)
                # If hire() succeeded (didn't return None on first line)
                if ret is not None:
                    # Hire was successful
                    v_open.filled = True

                    trans_type, origin_occ, dest_occ, ue_dur, pre_wage, new_wage, age, female, longterm, new_entrant = ret

                    if not simple_res:
                        # Keep the legacy UE-only spell record for backward compatibility
                        if trans_type == 'UE':
                            ue_spell_records.append((t+1, origin_occ, dest_occ, ue_dur))

                        # Rich per-event log (UE + EE)
                        transition_records.append((t+1, trans_type, origin_occ, dest_occ,
                                                   ue_dur, pre_wage, new_wage, age, female, longterm, new_entrant))
                else:
                    # Vacancy had applicants this period but none could be hired
                    # (its sampled/available applicants were already hired by a
                    # competing vacancy earlier in this period's shuffle).
                    net[v_open.occupation_id].failed_hires += 1
            assert(len(v_open.applicants) == 0)


        # Close a vacancy after it's been open for 6 months
        vacs_temp = [v for v in vacs_temp if not(v.filled) and v.time_open < 6]
        if vac_wage_update:
            for v in vacs_temp:
                v.wage *= 1.01

        # Build a count lookup once (O(n_vacs)) so curr_vacs below is O(1) per occupation
        vac_count_by_occ = defaultdict(int)
        for v in vacs_temp:
            vac_count_by_occ[v.occupation_id] += 1

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
        if not simple_res:
            _f_wages_t = []
            _m_wages_t = []
        for occ in net:
            if isinstance(curr_bus_cy, np.ndarray):
                occ_shock = curr_bus_cy[occ.occupation_id]
            else:
                occ_shock = curr_bus_cy
            # Only consider unemployed workers who have d_wage_offer attribute
            if not simple_res:
                u_rel_wage = sum(wrkr.ue_rel_wage for wrkr in occ.list_of_employed if wrkr.hired and wrkr.ue_rel_wage is not None)
                e_rel_wage = sum(wrkr.ee_rel_wage for wrkr in occ.list_of_employed if wrkr.hired and wrkr.ee_rel_wage is not None)
                ue = len([w for w in occ.list_of_employed if w.hired and w.ue_rel_wage is not None])
                ee = len([w for w in occ.list_of_employed if w.hired and w.ee_rel_wage is not None])
            # Update time_unemployed and long-term unemployed status of unemployed workers
            # Remove protected "hired" attribute of employed workers
            occ.update_workers()
            if not simple_res:
                for wrkr in occ.list_of_unemployed:
                    ue_stock_records.append((t+1, occ.occupation_id, wrkr.time_unemployed, wrkr.longterm_unemp))
            retired = occ.retire_workers()
            retired_all += retired
            # Assert that all unemployed people have spent 1 or more time periods unemployed
            assert(sum([worker.time_unemployed <= 0 for worker in occ.list_of_unemployed]) == 0)
            # Assert that all employed people have spent 0 time periods unemployed
            assert(sum([worker.time_unemployed <= 0 for worker in occ.list_of_employed]) == len(occ.list_of_employed))
            emp = len(occ.list_of_employed)
            curr_vacs = vac_count_by_occ[occ.occupation_id]
            occ.current_demand = (curr_vacs + emp)
            # If real-world vacancy rate is greater than the current vacancy rate, then we create new vacancies
            if not endog_vac_prob:
                vac_prob = max(0, vr_t - (curr_vacs/(occ.target_demand + 1)))
                # #vac_prob = d_v + (1-d_v)*((gamma_v * max(0, occ.target_demand*(occ_shock) - occ.current_demand)) / (emp + 1))
                vacs_create = int(np.random.binomial(occ.target_demand, np.clip(vac_prob, 0, 1)))
            else:
                # ---------------------------------------------------------------
                # Integrated year-over-year vacancy signal  (g'_t)
                #
                # Signal:  g'_t = (1/12) * Σ_{k=0}^{11} [td_{t-k} - td_{t-k-12}]
                #
                # After appending td_now the deque layout is:
                #   index  0  = td_{t-23}  …  index 11 = td_{t-12}
                #   index 12  = td_{t-11}  …  index 23 = td_{t}
                #
                # So each YoY pair is:  deque[23-k] - deque[11-k]  for k=0..11
                #   k=0  → td_{t}    - td_{t-12}
                #   k=11 → td_{t-11} - td_{t-23}
                #
                # Forecast:  projected_td = td_now + g'_t
                # Vacancies: vacs_gap = projected_td - current_demand
                #
                # This opens vacancies to close the gap between where demand is
                # heading next period and current headcount (employed + open vacs).
                # ---------------------------------------------------------------
                td_now = float(occ.target_demand * occ_shock)

                # Push current period into history (oldest entry is automatically dropped)
                occ.td_history.append(td_now)

                # Integrated YoY growth (absolute, same units as headcount)
                # g_prime = (1.0 / 12.0) * sum(
                #     occ.td_history[23 - k] - occ.td_history[11 - k]
                #     for k in range(12)
                # )
                g_prime_norm = (1.0 / 12.0) * sum(
                        (occ.td_history[23 - k] - occ.td_history[11 - k]) / (occ.td_history[11 - k] + 1e-6)
                        for k in range(12)
                    )                
                
                # We apply a linear weighting so that more recent demand changes affect behaviour more
                weight_denom = 78 # hard-code 78 to avoid recalculating the sum at each step sum(range(1, 13))  # normalisation factor so that weights sum to 1
                weighted_g_prime_norm = sum((12 - k) / 78 * (occ.td_history[23 - k] - occ.td_history[11 - k]) / (occ.td_history[11 - k] + 1e-6) 
                                            for k in range(12))


                # Because we are opening the vacancies to be filled in t+1 we therefore
                # project next period's demand and open vacancies to close the gap
                # Ebba changed this to td_now
                
                #projected_td = td_now + gamma_v*g_prime

                # # projected_td = float(occ.target_demand) + gamma_v*g_prime
                if t == delay:
                    occ.target_vr = 0.032
                if t >= delay:
                    if vac_method == "g_prime_norm_weighted":
                        signal = weighted_g_prime_norm
                    else:
                        signal = g_prime_norm

                    if vac_method == "cost_red":
                        # d_v is the geometric growth rate of gamma_v
                        gamma_v_t = np.clip(gamma_v * (1 + d_v) ** (t - delay), 0, 1)
                        occ.target_vr += gamma_v_t * signal
                    else:
                        # g_prime_norm / g_prime_norm_weighted: d_v is secular drift (0 = no drift)
                        occ.target_vr += d_v + gamma_v * signal

                target_vr = max(0.0, occ.target_vr)
                curr_vr = curr_vacs / (td_now + 1)
                vac_prob = max(0.0, target_vr - curr_vr)
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
                wage_median_occ = np.median([wrkr.wage for wrkr in occ.list_of_employed]) if emp > 0 else np.nan
                wages_occ = sum(wrkr.wage for wrkr in occ.list_of_employed)
                seps = occ.separated + retired
                retirees = retired
                hires = occ.hired
                failed_hires = occ.failed_hires
                entry_level_entrants = entry_alloc.get(occ.occupation_id, 0) if occ.entry_level else 0
                female_wages = sum(wrkr.wage for wrkr in occ.list_of_employed if wrkr.female)
                male_wages   = sum(wrkr.wage for wrkr in occ.list_of_employed if not wrkr.female)
                female_empl  = sum(1 for wrkr in occ.list_of_employed if wrkr.female)
                male_empl    = sum(1 for wrkr in occ.list_of_employed if not wrkr.female)
                _f_wages_t.extend(wrkr.wage for wrkr in occ.list_of_employed if wrkr.female)
                _m_wages_t.extend(wrkr.wage for wrkr in occ.list_of_employed if not wrkr.female)

                ### UPDATE INDICATOR RECORD
                record.append([t+1, occ.occupation_id, empl, unemp, empl + unemp, vacs_occ, n_ltue, curr_demand, t_demand, emp_seekers, unemp_seekers, wages_occ, u_rel_wage, e_rel_wage, ue, ee, seps, hires, failed_hires, vacs_wage, wage_occ, retirees, entry_level_entrants, female_wages, male_wages, female_empl, male_empl])

                # Annual snapshot of individual worker wages and ages (every 12 steps after delay)
                if t >= delay and (t - delay) % 12 == 0:
                    yr = (t - delay) // 12
                    for wrkr in occ.list_of_employed:
                        if not hasattr(wrkr, 'worker_id'):
                            wrkr.worker_id = _next_wid
                            _next_wid += 1
                        wage_age_records.append((t+1, yr, occ.occupation_id, wrkr.worker_id, wrkr.wage, wrkr.age,
                                                 getattr(wrkr, 'entry_time_step', None)))

            # if occ.occupation_id == 27 and endog_vac_prob:
            #     print(f"t={t} occ={occ.occupation_id} td_base={occ.target_demand:.0f} "
            #         f"g_norm={g_prime_norm:.4f} vacs_created={vacs_create:.0f} "
            #         f"curr_demand={occ.current_demand:.0f} target_vr = {occ.target_vr:.4f}")

        if simple_res:
            ### UPDATE INDICATOR RECORD
            record = np.append(record, 
                        np.array([[t+1, empl, unemp, empl + unemp, len(vacs_temp), n_ltue, t_demand, seps, emp_seekers, unemp_seekers]]), axis=0)

        else:
            record_temp_df = pd.DataFrame(record, columns=['Time Step', 'Occupation', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Current_Demand', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers', 'Total_Wages', 'U_Rel_Wage', 'E_Rel_Wage', 'UE_Transitions', 'EE_Transitions', "Separations", "Hires", "Failed_Hires", "Mean Vacancy Offer", "Mean Occupational Wage", "Retirees", "Entry_Level_Hires", "Female_Total_Wages", "Male_Total_Wages", "Female_Employment", "Male_Employment"])
            record_df = record_temp_df[record_temp_df['Time Step'] >= delay]
            grouped = record_df.groupby('Time Step').sum().reset_index()

            grouped['UER'] = grouped['Unemployment'] / grouped['Workers']
            grouped['Mean_Economy_Wage'] = grouped['Total_Wages'] / grouped['Employment']
            grouped['U_REL_WAGE_MEAN'] = grouped['U_Rel_Wage'] / grouped['UE_Transitions']
            grouped['E_REL_WAGE_MEAN'] = grouped['E_Rel_Wage'] / grouped['EE_Transitions']
            grouped['UE_Trans_Rate'] = grouped['UE_Transitions'] / grouped['Workers']
            grouped['EE_Trans_Rate'] = grouped['EE_Transitions'] / grouped['Workers']
            grouped['VACRATE'] = grouped['Vacancies'] / (grouped['Vacancies'] + grouped['Employment'])
            grouped['Hires Rate'] = grouped['Hires'] / grouped['Employment']
            grouped['Separations Rate'] = (grouped['Separations'] + grouped['EE_Transitions']) / grouped['Employment']
            grouped['E-U Rate'] = grouped['Separations'] / grouped['Employment']
            grouped['Gender_Pay_Gap'] = (
                1 - (grouped['Female_Total_Wages'] / grouped['Female_Employment'])
                  / (grouped['Male_Total_Wages']   / grouped['Male_Employment'])
            )
            gender_median_records.append({
                'Time Step': t + 1,
                'Female_Median_Wage': np.median(_f_wages_t) if _f_wages_t else np.nan,
                'Male_Median_Wage':   np.median(_m_wages_t) if _m_wages_t else np.nan,
            })

            data = {'UER': grouped['UER'], 'VACRATE': grouped['VACRATE'], }#, 'SEPSRATE': grouped['Separations Rate']}
            avg_wage_offer_diff_df = pd.DataFrame(avg_wage_offer_diff_records)

            if app_load_records:
                app_load_df = pd.concat(app_load_records, ignore_index=True)
            else:
                app_load_df = pd.DataFrame(columns=['Time Step','Occupation','OpenVacs','TotalApplicants','MeanAppsPerVac','MedianAppsPerVac'])

            ue_spell_origin_df = pd.DataFrame(ue_spell_records, columns=['Time Step', 'OriginOccupation', 'DestinationOccupation', 'UEDuration'])
            ue_stock_df = pd.DataFrame(ue_stock_records, columns=['Time Step', 'Occupation', 'UEDuration', 'LongtermUnemp'])
            transition_df = pd.DataFrame(transition_records, columns=[
                'Time Step', 'TransitionType', 'OriginOccupation', 'DestinationOccupation',
                'UEDuration', 'PreSepWage', 'AcceptedWage', 'Age', 'Female', 'LongtermUnemp',
                'NewEntrant'])
            wage_age_df = pd.DataFrame(wage_age_records, columns=['Time Step', 'Year', 'Occupation', 'Worker_ID', 'Wage', 'Age', 'Entry_Time_Step'])

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
        gender_median_df = pd.DataFrame(gender_median_records)
        gender_median_df = gender_median_df[gender_median_df['Time Step'] >= delay]
        grouped = grouped.merge(gender_median_df, on='Time Step', how='left')
        grouped['Gender_Median_Pay_Gap'] = (
            1 - grouped['Female_Median_Wage'] / grouped['Male_Median_Wage']
        )
        return record_df, grouped, net, data, seekers_rec, avg_wage_offer_diff_df, app_load_df, ue_spell_origin_df, vacs_temp, wage_age_df, ue_stock_df, transition_df

