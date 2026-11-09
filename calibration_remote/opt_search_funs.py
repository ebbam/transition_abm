"""
opt_search_funs.py

Optimizing-search benchmark track (referee response), kept apart from the data-driven model:
  - marginal-benefit application rules for unemployed workers ("sticky", "bayes", "bayes_month",
    "fixed", "rational"; see job_search_notebook.ipynb / "Formalizing the Behavioral Mechanisms")
  - the central-planner benchmark ("planner")
  - opt_search_and_apply: worker search with these rules (the data-driven worker.search_and_apply in
    abm_funs.py plus the optimizing branch and search diagnostics)
Used by model_fun_opt.run_single_opt. The data-driven model (abm_funs.py, model_fun.run_single_local)
does not import this module.
"""

import math
import random
import heapq

import numpy as np

from abm_funs import reservation_wage, res_wage_dat, applications_sent, util


#### Optimizing search benchmark (unemployed workers) ####
# Replaces the empirical applications_sent() draw with the marginal-benefit rule
# from job_search_notebook.ipynb / "Formalizing the Behavioral Mechanisms":
#   p_t     = beta_t * p_bar                     (subjective; no market-tightness signal)
#   dV_a    = (w_a - R_tau) * rho_tilde_a        (surplus over the reservation wage)
#   MB_a    = (1 - p_t)^(a-1) * p_t * dV_a       (vacancies ranked by dV, descending)
#   A*      = max{a <= A_bar : MB_a >= c},  c = kappa * w_i,  floored at a_min
# Two belief rules, both driven only by the worker's own failed applications:
#   "sticky": eq. (10) with h = 0, one update per month of unemployment
#   "bayes":  Beta-Bernoulli posterior mean after n failed applications this spell
#   "bayes_month": same, but one failure per month unemployed (N0 then in pseudo-months)
# Non-behavioural benchmarks (usually with use_res_wage=False):
#   "fixed":    beliefs never erode;  "rational": full information on last period's competition

OPT_SEARCH_DEFAULTS = {
    'p_bar': 0.10,   # subjective per-application success probability at full confidence
    'kappa': 0.005,  # per-application cost as a fraction of the worker's last wage
    'beta0': 1.0,    # "sticky": initial confidence
    'omega': 0.10,   # "sticky": decay of the learning rate alpha_tau = exp(-omega*tau)
    'r': 0.065,      # "sticky": reference outcome; erosion per failed month
    'N0': 20.0,      # "bayes": prior strength, in pseudo-applications
    'A_bar': 30,     # application ceiling (matches MAX_VACS sampled vacancies)
    'a_min': 1,      # minimum applications per month (UI search requirement)
    'use_res_wage': True,  # False: no reservation-wage filter, surplus measured from zero
    'planner_objective': 'surplus',  # "planner" rule: 'matches' (every feasible pair weight 1) or 'surplus'
}

def belief_sticky(tau, beta0, omega, r):
    """Eq. (10) iterated from beta0 with h = 0 for tau months:
    beta_tau = clip(beta0 - r * sum_{s=1..tau} exp(-omega*s), 0, 1)."""
    if tau <= 0:
        return float(np.clip(beta0, 0.0, 1.0))
    q = math.exp(-omega)
    geom = q * (1.0 - q**tau) / (1.0 - q) if q < 1.0 else float(tau)
    return float(np.clip(beta0 - r * geom, 0.0, 1.0))

def belief_bayes(n_fail, N0):
    """Posterior mean of per-application success relative to the prior mean
    after n_fail failures under a Beta prior of strength N0: N0 / (N0 + n_fail)."""
    return N0 / (N0 + n_fail)

def optimal_apps(p, dV_ranked, c, a_min=1, A_bar=None):
    """A* for a list ranked best-first, capped at A_bar and floored at min(a_min, len(dV_ranked)).
    Scalar p (common success probability, list ranked by dV): A* = max{a : (1-p)^(a-1) p dV_a >= c}.
    Vector p (one probability per ranked vacancy): MB_a = prod_{k<a}(1-p_k) p_a dV_a, and the worker
    keeps applying while MB_a >= c (stops at the first application that is not worth its cost)."""
    dV = np.asarray(dV_ranked, dtype=float)
    if A_bar is not None:
        dV = dV[:A_bar]
    n = len(dV)
    if n == 0:
        return 0
    if np.ndim(p) == 0:
        mb = (1.0 - p) ** np.arange(n) * p * dV
        ok = np.nonzero(mb >= c)[0]
        a_star = int(ok[-1]) + 1 if ok.size else 0
    else:
        pv = np.asarray(p, dtype=float)[:n]
        survive = np.concatenate(([1.0], np.cumprod(1.0 - pv)[:-1]))
        fail = np.nonzero(survive * pv * dV < c)[0]
        a_star = int(fail[0]) if fail.size else n
    return max(a_star, min(a_min, n))

def rho_tilde_row(neigh_weights, own_id):
    """Within-row percentile rescaling of occupational similarity.
    Neighbours (weight > 0, excluding own) get rank/n in (0, 1] (ties share the
    upper rank); the worker's own occupation is set to 1."""
    w = np.asarray(neigh_weights, dtype=float)
    out = np.zeros_like(w)
    mask = w > 0
    mask[own_id] = False
    n = int(mask.sum())
    if n > 0:
        vals = w[mask]
        # rank with ties -> max rank: count of neighbour weights <= each value
        out[mask] = np.searchsorted(np.sort(vals), vals, side='right') / n
    out[own_id] = 1.0
    return out

def _get_rho_tilde(occ):
    # Computed lazily and cached so existing (pickled/warm-started) networks work unchanged
    rt = getattr(occ, 'rho_tilde', None)
    if rt is None:
        rt = rho_tilde_row(occ.list_of_neigh_weights, occ.occupation_id)
        occ.rho_tilde = rt
    return rt

def optimal_application_set(wrkr, candidates, rho_tilde, res_wage, rule, prm, net=None):
    """Rank candidate vacancies by expected surplus and choose the top A*.
    Returns (chosen_vacancies, A_star, belief).

    res_wage=None means no reservation wage: candidates are all sampled vacancies and the
    surplus is measured from zero (dV_a = w_a * rho_a); otherwise dV_a = (w_a - R) * rho_a.
    Rules:
      "sticky" / "bayes" / "bayes_month": subjective beliefs eroded by own failures (see above)
      "fixed":    non-behavioural benchmark, beliefs never erode (p = beta0 * p_bar)
      "rational": non-behavioural benchmark, full information: p for each vacancy is last
                  period's mean chance of being picked in its occupation (competition_last),
                  so this rule DOES use market tightness (a deliberate contrast)
    """
    if rule == 'sticky':
        belief = belief_sticky(max(wrkr.time_unemployed - 1, 0),
                               prm['beta0'], prm['omega'], prm['r'])
    elif rule == 'bayes':
        belief = belief_bayes(getattr(wrkr, 'n_fail', 0), prm['N0'])
    elif rule == 'bayes_month':
        # One failure per month still unemployed, regardless of applications sent
        belief = belief_bayes(max(wrkr.time_unemployed - 1, 0), prm['N0'])
    elif rule == 'fixed':
        belief = prm['beta0']
    elif rule == 'rational':
        belief = np.nan
    else:
        raise ValueError(f"Unknown u_search_rule '{rule}'")

    if not candidates:
        return [], 0, belief
    if res_wage is None:
        dV = np.array([v.wage * rho_tilde[v.occupation_id] for v in candidates])
    else:
        dV = np.array([(v.wage - res_wage) * rho_tilde[v.occupation_id] for v in candidates])

    if rule == 'rational':
        p_all = np.array([getattr(net[v.occupation_id], 'competition_last', np.nan) for v in candidates],
                         dtype=float)
        p_all = np.where(np.isfinite(p_all) & (p_all > 0), np.clip(p_all, 0.0, 1.0), prm['p_bar'])
        order = np.argsort(p_all * dV)[::-1]          # highest expected value first
        p = p_all[order]
    else:
        p = belief * prm['p_bar']
        order = np.argsort(dV)[::-1]
    a_star = optimal_apps(p, dV[order], prm['kappa'] * wrkr.wage,
                          a_min=prm['a_min'], A_bar=prm['A_bar'])
    # Perceived probability that all a_star applications fail (read by model_fun_opt's duration record)
    wrkr.perceived_fail = float(np.prod(1.0 - np.broadcast_to(p, (len(order),))[:a_star]))
    return [candidates[k] for k in order[:a_star]], a_star, belief


#### Central-planner benchmark (u_search_rule = "planner") ####
# Each month the planner allocates the applications of every seeker: the unemployed and the employed
# who decided to search (the OTJ search decision is unchanged). With random hiring from the applicant
# pool (dumb_hire=False) a second application to a vacancy cannot raise its fill probability and a
# worker who wins two vacancies leaves one unfilled, so the efficient allocation is a one-to-one
# assignment: each seeker sends at most one application, each vacancy receives at most one.
#   feasible pair (i, v): rho_tilde_i[occ(v)] > 0 (any vacancy in a connected occupation),
#                         and w_v >= b_i if use_res_wage
#   weight: 'matches' -> 1 for an unemployed seeker, 1 - eps for an employed one, with eps small enough that
#                        the number of matches comes first and the unemployed win ties
#           'surplus' -> (w_v - b_i) * rho_tilde_i[occ(v)]  (pairs with weight <= 0 dropped)
#   b_i with use_res_wage:    unemployed: reservation wage (same draw as search_and_apply)
#                             employed:   1.05 x current wage (mean threshold in emp_search_and_apply)
#   b_i without use_res_wage: unemployed: 0;  employed: current wage (the wage given up by moving)
# Solved as a maximum-weight assignment (scipy linear_sum_assignment), ties broken at random.

def planner_seeker(wrkr, employed, prm):
    """(worker, employed, b_i) recorded when the worker searches (before separations that month).
    Clears this month's search outputs, which the occupation loop reads before the assignment."""
    wrkr.apps_sent = 0
    wrkr.d_wage_offer = np.nan
    if prm['use_res_wage']:
        b = wrkr.wage * (1.05 if employed else reservation_wage(wrkr.time_unemployed, res_wage_dat))
    else:
        b = wrkr.wage if employed else 0.0
    return (wrkr, employed, b)

def planner_assign(seekers, vacancies_by_occ, net, prm):
    """Assign seekers to open vacancies one-to-one (see above) and submit the applications.
    Returns (applications from unemployed, applications from employed)."""
    from scipy.optimize import linear_sum_assignment
    for wrkr, _, _ in seekers:
        wrkr.apps_sent = 0
        wrkr.n_acceptable = 0
        wrkr.perceived_fail = np.nan
        wrkr.d_wage_offer = np.nan
    vac_list = [v for vs in vacancies_by_occ.values() for v in vs]
    if not seekers or not vac_list:
        return 0, 0
    wages = np.array([v.wage for v in vac_list], dtype=float)
    vocc = np.array([v.occupation_id for v in vac_list])
    order = random.sample(range(len(seekers)), len(seekers))   # random tie-breaking between seekers
    W = np.zeros((len(seekers), len(vac_list)))
    # k + 1 matches always beat k: (k + 1)(1 - eps) > k for every k + 1 <= min(rows, cols)
    eps = 0.5 / (min(W.shape) + 1)
    for r, k in enumerate(order):
        wrkr, employed, b = seekers[k]
        rho = _get_rho_tilde(net[wrkr.occupation_id])[vocc]
        ok = rho > 0
        if prm['use_res_wage']:
            ok &= wages >= b
        if prm['planner_objective'] == 'matches':
            W[r] = ok * ((1.0 - eps) if employed else 1.0)
        elif prm['planner_objective'] == 'surplus':
            W[r] = np.where(ok, np.maximum((wages - b) * rho, 0.0), 0.0)
        else:
            raise ValueError(f"Unknown planner_objective '{prm['planner_objective']}'")
        wrkr.n_acceptable = int((W[r] > 0).sum())
    rows, cols = linear_sum_assignment(W, maximize=True)
    n_u = n_e = 0
    for r, c in zip(rows, cols):
        if W[r, c] <= 0:
            continue
        wrkr, employed, _ = seekers[order[r]]
        vac_list[c].applicants.append(wrkr)
        if employed:
            n_e += 1
        else:
            wrkr.apps_sent = 1
            n_u += 1
    return n_u, n_e



#### Worker search with the optimizing rules ####
def opt_search_and_apply(
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
    below_resw_prob: float = 0.15,
    u_search_rule: str = 'empirical',
    opt_params=None
):
    MAX_VACS = 30
    wrkr_occ = wrkr.occupation_id

    # local-bindings for speed
    rand = random.random
    r_choice = random.choice
    r_sample = random.sample
    #neigh_probs = net[wrkr_occ].list_of_neigh_weights
    #occ_count = len(neigh_probs)

    # Build (vacancy, weight) pairs in one pass
    all_vacs = []
    weights = []
    
    # Compute reservation wage once if needed
    if wage_prefs:
        res_wage = wrkr.wage * reservation_wage(wrkr.time_unemployed, res_wage_dat)
    else:
        res_wage = np.nan


    for occ_id, occ_prob in net[wrkr_occ].neigh_nonzero:
        vacs = vacancies_by_occ.get(occ_id)
        if not vacs:
            continue
        all_vacs.extend(vacs)
        weights.extend([occ_prob] * len(vacs))

    # for occ_id, occ_prob in enumerate(neigh_probs):
    #     if occ_prob == 0:
    #         continue
    #     vacs = vacancies_by_occ.get(occ_id)
    #     if not vacs:
    #         continue
        
    #     # Include all vacancies (no filtering here yet)
    #     all_vacs.extend(vacs)
    #     weights.extend([occ_prob] * len(vacs))

    # Diagnostics: distinct vacancies found / acceptable this month (read by model_fun_opt's duration record)
    wrkr.n_found = 0
    wrkr.n_acceptable = 0
    # Optimizing rules only: perceived probability that this month's applications all fail (set in
    # optimal_application_set; 1 if none are sent). NaN under the empirical rule, which has no beliefs.
    wrkr.perceived_fail = np.nan if u_search_rule == 'empirical' else 1.0
    if not all_vacs:
        wrkr.apps_sent = 0
        wrkr.d_wage_offer = np.nan
        return 0

    n_to_sample = min(MAX_VACS, len(all_vacs))

    # Sampling logic
    use_weights = not net[wrkr_occ].uniform_weights
    #use_weights = any(w != weights[0] for w in weights) if weights else False

    if use_weights:
        found_vacs = random.choices(all_vacs, weights=weights, k=n_to_sample)
    else:
        if n_to_sample == len(all_vacs):
            found_vacs = list(all_vacs)
        else:
            found_vacs = r_sample(all_vacs, k=n_to_sample)

    wrkr.n_found = len({id(v) for v in found_vacs})
    wrkr.n_acceptable = wrkr.n_found

    # *** APPLY STRICT RESERVATION WAGE FILTER IF REQUESTED ***
    if wage_prefs and not np.isnan(res_wage) and strict_reservation_wage:
        found_vacs = [v for v in found_vacs if v.wage >= res_wage]
        
        # If no vacancies pass the filter, return early
        if not found_vacs:
            wrkr.apps_sent = 0
            wrkr.d_wage_offer = np.nan
            return 0

    # mean wage of sampled pool (after potential filtering)
    wages_iter = (v.wage for v in found_vacs)
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
                v_wage = v.wage #getattr(v, "wage", 0.0)
                if v_wage >= res_wage:
                    above_resw.append(v)
                else:
                    below_resw.append(v)
        else:
            above_resw = found_vacs
            below_resw = []
        wrkr.n_acceptable = len({id(v) for v in above_resw})

        if u_search_rule != 'empirical':
            # *** OPTIMIZING BENCHMARK: A* from the marginal-benefit rule ***
            # Replaces the empirical effort draw and the risk-aversion offset.
            prm = OPT_SEARCH_DEFAULTS if opt_params is None else {**OPT_SEARCH_DEFAULTS, **opt_params}
            use_rw = prm.get('use_res_wage', True)
            if use_rw:
                assert wage_prefs and not np.isnan(res_wage), \
                    "optimizing search rules with use_res_wage need wage_prefs=True"
            # Weighted sampling is with replacement; never apply twice to one vacancy
            seen = set()
            pool = above_resw if use_rw else found_vacs
            candidates = [v for v in pool if not (id(v) in seen or seen.add(id(v)))]
            wrkr.n_acceptable = len(candidates)
            chosen_vacs, wrkr.A_star, wrkr.belief = optimal_application_set(
                wrkr, candidates, _get_rho_tilde(net[wrkr_occ]), res_wage if use_rw else None,
                u_search_rule, prm, net=net)
        else:
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
            len_neigh_weights = len(neigh_weights)
            def utility_key(v):
                v_wage = v.wage #getattr(v, "wage", 0.0)
                occ_w = neigh_weights[v.occupation_id] if v.occupation_id < len_neigh_weights else 0.0
                return util(wrkr.wage, v_wage, occ_w)

            top_k = heapq.nlargest(want_k, above_resw, key=utility_key) if want_k > 0 else []
            chosen_vacs = top_k[adjusted_start:adjusted_start + adjusted_n_apps]

        # *** Probabilistically add some below-reservation vacancies ***
        # (not for optimizing rules without a reservation wage: every vacancy was already a candidate)
        no_rw_rule = u_search_rule != 'empirical' and not (opt_params or {}).get('use_res_wage', True)
        if below_resw and wage_prefs and not np.isnan(res_wage) and not strict_reservation_wage and not no_rw_rule:
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
        # when not disc, choose up to 4 random vacancies - median of data
        n_choice = min(len(found_vacs), 4)
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
