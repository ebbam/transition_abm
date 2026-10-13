# Import packages
import os

import numpy as np
import pandas as pd
import random as random
from copy import deepcopy 
import math as math
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
from scipy.stats import norm
import cProfile, pstats, io
from plot_style import TEXTWIDTH, save_plot_standard, apply_minor_gridlines, add_shared_legend
rng = np.random.default_rng()

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/data/"
plot_new = False
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
        raw = 1
    elif bin_label == "1 to 10":
        raw = 5.5 if expectation else random.randint(1, 10)
    elif bin_label == "11 to 20":
        raw = 15.5 if expectation else random.randint(11, 20)
    elif bin_label == "21 to 80":
        raw = 50.5 if expectation else random.randint(21, 80)
    elif bin_label == "81 or more":
        raw = 90.5 if expectation else random.randint(81, 100)
    else:
        raise ValueError(f"Unknown bin label: {bin_label}")
    return_val = max(1, round(raw/2))
    if return_val > 75:
        print(return_val)
    return return_val

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

if plot_new:
    # series_dict = {}
    # for i, a in enumerate([0.1, 0.05, 0.01]):
    #     efforts = []
    #     for t_unemp in range(36):
    #         apps = search_effort_alpha(t_unemp, 1, True, a)
    #         efforts.append(apps)

    #     series_dict[f"Alpha: {a}"] = {
    #         "x": list(range(0, len(efforts))),
    #         "y": efforts
    #         }

    imposed_efforts = []
    for t_unemp in range(40):
        apps = applications_sent(duration_months=t_unemp, duration_to_prob_dict=duration_to_prob_dict, expectation = True)
        imposed_efforts.append(apps)

    bls_df = {
        "x": list(range(0, len(imposed_efforts))),
        "y": imposed_efforts
        }
    
    plt.plot(bls_df['x'], bls_df['y'], color='purple', linestyle='-', linewidth=2, label='Mean Application Effort from BLS Survey Data')

    plt.axvline(x=24, color='black', linestyle='--', linewidth=1, label='2 years')
    plt.axvline(x=36, color='grey', linestyle='--', linewidth=1, label='3 years')
    plt.legend(loc='upper right')
    plt.suptitle("Applications Sent by Unemployment Duration")
    plt.tight_layout()
    plt.savefig('output/figures/applications_sent_by_unemployment_duration.png', dpi=300)
    plt.show()


    from collections import defaultdict

    # --- parameters you can change ---
    N = 50000                 # Monte-Carlo draws per duration (increase for smoother bands)
    max_duration = 40        # number of months to plot
    seed = 123               # reproducible sampling seed

    random.seed(seed)
    np.random.seed(seed)

    # --- helper that runs N stochastic draws for a given duration ---
    def sample_applications(duration, duration_to_prob_dict, n_draws=N):
        """Return an array of length n_draws of stochastic application counts for given duration."""
        samples = []
        for _ in range(n_draws):
            # call your function in stochastic mode (expectation=False)
            # ensure it returns an integer >= 1 as your original did
            samples.append(applications_sent(duration, duration_to_prob_dict, expectation=False))
        return np.array(samples, dtype=float)

    # --- run MC across durations and compute summary stats ---
    durations = list(range(max_duration))
    means = []
    medians = []
    p05 = []
    p25 = []
    p75 = []
    p95 = []

    for t in durations:
        # if duration missing in dict, your function returns 1; still sample
        arr = sample_applications(t, duration_to_prob_dict, n_draws=N)
        means.append(arr.mean())
        medians.append(np.median(arr))
        p05.append(np.percentile(arr, 5))
        p25.append(np.percentile(arr, 25))
        p75.append(np.percentile(arr, 75))
        p95.append(np.percentile(arr, 95))

    means = np.array(means)
    medians = np.array(medians)
    p05 = np.array(p05)
    p25 = np.array(p25)
    p75 = np.array(p75)
    p95 = np.array(p95)

    # --- plotting ---
    plt.figure(figsize=(6.27, 5.02))

    # shaded 90% band (5th to 95th percentile)
    #plt.fill_between(durations, p05, p95, color='lavender', alpha=0.6, label='90% interval (5-95th pct)')

    # shaded interquartile band
    plt.fill_between(durations, p25, p75, color='mediumpurple', alpha=0.3, label='IQR (25-75th pct)')

    # mean and median lines
    plt.plot(durations, means, color='purple', linestyle='-', linewidth=2, label='Mean Application Effort')
    plt.plot(durations, medians, color='darkviolet', linestyle='--', linewidth=1.5, label='Median')

    # vertical reference lines
    plt.axvline(x=24, color='black', linestyle='--', linewidth=1, label='2 years')
    plt.axvline(x=36, color='grey', linestyle='--', linewidth=1, label='3 years')

    plt.legend(loc='upper right')
    plt.title("Applications Sent by Unemployment Duration")
    plt.xlabel("Unemployment Duration (Months)")
    plt.ylabel("Applications Sent")
    #plt.gcf().set_size_inches(8, 6)
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Dropbox/Apps/Overleaf/ABM_Transitions/figs/applications_sent_by_unemployment_duration_with_uncertainty.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    # Alternative search effort function that is dictated by the time series of a business cycle
    def search_effort_ts(t_unemp, se):
        apps = max(0, round(10 - 100*(1-se)))
        return apps

#### Reservation Wage Adjustment Rate ######
# Load the data
# res_wage_dat = pd.read_csv(path + "res_wage_dists.csv")

# # Main function: given duration and bin probabilities, return application count
# def reservation_wage(duration_months,
#                              res_wage_data,
#                              expectation=False,
#                              # Either lm (no sample balancing), glm, eb (entropy balancing)
#                              balancing_method='lm'):
#     """
#     If expectation=True return the expected value (float);
#     otherwise return a random draw (int).
#     """
#     if duration_months not in res_wage_data['dur_unemp'].values:
#         #raise ValueError(f"No probability distribution for duration {duration_months}")
#         res_wage = np.random.normal(loc=0.5, scale=np.max(res_wage_data[f'{balancing_method}_se']), size=1)[0]
#         if expectation:
#             res_wage = 0.5
#     else:
#         mean_val = res_wage_data[f'{balancing_method}_preds_fit'][res_wage_data['dur_unemp'] == duration_months]
#         se_val = res_wage_data[f'{balancing_method}_se'][res_wage_data['dur_unemp'] == duration_months]

#         if expectation:
#             res_wage = mean_val.iloc[0]  # Extract scalar from Series
#         else:
#             res_wage = np.random.normal(loc=mean_val.iloc[0], scale=se_val.iloc[0], size=1)[0]
    
#     return res_wage

res_wage_dat = pd.read_csv(path + "res_wage_dists.csv")

# Pre-build lookup dict to avoid per-call pandas indexing in search_and_apply
_res_wage_lookup = {
    int(row['dur_unemp']): (float(row['lm_preds_fit']), float(row['lm_se']))
    for _, row in res_wage_dat.iterrows()
}
_res_wage_fallback_se = float(res_wage_dat['lm_se'].max())

# Main function: given duration, return reservation wage multiplier
def reservation_wage(duration_months,
                             res_wage_data,
                             expectation=False,
                             # Either lm (no sample balancing), glm, eb (entropy balancing)
                             balancing_method='lm'):
    """
    If expectation=True return the expected value (float);
    otherwise return a random draw (int).
    """
    dur = int(duration_months)
    if dur not in _res_wage_lookup:
        return 0.5 if expectation else np.random.normal(loc=0.5, scale=_res_wage_fallback_se)
    mean_val, se_val = _res_wage_lookup[dur]
    return mean_val if expectation else np.random.normal(loc=mean_val, scale=se_val)


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
    return [candidates[k] for k in order[:a_star]], a_star, belief


## Defining classes
# Potentially redundant use of IDs in the below classes...to check
class worker:
    def __init__(wrkr, occupation_id,
                 # employed,
                 longterm_unemp,
                 # time_employed,
                 time_unemployed, wage, hired, female, age, risk_av_score, ee_rel_wage, ue_rel_wage, applicants_sent, d_wage_offer, last_ue_duration,
                 ue_new_wage=None, ue_presep_wage=None, new_entrant=False):
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
        wrkr.new_entrant = new_entrant


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

        # Diagnostics: distinct vacancies found / acceptable this month (read by model_fun's duration record)
        wrkr.n_found = 0
        wrkr.n_acceptable = 0
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

    def emp_search_and_apply(wrkr, net, vac_list, disc, emp_apps, wage_prefs,   
                          mis_rate: float = 0.10,                 # PROBABILITY TO REPLACE A SAMPLED VAC WITH A GLOBAL RANDOM VAC
        unique_random: bool = False,            # TRY TO AVOID DUPLICATE RANDOM PICKS IF TRUE
        global_pool_override = None ):
        # A sample of relevant vacancies are found that are in neighboring occupations
        # Select different random sample of "relevant" vacancies found by each worker
        # found_vacs = random.sample(vac_list, min(len(vac_list), 30))
        MAX_VACS = 30
        wrkr_occ = wrkr.occupation_id
        # neigh_probs = net[wrkr_occ].list_of_neigh_weights  # Already normalized
        # occ_ids = list(range(len(neigh_probs)))

        # # Build a flat list of vacancies and their weights
        # all_vacs = []
        # weights = []

        # for occ_id, occ_prob in zip(occ_ids, neigh_probs):
        #     vacs = vac_list.get(occ_id, [])
        #     for v in vacs:
        #         all_vacs.append(v)
        #         weights.append(occ_prob)

        # Build a flat list of vacancies and their weights
        all_vacs = []
        weights = []

        for occ_id, occ_prob in net[wrkr_occ].neigh_nonzero:
            vacs = vac_list.get(occ_id, [])
            all_vacs.extend(vacs)
            weights.extend([occ_prob] * len(vacs))


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

        # if disc and wage_prefs:
        #     found_vacs = [v for v in found_vacs if v.wage >= np.random.normal(wrkr.wage*1.05, 0.05*wrkr.wage*1.05)]
        if disc and wage_prefs:
            thresholds = np.random.normal(wrkr.wage*1.05, 0.05*wrkr.wage*1.05, size=len(found_vacs))
            found_vacs = [v for v, thr in zip(found_vacs, thresholds) if v.wage >= thr]

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

            
class occupation:
    def __init__(occ, occupation_id, list_of_employed, list_of_unemployed, 
                 list_of_neigh_bool, list_of_neigh_weights, current_demand, 
                 target_demand, wage, wage_mu, wage_sigma, separated, hired, entry_level_bool, experience_age, seps_rate, competition_last, neigh_nonzero, uniform_weights):
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
        occ.neigh_nonzero = neigh_nonzero
        occ.uniform_weights = uniform_weights


    def separate_workers(occ, delta_u, gam, bus_cy):
        n_emp = len(occ.list_of_employed)
        if(n_emp != 0):
            sep_prob_base = delta_u + (1-delta_u)*((gam * max(0, n_emp - (occ.target_demand*bus_cy)))/(n_emp + 1))
            sep_prob_gamma = delta_u + occ.seps_rate*(1-delta_u)*((gam * max(0, n_emp - (occ.target_demand*bus_cy)))/(n_emp + 1))
            sep_prob_delta = occ.seps_rate*delta_u + (1-delta_u)*((gam * max(0, n_emp - (occ.target_demand*bus_cy)))/(n_emp + 1))
            w = np.random.binomial(n_emp, sep_prob_base) #sep_prob_delta) #sep_prob_base) #sep_prob_delta) # sep_prob_base
            occ.separated = int(w)
            separated_workers = random.sample(occ.list_of_employed, int(w))
            occ.list_of_unemployed.extend(separated_workers)
            #occ.list_of_unemployed = occ.list_of_unemployed + separated_workers
            #occ.list_of_employed = list(set(occ.list_of_employed) - set(separated_workers))
            sep_ids = {id(s) for s in separated_workers}
            occ.list_of_employed = [e for e in occ.list_of_employed if id(e) not in sep_ids]

    
    def update_workers(occ):
        # Possible for loop to replace
        for w in occ.list_of_unemployed:
            w.time_unemployed += 1
            # Chosen 6 months - can be modified - 27 weeks (~ 6 mos) according to BLS.
            # We call update_workers before we tally LTUE people so time spent unemployed will be 1 greater than actual time spent "applying"
            # Given unemployed individuals are given one time period of unemployment from the moment they are fired (ie. they have not navigated the market as unemployed yet)
            # we set the threshold for LTUER at 7 (not 6)
            w.longterm_unemp = w.time_unemployed >= 7
            w.ue_rel_wage = None
            w.ee_rel_wage = None
            w.hired = False
            # Failed applications this spell (still unemployed => every application sent failed);
            # the belief signal for the "bayes" optimizing search rule
            w.n_fail = getattr(w, 'n_fail', 0) + (w.apps_sent or 0)
            w.apps_sent = 0
            w.age += 0.083
        for e in occ.list_of_employed:
            e.hired = False
            e.time_unemployed = 0
            e.n_fail = 0  # a new unemployment spell starts with a fresh failure count
            e.ue_rel_wage = None
            e.ee_rel_wage = None
            e.apps_sent = 0
            e.age += 0.083
            # The equivalent of an annual 2% raise
            #e.wage = e.wage*(1.02**(1/12))

    def retire_workers(occ):
        """ Function to retire workers over the age of 65 """
        workers_pre = len(occ.list_of_unemployed) + len(occ.list_of_employed)
        occ.list_of_unemployed = [w for w in occ.list_of_unemployed if w.age <= 65]
        occ.list_of_employed = [e for e in occ.list_of_employed if e.age <= 65]
        workers_sep = workers_pre - (len(occ.list_of_unemployed) + len(occ.list_of_employed))
        return(workers_sep)

    
    def entry(occ, entry_tot, enter_emp=False, use_wage_dist=False, jg_wage=None, entry_time_step=None):
        """Add exactly `entry_tot` new workers to this occupation (if entry-level).

        use_wage_dist=False (default): entry wage = mean of bottom 5% employed.
        use_wage_dist=True:            entry wage drawn from occ lognormal distribution.
        jg_wage: when enter_emp=True, override the market wage draw with this value.
                 Pass "entry_anchor" to use the 25th-pct anchor computed here, or a float.
        entry_time_step: if given, tagged onto each created worker as
                 `.entry_time_step` -- a permanent marker of when they entered
                 the labour force, independent of the `new_entrant` flag (which
                 clears on first hire). Lets later worker-level snapshots
                 (e.g. wage_age_df) be linked back to an entry cohort.
        """
        if entry_tot <= 0 or not occ.entry_level:
            return

        emp_no = len(occ.list_of_employed)
        fem_prob = float(np.mean([wrkr.female for wrkr in occ.list_of_employed])) if emp_no > 0 else 0.5

        if not use_wage_dist:
            if emp_no == 0:
                fixed_wage = occ.wage * 0.95
            else:
                bottom_5 = max(1, int(emp_no * 0.05))
                fixed_wage = np.mean([w.wage for w in heapq.nsmallest(bottom_5, occ.list_of_employed, key=lambda x: x.wage)])

        entry_anchor = np.exp(occ.wage_mu - 0.6745 * occ.wage_sigma)  # 25th pct of occ wage dist

        for _ in range(entry_tot):
            new_wage = np.random.lognormal(occ.wage_mu, occ.wage_sigma) if use_wage_dist else fixed_wage
            fem = (random.random() < fem_prob)
            ra = 3 if fem else 7

            if enter_emp:
                if jg_wage == "entry_anchor":
                    emp_wage = entry_anchor
                elif jg_wage is not None:
                    emp_wage = float(jg_wage)
                else:
                    emp_wage = new_wage
                new_wrkr = worker(
                    occ.occupation_id,         # occupation_id
                    False,                     # longterm_unemp
                    0,                         # time_unemployed
                    emp_wage,                  # wage
                    False,                     # hired
                    fem,                       # female
                    occ.experience_age,        # age
                    abs(int(np.random.normal(ra, 2))),  # risk_av_score
                    1,                         # ee_rel_wage
                    1,                         # ue_rel_wage
                    0,                         # applicants_sent
                    0,                         # d_wage_offer
                    None,                      # last_ue_duration
                    new_entrant=True
                )
                new_wrkr.entry_time_step = entry_time_step
                occ.list_of_employed.append(new_wrkr)
            else:
                new_wrkr = worker(
                    occ.occupation_id,         # occupation_id
                    False,                     # longterm_unemp
                    0,                         # time_unemployed
                    entry_anchor,              # wage (25th pct of occ dist as reservation wage anchor)
                    False,                     # hired
                    fem,
                    occ.experience_age,        # age
                    abs(int(np.random.normal(ra, 2))),  # risk_av_score
                    1,                         # ee_rel_wage
                    1,                         # ue_rel_wage
                    0,                         # applicants_sent
                    0,                         # d_wage_offer
                    None,                      # last_ue_duration
                    new_entrant=True
                )
                new_wrkr.entry_time_step = entry_time_step
                occ.list_of_unemployed.append(new_wrkr)

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
        

    # Function to hire a worker from pool of vacancies    
    def hire(v, net, dumb_hire, ltu_hire_bias=0.0, entry_hire_bias=0.0):
        assert not (ltu_hire_bias > 0.0 and entry_hire_bias > 0.0), \
            "ltu_hire_bias and entry_hire_bias cannot both be positive — only one policy at a time."
        if dumb_hire:
            a = random.choice(v.applicants)
            assert not a.hired
        else:
            if ltu_hire_bias > 0.0 or entry_hire_bias > 0.0:
                # Preferential hiring: weight favoured applicants more heavily.
                # ltu_hire_bias and entry_hire_bias are mutually exclusive (see
                # assert above), so whichever policy is active targets its own
                # attribute only -- a worker who happens to ALSO carry the
                # other (inactive) attribute must not lose their boost to it.
                # e.g. a new entrant who's been searching long enough to also
                # be longterm_unemp must still get the entry_hire_bias boost
                # when only the entry policy is active.
                available = [w for w in v.applicants if not w.hired]
                if not available:
                    v.applicants.clear()
                    return None
                if entry_hire_bias > 0.0:
                    weights = [(1.0 + entry_hire_bias) if getattr(w, 'new_entrant', False) else 1.0
                               for w in available]
                else:
                    weights = [(1.0 + ltu_hire_bias) if getattr(w, 'longterm_unemp', False) else 1.0
                               for w in available]
                a = random.choices(available, weights=weights, k=1)[0]
            else:
                # Sample up to 3 unique applicants without replacement
                sample_size = min(1, len(v.applicants))
                sampled_applicants = random.sample(v.applicants, sample_size)

                a = None
                for candidate in sampled_applicants:
                    if not candidate.hired:
                        a = candidate
                        break

                if a is None:
                    v.applicants.clear()
                    return None

        assert not a.hired

        origin_occ = None
        ue_dur     = None
        longterm   = None

        # Capture worker state before any lists are mutated or wages overwritten
        src_occ     = a.occupation_id
        pre_wage    = a.wage
        age         = a.age
        female      = a.female
        new_entrant = getattr(a, 'new_entrant', False)

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
                ue_dur     = src_list[idx].time_unemployed  # spell length right now
                longterm   = src_list[idx].longterm_unemp

                net[v.occupation_id].list_of_employed.append(src_list.pop(idx))
                a.ue_rel_wage = v.wage / a.wage

            except ValueError:
                print("Indexing failed - worker not found in either employed or unemployed list")

        net[v.occupation_id].hired += 1
        a.occupation_id = v.occupation_id
        a.time_unemployed = 0          # reset after hire
        a.wage = v.wage
        a.hired = True
        a.new_entrant = False          # first hire clears the new-entrant flag
        v.applicants.clear()

        # Always return a rich tuple covering both UE and EE transitions.
        # TransitionType: 'UE' if came from unemployed list, 'EE' otherwise.
        # ue_dur / longterm are None for EE hires.
        trans_type    = 'UE' if origin_occ is not None else 'EE'
        final_origin  = origin_occ if origin_occ is not None else src_occ
        return (trans_type, final_origin, v.occupation_id, ue_dur,
                pre_wage, v.wage, age, female, longterm, new_entrant)

        
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
                         demand_target[i,0], wages[i], wage_mu[i], wage_sigma[i], 0, 0, entry_level[i], experience_age[i], sep_rates[i], np.nan, 
                         [(i, w) for i, w in enumerate(list(A[i])) if w > 0], 
                         len(set(w for w in list(A[i]) if w > 0)) == 1)
        
    
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
        return 1.0 / (1.0 + math.exp(-x))

def p_search_logit(comp, alpha=0.0, beta_C=-0.8): #age, beta_A=0.05, beta_CA=0.0, A0=40.0
    """
    Logit p = alpha + beta_A*(age-A0) + beta_C*comp + beta_CA*comp*(age-A0)
    No cycle term, no occ_shock.
    """
    logit_val = (alpha
                 #+ beta_A * (age - A0)
                 + beta_C * comp)
    return _sigmoid(logit_val)

def test_p_search_logit():
    """Unit test for search probability function"""
    beta_C_positive = 0.8  # Positive effect (new model)
    
    # Test 1: Higher comp → higher probability (with positive beta_C)
    prob_low = p_search_logit(comp=0.1, alpha=-1.58, beta_C=beta_C_positive)
    prob_high = p_search_logit(comp=0.5, alpha=-1.58, beta_C=beta_C_positive)
    assert prob_high > prob_low, f"Higher comp should increase prob: {prob_low:.4f} vs {prob_high:.4f}"
    
    # Test 2: Edge cases
    prob_zero = p_search_logit(comp=0.0, alpha=-1.58, beta_C=beta_C_positive)
    prob_one = p_search_logit(comp=1.0, alpha=-1.58, beta_C=beta_C_positive)
    assert prob_one > prob_zero, "comp=1.0 should give higher prob than comp=0.0"
    
    # Test 3: Probability bounds
    assert 0 <= prob_low <= 1, "Probability must be in [0,1]"
    assert 0 <= prob_high <= 1, "Probability must be in [0,1]"
    
    print("search function test passed!")

# Run test
test_p_search_logit()

if plot_new:
# import numpy as np
#import matplotlib.pyplot as plt

# def _sigmoid(x):
#     if x < -709:
#         return 0.0
#     elif x > 709:
#         return 1.0
#     else:
#         return 1.0 / (1.0 + math.exp(-x))

# def p_search_logit(age, comp, *, alpha=0.0, beta_A=0.05, beta_C=-0.8, beta_CA=0.0, A0=40.0):
#     """
#     Logit p = alpha + beta_A*(age-A0) + beta_C*comp + beta_CA*comp*(age-A0)
#     """
#     logit_val = (alpha + beta_A * (age - A0) + beta_C * comp)
#     return _sigmoid(logit_val)

    # Parameters from your model
    alpha = -4
    beta_A = 0
    A0 = 40.0

    # Competition range (VacsPerApp: 0 to 1)
    comp_range = np.linspace(0, 1, 100)

    # Different theta (beta_C) values to test
    theta_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50]
    #theta_values = [0.1, 0.25, 0.5, 0.75, 0.9]


    # Different ages to show age effects
    ages = [30, 40, 50]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Different theta values (fixed age=40)
    ax1 = axes[0]
    for theta in theta_values:
        probs = [p_search_logit(comp, alpha=alpha, beta_C=theta) 
                for comp in comp_range]
        ax1.plot(comp_range, probs, linewidth=2, label=rf'$\beta_c$={theta}')

    ax1.axhline(y=0.03, color='red', linestyle='--', linewidth=1, alpha=0.5, label='7% target')
    ax1.axvspan(0.033, 0.5, alpha=0.15, color='blue', label='Expected range\n(2-30 Seekers per Vac.)')
    ax1.set_xlabel('Competition Metric', fontsize=12)
    ax1.set_ylabel('Search Probability', fontsize=12)
    ax1.set_title(r'Effect of $\beta_c$ on Search Behavior (Age=40)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Panel 2: Different ages (fixed theta=5.0)
    # ax2 = axes[1]
    # theta_fixed = 5.0
    # for age in ages:
    #     probs = [p_search_logit(age, comp, alpha=alpha, beta_A=beta_A, beta_C=theta_fixed, A0=A0) 
    #              for comp in comp_range]
    #     ax2.plot(comp_range, probs, linewidth=2, label=f'Age {age}')

    # ax2.axhline(y=0.03, color='red', linestyle='--', linewidth=1, alpha=0.5, label='7% target')
    # ax2.axvspan(0.033, 0.333, alpha=0.15, color='blue', label='Expected range')
    # ax2.set_xlabel('Competition (VacsPerApp)', fontsize=12)
    # ax2.set_ylabel('Search Probability', fontsize=12)
    # ax2.set_title(rf'Age Effects on Search Behavior ($\beta_c$={theta_fixed})', fontsize=13, fontweight='bold')
    # ax2.legend(fontsize=10)
    # ax2.grid(True, alpha=0.3)
    # ax2.set_xlim([0, 1])
    # ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('search_probability_vs_competition.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Plot saved as 'search_probability_vs_competition.png'")

    # Print some specific values for verification
    print("\n" + "="*60)
    print("SAMPLE VALUES (Age=40, θ=5.0):")
    print("="*60)
    test_comps = [0.0, 0.033, 0.1, 0.333, 0.5, 1.0]
    for comp in test_comps:
        prob = p_search_logit(comp, alpha=alpha, beta_C=5.0)
        print(f"VacsPerApp = {comp:.3f} → Search Probability = {prob:.4f} ({prob*100:.2f}%)")