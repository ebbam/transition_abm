"""
Diagnostic script: compare vacancy creation mechanisms in isolation.

Left column  : HP-filtered shocks (stationary, mean-zero)
Right column : GDP-level shocks (YoY growth rate computed inside g_prime_norm)

Toggles
-------
use_fill_rate : True  → UER-linked fill rate active
                False → expiry only

Methods available: g_prime_norm, g_prime_norm_weighted, cost_red, ar1
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from statsmodels.tsa.filters import hp_filter
from collate_macro_vars import macro_observations

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"

# ============================================================
# TOGGLES
# ============================================================
use_fill_rate  = True    # True = UER-linked fill rate; False = expiry only

# ============================================================
# SHARED PARAMETERS
# ============================================================
calib_date     = ["2001-01-01", "2024-05-01"]
delay          = 25       # burn-in periods before recording
target_demand  = 10300    # single_node: employment(10000) + vacancies(300)
max_vac_age    = 6        # vacancy lifetime (months)
base_fill_rate = 0.80     # fill rate at mean UER → mean duration ≈ 1/base_fill_rate months

# ============================================================
# LOAD GDP DATA
# ============================================================
realgdp = pd.read_csv(path + "data/macro_vars/GDPC1.csv", delimiter=',', decimal='.')
realgdp["DATE"]        = pd.to_datetime(realgdp["DATE"])
realgdp['log_REALGDP'] = np.log(realgdp['GDPC1'])
cycle, trend           = hp_filter.hpfilter(realgdp['log_REALGDP'], lamb=129600)
realgdp['log_Cycle']   = cycle + 1    # HP cycle centred at 1
gdp_pd = realgdp.set_index('DATE').sort_index()

monthly_dates_full = pd.date_range(start=gdp_pd.index.min(), end=gdp_pd.index.max(), freq='MS')
gdp_monthly        = gdp_pd.reindex(monthly_dates_full).interpolate(method='linear')

# YoY log growth (available as a shock option)
gdp_monthly['yoy_growth'] = gdp_monthly['log_REALGDP'].diff(12) + 1

extended_start = pd.to_datetime(calib_date[0]) - pd.DateOffset(months=24 + delay)
gdp_slice = gdp_monthly[
    (gdp_monthly.index >= extended_start) &
    (gdp_monthly.index <= calib_date[1])
]

# Shock series
# gdp_window_hp  : HP-filtered log GDP cycle, centred at 1
# gdp_window_raw : raw real GDP levels (scale-invariant in g_prime_norm ratios)
# gdp_window_yoy : YoY log growth rate + 1 (available but not plotted by default)
gdp_window_hp  = gdp_slice['log_Cycle'].values
gdp_window_raw = gdp_slice['GDPC1'].values
gdp_window_yoy = gdp_slice['yoy_growth'].fillna(1.0).values

# occ_shocks_dat layout (matches network_input_builder single_node):
#   indices 0..23         → deque pre-fill
#   indices 24..24+delay-1 → burn-in
#   indices 24+delay..    → calibration window (recorded)
n_total    = len(gdp_window_hp)
time_steps = n_total - 24

# ============================================================
# PRE-COMPUTE g_prime_norm_mean FOR GDP LEVEL MODE
# Mean YoY growth rate over the calibration window, used to split
# the signal into secular (mean) and cyclical (deviation) components.
# ============================================================
_g_vals = []
_hist   = [target_demand * float(gdp_window_raw[k]) for k in range(24)]
_dq     = deque(_hist, maxlen=24)
for _t in range(time_steps - 1):
    _dq.append(target_demand * float(gdp_window_raw[24 + _t]))
    _g = (1.0 / 12.0) * sum(
        (_dq[23 - k] - _dq[11 - k]) / (_dq[11 - k] + 1e-6)
        for k in range(12)
    )
    if _t >= delay:
        _g_vals.append(_g)
g_prime_norm_mean_gdp = float(np.mean(_g_vals))
print(f"g_prime_norm_mean (gdp_level): {g_prime_norm_mean_gdp:.5f}")

# ============================================================
# OBSERVED VACANCY AND UER DATA
# ============================================================
obs = macro_observations.loc[
    (macro_observations['DATE'] >= calib_date[0]) &
    (macro_observations['DATE'] <= calib_date[1])
].dropna(subset=['VACRATE']).reset_index(drop=True)
vac_obs   = obs['VACRATE'].to_numpy()
uer_obs   = obs['UER'].to_numpy()
mean_uer  = uer_obs.mean()
dates_obs = pd.to_datetime(obs['DATE'])

# ============================================================
# MECHANISM SIMULATOR
# ============================================================
def run_mechanism(name, d_v, gamma_v,
                  shocks_arr, mode,
                  use_fill_rate=True, new_ar1=None,
                  d_growth_rt=0.0, g_norm_mean=0.0,
                  gamma_trend_v=0.0, d_cost_red_v=0.0,
                  rho_v=1.0, beta_v=0.2,
                  fill_rate_base=None):
    """
    Simulate vacancy stock for one mechanism.

    Parameters
    ----------
    name           : "g_prime_norm" | "g_prime_norm_weighted" | "cost_red" | "ar1" | "fixed_flow"
    d_v            : baseline vacancy rate (starting level and reset at t=delay)
    gamma_v        : cyclical elasticity
    shocks_arr     : shock series (hp or raw GDP levels)
    mode           : "hp" | "gdp_level"
    d_growth_rt    : secular drift per period (hp mode only)
    g_norm_mean    : pre-computed mean of g_prime_norm (gdp_level decomposition)
    gamma_trend_v  : secular elasticity (gdp_level mode)
    d_cost_red_v   : geometric growth rate of gamma (cost_red only)
    rho_v          : AR(1) persistence
    fill_rate_base : override base_fill_rate for this run (default: global base_fill_rate)
    beta_v        : AR(1) loading on shock difference

    Returns pd.DataFrame with columns: DATE, vac_rate, target_vr, g_norm
    """
    history    = [target_demand * float(shocks_arr[k]) for k in range(24)]
    td_history = deque(history, maxlen=24)
    rho_eff        = new_ar1 if new_ar1 is not None else rho_v
    _fill_rate_base = fill_rate_base if fill_rate_base is not None else base_fill_rate

    curr_vacs       = 300.0
    target_vr_state = d_v
    occ_shock_prev  = float(shocks_arr[24])
    vac_ages        = [0] * int(curr_vacs)
    target_vr       = d_v

    vac_rate_rec  = []
    target_vr_rec = []
    g_norm_rec    = []
    calib_dates   = pd.date_range(start=calib_date[0], end=calib_date[1], freq='MS')

    for t in range(time_steps - 1):
        occ_shock = float(shocks_arr[24 + t])
        td_history.append(target_demand * occ_shock)

        # g_prime_norm: equal-weight 12m YoY average
        g_prime_norm = (1.0 / 12.0) * sum(
            (td_history[23 - k] - td_history[11 - k]) / (td_history[11 - k] + 1e-6)
            for k in range(12)
        )
        # g_prime_norm_weighted: triangular weights (more weight on recent pairs)
        weighted_g_prime_norm = sum(
            ((12 - k) / 78.0) * (td_history[23 - k] - td_history[11 - k]) / (td_history[11 - k] + 1e-6)
            for k in range(12)
        )

        # UER-linked fill rate
        if use_fill_rate and len(vac_ages) > 0:
            uer_t       = uer_obs[t - delay] if t >= delay else mean_uer
            fill_rate_t = _fill_rate_base * (uer_t / mean_uer)
            n_filled    = int(np.random.binomial(len(vac_ages), np.clip(fill_rate_t, 0, 1)))
            vac_ages    = vac_ages[n_filled:]

        # Vacancy expiry
        vac_ages  = [a + 1 for a in vac_ages]
        vac_ages  = [a for a in vac_ages if a < max_vac_age]
        curr_vacs = len(vac_ages)

        # ---- g_prime_norm / g_prime_norm_weighted / cost_red ----
        if name in ("g_prime_norm", "g_prime_norm_weighted", "cost_red"):
            if t == delay:
                target_vr_state = d_v

            if t < delay:
                target_vr = d_v
            else:
                signal = weighted_g_prime_norm if name == "g_prime_norm_weighted" else g_prime_norm

                if name == "cost_red":
                    gamma_v_t = gamma_v * (1 + d_cost_red_v) ** (t - delay)
                    if mode == "gdp_level":
                        target_vr_state += gamma_trend_v * g_norm_mean + gamma_v_t * (signal - g_norm_mean)
                    else:
                        target_vr_state += d_growth_rt + gamma_v_t * signal
                elif mode == "gdp_level":
                    target_vr_state += gamma_trend_v * g_norm_mean + gamma_v * (signal - g_norm_mean)
                else:
                    target_vr_state += d_growth_rt + gamma_v * signal

                target_vr = max(0.0, target_vr_state)

        # ---- AR(1) ----
        elif name == "ar1":
            if t == delay:
                target_vr_state = d_v

            if t < delay:
                target_vr = d_v
            else:
                if mode == "gdp_level":
                    # Use demeaned g_prime_norm so the secular trend doesn't accumulate
                    signal          = g_prime_norm - g_norm_mean
                else:
                    # HP shocks are already mean-zero; use month-on-month difference
                    signal          = occ_shock - occ_shock_prev
                target_vr_state = rho_eff * target_vr_state + beta_v * signal
                target_vr       = max(0.0, target_vr_state)

            occ_shock_prev = occ_shock

        # ---- fixed_flow ----
        # Firms post at a constant flow rate (d_growth_rt drift) independent of current stock.
        # Vacancy rate is determined by equilibrium: posting inflow vs fill+expiry outflow.
        # When UER rises → fill rate rises → stock drains faster → vacancy rate falls.
        # Unlike gap-closing mechanisms, there is no compensating re-posting.
        elif name == "fixed_flow":
            if t == delay:
                target_vr_state = d_v
            target_vr_state += d_growth_rt
            target_vr = target_vr_state  # tracks the posting rate for plotting

        # ---- vacancy creation ----
        if name == "fixed_flow":
            # Post at the current flow rate regardless of stock level
            vac_prob = np.clip(target_vr_state, 0, 1)
        else:
            # Gap-closing: post enough to close the gap between current and target rate
            vac_prob = np.clip(max(0.0, target_vr - curr_vacs / (target_demand + 1)), 0, 1)

        vac_ages.extend([0] * int(np.random.binomial(target_demand, vac_prob)))
        curr_vacs = len(vac_ages)

        if t >= delay:
            vac_rate_rec.append(curr_vacs / (curr_vacs + target_demand))
            target_vr_rec.append(target_vr)
            g_norm_rec.append(g_prime_norm)

    n_rec = len(vac_rate_rec)
    return pd.DataFrame({
        'DATE':      calib_dates[:n_rec],
        'vac_rate':  vac_rate_rec,
        'target_vr': target_vr_rec,
        'g_norm':    g_norm_rec,
    })


# ============================================================
# RUN — HP MODE (left column)
# ============================================================
np.random.seed(42)

_hp_kw = dict(
    shocks_arr    = gdp_window_hp,
    mode          = "hp",
    use_fill_rate = use_fill_rate,
    d_growth_rt   = 0.0001,   # secular drift per period
    g_norm_mean   = 0.0,
    gamma_trend_v = 0.01,
    d_cost_red_v  = 0.003,
    rho_v         = 1,      # random walk
    beta_v        = 0.2,
)

mods_hp = {
    # fixed_flow: firms post at constant rate; fill rate does all cyclical work
    # fill_rate_base=0.35 ≈ JOLTS hires/openings ratio; leaves room for cyclical variation
    # d_v and d_growth_rt are posting rates, not target vacancy rates.
    # equilibrium vac_rate = posting_rate / (fill_rate_base + 1/max_vac_age)
    # d_v = 0.032 * (0.50 + 0.167) ≈ 0.021 → mean vac_rate ≈ 0.032
    "fixed flow + UER fill rate":
        run_mechanism("fixed_flow", d_v=0.015, gamma_v=0.0, fill_rate_base=0.5,
                      **{**_hp_kw, 'd_growth_rt': 0}),
    "g_prime_norm 12m equal-weight (γ=0.03)":
        run_mechanism("g_prime_norm",          d_v=0.032, gamma_v=0.03, **_hp_kw),
    # "g_prime_norm 12m triangular-weight (γ=0.03)":
    #     run_mechanism("g_prime_norm_weighted", d_v=0.032, gamma_v=0.03, **_hp_kw),
    "cost_red (γ=0.03; d=0.003)":
        run_mechanism("cost_red",              d_v=0.032, gamma_v=0.03, **_hp_kw),
    "AR(1) (ρ=1.0, β=0.2)":
        run_mechanism("ar1",                   d_v=0.032, gamma_v=0.03, **_hp_kw),
}

# ============================================================
# RUN — GDP LEVEL MODE (right column)
# ============================================================
_gdp_kw = dict(
    shocks_arr    = gdp_window_raw,
    mode          = "gdp_level",
    use_fill_rate = use_fill_rate,
    d_growth_rt   = 0.0,
    g_norm_mean   = g_prime_norm_mean_gdp,
    gamma_trend_v = 0.005,    # secular: 0.003 * 0.023 ≈ 0.00007/month drift
    d_cost_red_v  = 0.002,
    rho_v         = 1.003,
    beta_v        = 0.03
)

mods_gdp = {
    "fixed flow + UER fill rate":
        run_mechanism("fixed_flow", d_v=0.015, gamma_v=0.0, fill_rate_base=0.5,
                      **{**_gdp_kw, 'd_growth_rt': 0}),
    "g_prime_norm 12m equal-weight (γ=0.03)":
        run_mechanism("g_prime_norm",          d_v=0.032, gamma_v=0.03, **_gdp_kw),
    "g_prime_norm 12m triangular-weight (γ=0.03)":
        run_mechanism("g_prime_norm_weighted", d_v=0.032, gamma_v=0.03, **_gdp_kw),
    "cost_red (γ=0.03; d=0.002)":
        run_mechanism("cost_red",              d_v=0.032, gamma_v=0.03, **_gdp_kw),
    "AR(1) (ρ=0.99, β=0.25)":
        run_mechanism("ar1",                   d_v=0.032, gamma_v=0.03, **_gdp_kw),
}

# ============================================================
# PLOT: 4 rows × 2 columns
#   col 0 = HP shocks  |  col 1 = GDP level shocks
#   row 0 = vacancy rate
#   row 1 = target_vr state
#   row 2 = g_prime_norm signal
#   row 3 = GDP reference (log level + first difference)
# ============================================================
fill_label = "with fill rate" if use_fill_rate else "no fill rate"
fig, axes  = plt.subplots(3, 2, figsize=(16, 16), sharex=True)

def plot_vac_panel(ax, mod_list, title):
    for name, item in mod_list.items():
        ax.plot(item['DATE'], item['vac_rate'], linewidth=1, label=name)
    ax.plot(dates_obs, vac_obs, 'k--', linewidth=1.5, label='Observed (JOLTS)')
    ax.set_ylabel('Vacancy Rate')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plot_vac_panel(axes[0, 0], mods_hp,  f'HP-filtered shocks — vacancy rate ({fill_label})')
plot_vac_panel(axes[0, 1], mods_gdp, f'GDP-level shocks — vacancy rate ({fill_label})')

for col, mod_list, d_v_ref, label in [
    (0, mods_hp,  0.032, 'HP-filtered shocks'),
    (1, mods_gdp, 0.032, 'GDP-level shocks'),
]:
    for name, item in mod_list.items():
        axes[1, col].plot(item['DATE'], item['target_vr'], linewidth=1, label=name)
        axes[2, col].plot(item['DATE'], item['g_norm'],    linewidth=1, label=name)

    axes[1, col].axhline(d_v_ref, color='grey', linewidth=0.8, linestyle=':', label=f'd_v = {d_v_ref}')
    axes[1, col].set_ylabel('target_vr')
    axes[1, col].set_title(f'Target Vacancy Rate — {label}')
    axes[1, col].legend(fontsize=8)
    axes[1, col].grid(alpha=0.3)

    axes[2, col].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2, col].set_ylabel('g_prime_norm')
    axes[2, col].set_title(f'Demand Growth Signal — {label}')
    axes[2, col].legend(fontsize=8)
    axes[2, col].grid(alpha=0.3)

# # Row 3: GDP reference panels
# gdp_plot_dates = gdp_slice.index
# fdlog_gdp      = gdp_slice['log_REALGDP'].diff()

# axes[3, 0].plot(gdp_plot_dates, gdp_slice['log_REALGDP'], color='darkslategrey', linewidth=1)
# axes[3, 0].set_ylabel('log(Real GDP)')
# axes[3, 0].set_xlabel('Date')
# axes[3, 0].set_title('Log Real GDP')
# axes[3, 0].grid(alpha=0.3)

# axes[3, 1].plot(gdp_plot_dates, fdlog_gdp, color='steelblue', linewidth=1, label='Δlog(Real GDP)')
# axes[3, 1].axhline(fdlog_gdp.mean(), color='black', linewidth=0.8, linestyle='--',
#                    label=f'mean = {fdlog_gdp.mean():.4f}')
# axes[3, 1].axhline(0, color='grey', linewidth=0.6, linestyle=':')
# axes[3, 1].set_ylabel('Δlog(Real GDP)')
# axes[3, 1].set_xlabel('Date')
# axes[3, 1].set_title('First Difference of Log Real GDP')
# axes[3, 1].legend(fontsize=8)
# axes[3, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('output/figures/vacancy_mechanism_diagnostic.png', dpi=150)
plt.show()
print("Saved to output/figures/vacancy_mechanism_diagnostic.png")

# %%
