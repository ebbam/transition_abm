"""
Simple vacancy creation simulation using aggregate GDP shock (gdp_dat).
Tests the current model_fun.py vacancy mechanism in a controlled setting.

Vacancy rule (mirrors model_fun.py exactly):
  - A deque of length 24 stores td_now = target_demand * occ_shock each period
  - g'_t = (1/12) * sum_{k=0..11} [ td_history[23-k] - td_history[11-k] ]
         = average of the last 12 year-over-year monthly differences
  - projected_td = td_now + g'_t
  - vacs_gap     = max(0, projected_td - current_demand)   [current_demand = emp + curr_vacs]
  - vacs_create  = round(vacs_gap)

The deque is pre-filled with the 24 months of real historical shocks that
precede the burn-in period (cols 0..23 of the extended shock array), exactly
as in network_input_builder / model_fun.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.filters import hp_filter
from collections import deque

# ---------------------------------------------------------------------------
# Load GDP data
# ---------------------------------------------------------------------------
path = "/Users/ebbamark/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"
macro_observations = pd.read_csv(path + "data/macro_vars/collated_macro_observations.csv")
macro_observations["DATE"] = pd.to_datetime(macro_observations["DATE"])

# HP filter GDP
realgdp = macro_observations[["DATE", "REALGDP"]].dropna(subset=["REALGDP"]).reset_index()
realgdp['log_REALGDP'] = np.log2(realgdp['REALGDP'])
cycle, trend = hp_filter.hpfilter(realgdp['log_REALGDP'], lamb=129600)
realgdp['log_Cycle'] = cycle + 1

# Build full monthly series
gdp_dat_pd = realgdp.set_index('DATE').sort_index()
monthly_dates_full = pd.date_range(
    start=gdp_dat_pd.index.min(), end=gdp_dat_pd.index.max(), freq='MS')
gdp_dat_monthly = gdp_dat_pd.reindex(monthly_dates_full).interpolate(method='linear')

# ---------------------------------------------------------------------------
# Build the extended shock array (mirrors network_input_builder):
#   cols 0..23            → deque pre-fill  (24 months of real history)
#   cols 24..24+delay-1   → burn-in period  (delay months)
#   cols 24+delay..       → calibration window
# ---------------------------------------------------------------------------
calib_date  = ["2000-12-01", "2019-05-01"]
delay       = 24
extended_start = pd.to_datetime(calib_date[0]) - pd.DateOffset(months=24 + delay)

gdp_extended = gdp_dat_monthly.loc[
    (gdp_dat_monthly.index >= extended_start) &
    (gdp_dat_monthly.index <= calib_date[1]), 'log_Cycle'].values

# Calibration-window slice (for plotting alignment)
calib_mask   = (gdp_dat_monthly.index >= calib_date[0]) & \
               (gdp_dat_monthly.index <= calib_date[1])
gdp_calib    = gdp_dat_monthly.loc[calib_mask, 'log_Cycle'].values
calib_dates  = gdp_dat_monthly.loc[calib_mask].index

print(f"Extended shock array length : {len(gdp_extended)}")
print(f"Expected (24+delay+calib)   : 24 + {delay} + {len(gdp_calib)} = {24+delay+len(gdp_calib)}")
print(f"Shock range: {gdp_extended.min():.4f} to {gdp_extended.max():.4f}")

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
target_demand = 1000      # baseline employment
sep_rate      = 0.03      # monthly separation rate
matching_efficiency    = 0.35  # m in fill_rate = m * sqrt(U/V); calibrate to match mean vacancy rate
gamma_v_values = [0.0, 0.5, 1.0]

# Loop runs over cols 24.. of the extended array (burn-in + calibration).
# time_steps = total cols - 24 = delay + calib_length
time_steps = len(gdp_extended) - 24

# Dates for burn-in + calibration (for plotting)
burnin_dates = pd.date_range(
    end=pd.to_datetime(calib_date[0]) - pd.DateOffset(months=1),
    periods=delay, freq='MS')
plot_dates = burnin_dates.append(calib_dates)

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
results = {}

for gamma_v in gamma_v_values:
    print(f"\n=== gamma_v = {gamma_v} ===")

    # Pre-fill deque with cols 0..23 (real historical shocks before burn-in)
    td_history = deque(
        [target_demand * float(gdp_extended[k]) for k in range(24)],
        maxlen=24)

    emp      = target_demand
    unemp    = int(target_demand * 0.05)
    curr_vacs = 0

    emp_series       = []
    unemp_series     = []
    vacs_series      = []
    vac_rate_series  = []
    u_rate_series    = []
    fill_rate_series = []
    td_now_series    = []
    projected_series = []
    g_prime_series   = []
    vacs_create_series = []

    for t in range(time_steps - 1):
        # Index into extended array offset by 24 (skip deque pre-fill cols)
        occ_shock = float(gdp_extended[24 + t])
        td_now    = target_demand * occ_shock

        # --- g_prime: integrated YoY signal (mirrors model_fun.py exactly) ---
        td_history.append(td_now)
        g_prime = (1.0 / 12.0) * sum(
            td_history[23 - k] - td_history[11 - k]
            for k in range(12))

        # --- projected demand and vacancy gap ---
        projected_td = td_now + gamma_v * g_prime   # gamma_v=0 → pure demand gap
        current_demand = emp + curr_vacs
        vacs_gap    = max(0.0, projected_td - current_demand)
        vacs_create = int(np.round(vacs_gap))
        curr_vacs  += vacs_create

        # --- separations ---
        seps   = int(np.round(emp * sep_rate))
        emp   -= seps
        unemp += seps

        # Cobb-Douglas fill rate: f = m * sqrt(U/V)
        # Falls in tight markets (low U/V), rises in slack markets (high U/V)
        uv_ratio    = unemp / (curr_vacs + 1)
        fill_rate_t = min(1.0, matching_efficiency * np.sqrt(uv_ratio))

        # --- hiring ---
        hires     = min(int(np.round(curr_vacs * fill_rate_t)), curr_vacs, unemp)
        emp      += hires
        unemp    -= hires
        curr_vacs -= hires

        # --- rates ---
        vac_rate = curr_vacs / (emp + curr_vacs) if (emp + curr_vacs) > 0 else 0
        u_rate   = unemp    / (emp + unemp)      if (emp + unemp)      > 0 else 0

        emp_series.append(emp)
        unemp_series.append(unemp)
        vacs_series.append(curr_vacs)
        vac_rate_series.append(vac_rate)
        u_rate_series.append(u_rate)
        fill_rate_series.append(fill_rate_t)
        td_now_series.append(td_now)
        projected_series.append(projected_td)
        g_prime_series.append(g_prime)
        vacs_create_series.append(vacs_create)

    results[gamma_v] = dict(
        emp=emp_series, unemp=unemp_series, vacs=vacs_series,
        vac_rate=vac_rate_series, u_rate=u_rate_series,
        fill_rate=fill_rate_series, td_now=td_now_series,
        projected_td=projected_series, g_prime=g_prime_series,
        vacs_create=vacs_create_series)

    print(f"  Mean vac rate : {np.mean(vac_rate_series):.4f}")
    print(f"  Mean u rate   : {np.mean(u_rate_series):.4f}")
    print(f"  Mean g_prime  : {np.mean(g_prime_series):.2f}")
    print(f"  Max vacs/period: {np.max(vacs_create_series):.0f}")

# ---------------------------------------------------------------------------
# Load observed vacancy rate and unemployment rate
# ---------------------------------------------------------------------------
jolts = pd.read_csv(path + "data/macro_vars/JTSJOR.csv", delimiter=',', decimal='.')
jolts["DATE"]    = pd.to_datetime(jolts["DATE"])
jolts_calib      = jolts[(jolts['DATE'] >= calib_date[0]) & (jolts['DATE'] <= calib_date[1])].copy()
jolts_calib["VACRATE"] = jolts_calib['JTSJOR'] / 100

obs_uer = macro_observations[["DATE", "UNRATE"]].dropna(subset=["UNRATE"])
obs_uer = obs_uer[(obs_uer['DATE'] >= calib_date[0]) & (obs_uer['DATE'] <= calib_date[1])].copy()
obs_uer["UNRATE"] = obs_uer["UNRATE"] / 100

# GDP shock trimmed to simulation length (burn-in + calib, offset by 24)
gdp_plot = gdp_extended[24:24 + time_steps - 1]

# Trim plot_dates to match simulation output length (time_steps-1 rows)
plot_dates_trimmed = plot_dates[:time_steps - 1]

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# Panel 1: Vacancy rate
ax = axes[0]
for gamma_v in gamma_v_values:
    ax.plot(plot_dates_trimmed, results[gamma_v]['vac_rate'],
            label=f'Simulated (γ_v={gamma_v})', linewidth=2, alpha=0.8)
ax.plot(jolts_calib['DATE'], jolts_calib['VACRATE'],
        label='Observed (JOLTS)', linestyle='--', color='black', linewidth=2)
ax.axvline(x=pd.to_datetime(calib_date[0]), color='red', linestyle=':', alpha=0.5, label='Calib start')
ax.set_ylabel('Vacancy Rate')
ax.set_title("Vacancy Rate: Simulated vs. Observed")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: g_prime signal with GDP shock on twin axis
ax = axes[1]
ax2 = ax.twinx()
for gamma_v in gamma_v_values:
    ax.plot(plot_dates_trimmed, results[gamma_v]['g_prime'],
            label=f'g_prime (γ_v={gamma_v})', linewidth=2, alpha=0.8)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.6)
ax.axvline(x=pd.to_datetime(calib_date[0]), color='red', linestyle=':', alpha=0.5)
gdp_line, = ax2.plot(plot_dates_trimmed, gdp_plot,
                     label='GDP shock', color='black', linewidth=1.5,
                     linestyle='--', alpha=0.5)
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
ax.set_ylabel("g_prime (headcount units)")
ax2.set_ylabel("GDP shock multiplier", color='black', alpha=0.7)
ax.set_title("Integrated YoY Demand Growth Signal (g_prime) with GDP Shock")
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines + [gdp_line], labels + ['GDP shock'], fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Projected vs actual demand with GDP shock on twin axis (gamma_v=0)
ax = axes[2]
ax2 = ax.twinx()
gamma_v = 0.0
ax.plot(plot_dates_trimmed, results[gamma_v]['td_now'],
        label='td_now', linewidth=2, linestyle='--', color='steelblue')
ax.plot(plot_dates_trimmed, results[gamma_v]['projected_td'],
        label='projected_td', linewidth=2, color='darkorange')
ax.axhline(y=target_demand, color='gray', linestyle=':', alpha=0.6, label='Target demand')
ax.axvline(x=pd.to_datetime(calib_date[0]), color='red', linestyle=':', alpha=0.5)
gdp_line, = ax2.plot(plot_dates_trimmed, gdp_plot,
                     label='GDP shock', color='black', linewidth=1.5,
                     linestyle='--', alpha=0.5)
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
ax.set_ylabel('Demand (headcount)')
ax2.set_ylabel('GDP shock multiplier', color='black', alpha=0.7)
ax.set_title(f"Projected vs Effective Demand (γ_v={gamma_v}) with GDP Shock")
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines + [gdp_line], labels + ['GDP shock'], fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: Unemployment rate with observed UER (gamma_v=0)
ax = axes[3]
gamma_v = 0.0
ax.plot(plot_dates_trimmed, results[gamma_v]['u_rate'],
        label='Simulated UER', color='steelblue', linewidth=2)
ax.plot(obs_uer['DATE'], obs_uer['UNRATE'],
        label='Observed UER', color='black', linestyle='--', linewidth=2)
ax.axvline(x=pd.to_datetime(calib_date[0]), color='red', linestyle=':',
           alpha=0.5, label='Calib start')
ax.set_ylabel('Unemployment Rate')
ax.set_xlabel('Date')
ax.set_title(f"Unemployment Rate: Simulated vs. Observed (γ_v={gamma_v})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(path + 'vacancy_simulation_simple.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved.")
plt.show()