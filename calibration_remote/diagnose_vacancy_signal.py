"""
Diagnostic: does g_prime_norm cancel across occupations in full_omn vs single_node?

Runs purely on occ_shocks_data — no gamma_v, d_v, or model mechanics needed.
Computes per-occupation g_prime_norm at every time step, then plots:
  1. Employment-weighted mean signal over time vs observed VACRATE
  2. Cross-occupation spread (10th–90th pct band) to show how much cancellation occurs
  3. Direct comparison of full_omn vs single_node
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from network_input_builder import network_input_builder
from collate_macro_vars import macro_observations, recession_spans

CALIB_DATE  = ["2001-01-01", "2019-05-01"]
DELAY       = 24
NETWORKS    = ["full_omn", "single_node"]
SAVE        = False
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/figures/")

# If True, convert full_omn Bartik growth-rate shocks to a cumulative level
# index (cumprod, normalised to 1 at col 0) before computing g_prime_norm.
# This makes the shock series comparable to the HP-cycle level used in
# single_node, so g_prime_norm measures cyclical position rather than
# acceleration.
# Set False to use raw Bartik growth-rate shocks (original behaviour).
# Set True to convert full_omn shocks to cumprod level index — NOT recommended,
# see diagnostic results: it creates a non-stationary trending series.
CONVERT_TO_LEVELS = False

# If True, replace full_omn shocks before REPLACE_BEFORE with the single_node
# GDP HP cycle (broadcast to all occupations).  This removes the artefacts from
# linearly-interpolated annual BEA data in the pre-fill / burn-in period.
REPLACE_EARLY_SHOCKS  = True
REPLACE_BEFORE        = "2002-08-01"
TRANSITION_MONTHS     = 18   # months over which to blend GDP → Bartik before cutoff

# Grid of (d_v, gamma_v) pairs for the drift simulation
# d_v: constant monthly drift added to target_vr each step
# gamma_v: weight on g_prime_norm signal
D_V_GRID     = [-0.0002, 0.0, 0.0002, 0.0008]
GAMMA_V_GRID = [0.05, 0.15, 0.30]
D_INITIAL    = 0.032   # initial target_vr (matches model default)


def _make_date_index(calib_date, delay, n_cols):
    """Return the full monthly DatetimeIndex for an occ_shocks_dat array."""
    extended_start = pd.to_datetime(calib_date[0]) - pd.DateOffset(months=24 + delay)
    return pd.date_range(start=extended_start, periods=n_cols, freq="MS")


def _apply_early_replacement(shocks_fo, shocks_gdp_1d, calib_date, delay):
    """
    Replace early columns of the full_omn shock array with the GDP HP cycle,
    with a linear blend over TRANSITION_MONTHS before REPLACE_BEFORE.

    Before (REPLACE_BEFORE - TRANSITION_MONTHS) : 100% GDP cycle
    Blending window                              : linear GDP → Bartik
    From REPLACE_BEFORE onward                  : 100% Bartik (unchanged)

    Parameters
    ----------
    shocks_fo     : np.ndarray (n_occs, T)  — full_omn shock array (modified in-place copy)
    shocks_gdp_1d : np.ndarray (T,)         — single_node GDP HP cycle (1-D)

    Returns
    -------
    np.ndarray (n_occs, T) with early columns replaced/blended.
    """
    result      = shocks_fo.copy()
    dates_full  = _make_date_index(calib_date, delay, shocks_fo.shape[1])
    cutoff      = pd.to_datetime(REPLACE_BEFORE)
    trans_start = cutoff - pd.DateOffset(months=TRANSITION_MONTHS)

    # Find integer index of the transition start
    trans_idx = int(np.searchsorted(dates_full, trans_start))

    for i, d in enumerate(dates_full):
        if d < trans_start:
            result[:, i] = shocks_gdp_1d[i]
        elif d < cutoff:
            # Linear weight: 0 = pure GDP at trans_start, 1 = pure Bartik at cutoff
            w = (i - trans_idx) / TRANSITION_MONTHS
            result[:, i] = (1.0 - w) * shocks_gdp_1d[i] + w * shocks_fo[:, i]
        # else: pure Bartik — leave unchanged

    n_hard   = int((dates_full < trans_start).sum())
    n_blended = int(((dates_full >= trans_start) & (dates_full < cutoff)).sum())
    print(f"    Early replacement: {n_hard} cols pure GDP, "
          f"{n_blended} cols blended ({TRANSITION_MONTHS}-month ramp).")
    return result


def compute_g_prime_signals(nw, calib_date=CALIB_DATE, delay=DELAY,
                            convert_to_levels=CONVERT_TO_LEVELS,
                            gdp_shocks=None):
    """
    Simulate the demand-history deque forward and record g_prime_norm per
    occupation per time step.  No model mechanics, no calibrated parameters.

    Parameters
    ----------
    convert_to_levels : bool
        If True and nw != 'single_node', convert Bartik growth-factor shocks
        to a cumulative level index via cumprod (NOT recommended — see notes).
    gdp_shocks : np.ndarray of shape (1, T) or None
        If provided and REPLACE_EARLY_SHOCKS is True, columns before
        REPLACE_BEFORE are overwritten with the GDP HP cycle (broadcast to
        all occupations) to remove annual-interpolation artefacts.

    Returns
    -------
    df : pd.DataFrame
        Columns: t, date, occupation_id, target_demand, g_prime_norm
    date_index : pd.DatetimeIndex
        Monthly dates aligned to t=0 … T-1 of the calibration window.
    """
    print(f"  Building network: {nw}...")
    mod_data, net, vacs, occ_ids, occ_shocks_dat = network_input_builder(
        nw, complete=False, delay=delay, calib_date=calib_date
    )

    # Replace / blend early full_omn columns with GDP HP cycle if requested
    if REPLACE_EARLY_SHOCKS and nw != "single_node" and gdp_shocks is not None:
        print(f"  [{nw}] Applying early-shock replacement...")
        occ_shocks_dat = _apply_early_replacement(
            occ_shocks_dat, gdp_shocks[0], calib_date, delay
        )

    # Convert Bartik growth-rate shocks to cumulative level index.
    # single_node already uses an HP-cycle level series — skip it.
    if convert_to_levels and nw != "single_node":
        cum = np.cumprod(occ_shocks_dat, axis=1)
        occ_shocks_dat = cum / cum[:, [0]]
        print(f"  [{nw}] Converted Bartik growth factors → cumulative level index.")

    # occ_shocks_dat has shape (n_occs, 24 + delay + calib_length).
    # Cols 0..23  → deque pre-fill
    # Cols 24..   → loop steps (indexed as 24+t inside model)
    n_loop_steps = occ_shocks_dat.shape[1] - 24

    # Date index aligned to the loop (t=0 = calib_date[0] + delay months)
    calib_start   = pd.to_datetime(calib_date[0])
    loop_start    = calib_start  # col 24 corresponds to calib_date[0] minus delay
    date_index    = pd.date_range(
        start=loop_start - pd.DateOffset(months=delay),
        periods=n_loop_steps,
        freq="MS"
    )

    # Pre-fill deques with cols 0..23 of occ_shocks_dat (real shock history)
    for occ in net:
        td_base = float(occ.target_demand)
        history = [
            td_base * float(occ_shocks_dat[occ.occupation_id, k])
            for k in range(24)
        ]
        occ.td_history = deque(history, maxlen=24)

    records = []
    for t in range(n_loop_steps - 1):
        curr_shocks = np.take(occ_shocks_dat, indices=24 + t, axis=1)

        for occ in net:
            occ_shock = curr_shocks[occ.occupation_id]
            td_now    = float(occ.target_demand * occ_shock)
            occ.td_history.append(td_now)

            g_prime_norm = (1.0 / 12.0) * sum(
                (occ.td_history[23 - k] - occ.td_history[11 - k])
                / (occ.td_history[11 - k] + 1e-6)
                for k in range(12)
            )

            records.append({
                't':             t,
                'date':          date_index[t],
                'occupation_id': occ.occupation_id,
                'target_demand': occ.target_demand,
                'g_prime_norm':  g_prime_norm,
            })

    df = pd.DataFrame(records)
    return df, date_index


def summarise_signals(df):
    """
    Compute per-time-step summary statistics of g_prime_norm,
    weighted by target_demand (occupation size proxy).
    """
    out = []
    for t, grp in df.groupby('t'):
        w   = grp['target_demand'].values.astype(float)
        g   = grp['g_prime_norm'].values
        w   = w / w.sum()

        wmean = np.sum(w * g)
        p10   = np.percentile(g, 10)
        p25   = np.percentile(g, 25)
        p75   = np.percentile(g, 75)
        p90   = np.percentile(g, 90)
        std   = np.std(g)

        out.append(dict(
            t=t,
            date=grp['date'].iloc[0],
            wmean=wmean,
            p10=p10, p25=p25, p75=p75, p90=p90,
            std=std,
        ))
    return pd.DataFrame(out)


def simulate_target_vr(summ, d_v, gamma_v, d_initial=D_INITIAL, calib_date=CALIB_DATE):
    """
    Simulate the vacancy-rate accumulation mechanism using the pre-computed
    employment-weighted mean g_prime_norm signal.

    At each step (after the delay burn-in):
        target_vr_t = target_vr_{t-1} + d_v + gamma_v * wmean_t

    Parameters
    ----------
    summ : pd.DataFrame
        Output of summarise_signals() — must have columns 'date' and 'wmean'.
    d_v : float
        Constant monthly drift.
    gamma_v : float
        Weight on the g_prime_norm signal.
    d_initial : float
        Starting value of target_vr.

    Returns
    -------
    pd.DataFrame with columns 'date' and 'target_vr'.
    """
    # Restrict to calibration window (post-delay)
    s = summ[summ['date'] >= pd.to_datetime(calib_date[0])].copy().reset_index(drop=True)
    vr = np.empty(len(s))
    vr[0] = d_initial
    for i in range(1, len(s)):
        vr[i] = vr[i - 1] + d_v + gamma_v * s.loc[i - 1, 'wmean']
    s['target_vr'] = vr
    return s[['date', 'target_vr']]


def extract_shock_series(calib_date=CALIB_DATE, delay=DELAY):
    """
    Extract and summarise the raw shock arrays for full_omn and single_node,
    covering the full time range (pre-fill + calibration window).

    When REPLACE_EARLY_SHOCKS is True, also returns a version of full_omn
    with pre-REPLACE_BEFORE columns substituted by the GDP HP cycle.

    Returns
    -------
    pd.DataFrame with columns:
        date
        single_node               : the single HP-cycle shock value
        full_omn_raw_wmean        : employment-weighted mean of raw Bartik shocks
        full_omn_replaced_wmean   : same, but early cols replaced with GDP cycle
        full_omn_level_wmean      : employment-weighted mean after cumprod (for reference)
    """
    # ── single_node ──────────────────────────────────────────────────────────
    _, net_sn, _, _, shocks_sn = network_input_builder(
        "single_node", complete=False, delay=delay, calib_date=calib_date
    )
    n_cols     = shocks_sn.shape[1]
    dates_full = _make_date_index(calib_date, delay, n_cols)
    sn_series  = shocks_sn[0, :]

    # ── full_omn ─────────────────────────────────────────────────────────────
    _, net_fo, _, _, shocks_fo = network_input_builder(
        "full_omn", complete=False, delay=delay, calib_date=calib_date
    )
    weights = np.array([float(occ.target_demand) for occ in net_fo])
    weights = weights / weights.sum()

    fo_raw_wmean = weights @ shocks_fo

    # Apply early-period replacement (with smooth blend) for comparison
    shocks_fo_replaced = shocks_fo.copy()
    if REPLACE_EARLY_SHOCKS:
        shocks_fo_replaced = _apply_early_replacement(
            shocks_fo, shocks_sn[0], calib_date, delay
        )

    fo_replaced_wmean = weights @ shocks_fo_replaced

    # Cumulative level (for reference only)
    cum = np.cumprod(shocks_fo, axis=1)
    fo_level_wmean = weights @ (cum / cum[:, [0]])

    return pd.DataFrame({
        'date':                   dates_full,
        'single_node':            sn_series,
        'full_omn_raw_wmean':     fo_raw_wmean,
        'full_omn_replaced_wmean': fo_replaced_wmean,
        'full_omn_level_wmean':   fo_level_wmean,
    })


# ── load single_node GDP shocks once for optional early-period replacement ─────
print("\nLoading single_node GDP HP cycle for early-shock replacement...")
_, _, _, _, _gdp_shocks = network_input_builder(
    "single_node", complete=False, delay=DELAY, calib_date=CALIB_DATE
)

# ── run for each network ───────────────────────────────────────────────────────
results = {}
for nw in NETWORKS:
    print(f"\nComputing signals for {nw}...")
    gdp_arg = _gdp_shocks if nw != "single_node" else None
    df_raw, dates = compute_g_prime_signals(nw, gdp_shocks=gdp_arg)
    results[nw] = summarise_signals(df_raw)
    print(f"  Done. {len(df_raw)} occupation-step records.")

# ── observed VACRATE aligned to calibration window ────────────────────────────
obs = macro_observations.copy()
obs = obs[(obs['DATE'] >= CALIB_DATE[0]) & (obs['DATE'] <= CALIB_DATE[1])].dropna(subset=['VACRATE'])

# ── plot ──────────────────────────────────────────────────────────────────────
colors = {'full_omn': 'steelblue', 'single_node': 'darkorange'}

fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=False)

# ── Panel 1: employment-weighted mean signal, both networks ───────────────────
ax = axes[0]
for nw, summ in results.items():
    calib_mask = summ['date'] >= pd.to_datetime(CALIB_DATE[0])
    s = summ[calib_mask]
    ax.plot(s['date'], s['wmean'], color=colors[nw], linewidth=1.8, label=nw)
    ax.fill_between(s['date'], s['p25'], s['p75'],
                    color=colors[nw], alpha=0.15)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax.set_ylabel('g_prime_norm\n(employment-wtd mean ± IQR)', fontsize=9)
ax.set_title('Economy-wide weighted mean signal — does it track the cycle?',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.tick_params(axis='x', labelsize=8)

# ── Panel 2: cross-occupation spread (full_omn only) ─────────────────────────
ax = axes[1]
nw = 'full_omn'
summ = results[nw]
calib_mask = summ['date'] >= pd.to_datetime(CALIB_DATE[0])
s = summ[calib_mask]
ax.fill_between(s['date'], s['p10'], s['p90'],
                color=colors[nw], alpha=0.20, label='P10–P90')
ax.fill_between(s['date'], s['p25'], s['p75'],
                color=colors[nw], alpha=0.35, label='IQR')
ax.plot(s['date'], s['wmean'], color=colors[nw], linewidth=1.8, label='Wtd mean')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax.set_ylabel('g_prime_norm\n(full_omn cross-occ. spread)', fontsize=9)
ax.set_title('full_omn: cross-occupation spread of g_prime_norm\n'
             '(wide band = high idiosyncratic noise; if mean ≠ 0 signal exists)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.tick_params(axis='x', labelsize=8)

# ── Panel 3: observed VACRATE for visual comparison ───────────────────────────
ax = axes[2]
ax2 = ax.twinx()

for nw, summ in results.items():
    calib_mask = summ['date'] >= pd.to_datetime(CALIB_DATE[0])
    s = summ[calib_mask]
    ax.plot(s['date'], s['wmean'], color=colors[nw], linewidth=1.8,
            label=f'{nw} wtd mean signal')

ax2.plot(obs['DATE'], obs['VACRATE'], color='black', linewidth=1.5,
         linestyle='dotted', label='Observed VACRATE')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax.set_ylabel('g_prime_norm (wtd mean)', fontsize=9)
ax2.set_ylabel('Observed vacancy rate', fontsize=9, color='black')
ax.set_title('Weighted mean signal vs observed VACRATE\n'
             '(if signal tracks VACRATE direction, mechanism can work)',
             fontsize=10, fontweight='bold')

lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
ax.tick_params(axis='x', labelsize=8)

level_tag = " [full_omn shocks → cumulative levels]" if CONVERT_TO_LEVELS else " [raw growth-rate shocks]"
fig.suptitle(f'Vacancy signal diagnostic: g_prime_norm cancellation analysis{level_tag}',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])

if SAVE:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_PATH, 'vacancy_signal_diagnostic.jpg'),
                bbox_inches='tight', dpi=200)
    print(f"\nSaved to {OUTPUT_PATH}vacancy_signal_diagnostic.jpg")
else:
    plt.show()

# ── Figure 2: drift simulation — can target_vr track observed VACRATE? ────────
obs_calib = obs[obs['DATE'] >= pd.to_datetime(CALIB_DATE[0])].copy()

n_rows = len(NETWORKS)
n_cols = len(D_V_GRID)
fig2, axes2 = plt.subplots(n_rows, n_cols,
                            figsize=(5 * n_cols, 4 * n_rows),
                            sharex=True, sharey='row')

for r, nw in enumerate(NETWORKS):
    summ = results[nw]
    for c, d_v in enumerate(D_V_GRID):
        ax = axes2[r, c]
        ax2b = ax.twinx()

        for gamma_v in GAMMA_V_GRID:
            sim = simulate_target_vr(summ, d_v=d_v, gamma_v=gamma_v)
            ax.plot(sim['date'], sim['target_vr'],
                    linewidth=1.4,
                    label=f'γ_v={gamma_v}')

        ax2b.plot(obs_calib['DATE'], obs_calib['VACRATE'],
                  color='black', linewidth=1.2, linestyle='dotted',
                  label='Observed VACRATE')
        ax2b.set_ylabel('Observed VACRATE', fontsize=8, color='black')
        ax2b.tick_params(axis='y', labelsize=7)

        ax.axhline(D_INITIAL, color='grey', linewidth=0.7, linestyle=':')
        for _, row in recession_spans.iterrows():
            ax.axvspan(row['start'], row['end'], color='grey', alpha=0.15)
        ax.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
        ax.set_title(f'{nw}  |  d_v={d_v:+.4f}', fontsize=9, fontweight='bold')
        ax.set_ylabel('Simulated target_vr', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

        if r == 0 and c == n_cols - 1:
            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2b.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc='upper left')

fig2.suptitle(
    'Drift simulation: target_vr = target_vr + d_v + γ_v·g_prime_norm\n'
    'Can the signal + drift recover observed VACRATE?',
    fontsize=11, fontweight='bold'
)
fig2.tight_layout(rect=[0, 0, 1, 0.95])

if SAVE:
    plt.savefig(os.path.join(OUTPUT_PATH, 'vacancy_drift_simulation.jpg'),
                bbox_inches='tight', dpi=200)
    print(f"Saved drift simulation to {OUTPUT_PATH}vacancy_drift_simulation.jpg")
else:
    plt.show()

# ── Figure 3: raw shock structure ────────────────────────────────────────────
print("\nExtracting shock series for structure plot...")
shock_df = extract_shock_series()

fig3, axes3 = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

# Panel 1: single_node HP-cycle shock
ax = axes3[0]
ax.plot(shock_df['date'], shock_df['single_node'],
        color='darkorange', linewidth=1.6, label='single_node (HP-cycle level)')
ax.axhline(1.0, color='black', linewidth=0.7, linestyle='--')
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_ylabel('Shock value', fontsize=9)
ax.set_title('single_node shock: HP-filtered cyclical component of log real GDP (+1)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# Panel 2: full_omn raw Bartik — both original and with early-period replacement
ax = axes3[1]
ax.plot(shock_df['date'], shock_df['full_omn_raw_wmean'],
        color='steelblue', linewidth=1.4, alpha=0.5,
        label='full_omn raw (original)')
if REPLACE_EARLY_SHOCKS:
    ax.plot(shock_df['date'], shock_df['full_omn_replaced_wmean'],
            color='steelblue', linewidth=1.8,
            label=f'full_omn raw (GDP cycle before {REPLACE_BEFORE})')
    ax.axvline(pd.to_datetime(REPLACE_BEFORE), color='steelblue',
               linewidth=0.9, linestyle=':', alpha=0.7)
ax.axhline(1.0, color='black', linewidth=0.7, linestyle='--')
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_ylabel('Shock value', fontsize=9)
ax.set_title('full_omn shock: employment-weighted mean of raw Bartik shocks\n'
             f'(faint = original; solid = GDP cycle substituted before {REPLACE_BEFORE})',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# Panel 3: overlay of single_node and replaced full_omn for direct comparison
ax = axes3[2]
ax.plot(shock_df['date'], shock_df['single_node'],
        color='darkorange', linewidth=1.6, label='single_node (HP-cycle)')
col = 'full_omn_replaced_wmean' if REPLACE_EARLY_SHOCKS else 'full_omn_raw_wmean'
ax.plot(shock_df['date'], shock_df[col],
        color='steelblue', linewidth=1.6,
        label='full_omn (after replacement)' if REPLACE_EARLY_SHOCKS else 'full_omn raw')
ax.axhline(1.0, color='black', linewidth=0.7, linestyle='--')
if REPLACE_EARLY_SHOCKS:
    ax.axvline(pd.to_datetime(REPLACE_BEFORE), color='steelblue',
               linewidth=0.9, linestyle=':', alpha=0.7)
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_ylabel('Shock value', fontsize=9)
ax.set_title('Direct comparison: single_node vs full_omn (post-replacement)\n'
             'Amplitude difference explains weaker g_prime_norm in full_omn',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(shock_df['date'].min(), shock_df['date'].max())

replace_tag = f" [early shocks → GDP before {REPLACE_BEFORE}]" if REPLACE_EARLY_SHOCKS else ""
fig3.suptitle(f'Shock structure comparison: single_node vs full_omn{replace_tag}',
              fontsize=12, fontweight='bold')
fig3.tight_layout(rect=[0, 0, 1, 0.97])

if SAVE:
    plt.savefig(os.path.join(OUTPUT_PATH, 'shock_structure.jpg'),
                bbox_inches='tight', dpi=200)
    print(f"Saved shock structure to {OUTPUT_PATH}shock_structure.jpg")
else:
    plt.show()

# ── print summary statistics ──────────────────────────────────────────────────
print("\n=== Signal summary (calibration window, GFC period 2008-2010) ===")
for nw, summ in results.items():
    gfc = summ[(summ['date'] >= '2008-01-01') & (summ['date'] <= '2010-12-01')]
    print(f"\n{nw}:")
    print(f"  Wtd mean g_prime_norm (GFC): {gfc['wmean'].mean():.4f}")
    print(f"  Std of g_prime_norm  (GFC): {gfc['std'].mean():.4f}")
    print(f"  Signal-to-noise ratio:       {abs(gfc['wmean'].mean()) / gfc['std'].mean():.3f}")
    exp = summ[(summ['date'] >= '2004-01-01') & (summ['date'] <= '2007-01-01')]
    print(f"  Wtd mean g_prime_norm (exp): {exp['wmean'].mean():.4f}")

# ── Figure 4: GDP level signal vs HP cycle ────────────────────────────────────
# Build the normalised REALGDP level over the same extended window used in
# test_run_time.py when GDP_SIGNAL='gdp_level'.
print("\nBuilding GDP level signal for Figure 4...")
_gdp_csv_path = os.path.join(PROJECT_ROOT, "..", "data", "macro_vars", "GDPC1.csv")
_realgdp_raw  = pd.read_csv(_gdp_csv_path)
_realgdp_raw["DATE"]    = pd.to_datetime(_realgdp_raw["DATE"])
_realgdp_raw["REALGDP"] = _realgdp_raw["GDPC1"]
_realgdp_raw = _realgdp_raw.set_index("DATE").sort_index()[["REALGDP"]]

_extended_start = pd.to_datetime(CALIB_DATE[0]) - pd.DateOffset(months=24 + DELAY)
_monthly_idx    = pd.date_range(start=_extended_start, end=CALIB_DATE[1], freq="MS")
_realgdp_interp = _realgdp_raw.reindex(_monthly_idx).interpolate("linear")
_gdp_level_arr  = _realgdp_interp["REALGDP"].values
_gdp_norm       = _gdp_level_arr / _gdp_level_arr[0]   # normalised to 1 at start

# HP-cycle shock array (single_node) — already loaded as _gdp_shocks (shape 1×T)
_hp_dates = _make_date_index(CALIB_DATE, DELAY, _gdp_shocks.shape[1])
_hp_series = _gdp_shocks[0, :]   # shape (T,)

# Align GDP level array to same date range
_gdp_norm_series = pd.Series(_gdp_norm, index=_monthly_idx)

# Restrict to calibration window for plotting clarity
_calib_mask_hp  = _hp_dates >= pd.to_datetime(CALIB_DATE[0])
_calib_dates_hp = _hp_dates[_calib_mask_hp]
_hp_calib       = _hp_series[_calib_mask_hp]
_gdp_norm_calib = _gdp_norm_series.reindex(_calib_dates_hp)

fig4, axes4 = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Panel 1: both signals overlaid, VACRATE on twin axis
ax = axes4[0]
ax2 = ax.twinx()
ax.plot(_calib_dates_hp, _hp_calib,
        color='darkorange', linewidth=1.8, label='HP cycle (single_node signal)')
ax.axhline(1.0, color='darkorange', linewidth=0.7, linestyle='--', alpha=0.5)
ax.plot(_calib_dates_hp, _gdp_norm_calib,
        color='steelblue', linewidth=1.8, label='REALGDP / REALGDP₀ (gdp_level signal)')
ax2.plot(obs['DATE'], obs['VACRATE'],
         color='black', linewidth=1.4, linestyle='dotted', label='Observed VACRATE')
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_ylabel('Signal value (HP cycle ≈ 1; GDP level > 1)', fontsize=9)
ax2.set_ylabel('Observed vacancy rate', fontsize=9)
ax.set_title('HP-filtered cycle vs normalised REALGDP level as vacancy signal\n'
             '(HP cycle is stationary around 1; GDP level trends upward)',
             fontsize=10, fontweight='bold')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper left')
ax.tick_params(axis='x', labelsize=8)

# Panel 2: deviations from 1 — makes the cyclical component of GDP level visible
ax = axes4[1]
ax2 = ax.twinx()
_hp_dev  = _hp_calib - 1.0
_gdp_dev = _gdp_norm_calib - 1.0
ax.plot(_calib_dates_hp, _hp_dev,
        color='darkorange', linewidth=1.8, label='HP cycle − 1 (detrended, mean-zero)')
ax.plot(_calib_dates_hp, _gdp_dev,
        color='steelblue', linewidth=1.8, label='GDP level − 1 (growing trend)')
ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
ax2.plot(obs['DATE'], obs['VACRATE'],
         color='black', linewidth=1.4, linestyle='dotted', label='Observed VACRATE')
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_ylabel('Signal − 1 (deviation from baseline)', fontsize=9)
ax2.set_ylabel('Observed vacancy rate', fontsize=9)
ax.set_title('Deviations from baseline: HP cycle centres on zero; GDP level accumulates a trend\n'
             '(with vac_method="gdp_growth", gamma_v acts on the level not the cycle)',
             fontsize=10, fontweight='bold')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper left')
ax.tick_params(axis='x', labelsize=8)
ax.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))

fig4.suptitle('Figure 4: GDP level signal for vacancy mechanism\n'
              '(signal fed to model as vac_signal_dat when GDP_SIGNAL="gdp_level")',
              fontsize=12, fontweight='bold')
fig4.tight_layout(rect=[0, 0, 1, 0.96])

if SAVE:
    plt.savefig(os.path.join(OUTPUT_PATH, 'gdp_level_signal.jpg'),
                bbox_inches='tight', dpi=200)
    print(f"Saved GDP level signal figure to {OUTPUT_PATH}gdp_level_signal.jpg")
else:
    plt.show()

# ── Figure 5: g_prime_norm computed from GDP level ────────────────────────────
# Apply the same deque-based YoY growth-rate averaging formula used in the
# model, but driven by the normalised REALGDP level instead of the HP cycle.
# Pre-fill cols 0..23 of _gdp_norm; loop from col 24 onward.
# Since target_demand is constant in single_node, it cancels in the ratio and
# we can set td_base = 1.0.
print("\nComputing g_prime_norm from GDP level signal...")
_td_base  = 1.0
_gp_history = deque([_td_base * float(_gdp_norm[k]) for k in range(24)], maxlen=24)

_n_loop = len(_gdp_norm) - 24
_gdp_gp_vals  = []
_gdp_gp_dates = []

for t in range(_n_loop - 1):
    td_now = _td_base * float(_gdp_norm[24 + t])
    _gp_history.append(td_now)

    gp = (1.0 / 12.0) * sum(
        (_gp_history[23 - k] - _gp_history[11 - k]) / (_gp_history[11 - k] + 1e-6)
        for k in range(12)
    )
    _gdp_gp_vals.append(gp)
    _gdp_gp_dates.append(_monthly_idx[24 + t])

_gdp_gp_series = pd.Series(_gdp_gp_vals, index=pd.DatetimeIndex(_gdp_gp_dates))

# Restrict both series to calibration window
_calib_start = pd.to_datetime(CALIB_DATE[0])
_gdp_gp_calib = _gdp_gp_series[_gdp_gp_series.index >= _calib_start]

# HP-cycle g_prime_norm from existing results (single_node)
_hp_gp = results['single_node']
_hp_gp_calib = _hp_gp[_hp_gp['date'] >= _calib_start]

# Simulate target_vr from GDP g_prime_norm using a multiplicative update:
#   target_vr_t = target_vr_{t-1} * (1 + gamma_v * gp_gdp_t)
# If GDP grew 2% (gp=0.02) and gamma_v=0.5, vacancy rate grows by 1%.
def _simulate_vr_from_series(gp_series_calib, gamma_v, d_initial=D_INITIAL):
    vals = gp_series_calib.values
    dates = gp_series_calib.index
    vr = np.empty(len(vals))
    vr[0] = d_initial
    for i in range(1, len(vals)):
        vr[i] = vr[i - 1] * (1.0 + gamma_v * vals[i - 1])
    return pd.Series(vr, index=dates)

fig5, axes5 = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

# Panel 1: compare g_prime_norm signals side by side, with VACRATE on twin axis
ax = axes5[0]
ax2 = ax.twinx()
ax.plot(_hp_gp_calib['date'], _hp_gp_calib['wmean'],
        color='darkorange', linewidth=1.8, label='g_prime_norm (HP cycle)')
ax.plot(_gdp_gp_calib.index, _gdp_gp_calib.values,
        color='steelblue', linewidth=1.8, label='g_prime_norm (GDP level)')
ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
ax2.plot(obs['DATE'], obs['VACRATE'],
         color='black', linewidth=1.4, linestyle='dotted', label='Observed VACRATE')
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_xlim(_calib_start, pd.to_datetime(CALIB_DATE[1]))
ax.set_ylabel('g_prime_norm', fontsize=9)
ax2.set_ylabel('Observed vacancy rate', fontsize=9)
ax.set_title('g_prime_norm: HP cycle (mean-zero) vs GDP level (positive mean = built-in drift)\n'
             'GDP version is stationary but shifted up by ~avg GDP growth rate',
             fontsize=10, fontweight='bold')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper left')
ax.tick_params(axis='x', labelsize=8)

# Panel 2: simulate target_vr using GDP g_prime_norm, d_v=0, varying gamma_v
ax = axes5[1]
ax2 = ax.twinx()
_gamma_grid = [0.1, 0.3, 0.5, 1.0]
_colors_sim = plt.cm.Blues(np.linspace(0.4, 0.9, len(_gamma_grid)))
for gamma_v, col in zip(_gamma_grid, _colors_sim):
    sim_vr = _simulate_vr_from_series(_gdp_gp_calib, gamma_v=gamma_v)
    ax.plot(sim_vr.index, sim_vr.values, color=col, linewidth=1.6,
            label=f'γ_v={gamma_v}')
ax2.plot(obs['DATE'], obs['VACRATE'],
         color='black', linewidth=1.4, linestyle='dotted', label='Observed VACRATE')
ax.axhline(D_INITIAL, color='grey', linewidth=0.7, linestyle=':')
for _, row in recession_spans.iterrows():
    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
ax.set_xlim(_calib_start, pd.to_datetime(CALIB_DATE[1]))
ax.set_ylabel('Simulated target_vr', fontsize=9)
ax2.set_ylabel('Observed vacancy rate', fontsize=9)
ax.set_title('Simulated target_vr: multiplicative rule  target_vr_t = target_vr_{t-1} × (1 + γ_v × gp_gdp_t)\n'
             '2% GDP growth + γ_v=0.5 → 1% proportional rise in vacancy rate',
             fontsize=10, fontweight='bold')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper left')
ax.tick_params(axis='x', labelsize=8)

fig5.suptitle('Figure 5: g_prime_norm from GDP level — multiplicative vacancy update\n'
              '(target_vr grows proportionally to GDP growth rate, scaled by γ_v)',
              fontsize=12, fontweight='bold')
fig5.tight_layout(rect=[0, 0, 1, 0.96])

if SAVE:
    plt.savefig(os.path.join(OUTPUT_PATH, 'gdp_gp_norm_signal.jpg'),
                bbox_inches='tight', dpi=200)
    print(f"Saved GDP g_prime_norm figure to {OUTPUT_PATH}gdp_gp_norm_signal.jpg")
else:
    plt.show()
