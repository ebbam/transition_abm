"""
entry_policy_exercise.py

Policy exercises for the labour market transformation paper.

Section 1: Diagnostic — demand-gap analysis for entry-level occupations.
Section 2: Entry-level worker experience figure (4-panel baseline).
Section 3: Policy 1 — preferential hiring for long-term unemployed (WOTC-style).
Section 4: Policy 3 — job guarantee / entry-level on-ramps.

Run interactively in VS Code "Run in Interactive Window".
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from network_input_builder import network_input_builder
from model_fun import run_single_local
from collate_macro_vars import recession_spans, gdp_dat, macro_observations
from plot_style import (TEXTWIDTH, DEFAULT_ASPECT_RATIO, ROW_HEIGHT, save_plot_standard,
                         add_shared_legend, add_suptitle_spacing)

# ── Settings ──────────────────────────────────────────────────────────────────
NW          = "full_omn"
CALIB_DATE  = ["2001-01-01", "2024-05-01"]
DELAY       = 25
VAC_SETTING = "exog_vac"

SAVE_FIG5_TO_DROPBOX = True
DROPBOX_FIG5_PATH = os.path.expanduser(
    f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/{NW}/{VAC_SETTING}/figures/covid_oos"
)

PKL_PATH = os.path.join(
    PROJECT_ROOT,
    f"output/calib_final/{NW}/{VAC_SETTING}/figures/covid_oos/cached_results/model_results_N25.pkl"
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    f"output/calib_final/{NW}/{VAC_SETTING}/figures/covid_oos/entry_policy/"
)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Reference model for all per-occupation plots
REFERENCE_MODEL = "Behavioural w. Cyc. OTJ w. RW"

# ── Build static occupation characteristics (entry_level flag) ────────────────
print("Building static occupation characteristics from network...")
_mod_data, _net_temp, _vacs, _, _occ_shocks_dat = network_input_builder(NW, False, CALIB_DATE, delay=DELAY)
_static_rows = []
for _occ in _net_temp:
    _static_rows.append({
        'Occupation': _occ.occupation_id,
        'entry_level': bool(_occ.entry_level),
        'wage_mu':     float(_occ.wage_mu),
        'wage_sigma':  float(_occ.wage_sigma),
        'target_demand_init': float(_occ.target_demand),
    })
STATIC_OCC_DF = pd.DataFrame(_static_rows)
print(f"  {STATIC_OCC_DF['entry_level'].sum()} entry-level occupations, "
      f"{(~STATIC_OCC_DF['entry_level']).sum()} non-entry-level.")

# ── Load cached simulation results ────────────────────────────────────────────
print(f"\nLoading cache from {PKL_PATH} ...")
with open(PKL_PATH, 'rb') as f:
    cached_data = pickle.load(f)
sim = cached_data['filtered_sim_results'].get(REFERENCE_MODEL)
if sim is None:
    available = list(cached_data['filtered_sim_results'].keys())
    raise KeyError(f"Model '{REFERENCE_MODEL}' not found. Available: {available}")
sim = sim.merge(STATIC_OCC_DF[['Occupation', 'entry_level']], on='Occupation', how='left')
print(f"  Loaded {len(sim):,} occ-period rows for '{REFERENCE_MODEL}'.")

# ── Demand gap ────────────────────────────────────────────────────────────────
# Current_Demand is recorded at END of each time step (emp + open_vacs after hiring).
# The gap available to new entrants at START of step t ≈ Target_Demand[t] - Current_Demand[t-1].
# We approximate with the same-period lag: sort by Occupation and Time Step, then shift.
sim = sim.sort_values(['Occupation', 'Time Step']).reset_index(drop=True)
sim['Current_Demand_lag'] = sim.groupby('Occupation')['Current_Demand'].shift(1)
sim['demand_gap'] = sim['Target_Demand'] - sim['Current_Demand_lag']

# Drop the first time step per occupation (lag is NaN)
sim_gap = sim.dropna(subset=['demand_gap']).copy()

entry_df     = sim_gap[sim_gap['entry_level']]
non_entry_df = sim_gap[~sim_gap['entry_level']]

# ── Entrants-weighted gap ─────────────────────────────────────────────────────
# Entry_Level_Hires counts new entrants allocated to this occ in this period.
# We weight the gap by how many entrants were actually sent there.
entry_nonzero = entry_df[entry_df['Entry_Level_Hires'] > 0].copy()

print("\n=== Summary: demand gap at entry-level occupations ===")
for label, sub in [("Entry-level (all occ-periods)", entry_df),
                   ("Entry-level (periods with ≥1 entrant)", entry_nonzero),
                   ("Non-entry-level", non_entry_df)]:
    gap = sub['demand_gap']
    print(f"\n  {label}  (n={len(sub):,})")
    print(f"    % periods gap > 0:   {(gap > 0).mean()*100:.1f}%")
    print(f"    % periods gap ≤ 0:   {(gap <= 0).mean()*100:.1f}%")
    print(f"    Mean gap:            {gap.mean():.2f}")
    print(f"    Median gap:          {gap.median():.2f}")
    print(f"    P10 / P90:           {gap.quantile(.10):.2f} / {gap.quantile(.90):.2f}")

# Economy-wide per period
by_period = entry_df.groupby('Time Step').agg(
    total_gap=('demand_gap', 'sum'),
    total_entrants=('Entry_Level_Hires', 'sum'),
    n_pos_gap_occs=('demand_gap', lambda x: (x > 0).sum()),
    n_neg_gap_occs=('demand_gap', lambda x: (x <= 0).sum()),
).reset_index()
by_period['entrants_vs_gap'] = by_period['total_entrants'] / by_period['total_gap'].clip(lower=1)

print("\n=== Economy-wide per period (entry-level occs) ===")
print(f"  Mean new entrants / period:       {by_period['total_entrants'].mean():.1f}")
print(f"  Mean total demand gap / period:   {by_period['total_gap'].mean():.1f}")
print(f"  % periods gap > entrants:         {(by_period['total_gap'] > by_period['total_entrants']).mean()*100:.1f}%")
print(f"  % periods total gap ≤ 0:          {(by_period['total_gap'] <= 0).mean()*100:.1f}%")
print(f"  Mean ratio entrants / gap:        {by_period['entrants_vs_gap'].mean():.2f}")

# ── FIGURE 1: Distribution of demand gap at entry-level vs non-entry-level ───
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (label, sub, color) in zip(axes, [
    ("Entry-level occupations", entry_df, "#7b5ea7"),
    ("Non-entry-level occupations", non_entry_df, "#4a9aba"),
]):
    gap = sub['demand_gap'].clip(-50, 200)
    ax.hist(gap, bins=60, color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--', label='Gap = 0')
    ax.axvline(gap.mean(), color=color, linewidth=1.5, linestyle='-', label=f'Mean = {gap.mean():.1f}')
    pct_pos = (sub['demand_gap'] > 0).mean() * 100
    ax.set_title(f"{label}\n{pct_pos:.1f}% of occ-periods have positive gap", fontsize=11)
    ax.set_xlabel("Demand gap (Target − lagged Current Demand)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle("Demand slack available to new entrants at time of allocation", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "diag_D1_gap_distribution.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: diag_D1_gap_distribution.png")

# ── FIGURE 2: Time-series — economy-wide gap vs entrant flow ─────────────────
# Map Time Step to calendar dates: Time Step = DELAY corresponds to CALIB_DATE[0].
# After the lag shift, the first Time Step in by_period is DELAY+1 (one month later).
by_period = by_period.sort_values('Time Step').reset_index(drop=True)
_date_start = pd.to_datetime(CALIB_DATE[0])
by_period['DATE'] = by_period['Time Step'].apply(
    lambda ts: _date_start + pd.DateOffset(months=int(ts - DELAY))
)

fig, ax = plt.subplots(figsize=(12, 4))
# shade recession periods
for _, rec in recession_spans.iterrows():
    ax.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
               color='lightgrey', alpha=0.5, zorder=0)
ax.fill_between(by_period['DATE'], 0, by_period['total_gap'].clip(lower=0),
                alpha=0.3, color='#7b5ea7', label='Positive gap (capacity)')
ax.fill_between(by_period['DATE'], by_period['total_gap'].clip(upper=0), 0,
                alpha=0.3, color='#e05c5c', label='Negative gap (over capacity)')
ax.plot(by_period['DATE'], by_period['total_entrants'],
        color='black', linewidth=1.5, label='New entrants per period')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Workers (model units)", fontsize=10)
ax.set_xlim(by_period['DATE'].min(), by_period['DATE'].max())
ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title("Economy-wide entry-level demand gap vs new entrant flow", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "diag_D2_gap_vs_entrants_ts.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: diag_D2_gap_vs_entrants_ts.png")

# ── FIGURE 3: Per-occupation — % of periods with positive gap ─────────────────
occ_summary = entry_df.groupby('Occupation').agg(
    pct_pos_gap=('demand_gap', lambda x: (x > 0).mean()),
    mean_gap=('demand_gap', 'mean'),
    mean_td=('Target_Demand', 'mean'),
    mean_entrants=('Entry_Level_Hires', 'mean'),
).reset_index().merge(STATIC_OCC_DF[['Occupation', 'wage_mu']], on='Occupation', how='left')
occ_summary['median_wage'] = np.exp(occ_summary['wage_mu'])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel A: histogram of pct_pos_gap across entry-level occs
ax = axes[0]
ax.hist(occ_summary['pct_pos_gap'] * 100, bins=20, color='#7b5ea7', alpha=0.75,
        edgecolor='white', linewidth=0.3)
ax.axvline(50, color='black', linewidth=1.2, linestyle='--', label='50% threshold')
ax.axvline(occ_summary['pct_pos_gap'].mean() * 100, color='#7b5ea7', linewidth=1.5,
           linestyle='-', label=f"Mean = {occ_summary['pct_pos_gap'].mean()*100:.0f}%")
ax.set_xlabel("% of periods with positive demand gap", fontsize=10)
ax.set_ylabel("Number of entry-level occupations", fontsize=10)
ax.set_title("Distribution across entry-level occupations", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Panel B: scatter — occupation size vs pct_pos_gap, coloured by median wage
ax = axes[1]
sc = ax.scatter(occ_summary['mean_td'], occ_summary['pct_pos_gap'] * 100,
                c=np.log(occ_summary['median_wage']), cmap='viridis',
                alpha=0.7, s=40, edgecolors='none')
plt.colorbar(sc, ax=ax, label='log(Median wage)')
ax.axhline(50, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax.set_xlabel("Mean target demand (occ size)", fontsize=10)
ax.set_ylabel("% of periods with positive gap", fontsize=10)
ax.set_title("Occ size vs slack availability\n(colour = log median wage)", fontsize=11)
ax.grid(alpha=0.3)

fig.suptitle("Entry-level occupation: how often is there room for new entrants?",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "diag_D3_occ_gap_summary.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: diag_D3_occ_gap_summary.png")

# ── FIGURE 4: Entrant-weighted gap — where do entrants actually land? ─────────
# Weighted by Entry_Level_Hires so we see what the average new entrant experiences
entrant_weighted_gap = np.repeat(
    entry_nonzero['demand_gap'].values,
    entry_nonzero['Entry_Level_Hires'].astype(int).values
)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(np.clip(entrant_weighted_gap, -50, 200), bins=60,
        color='#7b5ea7', alpha=0.75, edgecolor='white', linewidth=0.3)
ax.axvline(0, color='black', linewidth=1.2, linestyle='--', label='Gap = 0 (no slack)')
pct_pos_w = (entrant_weighted_gap > 0).mean() * 100
ax.axvline(np.median(entrant_weighted_gap), color='#7b5ea7', linewidth=1.5, linestyle='-',
           label=f'Median = {np.median(entrant_weighted_gap):.1f}')
ax.set_title(f"Demand gap experienced by new entrants (entrant-weighted)\n"
             f"{pct_pos_w:.1f}% of new entrant allocations land in an occupation with positive slack",
             fontsize=11)
ax.set_xlabel("Demand gap at destination occupation", fontsize=10)
ax.set_ylabel("New entrant allocations (frequency)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "diag_D4_entrant_weighted_gap.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: diag_D4_entrant_weighted_gap.png")

print(f"\n✓ All diagnostic figures saved to {OUTPUT_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Entry-level worker experience (4-panel figure, one per model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 2: Entry-level worker experience figures ===")

from scipy.stats import gaussian_kde
from model_colors import MODEL_COLORS, _greens, _reds

# Reverse map: display name → short model id (for filename suffixes)
MODEL_SUFFIX = {
    "Non-behavioural":                   "nonbehav",
    "Non-behavioural w. OTJ":            "otj_nonbehav",
    "Behavioural w. Cyc. OTJ w. RW":     "otj_cyclical_e_disc",
    "Behavioural w.o. Cyc. OTJ w. RW":   "otj_disc",
    "Behavioural w. Cyc. OTJ w.o RW":    "otj_cyclical_e_disc_no_rw",
    "Behavioural w.o. Cyc. OTJ w.o RW":  "otj_disc_no_rw",
}
# Reverse: raw model ID → display name (for looking up baseline in cached_data pkl)
_CM_ID_TO_DISPLAY = {v: k for k, v in MODEL_SUFFIX.items()}

# Precompute occupation median wage lookup (same for all models)
occ_wage_info = STATIC_OCC_DF[['Occupation', 'wage_mu']].copy()
occ_wage_info['occ_median_wage'] = np.exp(occ_wage_info['wage_mu'])

MAX_DUR    = 36
dur_bins   = [0, 3, 6, 12, 24, 36]
dur_labels = ['1–3', '3–6', '6–12', '12–24', '24–36']

trans_all = cached_data.get('filtered_transition_dfs', {})


def _prep_entrant_transitions(trans_raw):
    """Filter and enrich a transition df to entry-level UE transitions."""
    trans = trans_raw.merge(
        occ_wage_info[['Occupation', 'occ_median_wage']],
        left_on='OriginOccupation', right_on='Occupation', how='left'
    )
    if 'NewEntrant' in trans.columns:
        mask = (trans['TransitionType'] == 'UE') & (trans['NewEntrant'] == True)
    else:
        mask = (trans['TransitionType'] == 'UE')
    et = trans[mask].copy()
    et['WageRatio'] = et['AcceptedWage'] / et['occ_median_wage']
    et = et[et['UEDuration'].between(1, MAX_DUR)].copy()
    et['dur_bin'] = pd.cut(et['UEDuration'], bins=dur_bins, labels=dur_labels)
    # Labor market entry date: hire date minus unemployment spell length
    _d0 = pd.to_datetime(CALIB_DATE[0])
    _offsets = (et['Time Step'] - et['UEDuration'] - DELAY).astype(int)
    et['entry_date'] = pd.to_datetime(
        [_d0 + pd.DateOffset(months=int(m)) for m in _offsets]
    )
    return et


def _prep_incumbent_transitions(trans_raw):
    """Non-new-entrant UE transitions whose destination is an entry-level occupation.
    These workers compete directly with new entrants for the same vacancies, so
    their spell durations reveal any crowding-out caused by entry_hire_bias.
    """
    _entry_occ_ids = set(STATIC_OCC_DF[STATIC_OCC_DF['entry_level']]['Occupation'])
    trans = trans_raw[trans_raw['TransitionType'] == 'UE'].copy()
    if 'NewEntrant' in trans.columns:
        trans = trans[trans['NewEntrant'] != True]
    if 'DestinationOccupation' in trans.columns:
        trans = trans[trans['DestinationOccupation'].isin(_entry_occ_ids)]
    return trans[trans['UEDuration'].between(1, MAX_DUR)].copy()


def plot_entry_experience(model_name, trans_raw, color, suffix, out_path,
                          overlays=None, base_occ_df=None):
    """Produce the 6-panel entry-level experience figure for one model.

    overlays    : list of (trans_raw, occ_df, color, label) tuples drawn on
                  top of the baseline in every panel.
    base_occ_df : per-occupation time-series df for the baseline (3rd element
                  of policy_results[scenario]).  If supplied, Panel F shows
                  the entry-level LTUER over time.
    """
    overlays = overlays or []

    et = _prep_entrant_transitions(trans_raw)
    entrant_label = ("NewEntrant == True" if 'NewEntrant' in trans_raw.columns
                     else "all UE transitions [NewEntrant flag missing — re-run pending]")
    print(f"  {model_name}: {len(et):,} transitions ({entrant_label})")

    # Pre-process overlay entrant transitions once
    overlay_ets = [(_prep_entrant_transitions(o_tr), o_col, o_lbl)
                   for o_tr, _, o_col, o_lbl in overlays]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    (ax_A, ax_B, ax_C), (ax_D, ax_E, ax_F) = axes
    if base_occ_df is None:
        ax_F.set_visible(False)

    # ── Panel A: time to first employment ─────────────────────────────────────
    x_range = np.arange(0, MAX_DUR + 1)

    def _survival(et_df):
        durs = np.sort(et_df['UEDuration'].values)
        return np.array([(durs > x).mean() * 100 for x in x_range])

    surv = _survival(et)
    ax_A.plot(x_range, surv, color=color, linewidth=2.5, label='Baseline')
    ax_A.fill_between(x_range, 0, surv, alpha=0.12, color=color)
    for o_et, o_col, o_lbl in overlay_ets:
        o_surv = _survival(o_et)
        ax_A.plot(x_range, o_surv, color=o_col, linewidth=2.5, linestyle='--', label=o_lbl)
    ax_A.axvline(6,  color='grey', linewidth=0.9, linestyle='--', alpha=0.7, label='6 months')
    ax_A.axvline(12, color='grey', linewidth=0.9, linestyle=':',  alpha=0.7, label='12 months')
    ax_A.set_xlabel("Months since labour market entry", fontsize=10)
    ax_A.set_ylabel("% not yet in first job", fontsize=10)
    ax_A.set_title("A.  Time to first employment", fontsize=11, fontweight='bold')
    ax_A.set_xlim(0, MAX_DUR); ax_A.set_ylim(0, 100)
    ax_A.legend(fontsize=9); ax_A.grid(axis='y', alpha=0.3)

    # ── Panel B: first-job wage quality (KDE) ─────────────────────────────────
    x_wr = np.linspace(0.05, 4.0, 500)
    wr = et['WageRatio'].clip(0.05, 4.0).dropna()
    if len(wr) >= 10:
        kde = gaussian_kde(wr, bw_method=0.15)
        ax_B.plot(x_wr, kde(x_wr), color=color, linewidth=2.5, label='Baseline')
        ax_B.fill_between(x_wr, 0, kde(x_wr), alpha=0.12, color=color)
    ax_B.axvline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.8,
                 label='Occupation median wage')
    ax_B.axvline(float(wr.mean()), color=color, linewidth=1.5, linestyle='-',
                 label=f'Baseline mean = {wr.mean():.2f}')
    for o_et, o_col, o_lbl in overlay_ets:
        o_wr = o_et['WageRatio'].clip(0.05, 4.0).dropna()
        if len(o_wr) >= 10:
            o_kde = gaussian_kde(o_wr, bw_method=0.15)
            ax_B.plot(x_wr, o_kde(x_wr), color=o_col, linewidth=2.5,
                      linestyle='--', label=o_lbl)
        ax_B.axvline(float(o_wr.mean()), color=o_col, linewidth=1.5, linestyle='--',
                     label=f'{o_lbl} mean = {o_wr.mean():.2f}')
    ax_B.set_xlabel("First-job wage / occupation median wage", fontsize=10)
    ax_B.set_ylabel("Density", fontsize=10)
    ax_B.set_title("B.  First-job wage quality", fontsize=11, fontweight='bold')
    ax_B.set_xlim(0.05, 3.0)
    ax_B.legend(fontsize=9); ax_B.grid(axis='y', alpha=0.3)

    # ── Panel C: first-job wage by entry cohort (quarterly) ───────────────────
    # Each point = mean WageRatio of all entrants who entered the labour market
    # in that quarter, plotted by entry date (not hire date).
    def _cohort_wages(et_df):
        g = et_df.copy()
        g['cohort'] = g['entry_date'].dt.to_period('Y').dt.to_timestamp()
        return g.groupby('cohort')['WageRatio'].agg(['mean', 'sem']).reset_index()

    cw = _cohort_wages(et)
    ax_C.plot(cw['cohort'], cw['mean'], color=color, linewidth=2,
              marker='o', markersize=5, label='Baseline')
    ax_C.fill_between(cw['cohort'],
                      cw['mean'] - 1.96 * cw['sem'],
                      cw['mean'] + 1.96 * cw['sem'],
                      color=color, alpha=0.15)
    for o_et, o_col, o_lbl in overlay_ets:
        o_cw = _cohort_wages(o_et)
        ax_C.plot(o_cw['cohort'], o_cw['mean'], color=o_col, linewidth=2,
                  marker='o', markersize=5, linestyle='--', label=o_lbl)
        ax_C.fill_between(o_cw['cohort'],
                          o_cw['mean'] - 1.96 * o_cw['sem'],
                          o_cw['mean'] + 1.96 * o_cw['sem'],
                          color=o_col, alpha=0.15)
    for _, rec in recession_spans.iterrows():
        ax_C.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                     color='lightgrey', alpha=0.4, zorder=0)
    ax_C.axhline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.8,
                 label='Occupation median wage')
    ax_C.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
    ax_C.set_xlabel("Labour market entry cohort (year)", fontsize=10)
    ax_C.set_ylabel("Mean first-job wage / occ. median", fontsize=10)
    ax_C.set_title("C.  First-job wages by entry cohort", fontsize=11, fontweight='bold')
    ax_C.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
    ax_C.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.setp(ax_C.get_xticklabels(), rotation=45, ha='right')
    ax_C.legend(fontsize=9); ax_C.grid(axis='y', alpha=0.3)

    # ── Panel D: joint distribution — duration vs wage ratio ──────────────────
    sample_n = min(4000, len(et))
    samp = et.sample(sample_n, random_state=42)
    ax_D.scatter(samp['UEDuration'], samp['WageRatio'].clip(0.05, 4.0),
                 alpha=0.12, s=7, color=color, rasterized=True)
    trend = (et.sort_values('UEDuration')
               .groupby('UEDuration')['WageRatio']
               .mean()
               .rolling(3, center=True, min_periods=1)
               .mean())
    ax_D.plot(trend.index, trend.values.clip(0.05, 4.0),
              color=color, linewidth=1.8, label='Baseline mean')
    for o_et, o_col, o_lbl in overlay_ets:
        o_trend = (o_et.sort_values('UEDuration')
                       .groupby('UEDuration')['WageRatio']
                       .mean()
                       .rolling(3, center=True, min_periods=1)
                       .mean())
        ax_D.plot(o_trend.index, o_trend.values.clip(0.05, 4.0),
                  color=o_col, linewidth=1.8, linestyle='--', label=o_lbl)
    ax_D.axhline(1.0, color='grey', linewidth=1, linestyle='--', alpha=0.7)
    ax_D.set_xlabel("Unemployment duration at first hire (months)", fontsize=10)
    ax_D.set_ylabel("First-job wage / occupation median wage", fontsize=10)
    ax_D.set_title("D.  Duration–wage joint distribution", fontsize=11, fontweight='bold')
    ax_D.set_xlim(0, MAX_DUR); ax_D.set_ylim(0.05, 3.5)
    ax_D.legend(fontsize=9); ax_D.grid(alpha=0.2)

    # ── Panel E: composition of hires into entry-level occupations ───────────────
    # Three categories: new entrants (UE, NewEntrant=True), unemployed incumbents
    # (UE, NewEntrant=False), and employed movers (EE).
    _entry_occ_ids = set(STATIC_OCC_DF[STATIC_OCC_DF['entry_level']]['Occupation'])

    def _entry_hire_counts(t_raw):
        t = t_raw[t_raw['DestinationOccupation'].isin(_entry_occ_ids)].copy()
        has_new = 'NewEntrant' in t.columns

        def _per_group(g):
            return pd.Series({
                'n_entrant': ((g['TransitionType'] == 'UE') &
                              (g['NewEntrant'] == True)).sum() if has_new else 0,
                'n_ue_inc':  ((g['TransitionType'] == 'UE') &
                              (g['NewEntrant'] != True)).sum() if has_new
                              else (g['TransitionType'] == 'UE').sum(),
                'n_ee':      (g['TransitionType'] == 'EE').sum(),
            })

        if 'sim_id' in t.columns:
            per_sim = t.groupby('sim_id').apply(_per_group).reset_index()
            return per_sim['n_entrant'].mean(), per_sim['n_ue_inc'].mean(), per_sim['n_ee'].mean()
        else:
            return (float(((t['TransitionType'] == 'UE') & (t['NewEntrant'] == True)).sum() if has_new else 0),
                    float(((t['TransitionType'] == 'UE') & (t['NewEntrant'] != True)).sum() if has_new
                          else (t['TransitionType'] == 'UE').sum()),
                    float((t['TransitionType'] == 'EE').sum()))

    _scenarios  = [('Baseline', trans_raw, color)] + \
                  [(o_lbl, o_tr, o_col) for o_tr, _, o_col, o_lbl in overlays]
    _bar_labels, _n_entrant, _n_ue_inc, _n_ee = [], [], [], []
    for s_lbl, s_tr, _ in _scenarios:
        nent, nui, nee = _entry_hire_counts(s_tr)
        _bar_labels.append(s_lbl)
        _n_entrant.append(nent); _n_ue_inc.append(nui); _n_ee.append(nee)

    _x = np.arange(len(_bar_labels))
    _w = 0.5
    _ent_col = '#7b5ea7'   # purple for new entrants
    _ui_col  = '#74c476'   # green  for unemployed incumbents
    _ee_col  = '#2c7bb6'   # blue   for employed movers (EE)
    ax_E.bar(_x, _n_entrant, _w, label='New entrant (UE)',          color=_ent_col, alpha=0.8)
    ax_E.bar(_x, _n_ue_inc,  _w, bottom=_n_entrant,                 label='Unemployed (prior experience)', color=_ui_col, alpha=0.8)
    ax_E.bar(_x, _n_ee,      _w, bottom=[e+u for e,u in zip(_n_entrant, _n_ue_inc)],
             label='Employed mover (EE)', color=_ee_col, alpha=0.8)

    for i, (nent, nui, nee) in enumerate(zip(_n_entrant, _n_ue_inc, _n_ee)):
        total = nent + nui + nee
        if total > 0:
            for val, bot, col in [(nent, 0,            _ent_col),
                                  (nui,  nent,          _ui_col),
                                  (nee,  nent + nui,    _ee_col)]:
                if val / total >= 0.04:
                    ax_E.text(i, bot + val / 2, f'{val/total*100:.0f}%',
                              ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax_E.set_xticks(_x)
    ax_E.set_xticklabels(_bar_labels, fontsize=9)
    ax_E.set_ylabel("Hires into entry-level occupations\n(mean per simulation)", fontsize=10)
    ax_E.set_title("E.  Entry-level hiring composition\n(by prior labor market status)",
                   fontsize=11, fontweight='bold')
    ax_E.legend(fontsize=9); ax_E.grid(axis='y', alpha=0.3)

    # ── Panel F: entry-level LTUER over time ──────────────────────────────────
    if base_occ_df is not None:
        _entry_ids = set(STATIC_OCC_DF[STATIC_OCC_DF['entry_level']]['Occupation'])

        def _entry_ltuer_ts(occ_df):
            e = occ_df[occ_df['Occupation'].isin(_entry_ids)].copy()
            agg = (e.groupby(['Time Step', 'sim_id'])[['LT Unemployed Persons', 'Unemployment']]
                    .sum().reset_index())
            agg['ltuer'] = agg['LT Unemployed Persons'] / agg['Unemployment'].clip(lower=1)
            ts = agg.groupby('Time Step')['ltuer'].agg(['mean', 'sem']).reset_index()
            ts['DATE'] = pd.date_range(start=CALIB_DATE[0], end=CALIB_DATE[1], periods=len(ts))
            return ts

        ts_base = _entry_ltuer_ts(base_occ_df)
        ax_F.plot(ts_base['DATE'], ts_base['mean'] * 100, color=color,
                  linewidth=2.5, label='Baseline')
        ax_F.fill_between(ts_base['DATE'],
                          (ts_base['mean'] - 1.96 * ts_base['sem']) * 100,
                          (ts_base['mean'] + 1.96 * ts_base['sem']) * 100,
                          alpha=0.15, color=color)

        for _, o_occ, o_col, o_lbl in overlays:
            ts_pol = _entry_ltuer_ts(o_occ)
            ax_F.plot(ts_pol['DATE'], ts_pol['mean'] * 100, color=o_col,
                      linewidth=2.5, linestyle='--', label=o_lbl)
            ax_F.fill_between(ts_pol['DATE'],
                              (ts_pol['mean'] - 1.96 * ts_pol['sem']) * 100,
                              (ts_pol['mean'] + 1.96 * ts_pol['sem']) * 100,
                              alpha=0.12, color=o_col)

        ax_F.set_xlabel("Date", fontsize=10)
        ax_F.set_ylabel("Entry-level LTUER (%)", fontsize=10)
        ax_F.set_title("F.  Long-term unemployment rate\n(entry-level workers)",
                       fontsize=11, fontweight='bold')
        ax_F.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
        ax_F.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        plt.setp(ax_F.get_xticklabels(), rotation=45, ha='right')
        ax_F.legend(fontsize=9); ax_F.grid(axis='y', alpha=0.3)

    policy_note = "" if not overlays else f"  |  policy overlays: {', '.join(l for _, _, _, l in overlays)}"
    fig.suptitle(
        f"Entry-level worker labour market experience — {model_name}\n"
        f"Source: {entrant_label}  |  n = {len(et):,}{policy_note}",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    fname = os.path.join(out_path, f"fig2_entry_experience_{suffix}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_entry_experience_{suffix}.png")


# ── Loop over all available models ────────────────────────────────────────────
for model_name, trans_raw in trans_all.items():
    suffix = MODEL_SUFFIX.get(model_name, model_name.replace(" ", "_").lower())
    color  = MODEL_COLORS.get(model_name, '#888888')
    plot_entry_experience(model_name, trans_raw, color, suffix, OUTPUT_PATH)

print(f"\n✓ Done.")

# ══════════════════════════════════════════════════════════════════════════════
# POLICY SETUP: shared infrastructure for Sections 3 and 4
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Policy exercise setup ===")

from copy import deepcopy
from scipy.stats import gaussian_kde as _gkde
from abm_funs import duration_to_prob_dict

# Calibrated params CSV — matches the exog_vac / covid_oos pkl run.
# vac_wage_update=True: not overridden in auto_run params dict, so model_fun default (True) applies.
CALIB_CSV = os.path.join(
    PROJECT_ROOT,
    "output/calib_final/full_omn/exog_vac/"
    "calibrated_params_all_theta_signal_exog_wage_update_True.csv"
)
_param_df = pd.read_csv(CALIB_CSV)
_ref = (_param_df[_param_df['model_cat'] == 'otj_cyclical_e_disc']
        .sort_values('Timestamp', ascending=False))

# vac_dat: VACRATE series trimmed to calibration window
_obs     = macro_observations.loc[
    (macro_observations['DATE'] >= CALIB_DATE[0]) &
    (macro_observations['DATE'] <= CALIB_DATE[1])
].dropna(subset=['VACRATE'])
_vac_dat = _obs['VACRATE'].to_numpy()

# Base params template — mirrors the settings used to generate model_results.pkl
REF_BASE_PARAMS = {
    # infrastructure (from network_input_builder)
    'mod_data':           _mod_data,
    'net_temp':           _net_temp,
    'vacs':               _vacs,
    'time_steps':         int(_occ_shocks_dat.shape[1]),
    'occ_shocks_data':    _occ_shocks_dat,
    'app_effort_dat':     duration_to_prob_dict,
    # time-series inputs
    'gdp_data':           gdp_dat,
    'bus_confidence_dat': gdp_dat,
    'vac_data':           _vac_dat,
    # run settings
    'delay':              DELAY,
    'simple_res':         False,
    'dumb_hire':          False,
    'mistake_rate':       0.1,
    'vac_method':         'exog',
    'vac_wage_update':    True,    # model_fun default; not overridden in auto_run
    'steady_state':       False,
    # model spec: Behavioural w. Cyc. OTJ w. RW (otj_cyclical_e_disc)
    'otj':                True,
    'cyc_otj':            True,
    'cyc_ue':             False,
    'disc':               True,
    'wage_prefs':         True,
    'emp_apps':           1,
    'strict_rw':          False,
    # vacancy opening (exog_vac → not used → None → np.nan internally)
    'd_v':                None,
    'gamma_v':            None,
    # calibrated params
    'gamma_u': float(_ref.loc[_ref['Parameter'] == 'gamma_u', 'Value'].iloc[0]),
    'd_u':     float(_ref.loc[_ref['Parameter'] == 'd_u',     'Value'].iloc[0]),
    'theta':   float(_ref.loc[_ref['Parameter'] == 'theta',   'Value'].iloc[0]),
}
print(f"  Reference model params: gamma_u={REF_BASE_PARAMS['gamma_u']:.4f}, "
      f"d_u={REF_BASE_PARAMS['d_u']:.5f}, theta={REF_BASE_PARAMS['theta']:.4f}")

N_POLICY_SIMS      = 5
N_TOPUP_SIMS_DEF   = 20   # topup added later; mirrored by N_TOPUP_SIMS after topup block
N_TOTAL_SIMS       = N_POLICY_SIMS + N_TOPUP_SIMS_DEF  # = 25
POLICY_PKL_PATH    = os.path.join(OUTPUT_PATH, "policy_results.pkl")
FORCE_POLICY_RERUN = False  # set True to re-run even if cache exists


def run_policy_scenario(scenario_name, policy_overrides, base_params, calib_date,
                        n_sims=N_POLICY_SIMS, sim_id_offset=0):
    """Run n_sims of the model with policy_overrides applied on top of base_params.

    Returns (transition_df, sim_grouped_df, occ_df) concatenated across all runs,
    each tagged with a 'sim_id' column. occ_df is per-occupation time series.
    sim_id_offset allows top-up runs to continue numbering from an existing batch.
    """
    print(f"  Running {n_sims} sims for '{scenario_name}'...")
    trans_runs   = []
    grouped_runs = []
    occ_runs     = []
    for i in range(n_sims):
        params = deepcopy(base_params)
        params.update(policy_overrides)
        result = run_single_local(**params)
        record_df, sim_grouped, *_, transition_df = result
        sim_grouped = sim_grouped.copy()
        sim_grouped['DATE'] = pd.date_range(
            start=calib_date[0], end=calib_date[1], periods=len(sim_grouped)
        )
        sim_grouped['sim_id'] = i + sim_id_offset
        transition_df = transition_df.copy()
        transition_df['sim_id'] = i + sim_id_offset
        record_df = record_df.copy()
        record_df['sim_id'] = i + sim_id_offset
        trans_runs.append(transition_df)
        grouped_runs.append(sim_grouped)
        occ_runs.append(record_df)
        if (i + 1) % max(1, n_sims // 5) == 0:
            print(f"    [{scenario_name}] {i+1}/{n_sims} done")
    return (pd.concat(trans_runs,   ignore_index=True),
            pd.concat(grouped_runs, ignore_index=True),
            pd.concat(occ_runs,     ignore_index=True))


# ── Baseline economy-wide time series (for Panel C LTUER) ────────────────────
# sim is per-occupation; sum across occupations to get economy-wide counts.
_bl_sum_cols = [c for c in ['LT Unemployed Persons', 'Unemployment', 'Employment', 'Workers']
                if c in sim.columns]
_cached_grouped = (sim.groupby('Time Step')[_bl_sum_cols].sum().reset_index())
_cached_grouped['DATE'] = pd.date_range(
    start=CALIB_DATE[0], end=CALIB_DATE[1], periods=len(_cached_grouped)
)
_cached_grouped['sim_id'] = 0

# ── Run / load policy scenarios ───────────────────────────────────────────────
if os.path.exists(POLICY_PKL_PATH) and not FORCE_POLICY_RERUN:
    print(f"Loading cached policy results from {POLICY_PKL_PATH}...")
    with open(POLICY_PKL_PATH, 'rb') as _f:
        policy_results = pickle.load(_f)
    print(f"  Loaded scenarios: {list(policy_results.keys())}")
else:
    policy_results = {}

    policy_scenarios = {
        # Section 3: Policy 1 — LTU preferential hiring
        'ltu_bias_0.5':    {'ltu_hire_bias': 0.5},
        'ltu_bias_1.0':    {'ltu_hire_bias': 1.0},
        # Section 3B2: Policy 2 — entry-level worker preferential hiring
        'entry_bias_0.5':  {'entry_hire_bias': 0.5},
        'entry_bias_1.0':  {'entry_hire_bias': 1.0},
        # Section 4: Policy 3 — job guarantee
        'jg_frac_0.25':    {'jg_frac': 0.25},
        'jg_frac_0.50':    {'jg_frac': 0.50},
    }

    for scenario, overrides in policy_scenarios.items():
        policy_results[scenario] = run_policy_scenario(
            scenario, overrides, REF_BASE_PARAMS, CALIB_DATE, N_POLICY_SIMS
        )
        print(f"  ✓ '{scenario}' complete")

    print(f"Saving policy results to {POLICY_PKL_PATH}...")
    with open(POLICY_PKL_PATH, 'wb') as _f:
        pickle.dump(policy_results, _f)
    print("  Saved.")

# ── Inject baseline into policy_results ──────────────────────────────────────
# 3rd element is per-occupation record; for baseline we use sim from the pkl
# (already per-occupation mean, tagged sim_id=0 for consistent unpacking)
_sim_occ = sim.copy()
_sim_occ['sim_id'] = 0
policy_results['baseline'] = (trans_all[REFERENCE_MODEL], _cached_grouped, _sim_occ)

# ── fig2 policy overlay: entry-level experience baseline vs entry_hire_bias ──
if 'entry_bias_0.5' in policy_results and 'entry_bias_1.0' in policy_results:
    _ref_suffix = MODEL_SUFFIX.get(REFERENCE_MODEL, REFERENCE_MODEL.replace(" ", "_").lower())
    _ref_color  = MODEL_COLORS.get(REFERENCE_MODEL, '#888888')
    _entry_overlays = [
        (policy_results['entry_bias_0.5'][0], policy_results['entry_bias_0.5'][2], _reds[0], 'Entry bias ×1.5'),
        (policy_results['entry_bias_1.0'][0], policy_results['entry_bias_1.0'][2], _reds[1], 'Entry bias ×2.0'),
    ]
    plot_entry_experience(
        REFERENCE_MODEL,
        trans_all[REFERENCE_MODEL],
        _ref_color,
        suffix=f"{_ref_suffix}_entry_bias",
        out_path=OUTPUT_PATH,
        overlays=_entry_overlays,
        base_occ_df=policy_results['baseline'][2],
    )
    # ── Entry bias diagnostic: new-entrant stock and conditional hiring rate ───
    _entry_diag_scenarios = {
        'Baseline':        (policy_results['baseline'][0],        policy_results['baseline'][2],        _ref_color),
        'Entry bias ×1.5': (policy_results['entry_bias_0.5'][0],  policy_results['entry_bias_0.5'][2],  _reds[0]),
        'Entry bias ×2.0': (policy_results['entry_bias_1.0'][0],  policy_results['entry_bias_1.0'][2],  _reds[1]),
    }

    def _newentrant_stock_and_rate(trans_df, occ_df):
        """Reconstruct new-entrant stock via cumulative inflow − outflow per sim."""
        has_sim = 'sim_id' in trans_df.columns

        # Inflow: new entrants added to unemployment pool each period
        inflow = (occ_df.groupby(['Time Step', 'sim_id'] if has_sim else ['Time Step'])
                        ['Entry_Level_Hires'].sum().reset_index())
        if not has_sim:
            inflow['sim_id'] = 0

        # Outflow: new-entrant UE hires per period per sim
        ne_hires = trans_df[(trans_df['TransitionType'] == 'UE') &
                            (trans_df.get('NewEntrant', pd.Series(False, index=trans_df.index)) == True)].copy()
        if 'NewEntrant' in trans_df.columns:
            ne_hires = trans_df[(trans_df['TransitionType'] == 'UE') &
                                (trans_df['NewEntrant'] == True)].copy()
        else:
            ne_hires = trans_df.iloc[0:0].copy()  # empty

        grp_cols = ['Time Step', 'sim_id'] if has_sim else ['Time Step']
        outflow = ne_hires.groupby(grp_cols).size().reset_index(name='outflow')
        if not has_sim:
            outflow['sim_id'] = 0

        all_ts   = sorted(occ_df['Time Step'].unique())
        all_sims = sorted(occ_df['sim_id'].unique()) if has_sim else [0]
        rows = []
        for s in all_sims:
            inf_s = inflow[inflow['sim_id'] == s].set_index('Time Step')['Entry_Level_Hires']
            out_s = outflow[outflow['sim_id'] == s].set_index('Time Step')['outflow'] if len(outflow) else pd.Series(dtype=float)
            stock = 0.0
            for t in all_ts:
                stock += float(inf_s.get(t, 0))
                hire_t = float(out_s.get(t, 0))
                rate   = hire_t / max(stock, 1)
                stock  = max(stock - hire_t, 0)
                rows.append({'Time Step': t, 'sim_id': s, 'stock': stock, 'rate': rate})

        df = pd.DataFrame(rows)
        # Map Time Step → DATE
        ts_map = (occ_df.drop_duplicates('Time Step')
                        .assign(DATE=lambda d: pd.date_range(
                            start=CALIB_DATE[0], end=CALIB_DATE[1], periods=len(d)))
                        .set_index('Time Step')['DATE'])
        df['DATE'] = df['Time Step'].map(ts_map)

        # Overall matching rate: total hires / total active seekers from occ_df
        grp_cols_u = ['Time Step', 'sim_id'] if has_sim else ['Time Step']
        seek_ps = (occ_df.groupby(grp_cols_u)[['Hires', 'Unemployed Seekers', 'Employed Seekers']]
                         .sum().reset_index())
        if not has_sim:
            seek_ps['sim_id'] = 0
        seek_ps['total_seekers'] = seek_ps['Unemployed Seekers'] + seek_ps['Employed Seekers']
        seek_ps['overall_rate']  = seek_ps['Hires'] / seek_ps['total_seekers'].clip(lower=1)
        seek_ps['DATE'] = seek_ps['Time Step'].map(ts_map)
        overall = seek_ps

        stock_ts   = df.groupby('DATE')['stock'].agg(['mean', 'sem']).reset_index()
        rate_ts    = df.groupby('DATE')['rate'].agg(['mean', 'sem']).reset_index()
        overall_ts = overall.groupby('DATE')['overall_rate'].agg(['mean', 'sem']).reset_index()
        return stock_ts, rate_ts, overall_ts

    fig_ed, (ax_ed1, ax_ed2, ax_ed3) = plt.subplots(1, 3, figsize=(18, 5))
    for label, (tr, oc, col) in _entry_diag_scenarios.items():
        ls = '--' if label != 'Baseline' else '-'
        stock_ts, rate_ts, overall_ts = _newentrant_stock_and_rate(tr, oc)
        ax_ed1.plot(stock_ts['DATE'], stock_ts['mean'], color=col, linewidth=2, linestyle=ls, label=label)
        ax_ed1.fill_between(stock_ts['DATE'],
                            stock_ts['mean'] - 1.96 * stock_ts['sem'],
                            stock_ts['mean'] + 1.96 * stock_ts['sem'],
                            alpha=0.12, color=col)
        ax_ed2.plot(rate_ts['DATE'], rate_ts['mean'] * 100, color=col, linewidth=2, linestyle=ls, label=label)
        ax_ed2.fill_between(rate_ts['DATE'],
                            (rate_ts['mean'] - 1.96 * rate_ts['sem']) * 100,
                            (rate_ts['mean'] + 1.96 * rate_ts['sem']) * 100,
                            alpha=0.12, color=col)
        ax_ed3.plot(overall_ts['DATE'], overall_ts['mean'] * 100, color=col, linewidth=2, linestyle=ls, label=label)
        ax_ed3.fill_between(overall_ts['DATE'],
                            (overall_ts['mean'] - 1.96 * overall_ts['sem']) * 100,
                            (overall_ts['mean'] + 1.96 * overall_ts['sem']) * 100,
                            alpha=0.12, color=col)

    for ax_ed in (ax_ed1, ax_ed2, ax_ed3):
        for _, rec in recession_spans.iterrows():
            ax_ed.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                          color='lightgrey', alpha=0.4, zorder=0)
        ax_ed.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
        ax_ed.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
        ax_ed.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        plt.setp(ax_ed.get_xticklabels(), rotation=45, ha='right')
        ax_ed.legend(fontsize=9); ax_ed.grid(axis='y', alpha=0.3)

    ax_ed1.set_xlabel("Date", fontsize=10)
    ax_ed1.set_ylabel("New entrant stock (workers)", fontsize=10)
    ax_ed1.set_title("A.  New-entrant unemployment pool\n(cumulative inflow − outflow)", fontsize=11, fontweight='bold')
    ax_ed2.set_xlabel("Date", fontsize=10)
    ax_ed2.set_ylabel("New-entrant hiring rate (% per period)", fontsize=10)
    ax_ed2.set_title("B.  Conditional hiring rate\n(new-entrant hires / new-entrant stock)", fontsize=11, fontweight='bold')
    ax_ed3.set_xlabel("Date", fontsize=10)
    ax_ed3.set_ylabel("Matching rate (% per period)", fontsize=10)
    ax_ed3.set_title("C.  Overall matching efficiency\n(all hires / all active seekers)", fontsize=11, fontweight='bold')

    fig_ed.suptitle(f"Entry bias diagnostic — {REFERENCE_MODEL}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    _fname_ed = f"fig_entry_bias_diagnostic_{_ref_suffix}.png"
    fig_ed.savefig(os.path.join(OUTPUT_PATH, _fname_ed), dpi=150, bbox_inches='tight')
    plt.close(fig_ed)
    print(f"  Saved: {_fname_ed}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Policy 1 — preferential hiring for long-term unemployed
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 3: Policy 1 — LTU preferential hiring ===")

_model_color = MODEL_COLORS.get(REFERENCE_MODEL, '#888888')
_ltu_scenarios = {
    'Baseline':             ('baseline',     _model_color),
    'LTU bias ×1.5':        ('ltu_bias_0.5', _greens[0]), # '#4a9aba'),
    'LTU bias ×2.0':        ('ltu_bias_1.0', _greens[1]) #'#1f78b4'),
}

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
ax3A, ax3B, ax3C = axes3

for label, (scenario, color) in _ltu_scenarios.items():
    trans_pol, grouped_pol, *_ = policy_results[scenario]

    # Filter to UE transitions by long-term unemployed workers
    ltu_hires = trans_pol[
        (trans_pol['TransitionType'] == 'UE') &
        (trans_pol['LongtermUnemp'] == True)
    ].copy()
    ltu_hires = ltu_hires[ltu_hires['UEDuration'].between(1, 36)]

    # Panel A: survival curve (time still unemployed at each duration)
    durs_ltu  = np.sort(ltu_hires['UEDuration'].values)
    x_range   = np.arange(0, 37)
    survival  = np.array([(durs_ltu > x).mean() * 100 for x in x_range])
    ax3A.plot(x_range, survival, color=color, linewidth=2, label=label)

    # Panel B: first-job wage KDE
    ltu_hires = ltu_hires.merge(
        occ_wage_info[['Occupation', 'occ_median_wage']],
        left_on='DestinationOccupation', right_on='Occupation', how='left'
    )
    wage_ratio = (ltu_hires['AcceptedWage'] / ltu_hires['occ_median_wage']).clip(0.05, 4.0).dropna()
    if len(wage_ratio) >= 10:
        kde  = _gkde(wage_ratio, bw_method=0.15)
        x_wr = np.linspace(0.05, 4.0, 400)
        ax3B.plot(x_wr, kde(x_wr), color=color, linewidth=2, label=label)

    # Panel C: LTUER time series (mean ± 95% CI across sims)
    # grouped_pol is already economy-wide (one row per time step per sim)
    ltuer_per_sim = grouped_pol.copy()
    ltuer_per_sim['LTUER'] = (ltuer_per_sim['LT Unemployed Persons'] /
                               ltuer_per_sim['Unemployment'].clip(lower=1))
    ltuer_agg = ltuer_per_sim.groupby('DATE')['LTUER'].agg(['mean', 'sem']).reset_index()
    ax3C.plot(ltuer_agg['DATE'], ltuer_agg['mean'] * 100, color=color, linewidth=2, label=label)
    ax3C.fill_between(ltuer_agg['DATE'],
                      (ltuer_agg['mean'] - 1.96 * ltuer_agg['sem']) * 100,
                      (ltuer_agg['mean'] + 1.96 * ltuer_agg['sem']) * 100,
                      alpha=0.15, color=color)

# Panel A formatting
ax3A.axvline(6,  color='grey', linewidth=0.9, linestyle='--', alpha=0.7)
ax3A.axvline(12, color='grey', linewidth=0.9, linestyle=':',  alpha=0.7)
ax3A.set_xlabel("Months unemployed at hire", fontsize=10)
ax3A.set_ylabel("% LTU workers not yet hired", fontsize=10)
ax3A.set_title("A.  Time to re-employment (LTU)", fontsize=11, fontweight='bold')
ax3A.set_xlim(0, 36); ax3A.set_ylim(0, 100)
ax3A.legend(fontsize=9); ax3A.grid(axis='y', alpha=0.3)

# Panel B formatting
ax3B.axvline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.8, label='Occ. median wage')
ax3B.set_xlabel("Accepted wage / occ. median wage", fontsize=10)
ax3B.set_ylabel("Density", fontsize=10)
ax3B.set_title("B.  First-job wage quality (LTU hires)", fontsize=11, fontweight='bold')
ax3B.set_xlim(0.05, 3.0)
ax3B.legend(fontsize=9); ax3B.grid(axis='y', alpha=0.3)

# Panel C formatting
for _, rec in recession_spans.iterrows():
    ax3C.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                 color='lightgrey', alpha=0.4, zorder=0)
ax3C.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax3C.set_xlabel("Date", fontsize=10)
ax3C.set_ylabel("LTUER (%)", fontsize=10)
ax3C.set_title("C.  Long-term unemployment rate", fontsize=11, fontweight='bold')
ax3C.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
ax3C.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax3C.get_xticklabels(), rotation=45, ha='right')
ax3C.legend(fontsize=9); ax3C.grid(axis='y', alpha=0.3)

fig3.suptitle(
    "Policy 1: Preferential hiring for long-term unemployed (WOTC-style)\n"
    f"Reference model: {REFERENCE_MODEL}  |  N = {N_TOTAL_SIMS} sims per scenario",
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "fig3_policy1_ltu_hiring.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig3_policy1_ltu_hiring.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Policy 3 — job guarantee / entry-level on-ramps
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 4: Policy 3 — job guarantee ===")

_jg_scenarios = {
    'Baseline':          ('baseline',      '#888888'),
    'JG 25% of entrants': ('jg_frac_0.25', '#fdbf6f'),
    'JG 50% of entrants': ('jg_frac_0.50', '#ff7f00'),
}

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
ax4A, ax4B = axes4

for label, (scenario, color) in _jg_scenarios.items():
    trans_pol, *_ = policy_results[scenario]

    # Filter to new entrants who found their first job (UE transition, NewEntrant==True)
    entrants = trans_pol[
        (trans_pol['TransitionType'] == 'UE') &
        (trans_pol['NewEntrant'] == True)
    ].copy()
    entrants = entrants[entrants['UEDuration'].between(1, 36)]
    entrants = entrants.merge(
        occ_wage_info[['Occupation', 'occ_median_wage']],
        left_on='DestinationOccupation', right_on='Occupation', how='left'
    )
    entrants['WageRatio'] = (entrants['AcceptedWage'] / entrants['occ_median_wage'])

    print(f"  {label}: {len(entrants):,} first-employment transitions")

    # Panel A: time to first employment — survival curve
    durs    = np.sort(entrants['UEDuration'].values)
    x_range = np.arange(0, 37)
    survival = np.array([(durs > x).mean() * 100 for x in x_range])
    ax4A.plot(x_range, survival, color=color, linewidth=2.5, label=label)
    ax4A.fill_between(x_range, 0, survival, alpha=0.08, color=color)

    # Panel B: first-job wage KDE
    wr = entrants['WageRatio'].clip(0.05, 4.0).dropna()
    if len(wr) >= 10:
        kde  = _gkde(wr, bw_method=0.15)
        x_wr = np.linspace(0.05, 4.0, 400)
        ax4B.plot(x_wr, kde(x_wr), color=color, linewidth=2.5, label=label)
        ax4B.fill_between(x_wr, 0, kde(x_wr), alpha=0.08, color=color)

# Panel A formatting
ax4A.axvline(6,  color='grey', linewidth=0.9, linestyle='--', alpha=0.7, label='6 months')
ax4A.axvline(12, color='grey', linewidth=0.9, linestyle=':',  alpha=0.7, label='12 months')
ax4A.set_xlabel("Months since labour market entry", fontsize=10)
ax4A.set_ylabel("% not yet in first job", fontsize=10)
ax4A.set_title("A.  Time to first employment", fontsize=11, fontweight='bold')
ax4A.set_xlim(0, 36); ax4A.set_ylim(0, 100)
ax4A.legend(fontsize=9); ax4A.grid(axis='y', alpha=0.3)

# Panel B formatting
ax4B.axvline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.8,
             label='Occ. median wage')
ax4B.set_xlabel("First-job wage / occ. median wage", fontsize=10)
ax4B.set_ylabel("Density", fontsize=10)
ax4B.set_title("B.  First-job wage quality", fontsize=11, fontweight='bold')
ax4B.set_xlim(0.05, 3.0)
ax4B.legend(fontsize=9); ax4B.grid(axis='y', alpha=0.3)

fig4.suptitle(
    "Policy 3: Job guarantee — direct placement of new entrants\n"
    f"Reference model: {REFERENCE_MODEL}  |  N = {N_TOTAL_SIMS} sims per scenario  |  "
    f"jg_wage = entry_anchor (25th pct. of occ. wage distribution)",
    fontsize=10, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "fig4_policy3_job_guarantee.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig4_policy3_job_guarantee.png")

print(f"\n✓ All policy figures saved to {OUTPUT_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3B: Policy 1 — combined summary figure (5 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 3B: Policy 1 combined summary figure ===")

fig5 = plt.figure(figsize=(18, 9))
gs5   = fig5.add_gridspec(2, 3, hspace=0.42, wspace=0.33)
ax5A1 = fig5.add_subplot(gs5[0, 0])   # A: survival — all unemployed
ax5A2 = fig5.add_subplot(gs5[0, 1])   # B: survival — LTU workers only
ax5C  = fig5.add_subplot(gs5[0, 2])   # C: wage scarring (all re-employed)
ax5D  = fig5.add_subplot(gs5[1, 0])   # D: joint distribution (all re-employed)
ax5F  = fig5.add_subplot(gs5[1, 1])   # E: aggregate LTUER
ax5G  = fig5.add_subplot(gs5[1, 2])   # F: occ. heterogeneity scatter

# Panel C stacked bar data collected in loop
_ltu_bar_data = {}   # label -> (n_ltu, n_non_ltu, color)

for idx, (label, (scenario, color)) in enumerate(_ltu_scenarios.items()):
    trans_pol, grouped_pol, *_ = policy_results[scenario]

    x_range = np.arange(0, 37)

    # ── All UE transitions (shared for A1, D, E) ─────────────────────────────
    all_ue = trans_pol[trans_pol['TransitionType'] == 'UE'].copy()
    all_ue = all_ue[all_ue['UEDuration'].between(1, 36)]
    all_ue = all_ue.merge(
        occ_wage_info[['Occupation', 'occ_median_wage']],
        left_on='DestinationOccupation', right_on='Occupation', how='left'
    )
    all_ue['WageRatio'] = all_ue['AcceptedWage'] / all_ue['occ_median_wage']

    # ── Panel A1: survival — all unemployed ───────────────────────────────────
    durs_all = np.sort(all_ue['UEDuration'].values)
    surv_all = np.array([(durs_all > x).mean() * 100 for x in x_range])
    ax5A1.plot(x_range, surv_all, color=color, linewidth=2, label=label)

    # ── LTU subset (shared for A2) ────────────────────────────────────────────
    ltu = all_ue[all_ue['LongtermUnemp'] == True].copy()

    # ── Panel A2: survival — LTU workers only ────────────────────────────────
    durs_ltu = np.sort(ltu['UEDuration'].values)
    surv_ltu = np.array([(durs_ltu > x).mean() * 100 for x in x_range])
    ax5A2.plot(x_range, surv_ltu, color=color, linewidth=2, label=label)

    # ── Panel C: mean per-sim economy-wide hire counts by LTU status ────────────
    def _count_c(g):
        return pd.Series({
            'n_ee':  (g['TransitionType'] == 'EE').sum(),
            'n_stu': ((g['TransitionType'] == 'UE') & (g['LongtermUnemp'] != True)).sum(),
            'n_ltu': ((g['TransitionType'] == 'UE') & (g['LongtermUnemp'] == True)).sum(),
        })

    if 'sim_id' in trans_pol.columns:
        _per_sim_c = trans_pol.groupby('sim_id').apply(_count_c).reset_index()
        n_ee_c  = _per_sim_c['n_ee'].mean()
        n_stu_c = _per_sim_c['n_stu'].mean()
        n_ltu_c = _per_sim_c['n_ltu'].mean()
    else:
        n_ee_c  = (trans_pol['TransitionType'] == 'EE').sum()
        n_stu_c = ((trans_pol['TransitionType'] == 'UE') & (trans_pol['LongtermUnemp'] != True)).sum()
        n_ltu_c = ((trans_pol['TransitionType'] == 'UE') & (trans_pol['LongtermUnemp'] == True)).sum()
    _ltu_bar_data[label] = (n_ee_c, n_stu_c, n_ltu_c, color)

    # ── Panel D: joint distribution — all unemployed ──────────────────────────
    sample_n = min(2000, len(all_ue))
    samp = all_ue.sample(sample_n, random_state=42) if len(all_ue) >= sample_n else all_ue
    ax5D.scatter(samp['UEDuration'], samp['WageRatio'].clip(0.05, 4.0),
                 alpha=0.12, s=6, color=color, rasterized=True)
    trend = (all_ue.sort_values('UEDuration')
                   .groupby('UEDuration')['WageRatio']
                   .mean()
                   .rolling(3, center=True, min_periods=1)
                   .mean())
    ax5D.plot(trend.index, trend.values.clip(0.05, 4.0),
              color=color, linewidth=1.8, label=label)

    # ── Panel F: aggregate LTUER ───────────────────────────────────────────
    ltuer_ps = grouped_pol.copy()
    ltuer_ps['LTUER'] = (ltuer_ps['LT Unemployed Persons'] /
                          ltuer_ps['Unemployment'].clip(lower=1))
    ltuer_agg = ltuer_ps.groupby('DATE')['LTUER'].agg(['mean', 'sem']).reset_index()
    ax5F.plot(ltuer_agg['DATE'], ltuer_agg['mean'] * 100, color=color, linewidth=2, label=label)
    ax5F.fill_between(ltuer_agg['DATE'],
                      (ltuer_agg['mean'] - 1.96 * ltuer_agg['sem']) * 100,
                      (ltuer_agg['mean'] + 1.96 * ltuer_agg['sem']) * 100,
                      alpha=0.15, color=color)

# ── Panel A1 formatting ───────────────────────────────────────────────────────
ax5A1.axvline(6,  color='grey', linewidth=0.9, linestyle='--', alpha=0.7)
ax5A1.axvline(12, color='grey', linewidth=0.9, linestyle=':',  alpha=0.7)
ax5A1.set_xlabel("Months unemployed at hire", fontsize=10)
ax5A1.set_ylabel("% unemployed not yet hired", fontsize=10)
ax5A1.set_title("A.  Time to re-employment (all unemployed)", fontsize=11, fontweight='bold')
ax5A1.set_xlim(0, 36); ax5A1.set_ylim(0, 100)
ax5A1.legend(fontsize=9); ax5A1.grid(axis='y', alpha=0.3)

# ── Panel A2 formatting ───────────────────────────────────────────────────────
ax5A2.axvline(6,  color='grey', linewidth=0.9, linestyle='--', alpha=0.7)
ax5A2.axvline(12, color='grey', linewidth=0.9, linestyle=':',  alpha=0.7)
ax5A2.set_xlabel("Months unemployed at hire", fontsize=10)
ax5A2.set_ylabel("% LTU workers not yet hired", fontsize=10)
ax5A2.set_title("B.  Time to re-employment (LTU workers only)", fontsize=11, fontweight='bold')
ax5A2.set_xlim(0, 36); ax5A2.set_ylim(0, 100)
ax5A2.legend(fontsize=9); ax5A2.grid(axis='y', alpha=0.3)

# ── Panel C: stacked bar — EE / non-LTU UE / LTU UE hires into entry-level ───
_c_lbls  = list(_ltu_bar_data.keys())
_c_n_ee  = [_ltu_bar_data[l][0] for l in _c_lbls]
_c_n_stu = [_ltu_bar_data[l][1] for l in _c_lbls]
_c_n_ltu = [_ltu_bar_data[l][2] for l in _c_lbls]
_c_bx    = np.arange(len(_c_lbls))
_c_ee_col  = '#2c7bb6'   # blue   for employed (EE)
_c_stu_col = '#74c476'   # green  for short-term unemployed
_c_ltu_col = '#7b5ea7'   # purple for long-term unemployed
ax5C.bar(_c_bx, _c_n_ee,  0.5, label='Employed (EE)',       color=_c_ee_col,  alpha=0.85)
ax5C.bar(_c_bx, _c_n_stu, 0.5, bottom=_c_n_ee,              label='Unemployed (non-LTU)', color=_c_stu_col, alpha=0.85)
ax5C.bar(_c_bx, _c_n_ltu, 0.5, bottom=[e+s for e,s in zip(_c_n_ee, _c_n_stu)],
         label='Long-term unemployed', color=_c_ltu_col, alpha=0.85)
for _ci, (_ne, _ns, _nl) in enumerate(zip(_c_n_ee, _c_n_stu, _c_n_ltu)):
    _ct = _ne + _ns + _nl
    if _ct > 0:
        for _val, _bot, _col in [(_ne, 0,          _c_ee_col),
                                  (_ns, _ne,         _c_stu_col),
                                  (_nl, _ne + _ns,   _c_ltu_col)]:
            if _val / _ct >= 0.04:
                ax5C.text(_ci, _bot + _val / 2, f'{_val/_ct*100:.0f}%',
                          ha='center', va='center', fontsize=8, color='white', fontweight='bold')
ax5C.set_xticks(_c_bx)
ax5C.set_xticklabels(_c_lbls, fontsize=9)
ax5C.set_ylabel("All hires (mean per simulation)", fontsize=10)
ax5C.set_title("C.  Economy-wide hiring composition\n(by prior labor market status)", fontsize=11, fontweight='bold')
ax5C.legend(fontsize=9); ax5C.grid(axis='y', alpha=0.3)

# ── Panel D formatting ────────────────────────────────────────────────────────
ax5D.axhline(1.0, color='grey', linewidth=1, linestyle='--', alpha=0.7)
ax5D.axvline(6, color='darkred', linewidth=1.4, linestyle=':', alpha=0.85,
             label='LTU threshold (≥6 months)')
ax5D.text(6.3, 3.3, 'LTU (≥6 months)', fontsize=8, color='darkred', va='top', ha='left')
ax5D.set_xlabel("Unemployment duration at hire (months)", fontsize=10)
ax5D.set_ylabel("Accepted wage / occ. median wage", fontsize=10)
ax5D.set_title("D.  Duration–wage joint distribution (all re-employed)", fontsize=11, fontweight='bold')
ax5D.set_xlim(0, 36); ax5D.set_ylim(0.05, 3.5)
ax5D.legend(fontsize=9); ax5D.grid(alpha=0.2)

# ── Panel E formatting ────────────────────────────────────────────────────────
for _, rec in recession_spans.iterrows():
    ax5F.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                 color='lightgrey', alpha=0.4, zorder=0)
ax5F.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax5F.set_xlabel("Date", fontsize=10)
ax5F.set_ylabel("LTUER (%)", fontsize=10)
ax5F.set_title("E.  Aggregate long-term unemployment rate", fontsize=11, fontweight='bold')
ax5F.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
ax5F.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax5F.get_xticklabels(), rotation=45, ha='right')
ax5F.legend(fontsize=9); ax5F.grid(axis='y', alpha=0.3)

# ── Panel F: occ. heterogeneity scatter — both policy scenarios ───────────────
_, _, _occ_base_5 = policy_results['baseline']

def _ltuer_mean(occ_df):
    g = (occ_df.groupby(['Occupation', 'sim_id'])
               .agg(ltu=('LT Unemployed Persons', 'mean'),
                    unemp=('Unemployment', 'mean'))
               .reset_index())
    g['ltuer'] = g['ltu'] / g['unemp'].clip(lower=0.1)
    return g.groupby('Occupation')['ltuer'].mean().reset_index()

# Occupation → SOC group crosswalk (X is 1-indexed; occupation_id = X - 1).
# Loaded once; reused for all aggregation levels.
_CROSSWALK_BY_NW = {
    'full_omn':       "data/occ_uer_ltuer_observed_w_labels.csv",
    'onet_wage_asym': "data/occ_uer_ltuer_observed_onet_w_labels.csv",
}
_cw_raw = pd.read_csv(_CROSSWALK_BY_NW.get(NW, "data/occ_uer_ltuer_observed_w_labels.csv"),
                      dtype={'SOC_major_adj': str, 'SOC_minor_adj': str, 'SOC_broad_adj': str})
_cw_raw['SOC_major_adj'] = pd.to_numeric(_cw_raw['SOC_major_adj'], errors='coerce')

def _build_grp_cw(agg_level):
    desc_col = agg_level.replace('_adj', '_description')
    return (_cw_raw[['X', agg_level, desc_col]]
            .drop_duplicates('X')
            .dropna(subset=[agg_level])
            .assign(Occupation=lambda d: d['X'].astype(int) - 1)
            .rename(columns={agg_level: 'OccGroup', desc_col: 'OccGroupLabel'})
            [['Occupation', 'OccGroup', 'OccGroupLabel']])

_ltuer_base5 = _ltuer_mean(_occ_base_5).rename(columns={'ltuer': 'ltuer_base'})

_AGG_LEVELS = ['SOC_major_adj', 'SOC_minor_adj', 'SOC_broad_adj']
_AGG_SUFFIX = {'SOC_major_adj': 'SOC_major', 'SOC_minor_adj': 'SOC_minor', 'SOC_broad_adj': 'SOC_broad'}

for _agg_level in _AGG_LEVELS:
    _occ_grp_cw = _build_grp_cw(_agg_level)

    ax5G.cla()   # clear only panel F; panels A–E are untouched
    for _scenario, _color, _lbl in [
        ('ltu_bias_0.5', _greens[0], 'LTU bias ×1.5'),
        ('ltu_bias_1.0', _greens[1], 'LTU bias ×2.0'),
    ]:
        _, _, _occ_pol_5 = policy_results[_scenario]
        _het5 = (_ltuer_base5
                 .merge(_ltuer_mean(_occ_pol_5).rename(columns={'ltuer': 'ltuer_pol'}), on='Occupation')
                 .merge(STATIC_OCC_DF[['Occupation', 'target_demand_init']], on='Occupation', how='left')
                 .merge(_occ_grp_cw, on='Occupation', how='left'))
        _het5['ltuer_change'] = _het5['ltuer_pol'] - _het5['ltuer_base']
        _het5 = _het5.dropna(subset=['OccGroup'])

        # Weighted aggregate to chosen SOC level
        _het5_grp = (_het5.groupby('OccGroup', as_index=False)
                     .apply(lambda g: pd.Series({
                         'ltuer_base':         np.average(g['ltuer_base'],   weights=g['target_demand_init'].clip(lower=1)),
                         'ltuer_change':       np.average(g['ltuer_change'], weights=g['target_demand_init'].clip(lower=1)),
                         'target_demand_init': g['target_demand_init'].sum(),
                     }))
                     .reset_index(drop=True))

        _x5 = _het5_grp['ltuer_base'] * 100
        _y5 = _het5_grp['ltuer_change'] * 100
        _w5 = _het5_grp['target_demand_init'].clip(lower=1)
        ax5G.scatter(_x5, _y5, c=_color, s=np.sqrt(_w5) * 4,
                     edgecolors='none', alpha=0.6, label=_lbl)
        _mask5 = _x5.notna() & _y5.notna()
        _fit5 = np.polyfit(_x5[_mask5], _y5[_mask5], 1, w=_w5[_mask5])
        _xline5 = np.linspace(_x5.min(), _x5.max(), 200)
        ax5G.plot(_xline5, np.polyval(_fit5, _xline5), color=_color, linewidth=1.8, linestyle='-')

    ax5G.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax5G.set_xlabel("Baseline LTUER (%)", fontsize=10)
    ax5G.set_ylabel("Change in LTUER (pp)", fontsize=10)
    ax5G.set_title(f"F.  Occ. heterogeneity: LTUER change\nby scenario ({_AGG_SUFFIX[_agg_level]})",
                   fontsize=10, fontweight='bold')
    ax5G.legend(fontsize=9); ax5G.grid(alpha=0.3)

    fig5.suptitle(
        "Policy 1: Preferential hiring for LTU workers — summary\n"
        f"Reference model: {REFERENCE_MODEL}  |  N = {N_TOTAL_SIMS} sims per scenario",
        fontsize=12, fontweight='bold'
    )
    _suffix = _AGG_SUFFIX[_agg_level]
    _fname5 = f"fig5_policy1_combined_{_suffix}.png"
    fig5.savefig(os.path.join(OUTPUT_PATH, _fname5), dpi=150, bbox_inches='tight')
    print(f"Saved: {_fname5}")

plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3B2: LTU bias diagnostic — stock and conditional hiring rate
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 3B2: LTU bias diagnostic ===")

def _ltu_stock_and_rate(trans_df, grouped_df, occ_df):
    """LTU stock, LTU conditional rate, and overall matching rate per sim, then averaged."""
    has_sim = 'sim_id' in trans_df.columns
    grp_g = ['Time Step', 'sim_id'] if has_sim else ['Time Step']
    grp_t = ['Time Step', 'sim_id'] if has_sim else ['Time Step']

    # LTU stock per (Time Step, sim_id)
    agg_ps = (grouped_df.groupby(grp_g)[['LT Unemployed Persons']]
                        .sum().reset_index())
    if not has_sim:
        agg_ps['sim_id'] = 0

    # LTU hires per (Time Step, sim_id)
    ltu_h = trans_df[(trans_df['TransitionType'] == 'UE') &
                     (trans_df['LongtermUnemp'] == True)]
    ltu_hires_ps = ltu_h.groupby(grp_t).size().reset_index(name='ltu_hires')
    if not has_sim:
        ltu_hires_ps['sim_id'] = 0

    merged = agg_ps.merge(ltu_hires_ps, on=['Time Step', 'sim_id'], how='left')
    merged['ltu_hires'] = merged['ltu_hires'].fillna(0)
    merged['ltu_rate']  = merged['ltu_hires'] / merged['LT Unemployed Persons'].clip(lower=1)

    ts_map = (grouped_df.drop_duplicates('Time Step').set_index('Time Step')['DATE'])
    merged['DATE'] = merged['Time Step'].map(ts_map)

    # Overall matching rate: total hires / total active seekers from occ_df
    grp_occ = ['Time Step', 'sim_id'] if 'sim_id' in occ_df.columns else ['Time Step']
    seek_ps = (occ_df.groupby(grp_occ)[['Hires', 'Unemployed Seekers', 'Employed Seekers']]
                     .sum().reset_index())
    if 'sim_id' not in seek_ps.columns:
        seek_ps['sim_id'] = 0
    seek_ps['total_seekers'] = seek_ps['Unemployed Seekers'] + seek_ps['Employed Seekers']
    seek_ps['match_rate']    = seek_ps['Hires'] / seek_ps['total_seekers'].clip(lower=1)
    seek_ps['DATE'] = seek_ps['Time Step'].map(ts_map)

    stock_ts   = merged.groupby('DATE')['LT Unemployed Persons'].agg(['mean', 'sem']).reset_index()
    rate_ts    = merged.groupby('DATE')['ltu_rate'].agg(['mean', 'sem']).reset_index()
    overall_ts = seek_ps.groupby('DATE')['match_rate'].agg(['mean', 'sem']).reset_index()
    return stock_ts, rate_ts, overall_ts

fig_ld, (ax_ld1, ax_ld2, ax_ld3) = plt.subplots(1, 3, figsize=(18, 5))
for label, (scenario, color) in _ltu_scenarios.items():
    trans_pol, grouped_pol, occ_pol = policy_results[scenario]
    ls = '--' if label != 'Baseline' else '-'
    stock_ts, rate_ts, overall_ts = _ltu_stock_and_rate(trans_pol, grouped_pol, occ_pol)
    ax_ld1.plot(stock_ts['DATE'], stock_ts['mean'], color=color, linewidth=2, linestyle=ls, label=label)
    ax_ld1.fill_between(stock_ts['DATE'],
                        stock_ts['mean'] - 1.96 * stock_ts['sem'],
                        stock_ts['mean'] + 1.96 * stock_ts['sem'],
                        alpha=0.12, color=color)
    ax_ld2.plot(rate_ts['DATE'], rate_ts['mean'] * 100, color=color, linewidth=2, linestyle=ls, label=label)
    ax_ld2.fill_between(rate_ts['DATE'],
                        (rate_ts['mean'] - 1.96 * rate_ts['sem']) * 100,
                        (rate_ts['mean'] + 1.96 * rate_ts['sem']) * 100,
                        alpha=0.12, color=color)
    ax_ld3.plot(overall_ts['DATE'], overall_ts['mean'] * 100, color=color, linewidth=2, linestyle=ls, label=label)
    ax_ld3.fill_between(overall_ts['DATE'],
                        (overall_ts['mean'] - 1.96 * overall_ts['sem']) * 100,
                        (overall_ts['mean'] + 1.96 * overall_ts['sem']) * 100,
                        alpha=0.12, color=color)

for ax_ld in (ax_ld1, ax_ld2, ax_ld3):
    for _, rec in recession_spans.iterrows():
        ax_ld.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                      color='lightgrey', alpha=0.4, zorder=0)
    ax_ld.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
    ax_ld.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
    ax_ld.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.setp(ax_ld.get_xticklabels(), rotation=45, ha='right')
    ax_ld.legend(fontsize=9); ax_ld.grid(axis='y', alpha=0.3)

ax_ld1.set_xlabel("Date", fontsize=10)
ax_ld1.set_ylabel("LTU workers (stock)", fontsize=10)
ax_ld1.set_title("A.  Long-term unemployment stock", fontsize=11, fontweight='bold')
ax_ld2.set_xlabel("Date", fontsize=10)
ax_ld2.set_ylabel("LTU hiring rate (% per period)", fontsize=10)
ax_ld2.set_title("B.  Conditional hiring rate\n(LTU hires / LTU stock)", fontsize=11, fontweight='bold')
ax_ld3.set_xlabel("Date", fontsize=10)
ax_ld3.set_ylabel("Matching rate (% per period)", fontsize=10)
ax_ld3.set_title("C.  Overall matching efficiency\n(all hires / all active seekers)", fontsize=11, fontweight='bold')

fig_ld.suptitle(f"LTU bias diagnostic — {REFERENCE_MODEL}", fontsize=11, fontweight='bold')
plt.tight_layout()
_fname_ld = f"fig_ltu_bias_diagnostic_{MODEL_SUFFIX.get(REFERENCE_MODEL, 'ref')}.png"
fig_ld.savefig(os.path.join(OUTPUT_PATH, _fname_ld), dpi=150, bbox_inches='tight')
plt.close(fig_ld)
print(f"  Saved: {_fname_ld}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3C: Policy 1 — occupational heterogeneity in LTUER
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 3C: Occupational heterogeneity — LTUER ===")

# Per-occupation mean LTUER = mean(LT Unemployed Persons) / mean(Unemployment).
# Baseline: from sim (pkl per-occ mean). Policy: from occ_df (record_df per sim).
def _occ_ltuer(occ_df):
    """Compute per-occupation mean LTUER averaged across time steps and sims."""
    g = (occ_df.groupby(['Occupation', 'sim_id'])
               .agg(ltu=('LT Unemployed Persons', 'mean'),
                    unemp=('Unemployment', 'mean'))
               .reset_index())
    g['ltuer'] = g['ltu'] / g['unemp'].clip(lower=0.1)
    return g.groupby('Occupation')['ltuer'].mean().reset_index()

_, _, occ_base = policy_results['baseline']
_, _, occ_pol  = policy_results['ltu_bias_1.0']

ltuer_base = _occ_ltuer(occ_base).rename(columns={'ltuer': 'ltuer_base'})
ltuer_pol  = _occ_ltuer(occ_pol).rename(columns={'ltuer': 'ltuer_pol'})

_het = (ltuer_base
        .merge(ltuer_pol, on='Occupation')
        .merge(STATIC_OCC_DF[['Occupation', 'wage_mu', 'entry_level', 'target_demand_init']],
               on='Occupation', how='left'))
_het['ltuer_change'] = _het['ltuer_pol'] - _het['ltuer_base']
_het['median_wage']  = np.exp(_het['wage_mu'])

print(f"  Occupations with reduced LTUER: "
      f"{(_het['ltuer_change'] < 0).sum()} / {len(_het)}")
print(f"  Mean LTUER change (pp): {_het['ltuer_change'].mean()*100:.3f}")
print(f"  Entry-level mean:   {_het.loc[_het['entry_level'],  'ltuer_change'].mean()*100:.3f} pp")
print(f"  Non-entry mean:     {_het.loc[~_het['entry_level'], 'ltuer_change'].mean()*100:.3f} pp")

fig6, axes6 = plt.subplots(1, 2, figsize=(13, 5))
ax6A, ax6B = axes6

# Panel A: scatter — baseline LTUER vs change under policy
_dot_colors = _het['entry_level'].map({True: '#7b5ea7', False: '#4a9aba'})
ax6A.scatter(
    _het['ltuer_base'] * 100,
    _het['ltuer_change'] * 100,
    c=_dot_colors,
    s=np.sqrt(_het['target_demand_init'].clip(lower=1)) * 4,
    alpha=0.55, edgecolors='none'
)
ax6A.axhline(0, color='black', linewidth=0.8, linestyle='--')
for _col, _lbl in [('#7b5ea7', 'Entry-level'), ('#4a9aba', 'Non-entry-level')]:
    ax6A.scatter([], [], c=_col, s=40, alpha=0.7, label=_lbl)
ax6A.set_xlabel("Baseline LTUER (%)", fontsize=10)
ax6A.set_ylabel("Change in LTUER (pp)", fontsize=10)
ax6A.set_title("A.  Baseline LTUER vs policy effect by occupation\n"
               "(size = occ. size; colour = entry-level status)", fontsize=10, fontweight='bold')
ax6A.legend(fontsize=9); ax6A.grid(alpha=0.3)

# Panel B: distribution of LTUER changes by entry-level status
_entry_ch    = _het.loc[_het['entry_level'],  'ltuer_change'] * 100
_nonentry_ch = _het.loc[~_het['entry_level'], 'ltuer_change'] * 100
ax6B.hist(_nonentry_ch, bins=30, color='#4a9aba', alpha=0.6,
          label=f'Non-entry-level (n={len(_nonentry_ch)})', edgecolor='white', linewidth=0.3)
ax6B.hist(_entry_ch, bins=30, color='#7b5ea7', alpha=0.7,
          label=f'Entry-level (n={len(_entry_ch)})', edgecolor='white', linewidth=0.3)
ax6B.axvline(0, color='black', linewidth=1, linestyle='--')
ax6B.axvline(_entry_ch.mean(), color='#7b5ea7', linewidth=1.8, linestyle='-',
             label=f'Entry mean = {_entry_ch.mean():.3f} pp')
ax6B.axvline(_nonentry_ch.mean(), color='#4a9aba', linewidth=1.8, linestyle='-',
             label=f'Non-entry mean = {_nonentry_ch.mean():.3f} pp')
ax6B.set_xlabel("Change in LTUER (percentage points)", fontsize=10)
ax6B.set_ylabel("Number of occupations", fontsize=10)
ax6B.set_title("B.  Distribution of LTUER changes across occupations\n"
               "(LTU bias ×2.0 vs baseline)", fontsize=10, fontweight='bold')
ax6B.legend(fontsize=9); ax6B.grid(axis='y', alpha=0.3)

fig6.suptitle(
    "Policy 1: Occupational heterogeneity in LTUER (LTU bias ×2.0 vs baseline)\n"
    f"Reference model: {REFERENCE_MODEL}  |  N = {N_TOTAL_SIMS} sims per scenario",
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "fig6_policy1_occ_heterogeneity.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig6_policy1_occ_heterogeneity.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Cross-model comparison — LTU bias ×2.0
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 5: Cross-model LTU bias ×2.0 comparison ===")

from model_colors import MODEL_COLORS_BY_ID

# Model flags: model_cat → flags that override REF_BASE_PARAMS
_CM_SPECS = {
    'otj_nonbehav':              {'otj': True,  'cyc_otj': False, 'cyc_ue': False, 'disc': False, 'wage_prefs': False, 'strict_rw': True},
    'otj_disc':                  {'otj': True,  'cyc_otj': False, 'cyc_ue': False, 'disc': True,  'wage_prefs': True,  'strict_rw': False},
    'otj_disc_no_rw':            {'otj': True,  'cyc_otj': False, 'cyc_ue': False, 'disc': True,  'wage_prefs': False, 'strict_rw': True},
    'otj_cyclical_e_disc':       {'otj': True,  'cyc_otj': True,  'cyc_ue': False, 'disc': True,  'wage_prefs': True,  'strict_rw': False},
    'otj_cyclical_e_disc_no_rw': {'otj': True,  'cyc_otj': True,  'cyc_ue': False, 'disc': True,  'wage_prefs': False, 'strict_rw': True},
}

_CM_DISPLAY = {
    'otj_nonbehav':              'Non-behav.\nw. OTJ',
    'otj_disc':                  'Behav. w.o.\nCyc. OTJ\nw. RW',
    'otj_disc_no_rw':            'Behav. w.o.\nCyc. OTJ\nw.o RW',
    'otj_cyclical_e_disc':       'Behav. w.\nCyc. OTJ\nw. RW\n(ref.)',
    'otj_cyclical_e_disc_no_rw': 'Behav. w.\nCyc. OTJ\nw.o RW',
}

CROSS_MODEL_PKL   = os.path.join(OUTPUT_PATH, "cross_model_ltu_bias.pkl")
FORCE_CROSS_RERUN = False

# Replicate auto_run_model_modified param loading: concatenate the two CSVs
# so that all model_cats have gamma_u/d_u, and cyclical_e models also have theta.
_calib_base_csv  = os.path.join(
    PROJECT_ROOT,
    "output/calib_final/full_omn/exog_vac/calibrated_params_all_exog_wage_update_True.csv"
)
_calib_theta_csv = CALIB_CSV  # already loaded path (theta_signal variant)
_cm_param_df = pd.concat(
    [pd.read_csv(_calib_base_csv), pd.read_csv(_calib_theta_csv)],
    ignore_index=True
).sort_values('Timestamp', ascending=False)


def _build_model_params(model_cat):
    """Construct base_params for model_cat — exact same logic as auto_run_model_modified."""
    _pf = _cm_param_df
    if 'cyclical_e' in model_cat:
        _calib = {
            'gamma_u': float(_pf.loc[(_pf['model_cat'] == model_cat) & (_pf['Parameter'] == 'gamma_u'), 'Value'].iloc[0]),
            'd_u':     float(_pf.loc[(_pf['model_cat'] == model_cat) & (_pf['Parameter'] == 'd_u'),     'Value'].iloc[0]),
            'theta':   float(_pf.loc[(_pf['model_cat'] == model_cat) & (_pf['Parameter'] == 'theta'),   'Value'].iloc[0]),
            'gamma_v': None,
            'd_v':     None,
        }
    else:
        _calib = {
            'gamma_u': float(_pf.loc[(_pf['model_cat'] == model_cat) & (_pf['Parameter'] == 'gamma_u'), 'Value'].iloc[0]),
            'd_u':     float(_pf.loc[(_pf['model_cat'] == model_cat) & (_pf['Parameter'] == 'd_u'),     'Value'].iloc[0]),
            'theta':   None,
            'gamma_v': None,
            'd_v':     None,
        }
    params = deepcopy(REF_BASE_PARAMS)
    params.update(_calib)
    params.update(_CM_SPECS[model_cat])
    return params


if os.path.exists(CROSS_MODEL_PKL) and not FORCE_CROSS_RERUN:
    print(f"Loading cross-model cache from {CROSS_MODEL_PKL} ...")
    with open(CROSS_MODEL_PKL, 'rb') as _f:
        cross_model_results = pickle.load(_f)
    print(f"  Loaded: {list(cross_model_results.keys())}")
else:
    cross_model_results = {}
    for _mc in _CM_SPECS:
        print(f"\n  [{_mc}] building params...")
        _bp = _build_model_params(_mc)
        print(f"    gamma_u={_bp['gamma_u']:.4f}  d_u={_bp['d_u']:.5f}")
        cross_model_results[f"{_mc}_baseline"] = run_policy_scenario(
            f"{_mc}_baseline", {}, _bp, CALIB_DATE, N_POLICY_SIMS
        )
        cross_model_results[f"{_mc}_ltu_bias"] = run_policy_scenario(
            f"{_mc}_ltu_bias", {'ltu_hire_bias': 1.0}, _bp, CALIB_DATE, N_POLICY_SIMS
        )
        print(f"    ✓ {_mc} done")
    print(f"\nSaving to {CROSS_MODEL_PKL} ...")
    with open(CROSS_MODEL_PKL, 'wb') as _f:
        pickle.dump(cross_model_results, _f)
    print("  Saved.")

# ══════════════════════════════════════════════════════════════════════════════
# TOP-UP RUNS: extend all policy scenarios from 5 → 25 simulations
# Runs 20 additional sims per scenario, merges into existing results in-memory.
# baseline (policy_results['baseline']) is left as-is: it comes from the N10
# calibration cache and already has 10 sims with calibration-quality seeds.
# ══════════════════════════════════════════════════════════════════════════════
N_EXISTING_SIMS     = N_POLICY_SIMS        # = 5
N_TOPUP_SIMS        = 20
POLICY_TOPUP_PKL    = os.path.join(OUTPUT_PATH, "policy_results_topup.pkl")
FORCE_TOPUP_RERUN   = False

def _concat_results(a, b):
    """Concatenate two (trans_df, grouped_df, occ_df) tuples."""
    return tuple(pd.concat([x, y], ignore_index=True) for x, y in zip(a, b))

_POLICY_SCENARIOS_TOPUP = {
    'ltu_bias_0.5':   {'ltu_hire_bias':   0.5},
    # ltu_bias_1.0 is omitted: reused from cross_model_results['otj_cyclical_e_disc_ltu_bias']
    'entry_bias_0.5': {'entry_hire_bias': 0.5},
    'entry_bias_1.0': {'entry_hire_bias': 1.0},
}

if os.path.exists(POLICY_TOPUP_PKL) and not FORCE_TOPUP_RERUN:
    print(f"Loading policy top-up cache ({POLICY_TOPUP_PKL}) ...")
    with open(POLICY_TOPUP_PKL, 'rb') as _f:
        _policy_topups = pickle.load(_f)
    print(f"  Loaded: {list(_policy_topups.keys())}")
else:
    _policy_topups = {}
    for _sc, _ov in _POLICY_SCENARIOS_TOPUP.items():
        print(f"\n  [{_sc}] running {N_TOPUP_SIMS} additional sims ...")
        _policy_topups[_sc] = run_policy_scenario(
            f"{_sc}_topup", _ov, REF_BASE_PARAMS, CALIB_DATE,
            n_sims=N_TOPUP_SIMS, sim_id_offset=N_EXISTING_SIMS
        )
    with open(POLICY_TOPUP_PKL, 'wb') as _f:
        pickle.dump(_policy_topups, _f)
    print(f"  Saved: {POLICY_TOPUP_PKL}")

for _sc, _extra in _policy_topups.items():
    if _sc in policy_results:
        policy_results[_sc] = _concat_results(policy_results[_sc], _extra)
_n_po = policy_results['ltu_bias_1.0'][0]['sim_id'].nunique()
print(f"  policy_results: {_n_po} sims per scenario after top-up.")

def _cross_topup_pkl(mc):
    return os.path.join(OUTPUT_PATH, f"cross_model_{mc}_topup.pkl")

_all_topup_exist = all(os.path.exists(_cross_topup_pkl(_mc)) for _mc in _CM_SPECS)
if _all_topup_exist and not FORCE_TOPUP_RERUN:
    print("Per-model cross-model top-up pkls found — will load on demand.")
else:
    for _mc in _CM_SPECS:
        _pkl_path = _cross_topup_pkl(_mc)
        if os.path.exists(_pkl_path) and not FORCE_TOPUP_RERUN:
            print(f"  [{_mc}] topup pkl already exists, skipping.")
            continue
        print(f"\n  [{_mc}] running {N_TOPUP_SIMS} top-up sims ...")
        _bp = _build_model_params(_mc)
        _mc_topup = {}
        _key = f"{_mc}_ltu_bias"
        _mc_topup[_key] = run_policy_scenario(
            f"{_key}_topup", {'ltu_hire_bias': 1.0}, _bp, CALIB_DATE,
            n_sims=N_TOPUP_SIMS, sim_id_offset=N_EXISTING_SIMS
        )
        with open(_pkl_path, 'wb') as _f:
            pickle.dump(_mc_topup, _f)
        print(f"  Saved: {_pkl_path}")
        del _mc_topup


def _load_cm_model(mc):
    """Baseline from cached_data pkl (N-sim calibration run); ltu_bias from
    cross_model pkl (5 sims) + per-model topup pkl (20 sims), merged.
    Returns (bl_tuple, po_tuple). Caller deletes when done to free RAM."""
    _display = _CM_ID_TO_DISPLAY[mc]
    _grp_raw = (cached_data['model_results_all_sims'].get(mc) or
                cached_data['model_results_all_sims'].get(_display))
    if isinstance(_grp_raw, list):
        _parts = []
        for _i, _df in enumerate(_grp_raw):
            _df = _df.copy()
            _df['sim_id'] = _i
            if 'DATE' not in _df.columns:
                _df['DATE'] = pd.date_range(
                    start=CALIB_DATE[0], end=CALIB_DATE[1], periods=len(_df))
            _parts.append(_df)
        _grp_bl = pd.concat(_parts, ignore_index=True)
    else:
        _grp_bl = _grp_raw
    _occ_bl = (cached_data.get('filtered_sim_results_all_sims') or {}).get(_display)
    if _occ_bl is None:
        _occ_bl = cached_data['filtered_sim_results'][_display]
    _bl = (
        cached_data['filtered_transition_dfs'][_display],
        _grp_bl,
        _occ_bl,
    )
    _po_ex = cross_model_results.get(f"{mc}_ltu_bias")
    _pkl_path = _cross_topup_pkl(mc)
    if os.path.exists(_pkl_path):
        with open(_pkl_path, 'rb') as _f:
            _topup = pickle.load(_f)
        _po = (_concat_results(_po_ex, _topup[f"{mc}_ltu_bias"])
               if _po_ex is not None else _topup[f"{mc}_ltu_bias"])
        del _topup
    else:
        _po = _po_ex
    return _bl, _po


# Assign reference model to policy_results['ltu_bias_1.0']
_REFERENCE_MODEL_ID = MODEL_SUFFIX[REFERENCE_MODEL]
_, _po_ref = _load_cm_model(_REFERENCE_MODEL_ID)
policy_results['ltu_bias_1.0'] = _po_ref
print(f"  policy_results['ltu_bias_1.0'] set from {_REFERENCE_MODEL_ID}_ltu_bias "
      f"({_po_ref[0]['sim_id'].nunique()} sims).")


def _cm_mean_ltuer(grouped_df):
    g = grouped_df.copy()
    g['ltuer'] = g['LT Unemployed Persons'] / g['Unemployment'].clip(lower=1)
    return float(g['ltuer'].mean()) * 100


def _cm_mean_match_rate(occ_df):
    grp = ['Time Step', 'sim_id'] if 'sim_id' in occ_df.columns else ['Time Step']
    ps  = (occ_df.groupby(grp)[['Hires', 'Unemployed Seekers', 'Employed Seekers']]
                 .sum().reset_index())
    ps['rate'] = ps['Hires'] / (ps['Unemployed Seekers'] + ps['Employed Seekers']).clip(lower=1)
    return float(ps['rate'].mean()) * 100


def _match_rate_per_sim(occ_df):
    """Per-sim mean overall matching rate. Returns None if no sim_id."""
    if 'sim_id' not in occ_df.columns or occ_df['sim_id'].nunique() <= 1:
        return None
    ps = (occ_df.groupby(['Time Step', 'sim_id'])[['Hires', 'Unemployed Seekers', 'Employed Seekers']]
               .sum().reset_index())
    ps['rate'] = ps['Hires'] / (ps['Unemployed Seekers'] + ps['Employed Seekers']).clip(lower=1)
    return ps.groupby('sim_id')['rate'].mean()

print("\n  Model            | Baseline LTUER | Policy LTUER | Reduction (pp) | Match eff. Δ")
print("  " + "-" * 80)
_cm_metrics = {}
for _mc in _CM_SPECS:
    _bl_m, _po_m = _load_cm_model(_mc)
    _, grp_bl, occ_bl = _bl_m
    _, grp_po, occ_po = _po_m
    _ltuer_bl = _cm_mean_ltuer(grp_bl)
    _ltuer_po = _cm_mean_ltuer(grp_po)
    _eff_bl   = _cm_mean_match_rate(occ_bl)
    _eff_po   = _cm_mean_match_rate(occ_po)
    _eff_imp  = (_eff_po / _eff_bl - 1) * 100 if _eff_bl > 0 else 0.0
    _eff_bl_s = _match_rate_per_sim(occ_bl)
    _eff_po_s = _match_rate_per_sim(occ_po)
    if _eff_bl_s is not None and _eff_po_s is not None:
        _common_s  = _eff_bl_s.index.intersection(_eff_po_s.index)
        _imp_s     = (_eff_po_s[_common_s] / _eff_bl_s[_common_s] - 1) * 100
        _se_eff    = float(_imp_s.std(ddof=1) / np.sqrt(len(_imp_s))) if len(_imp_s) > 1 else 0.0
    else:
        _se_eff = 0.0
    _cm_metrics[_mc] = {
        'ltuer_bl':     _ltuer_bl,
        'ltuer_po':     _ltuer_po,
        'ltuer_red_pp': _ltuer_bl - _ltuer_po,
        'eff_bl':       _eff_bl,
        'eff_imp_pct':  _eff_imp,
        'se_eff_imp':   _se_eff,
    }
    print(f"  {_mc:<28} {_ltuer_bl:>7.2f}%   {_ltuer_po:>7.2f}%   "
          f"{_ltuer_bl-_ltuer_po:>7.2f} pp   {_eff_imp:>+.1f}%")
    del _bl_m, _po_m, grp_bl, occ_bl, grp_po, occ_po

# ── Figure ─────────────────────────────────────────────────────────────────────
_cm_cats   = list(_CM_SPECS.keys())
_cm_labels = [_CM_DISPLAY[c] for c in _cm_cats]
_cm_x      = np.arange(len(_cm_cats))
_cm_colors = [MODEL_COLORS_BY_ID.get(c, '#aaaaaa') for c in _cm_cats]
_cm_edges  = ['black' if c == 'otj_cyclical_e_disc' else 'white' for c in _cm_cats]
_cm_lw     = [1.8     if c == 'otj_cyclical_e_disc' else 0.3     for c in _cm_cats]

fig_cm, (ax_cm1, ax_cm2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: LTUER reduction (pp)
_ltuer_red = [_cm_metrics[c]['ltuer_red_pp'] for c in _cm_cats]
_ltuer_bl  = [_cm_metrics[c]['ltuer_bl']     for c in _cm_cats]
_A_max     = max(_ltuer_red) if _ltuer_red else 1
bars_A = ax_cm1.bar(_cm_x, _ltuer_red, 0.6, color=_cm_colors, alpha=0.85,
                    edgecolor=_cm_edges, linewidth=_cm_lw)
for bar, bl in zip(bars_A, _ltuer_bl):
    ax_cm1.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + _A_max * 0.02,
                f"BL={bl:.1f}%",
                ha='center', va='bottom', fontsize=8)
ax_cm1.set_xticks(_cm_x)
ax_cm1.set_xticklabels(_cm_labels, fontsize=9)
ax_cm1.set_ylabel("LTUER reduction (pp)", fontsize=10)
ax_cm1.set_title("A.  Long-term unemployment rate reduction\n(LTU bias ×2.0 vs model-specific baseline)",
                 fontsize=11, fontweight='bold')
ax_cm1.axhline(0, color='black', linewidth=0.8)
ax_cm1.grid(axis='y', alpha=0.3)

# Panel B: matching efficiency improvement (%)
_eff_imp = [_cm_metrics[c]['eff_imp_pct'] for c in _cm_cats]
ax_cm2.bar(_cm_x, _eff_imp, 0.6, color=_cm_colors, alpha=0.85,
           edgecolor=_cm_edges, linewidth=_cm_lw)
ax_cm2.set_xticks(_cm_x)
ax_cm2.set_xticklabels(_cm_labels, fontsize=9)
ax_cm2.set_ylabel("Matching efficiency improvement (%)", fontsize=10)
ax_cm2.set_title("B.  Overall matching efficiency improvement\n(LTU bias ×2.0 vs model-specific baseline)",
                 fontsize=11, fontweight='bold')
ax_cm2.axhline(0, color='black', linewidth=0.8)
ax_cm2.grid(axis='y', alpha=0.3)

from matplotlib.patches import Patch as _Patch
_legend_h = [_Patch(facecolor=MODEL_COLORS_BY_ID.get(c, '#aaaaaa'), alpha=0.85,
                    edgecolor='black' if c == 'otj_cyclical_e_disc' else 'none',
                    linewidth=1.8 if c == 'otj_cyclical_e_disc' else 0,
                    label=_CM_DISPLAY[c].replace('\n', ' '))
             for c in _cm_cats]
ax_cm2.legend(handles=_legend_h, fontsize=8, loc='upper left')

fig_cm.suptitle(
    f"LTU bias ×2.0: policy effects across model specifications  |  N = {N_TOTAL_SIMS} sims per scenario",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
_fname_cm = "fig_cross_model_ltu_bias.png"
fig_cm.savefig(os.path.join(OUTPUT_PATH, _fname_cm), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {_fname_cm}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5B: Cross-model combined figure — 3×3
# Row 1: distributional panels (survival curves + LTUER time series)
# Row 2: scalar outcomes (LTUER reduction, matching efficiency, LTU hire share)
# Row 3: duration & wages (UE duration all, UE duration LTU, wage ratio LTU)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 5B: Cross-model combined figure ===")

# Exclude cyclical OTJ models
_CM_SPECS_NO_CYC = {k: v for k, v in _CM_SPECS.items() if 'cyclical_e' not in k}

_x_range = np.arange(0, 37)

def _hires_rate_ts(grp_df):
    """Mean hires rate (hires/employment) per DATE across sims."""
    agg = grp_df.groupby('DATE')['Hires Rate'].agg(['mean', 'sem']).reset_index()
    return agg

def _jfr_ts(grp_df):
    """UE job-finding rate = UE_Transitions/Unemployment per DATE (%). No N_occ inflation."""
    _g = grp_df.copy()
    _g['jfr'] = _g['UE_Transitions'] / _g['Unemployment'].clip(lower=1) * 100
    agg = _g.groupby('DATE')['jfr'].agg(['mean', 'sem']).reset_index()
    return agg

def _ltuer_ts(grp):
    g = grp.copy()
    g['ltuer'] = g['LT Unemployed Persons'] / g['Unemployment'].clip(lower=1)
    agg = g.groupby('DATE')['ltuer'].agg(['mean', 'sem']) * 100
    return agg

def _ltu_wr(tr):
    _t = tr[(tr['TransitionType'] == 'UE') & (tr['LongtermUnemp'] == True)].copy()
    _t = _t[_t['PreSepWage'] > 0].copy()
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    return _t['wr'].mean() if len(_t) > 0 else np.nan

def _ltu_wr_per_sim(tr):
    _t = tr[(tr['TransitionType'] == 'UE') & (tr['LongtermUnemp'] == True)].copy()
    _t = _t[_t['PreSepWage'] > 0].copy()
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    if 'sim_id' not in _t.columns or _t['sim_id'].nunique() <= 1:
        return None
    return _t.groupby('sim_id')['wr'].mean()

def _ue_nonltu_wr(tr):
    _t = tr[(tr['TransitionType'] == 'UE') & (tr['LongtermUnemp'] != True)].copy()
    _t = _t[_t['PreSepWage'] > 0]
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    return _t['wr'].mean() if len(_t) > 0 else np.nan

def _ue_nonltu_wr_per_sim(tr):
    _t = tr[(tr['TransitionType'] == 'UE') & (tr['LongtermUnemp'] != True)].copy()
    _t = _t[_t['PreSepWage'] > 0]
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    if 'sim_id' not in _t.columns or _t['sim_id'].nunique() <= 1:
        return None
    return _t.groupby('sim_id')['wr'].mean()

def _ee_wr(tr):
    _t = tr[tr['TransitionType'] == 'EE'].copy()
    _t = _t[_t['PreSepWage'] > 0]
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    return _t['wr'].mean() if len(_t) > 0 else np.nan

def _ee_wr_per_sim(tr):
    _t = tr[tr['TransitionType'] == 'EE'].copy()
    _t = _t[_t['PreSepWage'] > 0]
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    if 'sim_id' not in _t.columns or _t['sim_id'].nunique() <= 1:
        return None
    return _t.groupby('sim_id')['wr'].mean()

def _surv_ci(ue_df, x_range):
    """Per-sim survival curve mean and SEM. Falls back to pooled mean + zero SE if no sim_id."""
    if 'sim_id' not in ue_df.columns or ue_df['sim_id'].nunique() <= 1:
        vals = np.array([(ue_df['UEDuration'] > x).mean() * 100 for x in x_range])
        return vals, np.zeros(len(x_range))
    sims = ue_df['sim_id'].unique()
    by_sim = np.array([
        [(ue_df.loc[ue_df['sim_id'] == s, 'UEDuration'] > x).mean() * 100 for x in x_range]
        for s in sims
    ])
    return by_sim.mean(axis=0), by_sim.std(axis=0) / np.sqrt(len(sims))

# Pre-collect all treatment-effect metrics
_cm_te = {}
for _mc in _CM_SPECS_NO_CYC:
    _bl_te, _po_te = _load_cm_model(_mc)
    _tr_bl, _grp_bl, _occ_bl = _bl_te
    _tr_po, _grp_po, _occ_po = _po_te

    _ue_bl = _tr_bl[(_tr_bl['TransitionType'] == 'UE') & _tr_bl['UEDuration'].between(1, 36)]
    _ue_po = _tr_po[(_tr_po['TransitionType'] == 'UE') & _tr_po['UEDuration'].between(1, 36)]
    _ltu_bl     = _ue_bl[_ue_bl['LongtermUnemp'] == True]
    _ltu_po     = _ue_po[_ue_po['LongtermUnemp'] == True]
    _non_ltu_bl = _ue_bl[_ue_bl['LongtermUnemp'] != True]
    _non_ltu_po = _ue_po[_ue_po['LongtermUnemp'] != True]

    # Survival curves with per-sim CIs
    _surv_bl_all,  _se_bl_all  = _surv_ci(_non_ltu_bl, _x_range)
    _surv_po_all,  _se_po_all  = _surv_ci(_non_ltu_po, _x_range)
    _surv_bl_ltu,  _se_bl_ltu  = _surv_ci(_ltu_bl, _x_range)
    _surv_po_ltu,  _se_po_ltu  = _surv_ci(_ltu_po, _x_range)

    # LTUER time series with propagated SE
    _ts_bl  = _ltuer_ts(_grp_bl)
    _ts_po  = _ltuer_ts(_grp_po)
    _common = _ts_bl.index.intersection(_ts_po.index)
    _delta_ts = pd.DataFrame({
        'DATE':        _common,
        'delta_ltuer': (_ts_po.loc[_common, 'mean'] - _ts_bl.loc[_common, 'mean']).values,
        'se_ltuer':    np.sqrt(_ts_po.loc[_common, 'sem'].values**2
                               + _ts_bl.loc[_common, 'sem'].values**2),
    })

    # Δ LTU conditional hire rate and Δ overall matching efficiency with propagated SE
    _, _rate_bl, _overall_bl = _ltu_stock_and_rate(_tr_bl, _grp_bl, _occ_bl)
    _, _rate_po, _overall_po = _ltu_stock_and_rate(_tr_po, _grp_po, _occ_po)

    _rate_common = _rate_bl.set_index('DATE').index.intersection(
                    _rate_po.set_index('DATE').index)
    _delta_rate  = pd.DataFrame({
        'DATE':          _rate_common,
        'delta_ltu_rate': (_rate_po.set_index('DATE').loc[_rate_common, 'mean']
                           - _rate_bl.set_index('DATE').loc[_rate_common, 'mean']).values,
        'se_ltu_rate':    np.sqrt(_rate_po.set_index('DATE').loc[_rate_common, 'sem'].values**2
                                  + _rate_bl.set_index('DATE').loc[_rate_common, 'sem'].values**2),
    })

    _ov_common     = _overall_bl.set_index('DATE').index.intersection(
                      _overall_po.set_index('DATE').index)
    _delta_overall = pd.DataFrame({
        'DATE':           _ov_common,
        'delta_match_eff': (_overall_po.set_index('DATE').loc[_ov_common, 'mean']
                            - _overall_bl.set_index('DATE').loc[_ov_common, 'mean']).values,
        'se_match_eff':    np.sqrt(_overall_po.set_index('DATE').loc[_ov_common, 'sem'].values**2
                                   + _overall_bl.set_index('DATE').loc[_ov_common, 'sem'].values**2),
    })

    # Per-sim SEs for scalar bar metrics
    _has_sim = 'sim_id' in _ue_bl.columns and _ue_bl['sim_id'].nunique() > 1
    if _has_sim:
        _ns = _ue_bl['sim_id'].nunique()
        def _se_delta(gb, ga, col):
            se_b = gb.groupby('sim_id')[col].mean().std(ddof=1) / np.sqrt(_ns)
            se_a = ga.groupby('sim_id')[col].mean().std(ddof=1) / np.sqrt(_ns)
            return float(np.sqrt(se_b**2 + se_a**2))
        _ue_bl_s = _ue_bl.copy(); _ue_bl_s['is_ltu'] = (_ue_bl_s['LongtermUnemp']==True).astype(float)*100
        _ue_po_s = _ue_po.copy(); _ue_po_s['is_ltu'] = (_ue_po_s['LongtermUnemp']==True).astype(float)*100
        _se_ltu_share = _se_delta(_ue_bl_s, _ue_po_s, 'is_ltu')
        _se_dur_all   = _se_delta(_non_ltu_bl, _non_ltu_po, 'UEDuration')
        _se_ltu_dur   = _se_delta(_ltu_bl,    _ltu_po,    'UEDuration')
        _t_bl = _tr_bl[(_tr_bl['TransitionType']=='UE') & (_tr_bl['LongtermUnemp']==True)].copy()
        _t_bl = _t_bl.merge(occ_wage_info[['Occupation','occ_median_wage']],
                            left_on='DestinationOccupation', right_on='Occupation', how='left')
        _t_bl['wr'] = _t_bl['AcceptedWage'] / _t_bl['occ_median_wage']
        _t_po = _tr_po[(_tr_po['TransitionType']=='UE') & (_tr_po['LongtermUnemp']==True)].copy()
        _t_po = _t_po.merge(occ_wage_info[['Occupation','occ_median_wage']],
                            left_on='DestinationOccupation', right_on='Occupation', how='left')
        _t_po['wr'] = _t_po['AcceptedWage'] / _t_po['occ_median_wage']
        _se_wr_ltu = _se_delta(_t_bl, _t_po, 'wr')
    else:
        _se_ltu_share = _se_dur_all = _se_ltu_dur = _se_wr_ltu = 0.0

    # % change in wage ratio from each model's own baseline
    _wr_bl_s = _ltu_wr_per_sim(_tr_bl);  _wr_po_s = _ltu_wr_per_sim(_tr_po)
    if _wr_bl_s is not None and _wr_po_s is not None:
        _cs_wr = _wr_bl_s.index.intersection(_wr_po_s.index)
        _pct_ch = (_wr_po_s[_cs_wr] / _wr_bl_s[_cs_wr] - 1) * 100
        _se_pct_wr = float(_pct_ch.std(ddof=1) / np.sqrt(len(_pct_ch))) if len(_pct_ch) > 1 else 0.0
    else:
        _se_pct_wr = 0.0
    _wr_bl_m = _ltu_wr(_tr_bl);  _wr_po_m = _ltu_wr(_tr_po)
    _pct_wr_ltu = (_wr_po_m / _wr_bl_m - 1) * 100 if (_wr_bl_m and _wr_bl_m > 0) else 0.0

    # % change in wage retention — non-LTU unemployed (UE, LongtermUnemp != True)
    _unwr_bl_s = _ue_nonltu_wr_per_sim(_tr_bl);  _unwr_po_s = _ue_nonltu_wr_per_sim(_tr_po)
    if _unwr_bl_s is not None and _unwr_po_s is not None:
        _cs_unwr = _unwr_bl_s.index.intersection(_unwr_po_s.index)
        _pct_ch_un = (_unwr_po_s[_cs_unwr] / _unwr_bl_s[_cs_unwr] - 1) * 100
        _se_pct_unwr = float(_pct_ch_un.std(ddof=1) / np.sqrt(len(_pct_ch_un))) if len(_pct_ch_un) > 1 else 0.0
    else:
        _se_pct_unwr = 0.0
    _unwr_bl_m = _ue_nonltu_wr(_tr_bl);  _unwr_po_m = _ue_nonltu_wr(_tr_po)
    _pct_wr_ue_nonltu = (_unwr_po_m / _unwr_bl_m - 1) * 100 if (_unwr_bl_m and _unwr_bl_m > 0) else 0.0

    # % change in wage retention — employed movers (EE)
    _ewr_bl_s = _ee_wr_per_sim(_tr_bl);  _ewr_po_s = _ee_wr_per_sim(_tr_po)
    if _ewr_bl_s is not None and _ewr_po_s is not None:
        _cs_ewr = _ewr_bl_s.index.intersection(_ewr_po_s.index)
        _pct_ch_e = (_ewr_po_s[_cs_ewr] / _ewr_bl_s[_cs_ewr] - 1) * 100
        _se_pct_ewr = float(_pct_ch_e.std(ddof=1) / np.sqrt(len(_pct_ch_e))) if len(_pct_ch_e) > 1 else 0.0
    else:
        _se_pct_ewr = 0.0
    _ewr_bl_m = _ee_wr(_tr_bl);  _ewr_po_m = _ee_wr(_tr_po)
    _pct_wr_ee = (_ewr_po_m / _ewr_bl_m - 1) * 100 if (_ewr_bl_m and _ewr_bl_m > 0) else 0.0

    # Occ-level scatter: Δ LTUER vs baseline LTUER at SOC broad level
    _occ_grp_cw_broad = _build_grp_cw('SOC_broad_adj')
    _ltuer_base_mc = _ltuer_mean(_occ_bl).rename(columns={'ltuer': 'ltuer_base'})
    _ltuer_pol_mc  = _ltuer_mean(_occ_po).rename(columns={'ltuer': 'ltuer_pol'})
    _occ_sc = (_ltuer_base_mc
               .merge(_ltuer_pol_mc, on='Occupation')
               .merge(STATIC_OCC_DF[['Occupation', 'target_demand_init']], on='Occupation', how='left')
               .merge(_occ_grp_cw_broad, on='Occupation', how='left')
               .dropna(subset=['OccGroup']))
    _occ_sc['ltuer_change'] = _occ_sc['ltuer_pol'] - _occ_sc['ltuer_base']
    _occ_sc_grp = (_occ_sc.groupby('OccGroup', as_index=False)
                   .apply(lambda g: pd.Series({
                       'ltuer_base':         np.average(g['ltuer_base'],   weights=g['target_demand_init'].clip(lower=1)),
                       'ltuer_change':       np.average(g['ltuer_change'], weights=g['target_demand_init'].clip(lower=1)),
                       'target_demand_init': g['target_demand_init'].sum(),
                   }), include_groups=False)
                   .reset_index(drop=True))

    # Δ hires rate ts
    _hr_bl_idx   = _hires_rate_ts(_grp_bl).set_index('DATE')
    _hr_po_idx   = _hires_rate_ts(_grp_po).set_index('DATE')
    _hr_cmn      = _hr_bl_idx.index.intersection(_hr_po_idx.index)
    _delta_hr_cm = pd.DataFrame({
        'DATE':     _hr_cmn,
        'delta_hr': (_hr_po_idx.loc[_hr_cmn, 'mean'] - _hr_bl_idx.loc[_hr_cmn, 'mean']).values * 100,
        'se_hr':    np.sqrt(_hr_bl_idx.loc[_hr_cmn, 'sem']**2 + _hr_po_idx.loc[_hr_cmn, 'sem']**2).values * 100,
    })

    # Δ JFR ts (replaces broken matching efficiency delta)
    _jfr_bl_idx  = _jfr_ts(_grp_bl).set_index('DATE')
    _jfr_po_idx  = _jfr_ts(_grp_po).set_index('DATE')
    _jfr_cmn     = _jfr_bl_idx.index.intersection(_jfr_po_idx.index)
    _delta_jfr_cm = pd.DataFrame({
        'DATE':      _jfr_cmn,
        'delta_jfr': (_jfr_po_idx.loc[_jfr_cmn, 'mean'] - _jfr_bl_idx.loc[_jfr_cmn, 'mean']).values,
        'se_jfr':    np.sqrt(_jfr_bl_idx.loc[_jfr_cmn, 'sem']**2 + _jfr_po_idx.loc[_jfr_cmn, 'sem']**2).values,
    })

    _cm_te[_mc] = {
        'occ_scatter_broad':  _occ_sc_grp,
        'delta_surv_all':     _surv_po_all  - _surv_bl_all,
        'se_surv_all':        np.sqrt(_se_po_all**2  + _se_bl_all**2),
        'delta_surv_ltu':     _surv_po_ltu  - _surv_bl_ltu,
        'se_surv_ltu':        np.sqrt(_se_po_ltu**2  + _se_bl_ltu**2),
        'delta_ltuer_ts':     _delta_ts,
        'delta_ltu_rate_ts':  _delta_rate,
        'delta_match_eff_ts': _delta_overall,
        'delta_dur_all':      _non_ltu_po['UEDuration'].mean() - _non_ltu_bl['UEDuration'].mean(),
        'se_dur_all':         _se_dur_all,
        'delta_ltu_dur':      _ltu_po['UEDuration'].mean() - _ltu_bl['UEDuration'].mean(),
        'se_ltu_dur':         _se_ltu_dur,
        'delta_ltu_share':    (_ue_po['LongtermUnemp'] == True).mean() * 100
                              - (_ue_bl['LongtermUnemp'] == True).mean() * 100,
        'se_ltu_share':       _se_ltu_share,
        'delta_wr_ltu':       _ltu_wr(_tr_po) - _ltu_wr(_tr_bl),
        'se_wr_ltu':          _se_wr_ltu,
        'pct_wr_ltu':         _pct_wr_ltu,
        'se_pct_wr_ltu':      _se_pct_wr,
        'pct_wr_ue_nonltu':   _pct_wr_ue_nonltu,
        'se_pct_wr_ue_nonltu': _se_pct_unwr,
        'pct_wr_ee':          _pct_wr_ee,
        'se_pct_wr_ee':       _se_pct_ewr,
        'delta_hr_ts':        _delta_hr_cm,
        'delta_jfr_ts':       _delta_jfr_cm,
    }
    del _bl_te, _po_te, _tr_bl, _grp_bl, _occ_bl, _tr_po, _grp_po, _occ_po

# Figure (4 panels: 2×2 grid)
from matplotlib.gridspec import GridSpec as _GS_CB
_fig_cb_w, _fig_cb_h = TEXTWIDTH, ROW_HEIGHT * 2
fig_cb = plt.figure(figsize=(_fig_cb_w, _fig_cb_h))
_gs_cb = _GS_CB(2, 2, figure=fig_cb, hspace=0.45, wspace=0.33)
ax_C  = fig_cb.add_subplot(_gs_cb[0, 0])   # A: Δ LTUER ts
ax_GF = fig_cb.add_subplot(_gs_cb[0, 1])   # B: Δ UE duration bar
ax_HR = fig_cb.add_subplot(_gs_cb[1, 0])   # C: Δ hires rate ts
ax_I  = fig_cb.add_subplot(_gs_cb[1, 1])   # D: occ scatter

_cb_x      = np.arange(len(_CM_SPECS_NO_CYC))
_cb_labels = [_CM_DISPLAY[c] for c in _CM_SPECS_NO_CYC]
_cb_colors = [MODEL_COLORS_BY_ID.get(c, '#aaaaaa') for c in _CM_SPECS_NO_CYC]
# Shared constants for grouped bar charts (duration + wage retention panels)
_wr_models = list(_CM_SPECS_NO_CYC)
_n_mod     = len(_wr_models)
_bw        = 0.22

def _cb_bar(ax, vals, ylabel, title, se=None):
    _yerr = [1.96 * s for s in se] if se is not None else None
    ax.bar(_cb_x, vals, 0.6, color=_cb_colors, alpha=0.85, edgecolor='white', linewidth=0.3,
           yerr=_yerr, error_kw=dict(ecolor='black', elinewidth=1.2, capsize=4, capthick=1.2))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(_cb_x); ax.set_xticklabels(_cb_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

# ── Panel A: Δ LTUER time series ───────────────────────────────────────────
for _mc in _CM_SPECS_NO_CYC:
    _col = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _lbl = _CM_DISPLAY[_mc].replace('\n', ' ')
    _ts = _cm_te[_mc]['delta_ltuer_ts']
    ax_C.plot(_ts['DATE'], _ts['delta_ltuer'], color=_col, linewidth=1.0, label=_lbl)
    ax_C.fill_between(_ts['DATE'],
                      _ts['delta_ltuer'] - 1.96*_ts['se_ltuer'],
                      _ts['delta_ltuer'] + 1.96*_ts['se_ltuer'],
                      alpha=0.12, color=_col)

for _, rec in recession_spans.iterrows():
    ax_C.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                 color='lightgrey', alpha=0.4, zorder=0)
ax_C.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax_C.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax_C.set_xlabel("Date")
ax_C.set_ylabel("Δ LTUER (pp)")
ax_C.set_title("Δ Aggregate LTUER", fontweight='normal')
ax_C.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
ax_C.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax_C.get_xticklabels(), rotation=45, ha='right')
ax_C.grid(axis='y', alpha=0.3)

# ── Panel B: Δ UE duration by worker type ─────────────────────────────────────
_dur_groups   = ['LTU Workers', 'Non-LTU Workers']
_dur_keys_val = ['delta_ltu_dur', 'delta_dur_all']
_dur_keys_se  = ['se_ltu_dur',    'se_dur_all']
_dur_grp_x    = np.arange(len(_dur_groups))
_dur_offsets  = np.linspace(-(_n_mod - 1) / 2 * _bw, (_n_mod - 1) / 2 * _bw, _n_mod)

for _mi, _mc in enumerate(_wr_models):
    _vals = [_cm_te[_mc][_k] for _k in _dur_keys_val]
    _ses  = [_cm_te[_mc][_k] for _k in _dur_keys_se]
    _col  = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _lbl  = _CM_DISPLAY[_mc].replace('\n', ' ')
    ax_GF.bar(_dur_grp_x + _dur_offsets[_mi], _vals, _bw,
              color=_col, alpha=0.85, edgecolor='white', linewidth=0.3,
              yerr=[1.96 * s for s in _ses],
              error_kw=dict(ecolor='black', elinewidth=1.2, capsize=4, capthick=1.2),
              label=_lbl)

ax_GF.axhline(0, color='black', linewidth=0.8)
ax_GF.set_xticks(_dur_grp_x)
ax_GF.set_xticklabels(_dur_groups)
ax_GF.set_ylabel("Δ Mean UE Duration (months)")
ax_GF.set_title("Δ Mean UE Duration by Worker Type",
                fontweight='normal')
ax_GF.grid(axis='y', alpha=0.3)

# ── Panel C: Δ hires rate time series ─────────────────────────────────────────
for _mc in _CM_SPECS_NO_CYC:
    _col = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _lbl = _CM_DISPLAY[_mc].replace('\n', ' ')
    _dh  = _cm_te[_mc]['delta_hr_ts']
    ax_HR.plot(_dh['DATE'], _dh['delta_hr'], color=_col, linewidth=1.0, label=_lbl)
    ax_HR.fill_between(_dh['DATE'],
                       _dh['delta_hr'] - 1.96*_dh['se_hr'],
                       _dh['delta_hr'] + 1.96*_dh['se_hr'],
                       alpha=0.12, color=_col)

for _, rec in recession_spans.iterrows():
    ax_HR.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                  color='lightgrey', alpha=0.4, zorder=0)
ax_HR.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax_HR.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax_HR.set_xlabel("Date")
ax_HR.set_ylabel("Δ Hires Rate (pp)")
ax_HR.set_title("Δ Hires Rate", fontweight='normal')
ax_HR.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
ax_HR.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax_HR.get_xticklabels(), rotation=45, ha='right')
ax_HR.grid(axis='y', alpha=0.3)

# ── Panel D: occupational heterogeneity scatter (SOC broad) ───────────────────
for _mc in _CM_SPECS_NO_CYC:
    _col = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _sc  = _cm_te[_mc]['occ_scatter_broad']
    _x   = _sc['ltuer_base'] * 100
    _y   = _sc['ltuer_change'] * 100
    _w   = _sc['target_demand_init'].clip(lower=1)
    ax_I.scatter(_x, _y, c=_col, s=np.sqrt(_w) * 4,
                 edgecolors='none', alpha=0.6)
    _mask = _x.notna() & _y.notna()
    if _mask.sum() >= 2:
        _fit = np.polyfit(_x[_mask], _y[_mask], 1, w=_w[_mask])
        _xline = np.linspace(_x.min(), _x.max(), 200)
        ax_I.plot(_xline, np.polyval(_fit, _xline), color=_col, linewidth=1.8)

ax_I.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax_I.set_xlabel("Baseline LTUER (%)")
ax_I.set_ylabel("Δ LTUER (pp)")
ax_I.set_title("Δ LTUER\nby Occupation",
               fontweight='normal')
ax_I.grid(alpha=0.3)

# ── Single figure-level legend ─────────────────────────────────────────────────
_cb_legend_colors = {_CM_DISPLAY[c].replace('\n', ' '): MODEL_COLORS_BY_ID.get(c, '#aaaaaa')
                      for c in _CM_SPECS_NO_CYC}

fig_cb.suptitle(
    "Preferential Hiring Policy: Long-Term Unemployed",
    fontweight='bold'
)
plt.tight_layout()
add_suptitle_spacing(fig_cb)
add_shared_legend(fig_cb, _cb_legend_colors, ncol=len(_CM_SPECS_NO_CYC))

_fname_cb = "fig_cross_model_combined.png"
save_plot_standard(fig_cb, os.path.join(OUTPUT_PATH, _fname_cb), aspect_ratio=_fig_cb_h / _fig_cb_w)
save_plot_standard(fig_cb, os.path.join(OUTPUT_PATH, _fname_cb.replace('.png', '.pdf')), aspect_ratio=_fig_cb_h / _fig_cb_w)
if SAVE_FIG5_TO_DROPBOX:
    os.makedirs(DROPBOX_FIG5_PATH, exist_ok=True)
    save_plot_standard(fig_cb, os.path.join(DROPBOX_FIG5_PATH, _fname_cb.replace('.png', '.pdf')), aspect_ratio=_fig_cb_h / _fig_cb_w)
    print(f"✓ {_fname_cb.replace('.png', '.pdf')} saved to Dropbox.")
plt.show()
print(f"Saved: {_fname_cb} (PNG + PDF)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5E: Cross-model combined figure — entry bias
# Same 3×3 layout as fig_cross_model_combined; worker groups are entry-level
# unemployed / non-entry unemployed / employed movers.
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 5E: Cross-model combined figure — entry bias ===")

_eb_entry_ids = set(STATIC_OCC_DF[STATIC_OCC_DF['entry_level']]['Occupation'])

CROSS_ENTRY_PKL         = os.path.join(OUTPUT_PATH, "cross_model_entry_bias.pkl")
FORCE_CROSS_ENTRY_RERUN = False

if os.path.exists(CROSS_ENTRY_PKL) and not FORCE_CROSS_ENTRY_RERUN:
    print(f"Loading cross-model entry bias cache from {CROSS_ENTRY_PKL} ...")
    with open(CROSS_ENTRY_PKL, 'rb') as _f:
        _eb_policy_runs = pickle.load(_f)
    print(f"  Loaded: {list(_eb_policy_runs.keys())}")
else:
    _eb_policy_runs = {}
    for _mc in _CM_SPECS_NO_CYC:
        print(f"\n  [{_mc}] running entry bias ×2.0 (N={N_POLICY_SIMS}) ...")
        _bp = _build_model_params(_mc)
        _eb_policy_runs[f"{_mc}_entry_bias"] = run_policy_scenario(
            f"{_mc}_entry_bias", {'entry_hire_bias': 1.0}, _bp, CALIB_DATE, N_POLICY_SIMS
        )
        print(f"    ✓ {_mc} done")
    print(f"\nSaving to {CROSS_ENTRY_PKL} ...")
    with open(CROSS_ENTRY_PKL, 'wb') as _f:
        pickle.dump(_eb_policy_runs, _f)
    print("  Saved.")
# 

def _cross_entry_topup_pkl(mc):
    return os.path.join(OUTPUT_PATH, f"cross_model_{mc}_entry_topup.pkl")


_all_entry_topup_exist = all(
    os.path.exists(_cross_entry_topup_pkl(_mc)) for _mc in _CM_SPECS_NO_CYC
)
if _all_entry_topup_exist and not FORCE_CROSS_ENTRY_RERUN:
    print("Per-model entry-bias top-up pkls found — will load on demand.")
else:
    for _mc in _CM_SPECS_NO_CYC:
        _epkl_path = _cross_entry_topup_pkl(_mc)
        if os.path.exists(_epkl_path) and not FORCE_CROSS_ENTRY_RERUN:
            print(f"  [{_mc}] entry topup pkl already exists, skipping.")
            continue
        print(f"\n  [{_mc}] running {N_TOPUP_SIMS} entry-bias top-up sims ...")
        _bp = _build_model_params(_mc)
        _mc_etopup = {
            f"{_mc}_entry_bias": run_policy_scenario(
                f"{_mc}_entry_bias_topup", {'entry_hire_bias': 1.0}, _bp, CALIB_DATE,
                n_sims=N_TOPUP_SIMS, sim_id_offset=N_EXISTING_SIMS
            )
        }
        with open(_epkl_path, 'wb') as _f:
            pickle.dump(_mc_etopup, _f)
        print(f"  Saved: {_epkl_path}")
        del _mc_etopup


def _load_cm_entry(mc):
    """Baseline from cached_data pkl (N25); entry_bias from CROSS_ENTRY_PKL (5 sims) +
    per-model entry topup pkl (20 sims). Returns (bl_tuple, po_tuple)."""
    _display = _CM_ID_TO_DISPLAY[mc]
    _grp_raw = (cached_data['model_results_all_sims'].get(mc) or
                cached_data['model_results_all_sims'].get(_display))
    if isinstance(_grp_raw, list):
        _parts = []
        for _i, _df in enumerate(_grp_raw):
            _df = _df.copy()
            _df['sim_id'] = _i
            if 'DATE' not in _df.columns:
                _df['DATE'] = pd.date_range(
                    start=CALIB_DATE[0], end=CALIB_DATE[1], periods=len(_df))
            _parts.append(_df)
        _grp_bl = pd.concat(_parts, ignore_index=True)
    else:
        _grp_bl = _grp_raw
    _occ_bl = (cached_data.get('filtered_sim_results_all_sims') or {}).get(_display)
    if _occ_bl is None:
        _occ_bl = cached_data['filtered_sim_results'][_display]
    _bl = (cached_data['filtered_transition_dfs'][_display], _grp_bl, _occ_bl)
    _po_ex   = _eb_policy_runs.get(f"{mc}_entry_bias")
    _epkl    = _cross_entry_topup_pkl(mc)
    if os.path.exists(_epkl):
        with open(_epkl, 'rb') as _f:
            _etopup = pickle.load(_f)
        _po = (_concat_results(_po_ex, _etopup[f"{mc}_entry_bias"])
               if _po_ex is not None else _etopup[f"{mc}_entry_bias"])
        del _etopup
    else:
        _po = _po_ex
    return _bl, _po


# ── Helper: entry hire share time series (fraction of UE hires from entry-level origin)
def _entry_ltuer_ts(trans_df, grp_df):
    """Fraction of UE hires from entry-level occupations who were LTU, per period (%)."""
    ts_map = grp_df.drop_duplicates('Time Step').set_index('Time Step')['DATE']
    ue = trans_df[(trans_df['TransitionType'] == 'UE') &
                  (trans_df['OriginOccupation'].isin(_eb_entry_ids))].copy()
    ue['is_ltu'] = (ue['LongtermUnemp'] == True).astype(float) * 100
    if 'sim_id' in ue.columns and ue['sim_id'].nunique() > 1:
        grp = ue.groupby(['Time Step', 'sim_id'])['is_ltu'].mean().reset_index()
        grp['DATE'] = grp['Time Step'].map(ts_map)
        agg = grp.groupby('DATE')['is_ltu'].agg(
            mean=lambda x: x.mean(),
            sem=lambda x: x.std(ddof=1) / np.sqrt(len(x))
        ).reset_index()
    else:
        agg = ue.groupby('Time Step')['is_ltu'].mean().reset_index()
        agg['DATE'] = agg['Time Step'].map(ts_map)
        agg = agg.rename(columns={'is_ltu': 'mean'})
        agg['sem'] = 0.0
    return agg

def _entry_hire_share_ts(trans_df, grp_df):
    ts_map = grp_df.drop_duplicates('Time Step').set_index('Time Step')['DATE']
    ue = trans_df[trans_df['TransitionType'] == 'UE'].copy()
    ue['is_entry_orig'] = ue['OriginOccupation'].isin(_eb_entry_ids).astype(float)
    if 'sim_id' in ue.columns and ue['sim_id'].nunique() > 1:
        grp = ue.groupby(['Time Step', 'sim_id'])['is_entry_orig'].mean().reset_index()
        grp['DATE'] = grp['Time Step'].map(ts_map)
        agg = grp.groupby('DATE')['is_entry_orig'].agg(
            mean=lambda x: x.mean(),
            sem=lambda x: x.std(ddof=1) / np.sqrt(len(x))
        ).reset_index()
    else:
        agg = ue.groupby('Time Step')['is_entry_orig'].mean().reset_index()
        agg['DATE'] = agg['Time Step'].map(ts_map)
        agg = agg.rename(columns={'is_entry_orig': 'mean'})
        agg['sem'] = 0.0
    return agg.set_index('DATE')


# ── Helpers: wage ratio by entry-level status (AcceptedWage / PreSepWage) ────
def _ue_entry_wr(tr):
    _t = tr[(tr['TransitionType'] == 'UE') &
            tr['OriginOccupation'].isin(_eb_entry_ids)].copy()
    _t = _t[_t['PreSepWage'] > 0]
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    return _t['wr'].mean() if len(_t) > 0 else np.nan


def _ue_entry_wr_per_sim(tr):
    _t = tr[(tr['TransitionType'] == 'UE') &
            tr['OriginOccupation'].isin(_eb_entry_ids)].copy()
    _t = _t[_t['PreSepWage'] > 0]
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    if 'sim_id' not in _t.columns or _t['sim_id'].nunique() <= 1:
        return None
    return _t.groupby('sim_id')['wr'].mean()


def _ue_nonentry_wr(tr):
    _t = tr[(tr['TransitionType'] == 'UE') &
            ~tr['OriginOccupation'].isin(_eb_entry_ids)].copy()
    _t = _t[_t['PreSepWage'] > 0]
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    return _t['wr'].mean() if len(_t) > 0 else np.nan


def _ue_nonentry_wr_per_sim(tr):
    _t = tr[(tr['TransitionType'] == 'UE') &
            ~tr['OriginOccupation'].isin(_eb_entry_ids)].copy()
    _t = _t[_t['PreSepWage'] > 0]
    _t['wr'] = _t['AcceptedWage'] / _t['PreSepWage']
    if 'sim_id' not in _t.columns or _t['sim_id'].nunique() <= 1:
        return None
    return _t.groupby('sim_id')['wr'].mean()


# ── Treatment-effect metrics ──────────────────────────────────────────────────
_eb_metrics = {}
_eb_te      = {}

for _mc in _CM_SPECS_NO_CYC:
    _bl_eb, _po_eb = _load_cm_entry(_mc)
    _tr_bl_eb, _grp_bl_eb, _occ_bl_eb = _bl_eb
    _tr_po_eb, _grp_po_eb, _occ_po_eb = _po_eb

    _ue_bl_eb = _tr_bl_eb[(_tr_bl_eb['TransitionType'] == 'UE') &
                           _tr_bl_eb['UEDuration'].between(1, 36)]
    _ue_po_eb = _tr_po_eb[(_tr_po_eb['TransitionType'] == 'UE') &
                           _tr_po_eb['UEDuration'].between(1, 36)]

    # Split by OriginOccupation: entry-level vs non-entry
    _entry_bl_eb    = _ue_bl_eb[_ue_bl_eb['OriginOccupation'].isin(_eb_entry_ids)]
    _entry_po_eb    = _ue_po_eb[_ue_po_eb['OriginOccupation'].isin(_eb_entry_ids)]
    _nonentry_bl_eb = _ue_bl_eb[~_ue_bl_eb['OriginOccupation'].isin(_eb_entry_ids)]
    _nonentry_po_eb = _ue_po_eb[~_ue_po_eb['OriginOccupation'].isin(_eb_entry_ids)]

    # Survival curves
    _surv_bl_entry, _se_bl_entry = _surv_ci(_entry_bl_eb,    _x_range)
    _surv_po_entry, _se_po_entry = _surv_ci(_entry_po_eb,    _x_range)
    _surv_bl_non,   _se_bl_non   = _surv_ci(_nonentry_bl_eb, _x_range)
    _surv_po_non,   _se_po_non   = _surv_ci(_nonentry_po_eb, _x_range)

    # Entry hire share time series
    _ehire_ts_bl  = _entry_hire_share_ts(_tr_bl_eb, _grp_bl_eb)
    _ehire_ts_po  = _entry_hire_share_ts(_tr_po_eb, _grp_po_eb)
    _ehire_common = _ehire_ts_bl.index.intersection(_ehire_ts_po.index)
    _delta_ehire_ts = pd.DataFrame({
        'DATE':        _ehire_common,
        'delta_ehire': (_ehire_ts_po.loc[_ehire_common, 'mean']
                        - _ehire_ts_bl.loc[_ehire_common, 'mean']).values,
        'se_ehire':    np.sqrt(_ehire_ts_po.loc[_ehire_common, 'sem'].values**2
                               + _ehire_ts_bl.loc[_ehire_common, 'sem'].values**2),
    })

    # Entry-level LTUER time series (absolute, for panel C)
    _ltuer_entry_bl_ts = _entry_ltuer_ts(_tr_bl_eb, _grp_bl_eb)
    _ltuer_entry_po_ts = _entry_ltuer_ts(_tr_po_eb, _grp_po_eb)

    # Overall matching efficiency
    _, _, _overall_bl_eb = _ltu_stock_and_rate(_tr_bl_eb, _grp_bl_eb, _occ_bl_eb)
    _, _, _overall_po_eb = _ltu_stock_and_rate(_tr_po_eb, _grp_po_eb, _occ_po_eb)
    _oc_eb = _overall_bl_eb.set_index('DATE').index.intersection(
              _overall_po_eb.set_index('DATE').index)
    _delta_overall_eb = pd.DataFrame({
        'DATE':            _oc_eb,
        'delta_match_eff': (_overall_po_eb.set_index('DATE').loc[_oc_eb, 'mean']
                            - _overall_bl_eb.set_index('DATE').loc[_oc_eb, 'mean']).values,
        'se_match_eff':    np.sqrt(
            _overall_po_eb.set_index('DATE').loc[_oc_eb, 'sem'].values**2
            + _overall_bl_eb.set_index('DATE').loc[_oc_eb, 'sem'].values**2),
    })

    # JFR delta time series (replaces broken ME delta in panel E)
    _jfr_bl_idx  = _jfr_ts(_grp_bl_eb).set_index('DATE')
    _jfr_po_idx  = _jfr_ts(_grp_po_eb).set_index('DATE')
    _jfr_common  = _jfr_bl_idx.index.intersection(_jfr_po_idx.index)
    _delta_jfr_eb = pd.DataFrame({
        'DATE':      _jfr_common,
        'delta_jfr': (_jfr_po_idx.loc[_jfr_common, 'mean'] - _jfr_bl_idx.loc[_jfr_common, 'mean']).values,
        'se_jfr':    np.sqrt(_jfr_bl_idx.loc[_jfr_common, 'sem']**2
                             + _jfr_po_idx.loc[_jfr_common, 'sem']**2).values,
    })

    # Matching efficiency scalar
    _eff_bl_eb   = _cm_mean_match_rate(_occ_bl_eb)
    _eff_po_eb   = _cm_mean_match_rate(_occ_po_eb)
    _eff_imp_eb  = (_eff_po_eb / _eff_bl_eb - 1) * 100 if _eff_bl_eb > 0 else 0.0
    _eff_bl_s_eb = _match_rate_per_sim(_occ_bl_eb)
    _eff_po_s_eb = _match_rate_per_sim(_occ_po_eb)
    if _eff_bl_s_eb is not None and _eff_po_s_eb is not None:
        _cs_eff = _eff_bl_s_eb.index.intersection(_eff_po_s_eb.index)
        _is_eff = (_eff_po_s_eb[_cs_eff] / _eff_bl_s_eb[_cs_eff] - 1) * 100
        _se_eff_eb = float(_is_eff.std(ddof=1) / np.sqrt(len(_is_eff))) if len(_is_eff) > 1 else 0.0
    else:
        _se_eff_eb = 0.0
    _eb_metrics[_mc] = {'eff_imp_pct': _eff_imp_eb, 'se_eff_imp': _se_eff_eb}

    # Per-sim SEs
    _has_sim_eb = 'sim_id' in _ue_bl_eb.columns and _ue_bl_eb['sim_id'].nunique() > 1
    if _has_sim_eb:
        _ns_eb = _ue_bl_eb['sim_id'].nunique()

        def _eb_se_d(gb, ga, col):
            se_b = gb.groupby('sim_id')[col].mean().std(ddof=1) / np.sqrt(_ns_eb)
            se_a = ga.groupby('sim_id')[col].mean().std(ddof=1) / np.sqrt(_ns_eb)
            return float(np.sqrt(se_b**2 + se_a**2))

        _ue_bl_ehs = _ue_bl_eb.copy()
        _ue_bl_ehs['is_entry'] = _ue_bl_ehs['OriginOccupation'].isin(_eb_entry_ids).astype(float) * 100
        _ue_po_ehs = _ue_po_eb.copy()
        _ue_po_ehs['is_entry'] = _ue_po_ehs['OriginOccupation'].isin(_eb_entry_ids).astype(float) * 100
        _se_entry_share  = _eb_se_d(_ue_bl_ehs, _ue_po_ehs, 'is_entry')
        _se_dur_entry    = _eb_se_d(_entry_bl_eb,    _entry_po_eb,    'UEDuration')
        _se_dur_nonentry = _eb_se_d(_nonentry_bl_eb, _nonentry_po_eb, 'UEDuration')
        _ue_bl_ltu_el = _entry_bl_eb.copy()
        _ue_bl_ltu_el['is_ltu'] = (_ue_bl_ltu_el['LongtermUnemp'] == True).astype(float) * 100
        _ue_po_ltu_el = _entry_po_eb.copy()
        _ue_po_ltu_el['is_ltu'] = (_ue_po_ltu_el['LongtermUnemp'] == True).astype(float) * 100
        _se_ltuer_entry = _eb_se_d(_ue_bl_ltu_el, _ue_po_ltu_el, 'is_ltu')

        def _se_pct_wr_eb(bl_s, po_s):
            if bl_s is None or po_s is None:
                return 0.0
            cs  = bl_s.index.intersection(po_s.index)
            pct = (po_s[cs] / bl_s[cs] - 1) * 100
            return float(pct.std(ddof=1) / np.sqrt(len(pct))) if len(pct) > 1 else 0.0

        _se_pct_wr_entry    = _se_pct_wr_eb(_ue_entry_wr_per_sim(_tr_bl_eb),
                                             _ue_entry_wr_per_sim(_tr_po_eb))
        _se_pct_wr_nonentry = _se_pct_wr_eb(_ue_nonentry_wr_per_sim(_tr_bl_eb),
                                             _ue_nonentry_wr_per_sim(_tr_po_eb))
        _se_pct_wr_ee_eb    = _se_pct_wr_eb(_ee_wr_per_sim(_tr_bl_eb),
                                             _ee_wr_per_sim(_tr_po_eb))
    else:
        _se_entry_share = _se_dur_entry = _se_dur_nonentry = _se_ltuer_entry = 0.0
        _se_pct_wr_entry = _se_pct_wr_nonentry = _se_pct_wr_ee_eb = 0.0

    # Scalar entry hire share
    _ehs_bl = _ue_bl_eb['OriginOccupation'].isin(_eb_entry_ids).mean() * 100
    _ehs_po = _ue_po_eb['OriginOccupation'].isin(_eb_entry_ids).mean() * 100

    # % Δ wage retention
    _wr_entry_bl    = _ue_entry_wr(_tr_bl_eb);    _wr_entry_po    = _ue_entry_wr(_tr_po_eb)
    _wr_nonentry_bl = _ue_nonentry_wr(_tr_bl_eb); _wr_nonentry_po = _ue_nonentry_wr(_tr_po_eb)
    _wr_ee_bl_eb    = _ee_wr(_tr_bl_eb);          _wr_ee_po_eb    = _ee_wr(_tr_po_eb)
    _pct_wr_entry    = (_wr_entry_po    / _wr_entry_bl    - 1)*100 if _wr_entry_bl    and _wr_entry_bl    > 0 else 0.0
    _pct_wr_nonentry = (_wr_nonentry_po / _wr_nonentry_bl - 1)*100 if _wr_nonentry_bl and _wr_nonentry_bl > 0 else 0.0
    _pct_wr_ee_eb    = (_wr_ee_po_eb    / _wr_ee_bl_eb    - 1)*100 if _wr_ee_bl_eb    and _wr_ee_bl_eb    > 0 else 0.0

    # Occ-level scatter (Δ LTUER vs baseline LTUER at SOC broad level)
    _occ_grp_cw_eb = _build_grp_cw('SOC_broad_adj')
    _ltuer_base_eb = _ltuer_mean(_occ_bl_eb).rename(columns={'ltuer': 'ltuer_base'})
    _ltuer_pol_eb  = _ltuer_mean(_occ_po_eb).rename(columns={'ltuer': 'ltuer_pol'})
    _occ_sc_eb = (_ltuer_base_eb
                  .merge(_ltuer_pol_eb, on='Occupation')
                  .merge(STATIC_OCC_DF[['Occupation', 'target_demand_init']], on='Occupation', how='left')
                  .merge(_occ_grp_cw_eb, on='Occupation', how='left')
                  .dropna(subset=['OccGroup']))
    _occ_sc_eb['ltuer_change'] = _occ_sc_eb['ltuer_pol'] - _occ_sc_eb['ltuer_base']
    _occ_sc_grp_eb = (_occ_sc_eb.groupby('OccGroup', as_index=False)
                      .apply(lambda g: pd.Series({
                          'ltuer_base':         np.average(g['ltuer_base'],
                                                           weights=g['target_demand_init'].clip(lower=1)),
                          'ltuer_change':       np.average(g['ltuer_change'],
                                                           weights=g['target_demand_init'].clip(lower=1)),
                          'target_demand_init': g['target_demand_init'].sum(),
                      }), include_groups=False)
                      .reset_index(drop=True))

    _eb_te[_mc] = {
        'delta_surv_entry':     _surv_po_entry - _surv_bl_entry,
        'se_surv_entry':        np.sqrt(_se_po_entry**2 + _se_bl_entry**2),
        'delta_surv_nonentry':  _surv_po_non   - _surv_bl_non,
        'se_surv_nonentry':     np.sqrt(_se_po_non**2 + _se_bl_non**2),
        'delta_ehire_ts':       _delta_ehire_ts,
        'delta_match_eff_ts':   _delta_overall_eb,
        'delta_jfr_ts':         _delta_jfr_eb,
        'delta_dur_entry':      _entry_po_eb['UEDuration'].mean() - _entry_bl_eb['UEDuration'].mean(),
        'se_dur_entry':         _se_dur_entry,
        'delta_dur_nonentry':   _nonentry_po_eb['UEDuration'].mean() - _nonentry_bl_eb['UEDuration'].mean(),
        'se_dur_nonentry':      _se_dur_nonentry,
        'delta_entry_share':    _ehs_po - _ehs_bl,
        'se_entry_share':       _se_entry_share,
        'delta_ltuer_entry':    (_entry_po_eb['LongtermUnemp'] == True).mean() * 100
                                - (_entry_bl_eb['LongtermUnemp'] == True).mean() * 100,
        'se_ltuer_entry':       _se_ltuer_entry,
        'pct_wr_entry':         _pct_wr_entry,
        'se_pct_wr_entry':      _se_pct_wr_entry,
        'pct_wr_nonentry':      _pct_wr_nonentry,
        'se_pct_wr_nonentry':   _se_pct_wr_nonentry,
        'pct_wr_ee':            _pct_wr_ee_eb,
        'se_pct_wr_ee':         _se_pct_wr_ee_eb,
        'occ_scatter_broad':    _occ_sc_grp_eb,
        'hires_rate_bl_ts':     _hires_rate_ts(_grp_bl_eb),
        'hires_rate_po_ts':     _hires_rate_ts(_grp_po_eb),
        'ltuer_entry_bl_ts':    _ltuer_entry_bl_ts,
        'ltuer_entry_po_ts':    _ltuer_entry_po_ts,
        'newentrant_bl_ts':     _newentrant_stock_and_rate(_tr_bl_eb, _occ_bl_eb)[0],
        'newentrant_po_ts':     _newentrant_stock_and_rate(_tr_po_eb, _occ_po_eb)[0],
        'overall_bl_ts':        _overall_bl_eb,
        'overall_po_ts':        _overall_po_eb,
        'jfr_bl_ts':            _jfr_ts(_grp_bl_eb),
        'jfr_po_ts':            _jfr_ts(_grp_po_eb),
    }
    del _bl_eb, _po_eb, _tr_bl_eb, _grp_bl_eb, _occ_bl_eb, _tr_po_eb, _grp_po_eb, _occ_po_eb

# ── Hires rate and job-finding rate: baseline vs policy ──────────────────────
# Note: ME computed via _ltu_stock_and_rate (hires/total_seekers from occ_df) is
# mis-scaled because Unemployed/Employed Seekers in record_df are economy-wide
# totals stored in every occupation row; summing across occupations inflates the
# denominator by N_occ. JFR = UE_Transitions/Unemployment avoids this.
print("\n  Hires rate and job-finding rate — baseline vs policy (entry bias ×2.0)")
print(f"  {'Model':<35} {'HR base':>8} {'HR pol':>8} {'JFR base':>9} {'JFR pol':>9}")
print("  " + "─" * 73)
for _mc in _CM_SPECS_NO_CYC:
    _lbl    = _CM_DISPLAY[_mc].replace('\n', ' ')
    _hr_bl  = _eb_te[_mc]['hires_rate_bl_ts']['mean'].mean() * 100
    _hr_po  = _eb_te[_mc]['hires_rate_po_ts']['mean'].mean() * 100
    _jfr_bl = _eb_te[_mc]['jfr_bl_ts']['mean'].mean()
    _jfr_po = _eb_te[_mc]['jfr_po_ts']['mean'].mean()
    print(f"  {_lbl:<35} {_hr_bl:>7.2f}%  {_hr_po:>7.2f}%  {_jfr_bl:>8.2f}%  {_jfr_po:>8.2f}%")
print(f"  (HR = hires/employment [JOLTS-comparable]; JFR = UE transitions/unemployed)")


# ── Figure (2 rows: A full-width on top, B+C below) ──────────────────────────
from matplotlib.gridspec import GridSpec as _GS_EB
_fig_eb_w, _fig_eb_h = TEXTWIDTH, ROW_HEIGHT * 2
fig_eb  = plt.figure(figsize=(_fig_eb_w, _fig_eb_h))
_gs_eb  = _GS_EB(2, 2, figure=fig_eb, hspace=0.45, wspace=0.33)
ax_eb_E  = fig_eb.add_subplot(_gs_eb[0, :])   # A: Δ LTUER entry-level (full width)
ax_eb_D  = fig_eb.add_subplot(_gs_eb[1, 0])   # B: UE duration by worker type
ax_eb_G  = fig_eb.add_subplot(_gs_eb[1, 1])   # C: Δ hires rate ts

_n_mod_eb = len(_CM_SPECS_NO_CYC)
_bw_eb    = 0.7 / _n_mod_eb
_x_eb     = np.arange(_n_mod_eb)
_cols_eb  = [MODEL_COLORS_BY_ID.get(c, '#aaaaaa') for c in _CM_SPECS_NO_CYC]


def _cb_bar_eb(ax, vals, ylabel, title, se=None):
    _yerr = [1.96 * s for s in se] if se is not None else None
    ax.bar(_x_eb, vals, 0.6, color=_cols_eb, alpha=0.85, edgecolor='white', linewidth=0.3,
           yerr=_yerr, error_kw=dict(ecolor='black', elinewidth=1.2, capsize=4, capthick=1.2))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(_x_eb)
    ax.set_xticklabels([_CM_DISPLAY[c].replace('\n', ' ') for c in _CM_SPECS_NO_CYC])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)


# Panel B — grouped bar: UE duration by worker type
_dur_groups_eb  = ['Entry-Level\nUnemployed', 'Non-Entry\nUnemployed']
_dur_keys_val_eb = ['delta_dur_entry', 'delta_dur_nonentry']
_dur_keys_se_eb  = ['se_dur_entry',    'se_dur_nonentry']
_dur_grp_x_eb   = np.arange(len(_dur_groups_eb))
_dur_off_eb     = np.linspace(-(_n_mod_eb-1)/2*_bw_eb, (_n_mod_eb-1)/2*_bw_eb, _n_mod_eb)
for _mi, _mc in enumerate(_CM_SPECS_NO_CYC):
    _col  = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _vals = [_eb_te[_mc][_k] for _k in _dur_keys_val_eb]
    _ses  = [_eb_te[_mc][_k] for _k in _dur_keys_se_eb]
    ax_eb_D.bar(_dur_grp_x_eb + _dur_off_eb[_mi], _vals, _bw_eb,
                color=_col, alpha=0.85, edgecolor='white', linewidth=0.3,
                yerr=[1.96*s for s in _ses],
                error_kw=dict(ecolor='black', elinewidth=1.2, capsize=3, capthick=1.2))
ax_eb_D.axhline(0, color='black', linewidth=0.8)
ax_eb_D.set_xticks(_dur_grp_x_eb)
ax_eb_D.set_xticklabels(_dur_groups_eb, ha='center')
ax_eb_D.set_ylabel("Δ Mean UE Duration (months)")
ax_eb_D.set_title("Δ Mean UE Duration by Worker Type",
                  fontweight='normal')
ax_eb_D.grid(axis='y', alpha=0.3)

# Panel A — Δ entry-level LTUER time series (policy − baseline)
for _mc in _CM_SPECS_NO_CYC:
    _col      = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _lbl      = _CM_DISPLAY[_mc].replace('\n', ' ')
    _lt_bl    = _eb_te[_mc]['ltuer_entry_bl_ts'].set_index('DATE')
    _lt_po    = _eb_te[_mc]['ltuer_entry_po_ts'].set_index('DATE')
    _dates_lt = _lt_bl.index.intersection(_lt_po.index)
    _delta_lt = _lt_po.loc[_dates_lt, 'mean'] - _lt_bl.loc[_dates_lt, 'mean']
    _se_lt    = np.sqrt(_lt_bl.loc[_dates_lt, 'sem']**2 + _lt_po.loc[_dates_lt, 'sem']**2)
    ax_eb_E.plot(_dates_lt, _delta_lt, color=_col, linewidth=1.0, label=_lbl)
    ax_eb_E.fill_between(_dates_lt, _delta_lt - 1.96 * _se_lt, _delta_lt + 1.96 * _se_lt,
                         alpha=0.12, color=_col)
for _, rec in recession_spans.iterrows():
    ax_eb_E.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                    color='lightgrey', alpha=0.4, zorder=0)
ax_eb_E.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax_eb_E.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax_eb_E.set_xlabel("Date")
ax_eb_E.set_ylabel("Δ LTUER (pp)")
ax_eb_E.set_title("Δ LTUER: Entry-Level Unemployed",
                  fontweight='normal')
ax_eb_E.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
ax_eb_E.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax_eb_E.get_xticklabels(), rotation=45, ha='right')
ax_eb_E.grid(axis='y', alpha=0.3)

# Panel C — Δ hires rate (policy − baseline, pp)
for _mc in _CM_SPECS_NO_CYC:
    _col   = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _hr_bl = _eb_te[_mc]['hires_rate_bl_ts'].set_index('DATE')
    _hr_po = _eb_te[_mc]['hires_rate_po_ts'].set_index('DATE')
    _dates = _hr_bl.index.intersection(_hr_po.index)
    _delta_hr  = (_hr_po.loc[_dates, 'mean'] - _hr_bl.loc[_dates, 'mean']) * 100
    _se_hr     = np.sqrt(_hr_bl.loc[_dates, 'sem']**2 + _hr_po.loc[_dates, 'sem']**2) * 100
    ax_eb_G.plot(_dates, _delta_hr, color=_col, linewidth=0.8)
    ax_eb_G.fill_between(_dates, _delta_hr - 1.96 * _se_hr, _delta_hr + 1.96 * _se_hr,
                         alpha=0.12, color=_col)
for _, rec in recession_spans.iterrows():
    ax_eb_G.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                    color='lightgrey', alpha=0.4, zorder=0)
ax_eb_G.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax_eb_G.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax_eb_G.set_xlabel("Date")
ax_eb_G.set_ylabel("Δ Hires Rate (pp)")
ax_eb_G.set_title("Δ Hires Rate",
                  fontweight='normal')
ax_eb_G.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
ax_eb_G.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax_eb_G.get_xticklabels(), rotation=45, ha='right')
ax_eb_G.grid(axis='y', alpha=0.3)

# Shared legend + save
_eb_legend_colors = {_CM_DISPLAY[c].replace('\n', ' '): MODEL_COLORS_BY_ID.get(c, '#aaaaaa')
                      for c in _CM_SPECS_NO_CYC}
fig_eb.suptitle(
    "Preferential Hiring Policy: Entry-Level Workers",
    fontweight='bold'
)
plt.tight_layout()
add_suptitle_spacing(fig_eb)
add_shared_legend(fig_eb, _eb_legend_colors, ncol=len(_CM_SPECS_NO_CYC))

_fname_eb = "fig_cross_model_combined_entry_bias.png"
save_plot_standard(fig_eb, os.path.join(OUTPUT_PATH, _fname_eb), aspect_ratio=_fig_eb_h / _fig_eb_w)
save_plot_standard(fig_eb, os.path.join(OUTPUT_PATH, _fname_eb.replace('.png', '.pdf')), aspect_ratio=_fig_eb_h / _fig_eb_w)
if SAVE_FIG5_TO_DROPBOX:
    os.makedirs(DROPBOX_FIG5_PATH, exist_ok=True)
    save_plot_standard(fig_eb, os.path.join(DROPBOX_FIG5_PATH, _fname_eb.replace('.png', '.pdf')), aspect_ratio=_fig_eb_h / _fig_eb_w)
    print(f"✓ {_fname_eb.replace('.png', '.pdf')} saved to Dropbox.")
plt.show()
print(f"Saved: {_fname_eb} (PNG + PDF)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5D: Re-employment wage by unemployment duration — cross-model
# Panel A: baseline trend lines per model (structural wage scarring comparison)
# Panel B: Δ trend (policy − baseline) per model (treatment effect on scarring)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 5D: Re-employment wage by duration — cross-model ===")

_WD_EDGES    = [1, 4, 7, 10, 13, 19, 25, 37]   # right-exclusive
_WD_MIDPTS   = [2, 5, 8, 11, 15.5, 21.5, 30.5]
_WD_LABELS   = ['1–3', '4–6', '7–9', '10–12', '13–18', '19–24', '25–36']

def _wage_trend(trans):
    """Mean re-employment wage ratio grouped into duration bins."""
    ue = (trans[(trans['TransitionType'] == 'UE') & trans['UEDuration'].between(1, 36)]
          .copy()
          .merge(occ_wage_info[['Occupation', 'occ_median_wage']],
                 left_on='DestinationOccupation', right_on='Occupation', how='left'))
    ue['wr'] = ue['AcceptedWage'] / ue['occ_median_wage']
    ue['bin'] = pd.cut(ue['UEDuration'], bins=_WD_EDGES,
                       labels=_WD_MIDPTS, right=False).astype(float)
    return ue.groupby('bin')['wr'].mean()

fig_wd, (ax_wd_A, ax_wd_B) = plt.subplots(1, 2, figsize=(14, 6))

for _mc in _CM_SPECS_NO_CYC:
    _col      = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _lbl      = _CM_DISPLAY[_mc].replace('\n', ' ')
    _bl_wd, _po_wd = _load_cm_model(_mc)
    _tr_bl    = _bl_wd[0]
    _tr_po    = _po_wd[0]
    _trend_bl = _wage_trend(_tr_bl)
    _trend_po = _wage_trend(_tr_po)

    ax_wd_A.plot(_trend_bl.index, _trend_bl.values,
                 color=_col, linewidth=2, marker='o', markersize=4, label=_lbl)

    _common = _trend_bl.index.intersection(_trend_po.index)
    ax_wd_B.plot(_common, (_trend_po[_common] - _trend_bl[_common]).values,
                 color=_col, linewidth=2, marker='o', markersize=4, label=_lbl)
    del _bl_wd, _po_wd, _tr_bl, _tr_po

# Shared x-axis tick formatting
for _ax in (ax_wd_A, ax_wd_B):
    _ax.set_xticks(_WD_MIDPTS)
    _ax.set_xticklabels(_WD_LABELS, fontsize=9)
    _ax.set_xlim(0, 33)

# Panel A formatting
ax_wd_A.axhline(1.0, color='grey', linewidth=1, linestyle='--', alpha=0.7)
ax_wd_A.axvline(5, color='darkred', linewidth=1.2, linestyle=':', alpha=0.8,
                label='LTU threshold (≥6 months)')
ax_wd_A.set_xlabel("Unemployment duration at hire (months)", fontsize=10)
ax_wd_A.set_ylabel("Mean re-employment wage / occ. median wage", fontsize=10)
ax_wd_A.set_title("A.  Re-employment wage ratio by duration\n(baseline, by model)", fontsize=11, fontweight='bold')
ax_wd_A.legend(fontsize=9); ax_wd_A.grid(axis='y', alpha=0.3)

# Panel B formatting
ax_wd_B.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax_wd_B.axvline(5, color='darkred', linewidth=1.2, linestyle=':', alpha=0.8,
                label='LTU threshold (≥6 months)')
ax_wd_B.set_xlabel("Unemployment duration at hire (months)", fontsize=10)
ax_wd_B.set_ylabel("Δ mean wage ratio (policy − baseline)", fontsize=10)
ax_wd_B.set_title("B.  Δ Re-employment wage ratio by duration\n(LTU bias ×2.0 − baseline, by model)", fontsize=11, fontweight='bold')
ax_wd_B.legend(fontsize=9); ax_wd_B.grid(axis='y', alpha=0.3)

fig_wd.suptitle(
    f"Re-employment wage by unemployment duration — cross-model comparison  |  N = {N_TOTAL_SIMS} sims per scenario",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
_fname_wd = "fig_cross_model_wage_duration.png"
fig_wd.savefig(os.path.join(OUTPUT_PATH, _fname_wd), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {_fname_wd}")
