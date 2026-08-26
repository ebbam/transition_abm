"""
analyze_occ_shocks.py

Inventory and characterise occupation-specific demand shocks from
occupational_va_shocks_full_omn.csv.

Steps
-----
1. Load occ_ids; verify row-order alignment with shock matrix.
2. Targeted identification: match known episodes by keyword.
3. Discovery: rank by largest sustained decline (rolling 12-month window).
4. Write outputs: CSV, time-series plots, markdown summary.

Outputs written to: output/occ_shock_analysis/

Shock convention (from network_input_builder.py)
-------------------------------------------------
  raw_csv_value + 1 = demand multiplier centred at 1.0
  Deviation from baseline = shock - 1.0
  A DROP  → shock < 1.0  → deviation < 0
  Values are monthly interpolations of quarterly HP-filtered VA cycles.
"""

import os
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
SHOCKS_CSV     = os.path.join(PROJECT_ROOT, "data/occupational_va_shocks_full_omn.csv")
OCC_IDS_CSV    = os.path.expanduser(
    "~/OneDrive - Nexus365/GenerateOccMobNets/data/occ_names_employment_asec_occ.csv")
CROSSWALK_CSV  = os.path.join(PROJECT_ROOT, "data/occ_uer_ltuer_observed_w_labels.csv")
OUT_DIR        = os.path.join(PROJECT_ROOT, "output/occ_shock_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── configurable threshold ─────────────────────────────────────────────────────
# "Considerable drop" = rolling 12-month mean of (shock - 1.0) < DROP_THRESHOLD
# Override here before running.
DROP_THRESHOLD = -0.03      # -3 % below trend

# ── targeted episodes ──────────────────────────────────────────────────────────
EPISODES = {
    "Coal mining": {
        "keywords": ["coal", "mining", "extraction"],
        "window":   ("2011-01-01", "2016-12-01"),
    },
    "Textiles/apparel": {
        "keywords": ["textile", "apparel", "sewing", "garment"],
        "window":   ("2000-01-01", "2007-12-01"),
    },
    "Furniture/wood": {
        "keywords": ["furniture", "woodwork", "cabinet"],
        "window":   ("2000-01-01", "2010-12-01"),
    },
    "Construction trades": {
        "keywords": ["construction", "carpenter", "laborer", "mason"],
        "window":   ("2008-01-01", "2010-12-01"),
    },
    "Motor vehicles": {
        "keywords": ["motor vehicle", "assembler", "automotive"],
        "window":   ("2008-01-01", "2009-12-01"),
    },
    "Routine clerical": {
        "keywords": ["data entry", "typist", "bookkeeping", "teller",
                     "telephone operator", "switchboard", "travel agent"],
        "window":   ("2000-01-01", "2019-12-01"),
    },
    "Printing/publishing": {
        "keywords": ["printing", "press operator", "prepress", "photographic"],
        "window":   ("2008-01-01", "2019-12-01"),
    },
    "Retail": {
        "keywords": ["cashier", "retail sales"],
        "window":   ("2015-01-01", "2019-12-01"),
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and align
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1 — Loading data and verifying alignment")
print("=" * 70)

shock_raw   = pd.read_csv(SHOCKS_CSV, index_col=0)
shock_codes = shock_raw['acs_2010_code'].values
date_cols   = [c for c in shock_raw.columns if c != 'acs_2010_code']
n_occs      = len(shock_raw)

# raw CSV values are quarterly HP-cycle deviations; add 1 → multipliers at 1.0
df_quarterly        = shock_raw[date_cols].astype(float) + 1.0
df_quarterly.index  = range(n_occs)
df_quarterly.columns = pd.to_datetime(df_quarterly.columns)

# monthly interpolation (linear, same as network_input_builder)
shock_monthly = df_quarterly.T.resample('MS').interpolate('linear').T
monthly_dates = shock_monthly.columns   # DatetimeIndex
n_months      = len(monthly_dates)

# load occ_ids
if os.path.exists(OCC_IDS_CSV):
    occ_ids = pd.read_csv(OCC_IDS_CSV)[['Code', 'Label']]
    occ_ids.columns = ['acs_occ_code', 'label']
    occ_ids['id']   = occ_ids.index.values   # 0-indexed = OriginOccupation in model

    if len(occ_ids) == n_occs and np.array_equal(
            occ_ids['acs_occ_code'].values, shock_codes):
        print(f"✓ Alignment VERIFIED: {n_occs} occupations, ACS codes match in order.")
        alignment_ok = True
    else:
        print("✗ ALIGNMENT FAILED — codes do not match in order or row count differs.")
        print(f"  shock rows={n_occs}, occ_ids rows={len(occ_ids)}")
        sys.exit(
            "Cannot label occupations safely. "
            "Check that occ_names_employment_asec_occ.csv matches the shock matrix row order."
        )
else:
    # fall back to crosswalk (covers ~463/528)
    print(f"⚠ occ_ids file not found at:\n  {OCC_IDS_CSV}")
    print("  Falling back to occ_crosswalk for labels.")
    cw       = pd.read_csv(CROSSWALK_CSV)[['acs_occ_code', 'label']].drop_duplicates('acs_occ_code')
    label_map = cw.set_index('acs_occ_code')['label'].to_dict()
    occ_ids  = pd.DataFrame({
        'acs_occ_code': shock_codes,
        'label': [label_map.get(int(c), f"ACS_{c}") for c in shock_codes],
        'id': range(n_occs),
    })
    n_labelled = sum(c in label_map for c in shock_codes.astype(int))
    print(f"  Labels found for {n_labelled}/{n_occs} occupations.")
    alignment_ok = False   # alignment not verified via occ_ids

print(f"\nShock matrix: {n_occs} occupations × {n_months} months")
print(f"Date range  : {monthly_dates[0].date()} → {monthly_dates[-1].date()}")
print(f"\n{'id':>5}  {'acs_occ_code':>13}  label")
print("-" * 70)
for _, row in occ_ids.iterrows():
    print(f"{row['id']:>5}  {row['acs_occ_code']:>13}  {row['label']}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Targeted identification
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"STEP 2 — Targeted episode identification  (threshold={DROP_THRESHOLD})")
print("=" * 70)

# helper: keyword match (case-insensitive, any word in list)
def _keyword_match(label_series, keywords):
    mask = pd.Series(False, index=label_series.index)
    for kw in keywords:
        mask |= label_series.str.contains(kw, case=False, na=False)
    return mask

episode_records = []   # accumulate for CSV output

for ep_name, ep_cfg in EPISODES.items():
    keywords = ep_cfg['keywords']
    win_s    = pd.to_datetime(ep_cfg['window'][0])
    win_e    = pd.to_datetime(ep_cfg['window'][1])

    matched_mask = _keyword_match(occ_ids['label'], keywords)
    matched      = occ_ids[matched_mask]

    print(f"\n── {ep_name} ──")
    print(f"   Keywords : {keywords}")
    print(f"   Window   : {win_s.date()} → {win_e.date()}")

    if matched.empty:
        print("   ✗ No occupations matched — episode NOT found at this resolution.")
        episode_records.append({
            'episode': ep_name, 'occ_id': np.nan, 'label': 'NO MATCH',
            'acs_occ_code': np.nan,
            'trough_month': np.nan, 'min_shock': np.nan,
            'mean_shock_in_window': np.nan, 'peak_to_trough_drop': np.nan,
        })
        continue

    win_cols = monthly_dates[(monthly_dates >= win_s) & (monthly_dates <= win_e)]
    if len(win_cols) == 0:
        print(f"   ✗ No monthly data in window.")
        continue

    for _, occ_row in matched.iterrows():
        idx   = occ_row['id']
        lbl   = occ_row['label']
        series = shock_monthly.loc[idx]                   # full series (shock level)
        win_s_series  = series[win_cols]
        min_val   = win_s_series.min()
        trough_dt = win_s_series.idxmin()
        mean_val  = win_s_series.mean()
        # peak-to-trough within window: compare max before trough to trough
        pre_trough = series[series.index <= trough_dt]
        peak_val   = pre_trough.max() if len(pre_trough) > 0 else np.nan
        ptt        = peak_val - min_val

        # did the expected decline actually appear?
        appeared = min_val < (1.0 + DROP_THRESHOLD)

        print(f"\n   Occupation : [{idx}] {lbl}")
        print(f"   Trough     : {trough_dt.date()}  shock={min_val:.4f}  "
              f"(deviation {min_val-1:.4f})")
        print(f"   Mean shock : {mean_val:.4f}  (deviation {mean_val-1:.4f})")
        print(f"   Peak→trough: {ptt:.4f}")
        print(f"   Appeared?  : {'YES' if appeared else 'NO (below threshold)'}")

        episode_records.append({
            'episode': ep_name,
            'occ_id': idx,
            'label': lbl,
            'acs_occ_code': occ_row['acs_occ_code'],
            'trough_month': trough_dt.strftime('%Y-%m-%d') if pd.notna(trough_dt) else np.nan,
            'min_shock': round(min_val, 5),
            'mean_shock_in_window': round(mean_val, 5),
            'peak_to_trough_drop': round(ptt, 5),
        })


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Discovery: top-15 by largest sustained decline
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3 — Discovery: top-15 occupations by sustained 12-month decline")
print(f"         (minimum rolling 12-month mean of shock - 1.0)")
print("=" * 70)

ROLLING_WINDOW = 12   # months

deviation    = shock_monthly - 1.0   # deviation from baseline
rolling_mean = deviation.T.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean().T
min_rolling  = rolling_mean.min(axis=1)   # worst 12-month window per occupation

discovery_df = occ_ids[['id', 'acs_occ_code', 'label']].copy()
discovery_df['min_roll12_deviation'] = min_rolling.values
discovery_df['trough_month_roll12']  = rolling_mean.idxmin(axis=1).values

top15 = discovery_df.nsmallest(15, 'min_roll12_deviation')
print(f"\n{'Rank':>4}  {'id':>5}  {'acs':>6}  "
      f"{'min_roll12_dev':>16}  {'trough_month':>13}  label")
print("-" * 80)
for rank, (_, row) in enumerate(top15.iterrows(), 1):
    trough_str = pd.Timestamp(row['trough_month_roll12']).strftime('%Y-%m') \
        if pd.notna(row['trough_month_roll12']) else 'n/a'
    print(f"{rank:>4}  {int(row['id']):>5}  {int(row['acs_occ_code']):>6}  "
          f"{row['min_roll12_deviation']:>16.4f}  {trough_str:>13}  "
          f"{str(row['label'])[:45]}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Write outputs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4 — Writing outputs")
print("=" * 70)

# ── 4a: CSV of flagged occupation-episodes ────────────────────────────────────
episodes_df = pd.DataFrame(episode_records)
csv_path    = os.path.join(OUT_DIR, "occ_shock_episodes.csv")
episodes_df.to_csv(csv_path, index=False)
print(f"✓ Episode CSV   : {csv_path}")

# ── 4b: Discovery CSV ─────────────────────────────────────────────────────────
disc_csv = os.path.join(OUT_DIR, "occ_shock_discovery_ranking.csv")
discovery_df.sort_values('min_roll12_deviation').to_csv(disc_csv, index=False)
print(f"✓ Discovery CSV : {disc_csv}")

# ── 4c: Time-series plots for all matched occupations ─────────────────────────
# One figure per episode; all matched occupations on the same axes.
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def _plot_episode(ep_name, occ_rows, win_s, win_e, shock_monthly, monthly_dates, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap    = plt.get_cmap('tab10')

    for i, (_, row) in enumerate(occ_rows.iterrows()):
        idx    = row['id']
        series = shock_monthly.loc[idx]
        ax.plot(monthly_dates, series.values,
                color=cmap(i % 10), lw=1.2, alpha=0.85, label=str(row['label'])[:50])

    ax.axhline(1.0, color='black', ls='--', lw=0.9, alpha=0.6, label='Baseline (1.0)')
    ax.axhspan(win_s, win_e, alpha=0.10, color='steelblue', label='Episode window')
    ax.axhline(1.0 + DROP_THRESHOLD, color='red', ls=':', lw=0.8, alpha=0.7,
               label=f'Threshold ({DROP_THRESHOLD:+.0%})')

    ax.set_xlabel('Date')
    ax.set_ylabel('Demand shock (1.0 = on-trend)')
    ax.set_title(f"Occupation demand shocks — {ep_name}", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='lower left', ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    fname = ep_name.lower().replace('/', '_').replace(' ', '_') + '.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname

matched_per_episode = {}
for ep_name, ep_cfg in EPISODES.items():
    matched = occ_ids[_keyword_match(occ_ids['label'], ep_cfg['keywords'])]
    matched_per_episode[ep_name] = matched
    if matched.empty:
        continue
    win_s = pd.to_datetime(ep_cfg['window'][0])
    win_e = pd.to_datetime(ep_cfg['window'][1])
    fname = _plot_episode(ep_name, matched, win_s, win_e,
                          shock_monthly, monthly_dates, PLOT_DIR)
    print(f"✓ Plot saved    : plots/{fname}")

# ── 4d: Discovery plot (top-15) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
cmap    = plt.get_cmap('tab20')
for i, (_, row) in enumerate(top15.iterrows()):
    idx    = int(row['id'])
    series = shock_monthly.loc[idx]
    ax.plot(monthly_dates, series.values,
            color=cmap(i), lw=1.0, alpha=0.8,
            label=f"[{idx}] {str(row['label'])[:35]}")

ax.axhline(1.0, color='black', ls='--', lw=0.9, alpha=0.6)
ax.axhline(1.0 + DROP_THRESHOLD, color='red', ls=':', lw=0.8, alpha=0.7,
           label=f'Threshold ({DROP_THRESHOLD:+.0%})')
ax.set_xlabel('Date')
ax.set_ylabel('Demand shock (1.0 = on-trend)')
ax.set_title('Top-15 occupations by worst sustained 12-month demand decline', fontsize=10)
ax.legend(fontsize=6, frameon=False, loc='lower left', ncol=2)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
disc_plot = os.path.join(PLOT_DIR, 'discovery_top15.png')
fig.savefig(disc_plot, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Plot saved    : plots/discovery_top15.png")

# ── 4e: Markdown summary ──────────────────────────────────────────────────────
md_lines = [
    "# Occupational Demand Shock Analysis",
    "",
    f"**Shock data**: `occupational_va_shocks_full_omn.csv`  ",
    f"**Occupations**: {n_occs}  |  "
    f"**Date range**: {monthly_dates[0].strftime('%Y-%m')} → "
    f"{monthly_dates[-1].strftime('%Y-%m')}  ",
    f"**Alignment**: {'verified (ACS codes match in order)' if alignment_ok else 'fallback (crosswalk labels)'}  ",
    f"**Drop threshold** (rolling 12-month mean): `shock - 1.0 < {DROP_THRESHOLD}`",
    "",
    "---",
    "",
    "## Step 2 — Targeted Episodes",
    "",
]

for ep_name, ep_cfg in EPISODES.items():
    matched = matched_per_episode.get(ep_name, pd.DataFrame())
    win_str = f"{ep_cfg['window'][0][:7]} → {ep_cfg['window'][1][:7]}"
    md_lines += [f"### {ep_name}  *(window: {win_str})*", ""]

    if matched.empty:
        md_lines += [
            "**✗ NOT FOUND** — no occupation labels matched keywords "
            f"`{ep_cfg['keywords']}`.",
            "This episode does not appear at the ACS occupation resolution "
            "used in this model.",
            "",
        ]
        continue

    win_cols = monthly_dates[
        (monthly_dates >= pd.to_datetime(ep_cfg['window'][0])) &
        (monthly_dates <= pd.to_datetime(ep_cfg['window'][1]))
    ]
    md_lines.append(f"Matched {len(matched)} occupation(s):\n")
    for _, row in matched.iterrows():
        idx    = row['id']
        lbl    = row['label']
        series = shock_monthly.loc[idx, win_cols]
        min_v  = series.min()
        trough = series.idxmin()
        found  = min_v < (1.0 + DROP_THRESHOLD)

        md_lines += [
            f"- **[{idx}] {lbl}**  ",
            f"  Trough: `{trough.strftime('%Y-%m')}`, shock=`{min_v:.4f}` "
            f"(deviation `{min_v-1:+.4f}`)  ",
            f"  Expected episode {'**CONFIRMED**' if found else '**NOT confirmed**'} "
            f"(threshold `{DROP_THRESHOLD}`)  ",
            "",
        ]

md_lines += [
    "---",
    "",
    "## Step 3 — Discovery: Top-15 Sustained Declines",
    "",
    f"Ranked by minimum rolling 12-month mean of `(shock - 1.0)`.",
    "",
    "| Rank | id | ACS code | Min roll-12 deviation | Trough month | Label |",
    "|------|-----|----------|----------------------|--------------|-------|",
]
for rank, (_, row) in enumerate(top15.iterrows(), 1):
    trough_str = pd.Timestamp(row['trough_month_roll12']).strftime('%Y-%m') \
        if pd.notna(row['trough_month_roll12']) else 'n/a'
    md_lines.append(
        f"| {rank} | {int(row['id'])} | {int(row['acs_occ_code'])} | "
        f"{row['min_roll12_deviation']:.4f} | {trough_str} | "
        f"{str(row['label'])[:55]} |"
    )

md_path = os.path.join(OUT_DIR, "occ_shock_summary.md")
with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines) + '\n')
print(f"✓ Markdown      : {md_path}")

print(f"\n✓ All outputs written to: {OUT_DIR}")
print(f"  DROP_THRESHOLD = {DROP_THRESHOLD}  (override at top of script to change sensitivity)")
