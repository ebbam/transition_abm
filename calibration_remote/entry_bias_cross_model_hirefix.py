"""
entry_bias_cross_model_hirefix.py

Re-runs fig_cross_model_combined_entry_bias (entry_policy_exercise.py, Section 5E:
entry-level hire bias x2.0, three non-cyclical models) after the hire() fix in
abm_funs.py and with the 7% non-cyclical OTJ search share, restructured around new
entrants: panel A (Δ share of UE hires from entry-level occupations that were LTU)
becomes Δ LTUER of new entrants, panel B's worker types become new entrants vs other
unemployed, and panel C (Δ hires rate) becomes the % change in the monthly job-finding
rate of new entrants and other unemployed (PANEL_C = 'jfr', saved as
fig_cross_model_combined_entry_bias_jfr) or, as in the first version of this rerun,
Δ cumulative earnings of new entrants since entry (PANEL_C = 'earnings', saved as
fig_cross_model_combined_entry_bias). Models are ordered by MODEL_ORDER.

Stage 1: per model, 25 entry_hire_bias = 1.0 sims and 25 baseline (no bias) sims,
         one worker process per model, saved in batches of 5 so an interrupted run
         resumes from the last finished batch. The baseline is re-run rather than
         taken from model_results_N25.pkl so that both arms come from the same code;
         the baselines in entry_policy_hirefix/ are not reused either (they were run
         at the accidental 3% OTJ share). Each sim also keeps the annual per-worker
         wage snapshots (run_single_local's wage_age_df) of the entry cohort, for
         the new panel C, and per-month counts of unemployed and long-term
         unemployed new entrants from its ue_stock_df, for the new panel A.
         Pickles are gzip-compressed.
Stage 2: rebuilds the figure from Section 5E's own source (plus the setup and
         helpers it calls), with _load_cm_entry reading the new runs and panel C
         swapped for the lifetime-earnings panel.

Panel A: LTUER of new entrants = long-term unemployed (7+ months, abm_funs.py's
longterm_unemp) new entrants / unemployed new entrants, each month, where a new
entrant is a worker who has not been hired since entering (the new_entrant flag that
entry_hire_bias favours; model_fun.py's ue_stock_df records it). Mean across sims per
month, Δ = policy - baseline, band from sqrt(sem_bl^2 + sem_po^2).

Panel B: Δ mean duration of completed UE spells (1-36 months), split by the hire's
NewEntrant flag: a worker's first spell since entering counts as "New entrants", every
later spell (and every spell of workers present at the start) as "Other unemployed", so
one worker can contribute to both groups over time.

Panel C ('jfr'): monthly job-finding rate = UE hires in month t / the unemployed who
could be hired in month t, per sim averaged over the calibration period, then % change
of the policy mean over the baseline mean (delta-method SE). Within a month, entrants
join at the start and can be hired, workers separated that month only search the next,
and unemployed stocks are counted at the end, after hiring. So the risk set is last
month's end-of-month stock, plus that month's entrants (grouped's Entry_Level_Hires) for
new entrants: new entrants = new-entrant stock (t-1) + entrants (t); other unemployed =
unemployed (t-1) - new-entrant stock (t-1). Hires are split by the hire's NewEntrant flag.

Panel C ('earnings'): every entrant after the burn-in (the policy is on for the whole run), followed
from entry to the end of the simulation. A worker counts at year k since entry only if
the simulation reaches their year-k annual snapshot, so later years are drawn from
earlier entry cohorts; a year in which a worker is not employed at the snapshot counts
as $0. Per sim and year, the median of cumulative dollar earnings over the workers
observed that year, then Δ = policy - baseline of the mean across sims, as
exposure_entry_policy_exercise.py's Section 13 (year 0 not shown). Years are shown while
at least MIN_COHORT_SHARE of the year-0 entrants are still observed, so the last few
years (a handful of 2001 entrants) are left off. Entrants never employed at any snapshot
have no Worker_ID (model_fun.py assigns it at a worker's first employed snapshot) and so
are missing from both arms.
Sims are not seeded, so the two arms are independent draws and the band uses
sqrt(sem_bl^2 + sem_po^2), as in panels A and B.

All output goes to entry_policy_hirefix/entry_bias_7pct/; nothing else is touched.
"""

import os
import sys
import ast
import gzip
import pickle
import multiprocessing as mp

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ltu_cross_model_hirefix import EPE_PATH, EPE_SRC, _between, _SETUP_SRC

EB_DIR = os.path.join(
    PROJECT_ROOT, "output/calib_final/full_omn/exog_vac/figures/covid_oos/entry_policy_hirefix/entry_bias_7pct/"
)
EB_MODELS = ['otj_nonbehav', 'otj_disc', 'otj_disc_no_rw']   # Section 5E's _CM_SPECS_NO_CYC
N_WORKERS = 3
N_SIMS = 25
BATCH = 5
ARMS = {'entry_bias': {'entry_hire_bias': 1.0}, 'baseline': {}}

MODEL_ORDER = ['otj_nonbehav', 'otj_disc_no_rw', 'otj_disc']   # bars and legend: fewest to most behavioural mechanisms
PANEL_C = 'jfr'                # 'jfr': job-finding rate bars; 'earnings': cumulative earnings since entry

LE_DISPLAY_MIN_YEAR = 1        # as exposure_entry_policy_exercise.py: year 0 is mostly pre-first-job
MIN_COHORT_SHARE = 0.10        # last year shown: >= 10% of year-0 entrants still observed


def _batch_pkl(mc, arm, b):
    return os.path.join(EB_DIR, f"{mc}_{arm}_sims{b * BATCH:02d}-{(b + 1) * BATCH - 1:02d}.pkl.gz")


def _all_pkls(mc):
    return [_batch_pkl(mc, arm, b) for arm in ARMS for b in range(N_SIMS // BATCH)]


def _save(obj, path):
    with gzip.open(path, 'wb', compresslevel=1) as _f:
        pickle.dump(obj, _f, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path):
    with gzip.open(path, 'rb') as _f:
        return pickle.load(_f)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: entry-bias and baseline runs
# ══════════════════════════════════════════════════════════════════════════════
_setup_ns = {'__file__': EPE_PATH, '__name__': 'entry_policy_exercise_setup'}


def _run_batch(mc, overrides, base_params, sim_id_offset):
    """run_policy_scenario (entry_policy_exercise.py) for BATCH sims, also keeping the
    post-burn-in entrants' rows of each sim's wage_age_df (run_single_local's 10th output) and
    per-month counts of unemployed / long-term unemployed new entrants from its ue_stock_df (11th)."""
    import pandas as pd
    from copy import deepcopy
    _delay = _setup_ns['DELAY']
    _calib = _setup_ns['CALIB_DATE']
    trans_runs, grouped_runs, occ_runs, wage_age_runs, ne_stock_runs = [], [], [], [], []
    for i in range(BATCH):
        params = deepcopy(base_params)
        params.update(overrides)
        result = _setup_ns['run_single_local'](**params)
        record_df, sim_grouped, *_, transition_df = result
        wage_age_df = result[9]
        ue_stock_df = result[10]
        _sid = sim_id_offset + i
        sim_grouped = sim_grouped.copy()
        sim_grouped['DATE'] = pd.date_range(start=_calib[0], end=_calib[1], periods=len(sim_grouped))
        sim_grouped['sim_id'] = _sid
        transition_df = transition_df.copy(); transition_df['sim_id'] = _sid
        record_df = record_df.copy();         record_df['sim_id'] = _sid
        wage_age_df = wage_age_df[wage_age_df['Entry_Time_Step'] >= _delay + 1].copy()
        wage_age_df['sim_id'] = _sid
        _ne = ue_stock_df[ue_stock_df['NewEntrant'] == True]
        ne_stock = pd.DataFrame({
            'n_unemp': _ne.groupby('Time Step').size(),
            'n_ltu':   _ne.groupby('Time Step')['LongtermUnemp'].sum(),
        }).reset_index()
        ne_stock['sim_id'] = _sid
        trans_runs.append(transition_df)
        grouped_runs.append(sim_grouped)
        occ_runs.append(record_df)
        wage_age_runs.append(wage_age_df)
        ne_stock_runs.append(ne_stock)
    return (pd.concat(trans_runs, ignore_index=True), pd.concat(grouped_runs, ignore_index=True),
            pd.concat(occ_runs, ignore_index=True), pd.concat(wage_age_runs, ignore_index=True),
            pd.concat(ne_stock_runs, ignore_index=True))


def _run_model(mc):
    """Worker (forked, so it inherits _setup_ns): every missing batch of both arms for one model."""
    _bp = _setup_ns['_build_model_params'](mc)
    for arm, overrides in ARMS.items():
        for b in range(N_SIMS // BATCH):
            _path = _batch_pkl(mc, arm, b)
            if os.path.exists(_path):
                print(f"  [{mc}] {os.path.basename(_path)} exists, skipping.", flush=True)
                continue
            _save(_run_batch(mc, overrides, _bp, b * BATCH), _path)
            print(f"  [{mc}] saved {os.path.basename(_path)}", flush=True)
    return mc


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: the figure
# ══════════════════════════════════════════════════════════════════════════════
_LOADER_SRC = """
import gzip
from model_colors import MODEL_COLORS_BY_ID

def _concat_results(a, b):
    return tuple(pd.concat([x, y], ignore_index=True) for x, y in zip(a, b))

def _load_arm(mc, arm):
    _parts = None
    for _p in sorted(_p for _p in _all_pkls(mc) if os.path.basename(_p).startswith(f"{mc}_{arm}_")):
        with gzip.open(_p, 'rb') as _f:
            _b = pickle.load(_f)
        _parts = _b if _parts is None else _concat_results(_parts, _b)
    return _parts

_eb_wage_age = {}
_eb_ne_stock = {}
_eb_jfr = {}


def _eb_jfr_by_type(tr, grp, ne_stock):
    \"\"\"Per-sim mean monthly job-finding rate (%) of new entrants and other unemployed (see the module
    docstring): UE hires in month t / the unemployed who could be hired in month t.\"\"\"
    _ue = tr[tr['TransitionType'] == 'UE']
    _h = (_ue.assign(ne=_ue['NewEntrant'] == True)
             .groupby(['sim_id', 'Time Step', 'ne']).size()
             .unstack('ne', fill_value=0).reindex(columns=[False, True], fill_value=0))
    _g = grp[grp['DATE'].notna()].set_index(['sim_id', 'Time Step']).sort_index()
    _s = _g[['Unemployment']].join(ne_stock.set_index(['sim_id', 'Time Step'])['n_unemp'], how='left').fillna(0)
    _s = _s.groupby(level='sim_id').shift(1)            # stock at the start of the month
    _s['entrants'] = _g['Entry_Level_Hires']            # entrants join at the start of the month
    _m = _s.dropna().join(_h, how='left').fillna(0)
    return pd.DataFrame({
        'ne':    _m[True] / (_m['n_unemp'] + _m['entrants']).clip(lower=1) * 100,
        'other': _m[False] / (_m['Unemployment'] - _m['n_unemp']).clip(lower=1) * 100,
    }).groupby('sim_id').mean()


def _load_cm_entry(mc):
    _bl = _load_arm(mc, 'baseline')
    _po = _load_arm(mc, 'entry_bias')
    _eb_wage_age[mc] = {'baseline': _bl[3], 'entry_bias': _po[3]}
    _eb_ne_stock[mc] = {'baseline': (_bl[4], _bl[1]), 'entry_bias': (_po[4], _po[1])}
    _eb_jfr[mc] = {'baseline': _eb_jfr_by_type(_bl[0], _bl[1], _bl[4]),
                   'entry_bias': _eb_jfr_by_type(_po[0], _po[1], _po[4])}
    return _bl[:3], _po[:3]


def _eb_jfr_pct(mc):
    \"\"\"% change in the mean monthly job-finding rate (policy vs baseline) by worker type, with delta-method SEs
    (the two arms are independent).\"\"\"
    _bl, _po = _eb_jfr[mc]['baseline'], _eb_jfr[mc]['entry_bias']
    _out = {}
    for _k in ('ne', 'other'):
        _b, _p = _bl[_k].mean(), _po[_k].mean()
        _out[_k] = (_p / _b - 1) * 100
        _out[f'se_{_k}'] = (_p / _b) * np.sqrt((_po[_k].sem() / _p)**2 + (_bl[_k].sem() / _b)**2) * 100
        print(f"  [{mc}] {_k} monthly JFR: baseline {_b:.2f}%, entry bias {_p:.2f}%, "
              f"change {_out[_k]:+.1f}% (SE {_out[f'se_{_k}']:.1f})")
    return _out


def _cohort_cum_dollars_median(wage_age_df):
    \"\"\"Per sim and year since entry, the median cumulative dollar earnings of the entrants
    observed that year (see the module docstring), with the number of them.\"\"\"
    df = wage_age_df.dropna(subset=['Entry_Time_Step']).copy()
    df['years_since_entry'] = np.floor((df['Time Step'] - df['Entry_Time_Step']) / 12).astype(int)
    df = df[df['years_since_entry'] >= 0]
    df = df.groupby(['sim_id', 'Worker_ID', 'years_since_entry'])['Wage'].mean().reset_index()
    # Last year each worker is observable: annual snapshots fall at Time Steps DELAY+1+12j up to the
    # sim's last one; year k's snapshot is the first at or after Entry_Time_Step + 12k.
    _w = wage_age_df.groupby(['sim_id', 'Worker_ID'])['Entry_Time_Step'].first().reset_index()
    _last = wage_age_df.groupby('sim_id')['Time Step'].max().rename('last_ts')
    _w = _w.merge(_last, on='sim_id')
    _first_snap = DELAY + 1 + 12 * np.ceil((_w['Entry_Time_Step'] - DELAY - 1) / 12)
    _w['max_k'] = ((_w['last_ts'] - _first_snap) // 12).astype(int)
    full = _w.loc[_w.index.repeat(_w['max_k'] + 1), ['sim_id', 'Worker_ID']].copy()
    full['years_since_entry'] = full.groupby(['sim_id', 'Worker_ID']).cumcount()
    full = full.merge(df, on=['sim_id', 'Worker_ID', 'years_since_entry'], how='left')
    full['Wage'] = full['Wage'].fillna(0.0)
    full = full.sort_values(['sim_id', 'Worker_ID', 'years_since_entry'])
    full['cum_dollars'] = full.groupby(['sim_id', 'Worker_ID'])['Wage'].cumsum()
    return (full.groupby(['sim_id', 'years_since_entry'])['cum_dollars']
                .agg(cum_dollars='median', n_workers='size').reset_index())


def _ne_ltuer_ts(ne_stock, grp_df):
    \"\"\"LTUER of new entrants (%) by DATE: mean and SEM across sims (burn-in months dropped).\"\"\"
    ts_map = grp_df.drop_duplicates('Time Step').set_index('Time Step')['DATE']
    d = ne_stock.copy()
    d['ltuer'] = d['n_ltu'] / d['n_unemp'].clip(lower=1) * 100
    d['DATE'] = d['Time Step'].map(ts_map)
    return d.dropna(subset=['DATE']).groupby('DATE')['ltuer'].agg(['mean', 'sem'])


def _eb_ne_ltuer_delta(mc):
    _bl = _ne_ltuer_ts(*_eb_ne_stock[mc]['baseline'])
    _po = _ne_ltuer_ts(*_eb_ne_stock[mc]['entry_bias'])
    _dates = _bl.index.intersection(_po.index)
    print(f"  [{mc}] new-entrant LTUER, mean over the period: baseline {_bl.loc[_dates, 'mean'].mean():.2f}%, "
          f"entry bias {_po.loc[_dates, 'mean'].mean():.2f}%")
    return pd.DataFrame({'mean': _po.loc[_dates, 'mean'] - _bl.loc[_dates, 'mean'],
                         'sem':  np.sqrt(_po.loc[_dates, 'sem']**2 + _bl.loc[_dates, 'sem']**2)})


def _eb_lifetime_delta(mc):
    _ts, _n = {}, {}
    for _arm in ('baseline', 'entry_bias'):
        _ps = _cohort_cum_dollars_median(_eb_wage_age[mc][_arm])
        _ts[_arm] = _ps.groupby('years_since_entry')['cum_dollars'].agg(['mean', 'sem'])
        _n[_arm] = _ps.groupby('years_since_entry')['n_workers'].min()   # smallest sim
    _d = pd.DataFrame({'mean': _ts['entry_bias']['mean'] - _ts['baseline']['mean'],
                       'sem':  np.sqrt(_ts['entry_bias']['sem']**2 + _ts['baseline']['sem']**2)})
    _n = pd.concat(_n, axis=1).min(axis=1)
    _shown = (_d.index >= LE_DISPLAY_MIN_YEAR) & (_n.reindex(_d.index) >= MIN_COHORT_SHARE * _n.loc[0]).values
    print(f"  [{mc}] entrants per sim at year 0: {_n.loc[0]:,.0f}; years shown "
          f"{_d.index[_shown].min()}-{_d.index[_shown].max()} "
          f"(at the last: {_n.loc[_d.index[_shown].max()]:,.0f} entrants)")
    return _d[_shown]
"""

_PANEL_A_SRC = """# Panel A — Δ LTUER of new entrants (policy − baseline)
for _mc in _CM_SPECS_NO_CYC:
    _col = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _lbl = _CM_DISPLAY[_mc].replace('\\n', ' ')
    _d   = _eb_ne_ltuer_delta(_mc)
    ax_eb_E.plot(_d.index, _d['mean'], color=_col, linewidth=1.0, label=_lbl)
    ax_eb_E.fill_between(_d.index, _d['mean'] - 1.96 * _d['sem'], _d['mean'] + 1.96 * _d['sem'],
                         alpha=0.12, color=_col)
for _, rec in recession_spans.iterrows():
    ax_eb_E.axvspan(pd.to_datetime(rec['start']), pd.to_datetime(rec['end']),
                    color='lightgrey', alpha=0.4, zorder=0)
ax_eb_E.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax_eb_E.set_xlim(pd.to_datetime(CALIB_DATE[0]), pd.to_datetime(CALIB_DATE[1]))
ax_eb_E.set_xlabel("Date")
ax_eb_E.set_ylabel("Δ LTUER (pp)")
ax_eb_E.set_title("Δ LTUER: New Entrants",
                  fontweight='normal')
ax_eb_E.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(4))
ax_eb_E.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax_eb_E.get_xticklabels(), rotation=45, ha='right')
ax_eb_E.grid(axis='y', alpha=0.3)

"""

_PANEL_C_SRC = """# Panel C — % change in the monthly job-finding rate by worker type (policy vs baseline)
for _mi, _mc in enumerate(_CM_SPECS_NO_CYC):
    _jf = _eb_jfr_pct(_mc)
    ax_eb_G.bar(_dur_grp_x_eb + _dur_off_eb[_mi], [_jf['ne'], _jf['other']], _bw_eb,
                color=MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa'), alpha=0.85, edgecolor='white', linewidth=0.3,
                yerr=[1.96 * _jf['se_ne'], 1.96 * _jf['se_other']],
                error_kw=dict(ecolor='black', elinewidth=1.2, capsize=3, capthick=1.2))
ax_eb_G.axhline(0, color='black', linewidth=0.8)
ax_eb_G.set_xticks(_dur_grp_x_eb)
ax_eb_G.set_xticklabels(_dur_groups_eb, ha='center')
ax_eb_G.set_ylabel("% Δ Monthly Job-Finding Rate")
ax_eb_G.set_title("% Δ Job-Finding Rate\\nby Worker Type",
                  fontweight='normal')
ax_eb_G.grid(axis='y', alpha=0.3)

"""

_PANEL_C_EARNINGS_SRC = """# Panel C (earlier version) — Δ cumulative earnings of all entrants since entry (policy − baseline)
for _mc in _CM_SPECS_NO_CYC:
    _col = MODEL_COLORS_BY_ID.get(_mc, '#aaaaaa')
    _le  = _eb_lifetime_delta(_mc)
    ax_eb_G.plot(_le.index, _le['mean'], color=_col, linewidth=1.0, marker='o', markersize=2)
    ax_eb_G.fill_between(_le.index, _le['mean'] - 1.96 * _le['sem'], _le['mean'] + 1.96 * _le['sem'],
                         alpha=0.12, color=_col)
    print(f"  [{_mc}] Δ cumulative earnings at year {_le.index[-1]}: "
          f"${_le['mean'].iloc[-1]:,.0f} (SE ${_le['sem'].iloc[-1]:,.0f})")
ax_eb_G.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax_eb_G.set_xlabel("Years since entry")
ax_eb_G.set_ylabel("Δ Cumulative Earnings ($)")
ax_eb_G.set_title("Δ Cumulative Earnings: New Entrants",
                  fontweight='normal')
ax_eb_G.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
ax_eb_G.grid(axis='y', alpha=0.3)

"""


def _section_5b_helpers():
    """Section 5B's function definitions and its two setup assignments (_CM_SPECS_NO_CYC,
    _x_range), which Section 5E reuses; 5B's own runs and figure are left out."""
    _s = EPE_SRC.index('print("\\n=== Section 5B')
    _e = EPE_SRC.rindex("\n# ═", 0, EPE_SRC.index("# SECTION 5E: Cross-model combined figure")) + 1
    _keep = [n for n in ast.parse(EPE_SRC[_s:_e]).body
             if isinstance(n, ast.FunctionDef)
             or (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in ('_CM_SPECS_NO_CYC', '_x_range'))]
    assert len([n for n in _keep if isinstance(n, ast.Assign)]) == 2
    return "\n\n".join(ast.unparse(n) for n in _keep)


def _defs(*names):
    """The named function definitions from anywhere in entry_policy_exercise.py (some sit inside
    if-blocks of sections that are otherwise skipped here); each must be defined exactly once."""
    _found = {}
    for n in ast.walk(ast.parse(EPE_SRC)):
        if isinstance(n, ast.FunctionDef) and n.name in names:
            assert n.name not in _found, n.name
            _found[n.name] = ast.unparse(n)
    assert set(_found) == set(names), set(names) - set(_found)
    return "\n\n".join(_found[k] for k in names)


def _section_5e():
    src = _between(EPE_SRC, 'print("\\n=== Section 5E', "# SECTION 5D: Re-employment wage by unemployment")
    src = src[:src.rindex("\n# ═") + 1]
    # the original caches and their loader -> _LOADER_SRC's _load_cm_entry
    _cache = _between(src, "CROSS_ENTRY_PKL         =", "# ── Helper: entry hire share time series")
    src = src.replace(_cache, "")
    # panel B (and the 5E metrics built on the same split): UE spells by entry-level origin
    # occupation -> by the hire's NewEntrant flag (first spell since entry vs every other spell)
    for _old_split, _new_split in [
        ("_ue_bl_eb[_ue_bl_eb['OriginOccupation'].isin(_eb_entry_ids)]",  "_ue_bl_eb[_ue_bl_eb['NewEntrant'] == True]"),
        ("_ue_po_eb[_ue_po_eb['OriginOccupation'].isin(_eb_entry_ids)]",  "_ue_po_eb[_ue_po_eb['NewEntrant'] == True]"),
        ("_ue_bl_eb[~_ue_bl_eb['OriginOccupation'].isin(_eb_entry_ids)]", "_ue_bl_eb[_ue_bl_eb['NewEntrant'] != True]"),
        ("_ue_po_eb[~_ue_po_eb['OriginOccupation'].isin(_eb_entry_ids)]", "_ue_po_eb[_ue_po_eb['NewEntrant'] != True]"),
        ("['Entry-Level\\nUnemployed', 'Non-Entry\\nUnemployed']",           "['New\\nEntrants', 'Other\\nUnemployed']"),
    ]:
        assert src.count(_old_split) == 1, _old_split
        src = src.replace(_old_split, _new_split)
    # panel A: LTU share of entry-level-origin UE hires -> LTUER of new entrants
    _old_a = _between(src, "# Panel A — Δ entry-level LTUER time series (policy − baseline)",
                      "# Panel C — Δ hires rate (policy − baseline, pp)")
    src = src.replace(_old_a, _PANEL_A_SRC)
    # panel C: hires rate -> job-finding rate by worker type (or lifetime earnings)
    _old_c = _between(src, "# Panel C — Δ hires rate (policy − baseline, pp)", "# Shared legend + save")
    src = src.replace(_old_c, _PANEL_C_SRC if PANEL_C == 'jfr' else _PANEL_C_EARNINGS_SRC)
    if PANEL_C == 'jfr':
        for _old, _new in [
            ('_fname_eb = "fig_cross_model_combined_entry_bias.png"', '_fname_eb = "fig_cross_model_combined_entry_bias_jfr.png"'),
            ('ax_eb_D.set_title("Δ Mean UE Duration by Worker Type",', 'ax_eb_D.set_title("Δ Mean UE Duration\\nby Worker Type",'),
            ('"Preferential Hiring Policy: Entry-Level Workers"', '"Preferential Hiring Policy: New Entrants"'),
        ]:
            assert src.count(_old) == 1, _old
            src = src.replace(_old, _new)
    _old = "os.path.join(OUTPUT_PATH, _fname_eb"
    assert src.count(_old) == 2, _old
    return src.replace(_old, "os.path.join(EB_DIR, _fname_eb")


def _figure_source():
    return "\n".join([
        _SETUP_SRC.replace("SAVE_FIG5_TO_DROPBOX = True", "SAVE_FIG5_TO_DROPBOX = False"),
        _between(EPE_SRC, "occ_wage_info = STATIC_OCC_DF[['Occupation', 'wage_mu']].copy()", "MAX_DUR    = 36"),
        _between(EPE_SRC, "def _ltuer_mean(occ_df):", "_ltuer_base5 = _ltuer_mean"),
        _between(EPE_SRC, "def _ltu_stock_and_rate(", "fig_ld, (ax_ld1"),
        _section_5b_helpers(),
        f"_CM_SPECS_NO_CYC = {{k: _CM_SPECS_NO_CYC[k] for k in {MODEL_ORDER!r}}}",
        _defs('_cm_mean_match_rate', '_match_rate_per_sim', '_newentrant_stock_and_rate'),
        _LOADER_SRC,
        _section_5e(),
    ])


def _fig_globals():
    return {'__file__': EPE_PATH, '__name__': '__main__', 'EB_DIR': EB_DIR, '_all_pkls': _all_pkls,
            'LE_DISPLAY_MIN_YEAR': LE_DISPLAY_MIN_YEAR, 'MIN_COHORT_SHARE': MIN_COHORT_SHARE}


def main(figure_only=False):
    global _setup_ns
    os.makedirs(EB_DIR, exist_ok=True)

    if not figure_only:
        print("=== Stage 1: entry-bias and baseline runs (hire() fix, 7% OTJ) ===", flush=True)
        exec(compile(_SETUP_SRC, EPE_PATH, 'exec'), _setup_ns)
        _todo = [mc for mc in EB_MODELS if not all(os.path.exists(p) for p in _all_pkls(mc))]
        if _todo:
            with mp.get_context('fork').Pool(min(N_WORKERS, len(_todo))) as _pool:
                for _mc in _pool.imap_unordered(_run_model, _todo):
                    print(f"  ✓ {_mc} done", flush=True)
        del _setup_ns

    missing = [p for mc in EB_MODELS for p in _all_pkls(mc) if not os.path.exists(p)]
    assert not missing, f"Stage 1 incomplete, missing: {missing}"

    print("\n=== Stage 2: fig_cross_model_combined_entry_bias ===", flush=True)
    exec(compile(_figure_source(), EPE_PATH, 'exec'), _fig_globals())
    print(f"\n✓ Done. Output written to {EB_DIR}")


if __name__ == "__main__":
    main(figure_only="--figure-only" in sys.argv)
