# %%
import pandas as pd
import pyfixest as pf
import numpy as np
import os
import pickle

import cloudpickle
import joblib

def save_model(obj, path):
    """Try cloudpickle/joblib/pickle in order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if cloudpickle is not None:
        try:
            with open(path, "wb") as f:
                cloudpickle.dump(obj, f)
            return
        except Exception:
            pass
    if joblib is not None:
        try:
            joblib.dump(obj, path, compress=3)
            return
        except Exception:
            pass
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_model(path):
    """Load serialized object; try cloudpickle first then joblib then pickle."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if cloudpickle is not None:
        try:
            with open(path, "rb") as f:
                return cloudpickle.load(f)
        except Exception:
            pass
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


# --- Control flags ---
# Set new_models=True to re-run all regressions and overwrite cache.
# Set new_models=False to load previously saved results from cache_prefix.
new_models = True
server = True
if server:
    print("Running on server: setting new_models = True to fit fresh models.")
    new_models = True
    cache_prefix = 'reg_output/'
    df = pd.read_csv("lightcast_master.csv")
else:
    print("Running locally: set new_models = False to load models from cache.")
    new_models = False  # <-- uncomment to load from cache when running locally
    cache_prefix = '../data/lightcast_vacancies/reg_output/'
    df = pd.read_csv("../data/lightcast_vacancies/lightcast_master.csv")

os.makedirs(cache_prefix, exist_ok=True)


# %% [markdown]
# # Supply Side Wage Adjustment Mechanism Exploration
# 
# As a reminder, we have access to 24 million job postings between 2010-2024. We are interested in extracting a single parameter on which to ground a wage adjustment rate to vacancy duration. Below, I outline the regression results of a preliminary attempt. 
# 
# I double-checked the full dataset to see whether we can actually extract a panel of vacancies themselves, but I am now fairly certain that this is not possible unless we try to link vacancies with different IDs that we suspect migth be identical. Additionally, Lightcast de-duplicates the job openings themselves. Though I am not sure whether this applies month-by-month, I think it partly explains why we cannot link individual vacancy IDs over time. 
# 
# Therefore, after some discussion with Stefi, I have tried the following three options with variations in the estimation strategy listed below each number:
# 
# 1. Estimate a *cross-sectional* regression on a dataset where observations are individual vacancies. 
#     - Include/exclude year-month fixed effects.
#     - Down-weight longer vacancy durations due to potential selection effects.
# 2. Estimate a *panel* regression in which we link *firm-level* occupational wage offers over time, regressing them on *firm-level* occupational vacancy durations.
#     - Vary SOC-2, SOC-3, SOC-4, SOC-5 occupational categories.
#     - First-differences vs. levels.
# 3. Estimate a *panel* regression in which we link *industry-level* occupational wage offers over time, regressing them on *industry-level* occupational vacancy durations. 
#     - Vary SOC-2, SOC-3, SOC-4, SOC-5 occupational categories.
#     - Vary NAICS-2, 3, 4, 5 industrial categories.
#     - First-differences vs. levels.

# %%
#df = df[df["year"] <= 2018].copy()
df["log_wage"] = np.log(df["salary"])
df["ym"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)

# Duration in days, weeks, months
df["duration_days"]   = df["duration"]
df["duration_weeks"]  = df["duration"] / 7
df["duration_months"] = df["duration"] / 30.4

# Inverse duration weights (length-bias correction, based on days)
df["idw"] = 1 / df["duration_days"]

print(f"Rows before cleaning: {len(df):,}")

# Drop invalid rows
df = df[
    df["log_wage"].notna() &
    df["duration"].notna() &
    df["company"].notna() &
    (df["duration"] > 0) &
    np.isfinite(df["log_wage"])
].copy()

for col in ["company", "soc_3", "naics2", "msa", "ym"]:
    df[col] = df[col].astype("category")
    
print(f"Rows after cleaning: {len(df):,}")


# %% [markdown]
# ## Econometric Specifications
# 
# ### 1. Individual Vacancy-Level Regression
# 
# As a baseline, we estimate the following log-linear regression at the 
# individual vacancy level:
# 
# 
# $\ln w_{i,j,s,l,t} = \beta \cdot \text{dur}_{i,j,s,l,t} + \alpha_j + \alpha_s + \alpha_l + \alpha_t + \varepsilon_{i,j,s,l,t}$
# 
# where $\ln w_{i,j,s,l,t}$ is the log of the posted salary for vacancy 
# $i$ posted by firm $j$ in occupation $s$, location $l$ (MSA), at 
# year-month $t$. $\text{dur}_{i,j,s,l,t}$ is vacancy duration, measured either in days, weeks, and months. The coefficient of interest 
# $\beta$ captures the conditional association between duration and posted 
# wages within firms, occupations, locations, and time periods. $\alpha_j$, 
# $\alpha_s$, $\alpha_l$, and $\alpha_t$ denote firm, occupation, location, 
# and year-month fixed effects respectively. Standard errors are clustered 
# at the firm level. We also attempt excluding the time fixed effect as the wage response is likely aggregate.
# 
# To address length-biased sampling, whereby longer-duration vacancies are 
# mechanically over-represented in any cross-sectional scrape of job 
# postings, we re-estimate this specification with 
# inverse-duration weights $\omega_i = 1/\text{dur}_i$, which downweight 
# long-duration observations proportionally.
# 

# %%
# ===========================================================================
# PART 1: Individual vacancy-level regressions
# ===========================================================================
# new_models is inherited from the setup cell above.
# On a fresh server run (new_models=True) this will fit and save.
# Locally, set new_models=False in the setup cell to load from cache.

cs_cache = f'{cache_prefix}vacancy_cross_section_results.cpkl'

if new_models:
    fit_weeks = pf.feols(
        "log_wage ~ duration_weeks | company + soc_3 + naics2 + msa + ym",
        data=df, vcov={"CRV1": "company"})

    fit_weeks_idw = pf.feols(
        "log_wage ~ duration_weeks | company + soc_3 + naics2 + msa + ym",
        data=df, weights="idw", vcov={"CRV1": "company"})

    fit_weeks_no_ym = pf.feols(
        "log_wage ~ duration_weeks | company + soc_3 + naics2 + msa",
        data=df, vcov={"CRV1": "company"})

    fit_weeks_idw_no_ym = pf.feols(
        "log_wage ~ duration_weeks | company + soc_3 + naics2 + msa",
        data=df, weights="idw", vcov={"CRV1": "company"})

    fit_months = pf.feols(
        "log_wage ~ duration_months | company + soc_3 + naics2 + msa + ym",
        data=df, vcov={"CRV1": "company"})

    fit_months_idw = pf.feols(
        "log_wage ~ duration_months | company + soc_3 + naics2 + msa + ym",
        data=df, weights="idw", vcov={"CRV1": "company"})

    fit_months_no_ym = pf.feols(
        "log_wage ~ duration_months | company + soc_3 + naics2 + msa",
        data=df, vcov={"CRV1": "company"})

    fit_months_idw_no_ym = pf.feols(
        "log_wage ~ duration_months | company + soc_3 + naics2 + msa",
        data=df, weights="idw", vcov={"CRV1": "company"})

    all_mods = [
        fit_weeks, fit_weeks_idw, fit_weeks_no_ym, fit_weeks_idw_no_ym,
        fit_months, fit_months_idw, fit_months_no_ym, fit_months_idw_no_ym,
    ]
    save_model(all_mods, cs_cache)
    print(f"Saved: {cs_cache}")
else:
    all_mods = load_model(cs_cache)
    print(f"Loaded: {cs_cache}")

# %%
print("\n=== Individual Vacancy-Level Results ===")
pf.etable( # fit_days, fit_days_idw, fit_days_no_ym, #fit_days_idw_no_ym,
    all_mods,
    #[fit_weeks, fit_weeks_idw, fit_weeks_no_ym, fit_weeks_idw_no_ym, fit_months, fit_months_idw, fit_months_no_ym, fit_months_idw_no_ym],  
    labels={
        #"duration_days":   "Duration (days)",
        "log_wage": "(log) Wage",
        "duration_weeks":  "Duration (weeks)",
        "duration_months": "Duration (months)",
        "ym": "Time (Year-Month)",
        "company": "Firm",
        "msa": "Location (MSA)",
        "soc_3": "Occ. (SOC3)",
        "naics2": "Ind. (NAICS2)"
    },
    model_heads=[
        #"Days", "Days (IDW)", "Days", "Days (IDW)",
        "Weeks", "Weeks (IDW)", "Weeks", "Weeks (IDW)",
        "Months", "Months (IDW)", "Months", "Months (IDW)"
    ]
)

# %%
# ===========================================================================
# PART 2: Panel helper functions
# ===========================================================================

# Duration units used in all panel regressions
duration_vars = {
    "weeks":  "mean_dur_weeks",
    "months": "mean_dur_months",
}

def build_panel(df, group_vars, soc_col, min_cell_size):
    """
    Collapse df to a panel grouped by group_vars + soc_col + ym.
    Drops sparse cells (< min_cell_size vacancies) and singletons
    (cells observed in fewer than 2 months).
    Returns the collapsed panel and the cell identifier column name.
    """
    cell_id = "__".join(group_vars) + f"_{soc_col}"

    # observed=True: skip empty category cross-products (avoids OOM on high-cardinality columns)
    panel = (
        df.groupby(group_vars + [soc_col, "ym", "year", "month"], observed=True)
        .agg(
            mean_log_wage     = ("log_wage",        "mean"),
            mean_dur_days     = ("duration_days",   "mean"),
            mean_dur_weeks    = ("duration_weeks",  "mean"),
            mean_dur_months   = ("duration_months", "mean"),
            median_log_wage   = ("log_wage",        "median"),
            median_dur_days   = ("duration_days",   "median"),
            median_dur_weeks  = ("duration_weeks",  "median"),
            median_dur_months = ("duration_months", "median"),
            n_vacancies       = ("log_wage",        "count"),
        )
        .reset_index()
    )

    # Drop sparse cells
    panel = panel[panel["n_vacancies"] >= min_cell_size].copy()

    # Composite cell identifier
    panel[cell_id] = panel[group_vars + [soc_col]].astype(str).agg("_".join, axis=1)

    # Drop singleton cells (observed in only one month)
    period_counts = panel.groupby(cell_id, observed=True)["ym"].nunique()
    valid_cells   = period_counts[period_counts >= 2].index
    panel         = panel[panel[cell_id].isin(valid_cells)].copy()

    return panel, cell_id


def run_panel_models(panel, cell_var, duration_vars):
    """Run mean and cell-size-weighted models for each duration unit."""
    models = {}
    for unit, dur_col in duration_vars.items():
        models[f"Mean ({unit})"] = pf.feols(
            f"mean_log_wage ~ {dur_col} | {cell_var} + ym",
            data=panel, vcov={"CRV1": cell_var})
        models[f"Weighted ({unit})"] = pf.feols(
            f"mean_log_wage ~ {dur_col} | {cell_var} + ym",
            data=panel, weights="n_vacancies",
            vcov={"CRV1": cell_var})
    return models

# %% [markdown]
# ---
# 
# ### 2. Company--Occupation Panel
# 
# The individual vacancy-level specification compares *different* vacancies 
# within the same firm at a point in time, but cannot identify whether firms 
# *respond* to prolonged vacancy duration by adjusting posted wages over time. 
# To address this, we collapse the data to a firm-occupation-month panel and 
# estimate:
# 
# 
# $\bar{w}_{j,s,t} = \beta \cdot \overline{\text{dur}}_{j,s,t} + \alpha_{j,s} + \alpha_t + \varepsilon_{j,s,t}$
# 
# where $\bar{w}_{j,s,t}$ is the mean log posted wage across all vacancies 
# posted by firm $j$ in occupation $s$ at year-month $t$, and 
# $\overline{\text{dur}}_{j,s,t}$ is the corresponding mean vacancy duration. 
# The firm-occupation fixed effect $\alpha_{j,s}$ absorbs all time-invariant 
# heterogeneity at the firm-occupation level - including the structural 
# difficulty of filling a given role at a given firm and any firm-specific 
# compensation norms. Identification therefore comes from within-cell 
# variation over time: $\beta$ captures whether firm $j$ posts higher wages 
# for occupation $s$ in months when its vacancies in that occupation take 
# longer to fill than usual.
# 
# Cells with fewer than five vacancies in a given month are dropped prior to 
# estimation. Cells observed in only a single month are also dropped as they 
# provide no within-cell variation. We estimate this specification
# using mean duration entered in days, weeks, and months, and additionally 
# with observations weighted by cell size $n_{j,s,t}$ to account for 
# differences in the precision of cell-level means. Standard errors are 
# clustered at the firm-occupation cell level. The specification is estimated 
# separately for occupational aggregation at the 2-, 3-, 4-, and 5-digit SOC 
# level, providing a robustness check on the role of occupational 
# classification granularity.

# %% [markdown]
# ### Company - Occupation Panel

# %%
# ===========================================================================
# PART 3: Company-Occupation panels — loop over SOC levels
# ===========================================================================

MIN_CELL_SIZE = 5
SOC_LEVELS    = ["soc_2", "soc_3", "soc_4", "soc_5"]

print("\n" + "=" * 70)
print("COMPANY-OCCUPATION PANELS")
print("=" * 70)

co_results = {}   # fitted models keyed by soc level
co_panels  = {}   # collapsed panels keyed by soc level (reused in summary table)

for soc in SOC_LEVELS:
    cache_path = f'{cache_prefix}firm_occ_results_{soc}.cpkl'

    if not new_models and os.path.exists(cache_path):
        co_results[soc] = load_model(cache_path)
        print(f"\n--- SOC level: {soc} | loaded from cache ---")
        continue

    print(f"\n--- SOC level: {soc} ---")
    panel, cell_id = build_panel(
        df,
        group_vars    = ["company"],
        soc_col       = soc,
        min_cell_size = MIN_CELL_SIZE,
    )
    print(f"  {len(panel):,} rows | {panel[cell_id].nunique():,} cells "
          f"| cell size p50: {panel['n_vacancies'].median():.0f}")

    models = run_panel_models(panel, cell_id, duration_vars)
    co_results[soc] = models
    co_panels[soc]  = (panel, cell_id)   # cache for summary table

    save_model(models, cache_path)
    print(f"  Saved: {cache_path}")

for soc in SOC_LEVELS:
    if soc not in co_results:
        continue
    print(f"\n--- SOC level: {soc} ---")
    models = co_results[soc]
    pf.etable(
        list(models.values()),
        labels={
            "mean_dur_weeks":  "Mean duration (weeks)",
            "mean_dur_months": "Mean duration (months)",
        },
        model_heads=list(models.keys()),
        title=f"Company-Occupation Panel | {soc.upper()}"
    )

# %% [markdown]
# ### Industry - Occupation Panel

# %% [markdown]
# 
# ### 3. Industry--Occupation Panel
# 
# To address potential sparsity in the firm--occupation panel arising from 
# firms that post relatively few vacancies in a given occupation, we 
# additionally estimate the model at the industry--occupation--month level:
# 
# \begin{equation}
# \bar{w}_{k,s,t} = \beta \cdot \overline{\text{dur}}_{k,s,t} + 
# \alpha_{k,s} + \alpha_t + \varepsilon_{k,s,t}
# \label{eq:io_panel}
# \end{equation}
# 
# where $k$ indexes industry, $\bar{w}_{k,s,t}$ is the mean log posted wage 
# for industry $k$, occupation $s$ at year-month $t$, and 
# $\overline{\text{dur}}_{k,s,t}$ is the corresponding mean vacancy duration. 
# The industry--occupation fixed effect $\alpha_{k,s}$ absorbs all 
# time-invariant differences in wages and vacancy duration across 
# industry--occupation pairs. This specification trades within-firm 
# identification for larger and more stable cells, at the cost of 
# averaging over heterogeneity across firms within the same industry.
# 
# Equation~(\ref{eq:io_panel}) is estimated across all combinations of 
# industry aggregation (2- through 6-digit NAICS) and occupational 
# aggregation (2- through 5-digit SOC), yielding a $5 \times 4$ grid of 
# specifications. This allows us to assess whether results are sensitive to 
# the level of industry and occupational disaggregation. The same cell-size 
# and singleton filters described above are applied prior to estimation, and 
# standard errors are clustered at the industry--occupation cell level.

# %%
# ===========================================================================
# PART 4: Industry-Occupation panels — loop over SOC x NAICS levels
# ===========================================================================

print("\n" + "=" * 70)
print("INDUSTRY-OCCUPATION PANELS")
print("=" * 70)

NAICS_LEVELS = ["naics2", "naics3", "naics4", "naics5", "naics6"]

io_results = {}   # keyed by (naics, soc)
io_panels  = {}   # collapsed panels keyed by (naics, soc) — reused in summary table

for naics in NAICS_LEVELS:
    for soc in SOC_LEVELS:
        key        = (naics, soc)
        cache_path = f'{cache_prefix}ind_occ_{naics}_{soc}.cpkl'

        if not new_models and os.path.exists(cache_path):
            io_results[key] = load_model(cache_path)
            print(f"\n--- {naics} x {soc} | loaded from cache ---")
            continue

        print(f"\n--- NAICS level: {naics} | SOC level: {soc} ---")
        panel, cell_id = build_panel(
            df,
            group_vars    = [naics],
            soc_col       = soc,
            min_cell_size = MIN_CELL_SIZE,
        )
        print(f"  {len(panel):,} rows | {panel[cell_id].nunique():,} cells "
              f"| cell size p50: {panel['n_vacancies'].median():.0f}")

        if panel[cell_id].nunique() < 10:
            print("  Skipping — fewer than 10 cells after filtering.")
            continue

        models = run_panel_models(panel, cell_id, duration_vars)
        io_results[key] = models
        io_panels[key]  = (panel, cell_id)   # cache for summary table

        save_model(models, cache_path)
        print(f"  Saved: {cache_path}")

for naics in NAICS_LEVELS:
    for soc in SOC_LEVELS:
        key = (naics, soc)
        if key not in io_results:
            continue
        models = io_results[key]
        print(f"\n--- NAICS level: {naics} | SOC level: {soc} ---")
        pf.etable(
            list(models.values()),
            labels={
                "mean_dur_weeks":  "Mean duration (weeks)",
                "mean_dur_months": "Mean duration (months)",
            },
            model_heads=list(models.keys()),
            title=f"Industry-Occupation Panel | {naics.upper()} x {soc.upper()}"
        )

# %%
# ===========================================================================
# PART 5: Cross-SOC x NAICS summary table (weeks coefficient, mean model)
# ===========================================================================

print("\n=== Cross-NAICS x SOC comparison: Mean duration (weeks), Mean model ===")
print(f"{'NAICS':<10} {'SOC':<8} {'Coef':>10} {'SE':>10} {'p-value':>10} {'N':>12}")
print("-" * 62)

for (naics, soc), models in io_results.items():
    m    = models["Mean (weeks)"]
    coef = m.coef()["mean_dur_weeks"]
    se   = m.se()["mean_dur_weeks"]
    pval = m.pvalue()["mean_dur_weeks"]
    nobs = m.nobs
    print(f"{naics:<10} {soc:<8} {coef:>10.6f} {se:>10.6f} {pval:>10.4f} {nobs:>12,}")

# %%
# ===========================================================================
# Summary table across all panel specifications
# ===========================================================================
# Uses panels cached during the regression loops (co_panels / io_panels).
# If a panel wasn't cached (e.g. loaded from cache with new_models=False),
# it is rebuilt from the raw data on demand.

rows = []

# ---------------------------------------------------------------------------
# Company-Occupation panels
# ---------------------------------------------------------------------------
for soc in SOC_LEVELS:
    if soc not in co_results:
        continue

    if soc in co_panels:
        panel, cell_id = co_panels[soc]
    else:
        panel, cell_id = build_panel(df, group_vars=["company"],
                                     soc_col=soc, min_cell_size=MIN_CELL_SIZE)

    m = co_results[soc]["Mean (weeks)"]
    rows.append({
        "Panel type":         "Company-Occ",
        "NAICS level":        "—",
        "SOC level":          soc.upper(),
        "Cells":              panel[cell_id].nunique(),
        "Observations":       len(panel),
        "Mean duration (w)":  panel["mean_dur_weeks"].mean(),
        "Median duration (w)":panel["mean_dur_weeks"].median(),
        "Mean log wage":      panel["mean_log_wage"].mean(),
        "Median log wage":    panel["mean_log_wage"].median(),
        "Coef (weeks)":       m.coef()["mean_dur_weeks"],
        "SE":                 m.se()["mean_dur_weeks"],
        "p-value":            m.pvalue()["mean_dur_weeks"],
    })

# ---------------------------------------------------------------------------
# Industry-Occupation panels
# ---------------------------------------------------------------------------
for (naics, soc), models in io_results.items():
    key = (naics, soc)

    if key in io_panels:
        panel, cell_id = io_panels[key]
    else:
        panel, cell_id = build_panel(df, group_vars=[naics],
                                     soc_col=soc, min_cell_size=MIN_CELL_SIZE)

    m = models["Mean (weeks)"]
    rows.append({
        "Panel type":         "Industry-Occ",
        "NAICS level":        naics.upper(),
        "SOC level":          soc.upper(),
        "Cells":              panel[cell_id].nunique(),
        "Observations":       len(panel),
        "Mean duration (w)":  panel["mean_dur_weeks"].mean(),
        "Median duration (w)":panel["mean_dur_weeks"].median(),
        "Mean log wage":      panel["mean_log_wage"].mean(),
        "Median log wage":    panel["mean_log_wage"].median(),
        "Coef (weeks)":       m.coef()["mean_dur_weeks"],
        "SE":                 m.se()["mean_dur_weeks"],
        "p-value":            m.pvalue()["mean_dur_weeks"],
    })

# ---------------------------------------------------------------------------
# Assemble display version
# ---------------------------------------------------------------------------
summary = pd.DataFrame(rows).sort_values(
    ["Panel type", "NAICS level", "SOC level"], ascending=[False, True, True]
).reset_index(drop=True)

summary_display = summary.copy()
for col in ["Mean duration (w)", "Median duration (w)"]:
    summary_display[col] = summary_display[col].round(1)
for col in ["Mean log wage", "Median log wage"]:
    summary_display[col] = summary_display[col].round(3)
summary_display["Coef (weeks)"] = summary_display["Coef (weeks)"].round(6)
summary_display["SE"]           = summary_display["SE"].round(6)
summary_display["p-value"]      = summary_display["p-value"].round(4)
summary_display["Observations"] = summary_display["Observations"].apply(lambda x: f"{x:,}")
summary_display["Cells"]        = summary_display["Cells"].apply(lambda x: f"{x:,}")

print("\n=== Full Model Summary Table ===")
print(summary_display.to_string(index=False))

# ---------------------------------------------------------------------------
# Export — always save CSV (editable locally) and LaTeX
# ---------------------------------------------------------------------------
csv_path = f'{cache_prefix}model_summary_table.csv'
tex_path = f'{cache_prefix}model_summary_table.tex'

summary.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

latex = summary.to_latex(
    index        = False,
    float_format = "%.4f",
    caption      = (
        "Summary of panel specifications. Each row corresponds to one "
        "panel regression of mean log posted wage on mean vacancy duration "
        "(weeks), with cell and year-month fixed effects. "
        "Duration and wage statistics are computed at the cell-month level "
        "after applying the minimum cell size filter of "
        f"{MIN_CELL_SIZE} vacancies and dropping singleton cells."
    ),
    label         = "tab:model_summary",
    column_format = "llrrrrrrrrr",
)
with open(tex_path, "w") as f:
    f.write(latex)
print(f"Saved: {tex_path}")


