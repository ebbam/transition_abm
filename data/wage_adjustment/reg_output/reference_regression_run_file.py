# %%
import pandas as pd
import pyfixest as pf
import numpy as np
import os
import pickle
import cloudpickle
import joblib


# ===========================================================================
# SERIALISATION HELPERS
# ===========================================================================

def save_model(obj, path):
    """Try cloudpickle / joblib / pickle in order."""
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
    """Load serialised object; try cloudpickle first, then joblib, then pickle."""
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


# ===========================================================================
# CONTROL FLAGS  ← only section you should need to edit between runs
# ===========================================================================
# new_models : True  → re-fit all models and overwrite cache
#              False → load previously saved results from cache (local use)
# server     : True  → assume HPC / server environment (forces new_models=True)
#              False → local environment
#
# NOTE: fd is no longer set here. The script loops over fd in [False, True]
# and runs both the levels and first-differences estimators automatically.

server = True
run_cross_section = False
save_models = False

if server:
    print("Running on server: forcing new_models = True.")
    new_models   = True
    cache_prefix = "reg_output/"
    df = pd.read_csv("lightcast_master.csv")
else:
    print("Running locally: set new_models = False to load from cache.")
    new_models   = False
    cache_prefix = "../data/lightcast_vacancies/reg_output/"
    df = pd.read_csv("../data/lightcast_vacancies/lightcast_master.csv")

os.makedirs(cache_prefix, exist_ok=True)


# %% [markdown]
# # Supply Side Wage Adjustment Mechanism Exploration
#
# As a reminder, we have access to 24 million job postings between 2010-2024.
# We are interested in extracting a single parameter on which to ground a wage
# adjustment rate to vacancy duration. Below, I outline the regression results
# of a preliminary attempt.
#
# I double-checked the full dataset to see whether we can actually extract a
# panel of vacancies themselves, but I am now fairly certain that this is not
# possible unless we try to link vacancies with different IDs that we suspect
# might be identical. Additionally, Lightcast de-duplicates the job openings
# themselves. Though I am not sure whether this applies month-by-month, I
# think it partly explains why we cannot link individual vacancy IDs over time.
#
# Therefore, after some discussion with Stefi, I have tried the following three
# options with variations in the estimation strategy listed below each number:
#
# 1. Estimate a *cross-sectional* regression on a dataset where observations
#    are individual vacancies.
#    - Include/exclude year-month fixed effects.
#    - Down-weight longer vacancy durations due to potential selection effects.
# 2. Estimate a *panel* regression in which we link *firm-level* occupational
#    wage offers over time, regressing them on *firm-level* occupational
#    vacancy durations.
#    - Vary SOC-2, SOC-3, SOC-4, SOC-5 occupational categories.
#    - First-differences vs. levels.
# 3. Estimate a *panel* regression in which we link *industry-level*
#    occupational wage offers over time, regressing them on *industry-level*
#    occupational vacancy durations.
#    - Vary SOC-2, SOC-3, SOC-4, SOC-5 occupational categories.
#    - Vary NAICS-2, 3, 4, 5 industrial categories.
#    - First-differences vs. levels.


# ===========================================================================
# DATA PREPARATION  (single cleaning block)
# ===========================================================================
print("Preparing data...")
# %%
df["log_wage"] = np.log(df["salary"])
df["ym"]       = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)

# Duration in multiple units
df["duration_days"]   = df["duration"]
df["duration_weeks"]  = df["duration"] / 7
df["duration_months"] = df["duration"] / 30.4

# Inverse-duration weights for length-bias correction
df["idw"] = 1 / df["duration_days"]

print(f"Rows before cleaning: {len(df):,}")

df = df[
    df["log_wage"].notna()  &
    df["duration"].notna()  &
    df["company"].notna()   &
    (df["duration"] > 0)    &
    np.isfinite(df["log_wage"])
].copy()

for col in ["company", "soc_3", "naics2", "msa", "ym"]:
    df[col] = df[col].astype("category")

df["month"] = df["month"].astype("category")

print(f"Rows after cleaning:  {len(df):,}")


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
# year-month $t$. $\text{dur}_{i,j,s,l,t}$ is vacancy duration, measured in months. The coefficient of interest 
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


# ===========================================================================
# PART 1: Individual vacancy-level (cross-sectional) regressions
# ===========================================================================
# Months only. This section does not depend on the FD/levels choice and
# therefore runs once, before the estimator loop below.

if run_cross_section:
    print("Running cross-sectional regressions...")
    # %%
    cs_cache = f"{cache_prefix}vacancy_cross_section_results_months.cpkl"

    # Fixed-effect specifications: stepwise adding controls.
    # With year-month FE: occ+ind+loc → +ym → +firm
    ym_specs = [
        "soc_3 + naics2 + msa",
        "soc_3 + naics2 + msa + ym",
        "soc_3 + naics2 + msa + ym + company",
    ]
    # Without year-month FE: occ+ind+loc → +month → +firm → +year
    no_ym_specs = [
        "soc_3 + naics2 + msa",
        "soc_3 + naics2 + msa + month",
        "soc_3 + naics2 + msa + month + company",
        "soc_3 + naics2 + msa + month + company + year",
    ]

    print("Running individual vacancy-level regressions...")

    if new_models:
        individual_models = []
        model_names       = []

        for spec in ym_specs:
            individual_models.append(pf.feols(
                f"log_wage ~ duration_months | {spec}",
                data=df, vcov={"CRV1": "company"}))
            model_names.append(f"Base | {spec.split('+')[-1].strip()}")

        for spec in ym_specs:
            individual_models.append(pf.feols(
                f"log_wage ~ duration_months | {spec}",
                data=df, weights="idw", vcov={"CRV1": "company"}))
            model_names.append(f"IDW | {spec.split('+')[-1].strip()}")

        for spec in no_ym_specs:
            individual_models.append(pf.feols(
                f"log_wage ~ duration_months | {spec}",
                data=df, vcov={"CRV1": "company"}))
            model_names.append(f"No-YM | {spec.split('+')[-1].strip()}")

        for spec in no_ym_specs:
            individual_models.append(pf.feols(
                f"log_wage ~ duration_months | {spec}",
                data=df, weights="idw", vcov={"CRV1": "company"}))
            model_names.append(f"No-YM IDW | {spec.split('+')[-1].strip()}")

        if save_models:
            save_model((individual_models, model_names), cs_cache)
            print(f"Saved: {cs_cache}")
    else:
        individual_models, model_names = load_model(cs_cache)
        print(f"Loaded: {cs_cache}")

    # all_mods is a flat list of individual Feols objects — no .to_list() needed
    all_mods = individual_models

    # --- LaTeX table ---
    tex = pf.etable(all_mods, type="tex",
                    labels={"duration_months": "Duration (months)"},
                    model_heads=model_names)
    with open(f"{cache_prefix}table_cross_section_months.tex", "w") as f:
        f.write(tex)

    # --- CSV export (models are already individual; no flattening needed) ---
    for name, m in zip(model_names, individual_models):
        for var in m.coef().index:
            rows.append({
                "model": name, "variable": var,
                "coef": m.coef()[var], "se": m.se()[var],
                "pvalue": m.pvalue()[var], "nobs": m._N,
            })
    pd.DataFrame(rows).to_csv(
        f"{cache_prefix}coefs_cross_section_months.csv", index=False)

    tidy_rows = []
    for name, m in zip(model_names, individual_models):
        tidy = m.tidy()
        tidy["model"] = name
        tidy_rows.append(tidy)
    pd.concat(tidy_rows).to_csv(
        f"{cache_prefix}coefs_cross_section_months_tidy.csv", index=True)

    # --- Console table ---
    print("\n=== Individual Vacancy-Level Results ===")
    pf.etable(
        all_mods,
        model_heads=model_names,
        labels={
            "log_wage":        "(log) Wage",
            "duration_months": "Duration (months)",
            "ym":              "Time (Year-Month)",
            "company":         "Firm",
            "msa":             "Location (MSA)",
            "soc_3":           "Occ. (SOC3)",
            "naics2":          "Ind. (NAICS2)",
        },
    )

    del all_mods


# ===========================================================================
# PART 2: Panel helper functions
# ===========================================================================

# %%
# Months only for all panel regressions.
duration_vars = {
    "months": "mean_dur_months",
}

# Labels for etable — cover both levels and FD variable names, months only
dur_labels = {
    "d_mean_dur_months": "Δ Mean duration (months)",
    "mean_dur_months":   "Mean duration (months)",
}


def build_panel(df, group_vars, soc_col, min_cell_size):
    """
    Collapse df to a panel grouped by group_vars + soc_col + ym.

    Drops sparse cells (< min_cell_size vacancies) and singleton cells
    (observed in fewer than 2 months).
    Returns the collapsed panel and the composite cell identifier column name.
    """
    cell_id = "__".join(group_vars) + f"_{soc_col}"

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

    # Drop singleton cells (no within-cell variation)
    period_counts = panel.groupby(cell_id, observed=True)["ym"].nunique()
    valid_cells   = period_counts[period_counts >= 2].index
    panel         = panel[panel[cell_id].isin(valid_cells)].copy()

    return panel, cell_id


def apply_first_differences(panel, cell_id):
    """
    Sort panel by cell and time, then first-difference mean log wage and all
    mean duration columns within each cell.

    Drops the first observation per cell (NaN after differencing) and any cell
    left with fewer than 2 FD observations.
    """
    panel = panel.sort_values([cell_id, "ym"]).copy()

    for col in ["mean_log_wage", "mean_dur_weeks", "mean_dur_months"]:
        panel[f"d_{col}"] = panel.groupby(cell_id, observed=True)[col].diff()

    # Drop first observation per cell
    panel = panel.dropna(subset=["d_mean_log_wage", "d_mean_dur_months"]).copy()

    # Require at least 2 usable FD observations per cell
    fd_counts   = panel.groupby(cell_id, observed=True)["d_mean_log_wage"].count()
    valid_cells = fd_counts[fd_counts >= 2].index
    panel       = panel[panel[cell_id].isin(valid_cells)].copy()

    return panel


def run_panel_models(panel, cell_var, duration_vars, fd=False):
    """
    Fit mean and cell-size-weighted models for each duration unit.

    fd=False (levels):
        mean_log_wage ~ mean_dur | cell_var + ym
    fd=True (first differences):
        d_mean_log_wage ~ d_mean_dur | ym
        Cell FE drops out by construction; ym FE is retained.
    """
    models = {}
    for unit, dur_col in duration_vars.items():
        if fd:
            dep_var  = "d_mean_log_wage"
            expl_var = f"d_{dur_col}"
            formula  = f"{dep_var} ~ {expl_var} | ym"
        else:
            dep_var  = "mean_log_wage"
            expl_var = dur_col
            formula  = f"{dep_var} ~ {expl_var} | {cell_var} + ym"

        models[f"Mean ({unit})"] = pf.feols(
            formula, data=panel, vcov={"CRV1": cell_var})

        models[f"Weighted ({unit})"] = pf.feols(
            formula, data=panel, weights="n_vacancies",
            vcov={"CRV1": cell_var})

    return models


# ===========================================================================
# PRE-BUILD ALL PANELS ONCE
# build_panel runs a full groupby on the 3 GB raw dataframe. By building
# every panel here — before the estimator loop and before the robustness
# sections — we call it exactly 24 times instead of ~200+.
# df is then deleted to free ~3 GB of memory before any feols calls.
# ===========================================================================

MIN_CELL_SIZE = 5
SOC_LEVELS    = ["soc_2", "soc_3", "soc_4", "soc_5"]
NAICS_LEVELS  = ["naics2", "naics3", "naics4", "naics5", "naics6"]

print("\nPre-building all cell-level panels from raw data...")

raw_panels = {}   # keyed by (group_vars_tuple, soc_col) → (panel, cell_id)

for soc in SOC_LEVELS:
    key = (("company",), soc)
    print(f"  Building company × {soc}...")
    raw_panels[key] = build_panel(df, group_vars=["company"],
                                  soc_col=soc, min_cell_size=MIN_CELL_SIZE)

for naics in NAICS_LEVELS:
    for soc in SOC_LEVELS:
        key = ((naics,), soc)
        print(f"  Building {naics} × {soc}...")
        raw_panels[key] = build_panel(df, group_vars=[naics],
                                      soc_col=soc, min_cell_size=MIN_CELL_SIZE)

print("All panels built. Freeing raw data from memory...")
del df
print("Raw data freed.\n")


# ===========================================================================
# PARTS 3–6: Loop over both estimators (levels then first differences)
# ===========================================================================

for fd in [False, True]:
    print(f"Running FD: {fd}")
    # Derived variables that depend on which estimator we are running
    fd_suffix = "_fd" if fd else ""
    dur_key   = "d_mean_dur_months" if fd else "mean_dur_months"
    dur_lbl   = "Δ Mean duration (months)" if fd else "Mean duration (months)"
    wage_col  = "d_mean_log_wage"   if fd else "mean_log_wage"
    dur_col   = "d_mean_dur_months" if fd else "mean_dur_months"

    print("\n" + "#" * 70)
    print(f"# ESTIMATOR: {'FIRST DIFFERENCES' if fd else 'LEVELS'}")
    print("#" * 70)

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
    # using mean duration in months, and additionally with observations weighted 
    # by cell size $n_{j,s,t}$ to account for differences in the precision of 
    # cell-level means. Standard errors are clustered at the firm-occupation cell 
    # level. The specification is estimated separately for occupational aggregation 
    # at the 2-, 3-, 4-, and 5-digit SOC level, providing a robustness check on 
    # the role of occupational classification granularity.

    # =========================================================================
    # PART 3: Company × Occupation panels — loop over SOC levels
    # =========================================================================

    print("Running company-occ panel regressions...")

    MIN_CELL_SIZE = 5
    SOC_LEVELS    = ["soc_2", "soc_3", "soc_4", "soc_5"]

    print("\n" + "=" * 70)
    print(f"COMPANY-OCCUPATION PANELS {'(FIRST DIFFERENCES)' if fd else '(LEVELS)'}")
    print("=" * 70)

    co_results = {}
    co_panels  = {}

    for soc in SOC_LEVELS:
        cache_path = f"{cache_prefix}firm_occ_results_{soc}{fd_suffix}.cpkl"

        if not new_models and os.path.exists(cache_path):
            co_results[soc] = load_model(cache_path)
            print(f"\n--- SOC level: {soc} | loaded from cache ---")
            continue

        print(f"\n--- SOC level: {soc} ---")
        panel, cell_id = raw_panels[(("company",), soc)]
        panel = panel.copy()   # never mutate the cached panel

        if fd:
            panel = apply_first_differences(panel, cell_id)

        print(f"  {len(panel):,} rows | {panel[cell_id].nunique():,} cells "
              f"| cell size p50: {panel['n_vacancies'].median():.0f}")

        models = run_panel_models(panel, cell_id, duration_vars, fd=fd)
        co_results[soc] = models
        co_panels[soc]  = (panel, cell_id)

        if save_models:
            save_model(models, cache_path)
            print(f"  Saved: {cache_path}")

    for soc in SOC_LEVELS:
        if soc not in co_results:
            continue
        cache_path = f"{cache_prefix}firm_occ_results_{soc}{fd_suffix}.cpkl"

        print(f"\n--- SOC level: {soc} ---")
        models = co_results[soc]
        pf.etable(
            list(models.values()),
            labels      = dur_labels,
            model_heads = list(models.keys()),
            title       = f"Company-Occupation Panel {'FD' if fd else 'Levels'} | {soc.upper()}",
        )

        tidy_rows = []
        for model_name, m in models.items():
            t = m.tidy().reset_index()
            t["model"] = model_name
            tidy_rows.append(t)
        pd.concat(tidy_rows).to_csv(
            cache_path.replace(".cpkl", "_tidy.csv"), index=False)

        tex = pf.etable(list(models.values()), type="tex",
                        model_heads=list(models.keys()))
        with open(cache_path.replace(".cpkl", ".tex"), "w") as f:
            f.write(tex)

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

    # =========================================================================
    # PART 4: Industry × Occupation panels — loop over SOC x NAICS levels
    # =========================================================================

    NAICS_LEVELS = ["naics2", "naics3", "naics4", "naics5", "naics6"]

    print("\n" + "=" * 70)
    print(f"INDUSTRY-OCCUPATION PANELS {'(FIRST DIFFERENCES)' if fd else '(LEVELS)'}")
    print("=" * 70)

    io_results = {}
    io_panels  = {}

    print("Running industry-occupation panel regressions...")

    for naics in NAICS_LEVELS:
        for soc in SOC_LEVELS:
            key        = (naics, soc)
            cache_path = f"{cache_prefix}ind_occ_{naics}_{soc}{fd_suffix}.cpkl"

            if not new_models and os.path.exists(cache_path):
                io_results[key] = load_model(cache_path)
                print(f"\n--- {naics} x {soc} | loaded from cache ---")
                continue

            print(f"\n--- NAICS level: {naics} | SOC level: {soc} ---")
            panel, cell_id = raw_panels[((naics,), soc)]
            panel = panel.copy()   # never mutate the cached panel

            if fd:
                panel = apply_first_differences(panel, cell_id)

            print(f"  {len(panel):,} rows | {panel[cell_id].nunique():,} cells "
                  f"| cell size p50: {panel['n_vacancies'].median():.0f}")

            if panel[cell_id].nunique() < 10:
                print("  Skipping — fewer than 10 cells after filtering.")
                continue

            models = run_panel_models(panel, cell_id, duration_vars, fd=fd)
            io_results[key] = models
            io_panels[key]  = (panel, cell_id)

            if save_models:
                save_model(models, cache_path)
                print(f"  Saved: {cache_path}")

    for naics in NAICS_LEVELS:
        for soc in SOC_LEVELS:
            key = (naics, soc)
            if key not in io_results:
                continue
            cache_path = f"{cache_prefix}ind_occ_{naics}_{soc}{fd_suffix}.cpkl"

            models = io_results[key]
            print(f"\n--- NAICS level: {naics} | SOC level: {soc} ---")
            pf.etable(
                list(models.values()),
                labels      = dur_labels,
                model_heads = list(models.keys()),
                title       = (f"Industry-Occupation Panel {'FD' if fd else 'Levels'} | "
                               f"{naics.upper()} x {soc.upper()}"),
            )

            tidy_rows = []
            for model_name, m in models.items():
                t = m.tidy().reset_index()
                t["model"] = model_name
                tidy_rows.append(t)
            pd.concat(tidy_rows).to_csv(
                cache_path.replace(".cpkl", "_tidy.csv"), index=False)

            tex = pf.etable(list(models.values()), type="tex",
                            model_heads=list(models.keys()))
            with open(cache_path.replace(".cpkl", ".tex"), "w") as f:
                f.write(tex)

    # =========================================================================
    # PART 5: Cross-SOC × NAICS comparison table (months coefficient, mean model)
    # =========================================================================

    print(f"\n=== Cross-NAICS x SOC comparison: {dur_lbl}, Mean model ===")
    print(f"{'NAICS':<10} {'SOC':<8} {'Coef':>10} {'SE':>10} {'p-value':>10} {'N':>12}")
    print("-" * 62)

    for (naics, soc), models in io_results.items():
        m    = models["Mean (months)"]
        coef = m.coef()[dur_key]
        se   = m.se()[dur_key]
        pval = m.pvalue()[dur_key]
        nobs = m._N
        print(f"{naics:<10} {soc:<8} {coef:>10.6f} {se:>10.6f} {pval:>10.4f} {nobs:>12,}")

    # =========================================================================
    # PART 6: Full summary table across all panel specifications
    # =========================================================================

    rows = []

    # --- Company × Occupation panels ---
    for soc in SOC_LEVELS:
        if soc not in co_results:
            continue

        if soc in co_panels:
            panel, cell_id = co_panels[soc]
        else:
            panel, cell_id = raw_panels[(("company",), soc)]
            panel = panel.copy()
            if fd:
                panel = apply_first_differences(panel, cell_id)

        m = co_results[soc]["Mean (months)"]
        rows.append({
            "Panel type":          "Company-Occ",
            "Estimator":           "FD" if fd else "Levels",
            "NAICS level":         "—",
            "SOC level":           soc.upper(),
            "Cells":               panel[cell_id].nunique(),
            "Observations":        len(panel),
            "Mean duration (m)":   panel[dur_col].mean(),
            "Median duration (m)": panel[dur_col].median(),
            "Mean log wage":       panel[wage_col].mean(),
            "Median log wage":     panel[wage_col].median(),
            "Coef (months)":       m.coef()[dur_key],
            "SE":                  m.se()[dur_key],
            "p-value":             m.pvalue()[dur_key],
        })

    # --- Industry × Occupation panels ---
    for (naics, soc), models in io_results.items():
        key = (naics, soc)

        if key in io_panels:
            panel, cell_id = io_panels[key]
        else:
            panel, cell_id = raw_panels[((naics,), soc)]
            panel = panel.copy()
            if fd:
                panel = apply_first_differences(panel, cell_id)

        m = models["Mean (months)"]
        rows.append({
            "Panel type":          "Industry-Occ",
            "Estimator":           "FD" if fd else "Levels",
            "NAICS level":         naics.upper(),
            "SOC level":           soc.upper(),
            "Cells":               panel[cell_id].nunique(),
            "Observations":        len(panel),
            "Mean duration (m)":   panel[dur_col].mean(),
            "Median duration (m)": panel[dur_col].median(),
            "Mean log wage":       panel[wage_col].mean(),
            "Median log wage":     panel[wage_col].median(),
            "Coef (months)":       m.coef()[dur_key],
            "SE":                  m.se()[dur_key],
            "p-value":             m.pvalue()[dur_key],
        })

    summary = pd.DataFrame(rows).sort_values(
        ["Panel type", "NAICS level", "SOC level"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    # --- Formatted display copy ---
    summary_display = summary.copy()
    for col in ["Mean duration (m)", "Median duration (m)"]:
        summary_display[col] = summary_display[col].round(3)
    for col in ["Mean log wage", "Median log wage"]:
        summary_display[col] = summary_display[col].round(4)
    summary_display["Coef (months)"] = summary_display["Coef (months)"].round(6)
    summary_display["SE"]            = summary_display["SE"].round(6)
    summary_display["p-value"]       = summary_display["p-value"].round(4)
    summary_display["Observations"]  = summary_display["Observations"].apply(lambda x: f"{x:,}")
    summary_display["Cells"]         = summary_display["Cells"].apply(lambda x: f"{x:,}")

    print("\n=== Full Model Summary Table ===")
    print(summary_display.to_string(index=False))

    # --- Export ---
    csv_path = f"{cache_prefix}model_summary_table{fd_suffix}.csv"
    tex_path = f"{cache_prefix}model_summary_table{fd_suffix}.tex"

    summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    estimator_note = (
        "First-differences estimator. The dependent variable is the month-on-month "
        "change in mean log wage within each cell; the regressor is the month-on-month "
        "change in mean vacancy duration. Cell fixed effects drop out by construction; "
        "year-month fixed effects are retained."
        if fd else
        "Within-estimator (levels). Cell and year-month fixed effects included."
    )

    latex = summary.to_latex(
        index        = False,
        float_format = "%.4f",
        caption      = (
            f"Summary of panel specifications. {estimator_note} "
            "Duration and wage statistics are computed at the cell-month level "
            f"after applying the minimum cell size filter of {MIN_CELL_SIZE} "
            "vacancies and dropping singleton cells."
        ),
        label         = f"tab:model_summary{fd_suffix}",
        column_format = "lllrrrrrrrrr",
    )
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"Saved: {tex_path}")

# ===========================================================================
# ROBUSTNESS SPECIFICATIONS
# Append this block after the main `for fd in [False, True]:` loop.
#
# These four specifications extend the baseline FD panel models to address
# the simultaneity / reverse-causality concerns with the contemporaneous FD:
#
# PART 7  — Lagged FD:        Δwage_t  ~  Δdur_{t-1}
#            Last period's duration change predicts this period's wage change.
#            Directly sidesteps the within-period simultaneity problem.
#
# PART 8  — Flipped regression: Δdur_t  ~  Δwage_t
#            Asks whether wage increases predict faster vacancy filling.
#            The inverted coefficient gives an alternative recovery of the
#            wage-adjustment elasticity.
#
# PART 9  — Threshold dummy:  Δwage_t  ~  Δ1(dur_t > k)
#            Replaces continuous duration with a binary "long vacancy"
#            indicator at 1-, 2-, and 3-month thresholds.
#            Robust to functional-form assumptions.
#
# PART 10 — Lower-bound comparison table
#            Places the contemporaneous FD estimate (already saved) alongside
#            the lagged FD estimate to make the case that the former is a
#            conservative lower bound on the true adjustment parameter.
# ===========================================================================

# ---------------------------------------------------------------------------
# ROBUSTNESS HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def add_lagged_duration(panel, cell_id):
    """
    Add d_mean_dur_months_lag: a one-period lag of the already-computed
    first difference of mean duration, within each cell.
    Drops the first usable FD observation per cell (now NaN after lagging)
    and removes cells with fewer than 2 remaining observations.
    """
    panel = panel.sort_values([cell_id, "ym"]).copy()
    panel["d_mean_dur_months_lag"] = (
        panel.groupby(cell_id, observed=True)["d_mean_dur_months"].shift(1)
    )
    panel = panel.dropna(subset=["d_mean_dur_months_lag"]).copy()

    counts      = panel.groupby(cell_id, observed=True)["d_mean_log_wage"].count()
    valid_cells = counts[counts >= 2].index
    panel       = panel[panel[cell_id].isin(valid_cells)].copy()
    return panel


def add_threshold_dummy(panel_raw, cell_id, threshold_months):
    """
    Create a binary 'long vacancy' dummy (mean_dur_months > threshold_months)
    in the pre-FD panel, then first-difference both the dummy and log wage
    within each cell. Returns the differenced panel ready for estimation.
    threshold_months : numeric — e.g. 1, 2, or 3
    """
    panel = panel_raw.sort_values([cell_id, "ym"]).copy()
    panel["long_vac"] = (panel["mean_dur_months"] > threshold_months).astype(float)

    for col in ["mean_log_wage", "long_vac"]:
        panel[f"d_{col}"] = panel.groupby(cell_id, observed=True)[col].diff()

    panel = panel.dropna(subset=["d_mean_log_wage", "d_long_vac"]).copy()

    counts      = panel.groupby(cell_id, observed=True)["d_mean_log_wage"].count()
    valid_cells = counts[counts >= 2].index
    panel       = panel[panel[cell_id].isin(valid_cells)].copy()
    return panel


def run_robustness_models(panel, cell_var, spec):
    """
    Fit mean and cell-size-weighted models for one robustness specification.
    spec options:
        "lagged"    — d_mean_log_wage ~ d_mean_dur_months_lag | ym
        "flipped"   — d_mean_dur_months ~ d_mean_log_wage     | ym
        "threshold" — d_mean_log_wage ~ d_long_vac            | ym
    """
    formulas = {
        "lagged":    "d_mean_log_wage ~ d_mean_dur_months_lag | ym",
        "flipped":   "d_mean_dur_months ~ d_mean_log_wage     | ym",
        "threshold": "d_mean_log_wage ~ d_long_vac            | ym",
    }
    formula = formulas[spec]
    expl_var = formula.split("~")[1].split("|")[0].strip()

    models = {
        "Mean":     pf.feols(formula, data=panel, vcov={"CRV1": cell_var}),
        "Weighted": pf.feols(formula, data=panel, weights="n_vacancies",
                             vcov={"CRV1": cell_var}),
    }
    return models, expl_var


def collect_coefs(results_dict, expl_var, panel_type, extra_cols=None):
    """
    Flatten a {key: {model_name: feols}} results dict into a tidy DataFrame.
    extra_cols : dict of additional columns to add to every row (e.g. threshold)
    """
    rows = []
    for key, models in results_dict.items():
        for model_name, m in models.items():
            if expl_var not in m.coef().index:
                continue
            row = {
                "Panel type":  panel_type,
                "Key":         str(key),
                "Model":       model_name,
                "Coef":        m.coef()[expl_var],
                "SE":          m.se()[expl_var],
                "p-value":     m.pvalue()[expl_var],
                "N":           m._N,
            }
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# PART 7: Lagged FD  —  Δwage_t  ~  Δdur_{t-1}
# ===========================================================================
# Build FD panels fresh (build_panel is cheap; it's the vacancy-level feols
# that caused OOM, not the groupby aggregation).

print("\n" + "#" * 70)
print("# PART 7: LAGGED FD SPECIFICATION  (Δwage_t ~ Δdur_{t-1})")
print("#" * 70)

lag_co_results = {}
lag_io_results = {}

# --- Company × Occupation ---
print("\n" + "=" * 70)
print("COMPANY-OCCUPATION PANELS (LAGGED FD)")
print("=" * 70)

for soc in SOC_LEVELS:
    cache_path = f"{cache_prefix}firm_occ_lagged_{soc}.cpkl"

    if not new_models and os.path.exists(cache_path):
        lag_co_results[soc] = load_model(cache_path)
        print(f"\n--- SOC level: {soc} | loaded from cache ---")
        continue

    print(f"\n--- SOC level: {soc} ---")
    panel, cell_id = raw_panels[(("company",), soc)]
    panel    = apply_first_differences(panel.copy(), cell_id)
    panel_lg = add_lagged_duration(panel, cell_id)

    print(f"  {len(panel_lg):,} rows | {panel_lg[cell_id].nunique():,} cells")

    models, expl_var = run_robustness_models(panel_lg, cell_id, "lagged")
    lag_co_results[soc] = models

    if save_models:
        save_model(models, cache_path)
        print(f"  Saved: {cache_path}")

    pf.etable(list(models.values()),
              labels={"d_mean_dur_months_lag": "Δ Mean duration (months, lag)"},
              model_heads=list(models.keys()),
              title=f"Company-Occ Lagged FD | {soc.upper()}")

    tidy_rows = []
    for name, m in models.items():
        t = m.tidy().reset_index(); t["model"] = name; tidy_rows.append(t)
    pd.concat(tidy_rows).to_csv(
        cache_path.replace(".cpkl", "_tidy.csv"), index=False)

# --- Industry × Occupation ---
print("\n" + "=" * 70)
print("INDUSTRY-OCCUPATION PANELS (LAGGED FD)")
print("=" * 70)

for naics in NAICS_LEVELS:
    for soc in SOC_LEVELS:
        key        = (naics, soc)
        cache_path = f"{cache_prefix}ind_occ_lagged_{naics}_{soc}.cpkl"

        if not new_models and os.path.exists(cache_path):
            lag_io_results[key] = load_model(cache_path)
            print(f"\n--- {naics} x {soc} | loaded from cache ---")
            continue

        print(f"\n--- NAICS level: {naics} | SOC level: {soc} ---")
        panel, cell_id = raw_panels[((naics,), soc)]
        panel = panel.copy()

        if panel[cell_id].nunique() < 10:
            print("  Skipping — fewer than 10 cells after filtering.")
            continue

        panel    = apply_first_differences(panel, cell_id)
        panel_lg = add_lagged_duration(panel, cell_id)

        print(f"  {len(panel_lg):,} rows | {panel_lg[cell_id].nunique():,} cells")

        models, expl_var = run_robustness_models(panel_lg, cell_id, "lagged")
        lag_io_results[key] = models

        if save_models:
            save_model(models, cache_path)
            print(f"  Saved: {cache_path}")

        pf.etable(list(models.values()),
                  labels={"d_mean_dur_months_lag": "Δ Mean duration (months, lag)"},
                  model_heads=list(models.keys()),
                  title=f"Industry-Occ Lagged FD | {naics.upper()} x {soc.upper()}")

        tidy_rows = []
        for name, m in models.items():
            t = m.tidy().reset_index(); t["model"] = name; tidy_rows.append(t)
        pd.concat(tidy_rows).to_csv(
            cache_path.replace(".cpkl", "_tidy.csv"), index=False)

# Save tidy summary for Part 7
lag_co_tidy = collect_coefs(lag_co_results, "d_mean_dur_months_lag",
                             "Company-Occ", {"Spec": "Lagged FD"})
lag_io_tidy = collect_coefs(lag_io_results, "d_mean_dur_months_lag",
                             "Industry-Occ", {"Spec": "Lagged FD"})
lag_summary = pd.concat([lag_co_tidy, lag_io_tidy], ignore_index=True)
lag_summary.to_csv(f"{cache_prefix}robustness_lagged_fd.csv", index=False)
print(f"\nSaved: {cache_prefix}robustness_lagged_fd.csv")


# ===========================================================================
# PART 8: Flipped regression  —  Δdur_t  ~  Δwage_t
# ===========================================================================
# The coefficient γ answers: "when wages rise by 1 log point, how does
# duration change?" 1/γ gives an alternative estimate of the wage-adjustment
# elasticity comparable to the main β. Both are reported in the comparison
# table in Part 10.

print("\n" + "#" * 70)
print("# PART 8: FLIPPED REGRESSION  (Δdur_t ~ Δwage_t)")
print("#" * 70)

flip_co_results = {}
flip_io_results = {}

# --- Company × Occupation ---
print("\n" + "=" * 70)
print("COMPANY-OCCUPATION PANELS (FLIPPED FD)")
print("=" * 70)

for soc in SOC_LEVELS:
    cache_path = f"{cache_prefix}firm_occ_flipped_{soc}.cpkl"

    if not new_models and os.path.exists(cache_path):
        flip_co_results[soc] = load_model(cache_path)
        print(f"\n--- SOC level: {soc} | loaded from cache ---")
        continue

    print(f"\n--- SOC level: {soc} ---")
    panel, cell_id = raw_panels[(("company",), soc)]
    panel = apply_first_differences(panel.copy(), cell_id)

    print(f"  {len(panel):,} rows | {panel[cell_id].nunique():,} cells")

    models, expl_var = run_robustness_models(panel, cell_id, "flipped")
    flip_co_results[soc] = models

    if save_models:
        save_model(models, cache_path)
        print(f"  Saved: {cache_path}")

    pf.etable(list(models.values()),
              labels={"d_mean_log_wage": "Δ Mean log wage"},
              model_heads=list(models.keys()),
              title=f"Company-Occ Flipped FD | {soc.upper()}")

    tidy_rows = []
    for name, m in models.items():
        t = m.tidy().reset_index(); t["model"] = name; tidy_rows.append(t)
    pd.concat(tidy_rows).to_csv(
        cache_path.replace(".cpkl", "_tidy.csv"), index=False)

# --- Industry × Occupation ---
print("\n" + "=" * 70)
print("INDUSTRY-OCCUPATION PANELS (FLIPPED FD)")
print("=" * 70)

for naics in NAICS_LEVELS:
    for soc in SOC_LEVELS:
        key        = (naics, soc)
        cache_path = f"{cache_prefix}ind_occ_flipped_{naics}_{soc}.cpkl"

        if not new_models and os.path.exists(cache_path):
            flip_io_results[key] = load_model(cache_path)
            print(f"\n--- {naics} x {soc} | loaded from cache ---")
            continue

        print(f"\n--- NAICS level: {naics} | SOC level: {soc} ---")
        panel, cell_id = raw_panels[((naics,), soc)]
        panel = panel.copy()

        if panel[cell_id].nunique() < 10:
            print("  Skipping — fewer than 10 cells after filtering.")
            continue

        panel = apply_first_differences(panel, cell_id)

        print(f"  {len(panel):,} rows | {panel[cell_id].nunique():,} cells")

        models, expl_var = run_robustness_models(panel, cell_id, "flipped")
        flip_io_results[key] = models

        if save_models:
            save_model(models, cache_path)
            print(f"  Saved: {cache_path}")

        pf.etable(list(models.values()),
                  labels={"d_mean_log_wage": "Δ Mean log wage"},
                  model_heads=list(models.keys()),
                  title=f"Industry-Occ Flipped FD | {naics.upper()} x {soc.upper()}")

        tidy_rows = []
        for name, m in models.items():
            t = m.tidy().reset_index(); t["model"] = name; tidy_rows.append(t)
        pd.concat(tidy_rows).to_csv(
            cache_path.replace(".cpkl", "_tidy.csv"), index=False)

# Save tidy summary for Part 8
flip_co_tidy = collect_coefs(flip_co_results, "d_mean_log_wage",
                              "Company-Occ", {"Spec": "Flipped FD"})
flip_io_tidy = collect_coefs(flip_io_results, "d_mean_log_wage",
                              "Industry-Occ", {"Spec": "Flipped FD"})
flip_summary = pd.concat([flip_co_tidy, flip_io_tidy], ignore_index=True)
flip_summary.to_csv(f"{cache_prefix}robustness_flipped_fd.csv", index=False)
print(f"\nSaved: {cache_prefix}robustness_flipped_fd.csv")


# ===========================================================================
# PART 9: Threshold dummy  —  Δwage_t  ~  Δ1(dur_t > k months)
# ===========================================================================
# Run at three thresholds: 1, 2, and 3 months. For each, the dummy equals 1
# in months where the cell's mean vacancy duration exceeds the threshold.
# The FD of this dummy captures transitions into / out of "long vacancy" status.

print("\n" + "#" * 70)
print("# PART 9: THRESHOLD DUMMY  (Δwage_t ~ Δ1(dur_t > k months))")
print("#" * 70)

THRESHOLDS = [1, 2, 3]   # months

thresh_co_results = {k: {} for k in THRESHOLDS}
thresh_io_results = {k: {} for k in THRESHOLDS}

# --- Company × Occupation ---
print("\n" + "=" * 70)
print("COMPANY-OCCUPATION PANELS (THRESHOLD DUMMY)")
print("=" * 70)

for soc in SOC_LEVELS:
    # Retrieve pre-built panel; copy so threshold dummy additions don't mutate cache
    panel_raw, cell_id = raw_panels[(("company",), soc)]
    panel_raw = panel_raw.copy()

    for k in THRESHOLDS:
        cache_path = f"{cache_prefix}firm_occ_threshold{k}m_{soc}.cpkl"

        if not new_models and os.path.exists(cache_path):
            thresh_co_results[k][soc] = load_model(cache_path)
            print(f"\n--- SOC: {soc} | threshold: {k}m | loaded from cache ---")
            continue

        print(f"\n--- SOC level: {soc} | threshold: {k} month(s) ---")
        panel_t = add_threshold_dummy(panel_raw, cell_id, threshold_months=k)

        print(f"  {len(panel_t):,} rows | {panel_t[cell_id].nunique():,} cells"
              f" | long-vac share: {panel_t['long_vac'].mean():.2%}")

        models, expl_var = run_robustness_models(panel_t, cell_id, "threshold")
        thresh_co_results[k][soc] = models

        if save_models:
            save_model(models, cache_path)
            print(f"  Saved: {cache_path}")

        pf.etable(list(models.values()),
                  labels={"d_long_vac": f"Δ Long vacancy (>{k}m)"},
                  model_heads=list(models.keys()),
                  title=f"Company-Occ Threshold (>{k}m) | {soc.upper()}")

        tidy_rows = []
        for name, m in models.items():
            t = m.tidy().reset_index(); t["model"] = name; tidy_rows.append(t)
        pd.concat(tidy_rows).to_csv(
            cache_path.replace(".cpkl", "_tidy.csv"), index=False)

# --- Industry × Occupation ---
print("\n" + "=" * 70)
print("INDUSTRY-OCCUPATION PANELS (THRESHOLD DUMMY)")
print("=" * 70)

for naics in NAICS_LEVELS:
    for soc in SOC_LEVELS:
        key = (naics, soc)
        panel_raw, cell_id = raw_panels[((naics,), soc)]
        panel_raw = panel_raw.copy()

        if panel_raw[cell_id].nunique() < 10:
            print(f"  Skipping {naics} x {soc} — fewer than 10 cells.")
            continue

        for k in THRESHOLDS:
            cache_path = f"{cache_prefix}ind_occ_threshold{k}m_{naics}_{soc}.cpkl"

            if not new_models and os.path.exists(cache_path):
                thresh_io_results[k][key] = load_model(cache_path)
                print(f"\n--- {naics} x {soc} | threshold: {k}m | loaded ---")
                continue

            print(f"\n--- NAICS: {naics} | SOC: {soc} | threshold: {k} month(s) ---")
            panel_t = add_threshold_dummy(panel_raw, cell_id, threshold_months=k)

            print(f"  {len(panel_t):,} rows | {panel_t[cell_id].nunique():,} cells"
                  f" | long-vac share: {panel_t['long_vac'].mean():.2%}")

            models, expl_var = run_robustness_models(panel_t, cell_id, "threshold")
            thresh_io_results[k][key] = models

            if save_models:
                save_model(models, cache_path)
                print(f"  Saved: {cache_path}")

            pf.etable(list(models.values()),
                      labels={"d_long_vac": f"Δ Long vacancy (>{k}m)"},
                      model_heads=list(models.keys()),
                      title=f"Ind-Occ Threshold (>{k}m) | {naics.upper()} x {soc.upper()}")

            tidy_rows = []
            for name, m in models.items():
                t = m.tidy().reset_index(); t["model"] = name; tidy_rows.append(t)
            pd.concat(tidy_rows).to_csv(
                cache_path.replace(".cpkl", "_tidy.csv"), index=False)

# Save tidy summary for Part 9
thresh_rows = []
for k in THRESHOLDS:
    co_tidy = collect_coefs(thresh_co_results[k], "d_long_vac",
                             "Company-Occ", {"Spec": f"Threshold >{k}m"})
    io_tidy = collect_coefs(thresh_io_results[k], "d_long_vac",
                             "Industry-Occ", {"Spec": f"Threshold >{k}m"})
    thresh_rows.extend([co_tidy, io_tidy])
thresh_summary = pd.concat(thresh_rows, ignore_index=True)
thresh_summary.to_csv(f"{cache_prefix}robustness_threshold.csv", index=False)
print(f"\nSaved: {cache_prefix}robustness_threshold.csv")


# ===========================================================================
# PART 10: Lower-bound comparison table
# ===========================================================================
# Places the contemporaneous FD estimate (saved by the main loop) alongside
# the lagged FD estimate to make the conservative-lower-bound argument.
# Also reports 1/γ from the flipped regression as a third estimate.
# All three should be positive if the wage-adjustment mechanism is real;
# the contemporaneous estimate is expected to be the smallest of the three.

print("\n" + "#" * 70)
print("# PART 10: LOWER-BOUND COMPARISON TABLE")
print("#" * 70)

# --- Load contemporaneous FD summary (already saved by main loop) ---
contemp = pd.read_csv(f"{cache_prefix}model_summary_table_fd.csv")

# Standardise column names to match the tidy frames built above
contemp_rows = []
for _, row in contemp.iterrows():
    contemp_rows.append({
        "Panel type": row["Panel type"],
        "Key":        (row["NAICS level"] + "_" + row["SOC level"])
                      if row["Panel type"] == "Industry-Occ"
                      else row["SOC level"],
        "Model":      "Mean",
        "Coef":       row["Coef (months)"],
        "SE":         row["SE"],
        "p-value":    row["p-value"],
        "N":          row["Observations"],
        "Spec":       "Contemporaneous FD",
    })
contemp_tidy = pd.DataFrame(contemp_rows)

# --- Restrict all frames to Mean model only for comparability ---
lag_mean   = lag_summary[lag_summary["Model"] == "Mean"].copy()
flip_mean  = flip_summary[flip_summary["Model"] == "Mean"].copy()

# For the flipped regression, the comparable quantity is 1/γ.
# We report γ as-is and flag its interpretation in the table.
flip_mean  = flip_mean.copy()
flip_mean["Coef_inverted"] = 1 / flip_mean["Coef"]

# --- Assemble comparison ---
comparison = pd.concat([contemp_tidy, lag_mean], ignore_index=True)

# Print console summary
print(f"\n{'Spec':<22} {'Panel type':<14} {'Key':<20} "
      f"{'Coef':>10} {'SE':>10} {'p-value':>10} {'N':>12}")
print("-" * 92)

for spec_label in ["Contemporaneous FD", "Lagged FD"]:
    sub = comparison[comparison["Spec"] == spec_label]
    for _, r in sub.iterrows():
        print(f"{spec_label:<22} {r['Panel type']:<14} {str(r['Key']):<20} "
              f"{r['Coef']:>10.6f} {r['SE']:>10.6f} "
              f"{r['p-value']:>10.4f} {str(r['N']):>12}")
    print()

print("\n--- Flipped regression (γ) and implied wage-adjustment (1/γ) ---")
print(f"{'Panel type':<14} {'Key':<20} {'γ (Δdur~Δwage)':>16} "
      f"{'1/γ':>10} {'p-value':>10}")
print("-" * 76)
for _, r in flip_mean.iterrows():
    print(f"{r['Panel type']:<14} {str(r['Key']):<20} "
          f"{r['Coef']:>16.6f} {r['Coef_inverted']:>10.6f} "
          f"{r['p-value']:>10.4f}")

# --- Export ---
csv_path = f"{cache_prefix}robustness_comparison_table.csv"
comparison.to_csv(csv_path, index=False)

flip_export = flip_mean[["Panel type", "Key", "Coef", "SE", "p-value", "N",
                          "Coef_inverted"]].rename(
    columns={"Coef": "gamma", "Coef_inverted": "1_over_gamma"})
flip_export["Spec"] = "Flipped FD"
flip_export.to_csv(f"{cache_prefix}robustness_flipped_inverted.csv", index=False)

print(f"\nSaved: {csv_path}")
print(f"Saved: {cache_prefix}robustness_flipped_inverted.csv")

# --- LaTeX table (contemporaneous vs lagged, Mean model only) ---
latex_rows = []
for _, r in comparison[comparison["Model"] == "Mean"].iterrows():
    latex_rows.append({
        "Specification":  r["Spec"],
        "Panel type":     r["Panel type"],
        "Key":            r["Key"],
        "Coefficient":    r["Coef"],
        "SE":             r["SE"],
        "p-value":        r["p-value"],
        "Observations":   r["N"],
    })
latex_df = pd.DataFrame(latex_rows)
latex = latex_df.to_latex(
    index=False,
    float_format="%.6f",
    caption=(
        r"Contemporaneous vs.\ lagged FD estimates of the wage-adjustment "
        "coefficient on vacancy duration (months). The contemporaneous "
        "estimate is expected to be a conservative lower bound due to "
        "within-period simultaneity: higher wages attract applicants in "
        "the same month, compressing observed duration and attenuating "
        r"$\hat{\beta}$. The lagged specification "
        r"($\Delta \bar{w}_t \sim \Delta \overline{\text{dur}}_{t-1}$) "
        "sidesteps this by using last period's duration change as the regressor."
    ),
    label="tab:robustness_comparison",
    column_format="llllrrr",
)
with open(f"{cache_prefix}robustness_comparison_table.tex", "w") as f:
    f.write(latex)
print(f"Saved: {cache_prefix}robustness_comparison_table.tex")