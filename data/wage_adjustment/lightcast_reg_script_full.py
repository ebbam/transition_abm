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
        panel, cell_id = build_panel(
            df,
            group_vars    = ["company"],
            soc_col       = soc,
            min_cell_size = MIN_CELL_SIZE,
        )

        if fd:
            panel = apply_first_differences(panel, cell_id)

        print(f"  {len(panel):,} rows | {panel[cell_id].nunique():,} cells "
              f"| cell size p50: {panel['n_vacancies'].median():.0f}")

        models = run_panel_models(panel, cell_id, duration_vars, fd=fd)
        co_results[soc] = models
        co_panels[soc]  = (panel, cell_id)

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
            panel, cell_id = build_panel(
                df,
                group_vars    = [naics],
                soc_col       = soc,
                min_cell_size = MIN_CELL_SIZE,
            )

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
            panel, cell_id = build_panel(df, group_vars=["company"],
                                         soc_col=soc, min_cell_size=MIN_CELL_SIZE)
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
            panel, cell_id = build_panel(df, group_vars=[naics],
                                         soc_col=soc, min_cell_size=MIN_CELL_SIZE)
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