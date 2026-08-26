# %% [markdown]
# ### Summarising Calibration Results

# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, gaussian_kde
from matplotlib.lines import Line2D
import pyabc
from plot_style import (TEXTWIDTH, DEFAULT_ASPECT_RATIO, ROW_HEIGHT_3, save_plot_standard,
                         add_shared_legend, add_suptitle_spacing)
from network_input_builder import *
path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/"
dropbox = True

calib_source_folder = "calib_final"


if dropbox:
    dropbox_path = f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/"
else:
    dropbox_path = ""

#prefix = "cos_calib_smart_hire_3/"
prefix = "calib_final/"
vac_rate_setting = "exog_vac/"
if "smart_hire" in prefix:
    add_suffix = "smart_hire"
    print(add_suffix)
else:
    add_suffix = ""

# if add_suffix != "smart_hire":
#     raise KeyError

if prefix == "calib_final/":
    if vac_rate_setting == "endog_vac_drift/":
        add_suffix = "_wages_replace_early_shocks_endog_vac_distance_weighted_vr"
    elif vac_rate_setting == "cost_red/":
        add_suffix = "_wages_replace_early_shocks_cost_red_distance_weighted_vr"
    elif vac_rate_setting == "exog_vac/":
        add_suffix = "_exog_wage_update_True"

# %%

#################################################################
### Support Functions ###########################################
#################################################################
# replace this function with your actual loader that returns (df, w)
def load_model_distribution(db_path):
    """
    Return (df, w) for a model. Replace this with your code that
    reads history and calls history.get_distribution(t=history.max_t).
    df: pandas.DataFrame with columns 'd_u', 'gamma_u', optional 'theta'
    w: 1D numpy array of weights aligned with df.index
    """
    # --- PLACEHOLDER: replace with your actual loader ---
    #history1 = pyabc.History("sqlite:///" + f"output/{prefix}{which_params}/{name}{add_suffix}.db")
    #latest_run_id = history1._find_latest_id()
    #print(latest_run_id)
    history = pyabc.History("sqlite:///" + f"output/{prefix}{which_params}/{vac_rate_setting}{name}{add_suffix}.db")
    print(f"output/{prefix}{which_params}/{vac_rate_setting}{name}{add_suffix}.db")
    df, w = history.get_distribution(t=history.max_t)
    if len(df) == 0:
        print(f'{name} empty')

    return df, w

# safe kde wrapper: uses seaborn weights when available, else resamples
def plot_kde_on_ax(ax, x, y=None, weights=None, label=None, color=None, bw_adjust=1.0, fill=False, levels=6, alpha = 1):
    """
    If y is None -> 1d kde; if y provided -> 2d kde contour.
    Tries to use sns.kdeplot(weights=...). If seaborn does not support weights for your version,
    falls back to weighted resampling.
    """
    use_seaborn_weights = True
    try:
        # try a direct call - will raise if weights unsupported
        if y is None:
            sns.kdeplot(x=x, ax=ax, weights=weights, label=label, color=color, bw_adjust=bw_adjust, fill=fill, alpha = alpha)
        else:
            sns.kdeplot(x=x, y=y, ax=ax, weights=weights, color=color, levels=levels, bw_adjust=bw_adjust, fill=fill, alpha = alpha)
        return
    except TypeError:
        use_seaborn_weights = False

    # fallback: weighted resample
    n_resample = min(4000, max(500, int(len(x) * 0.1)))  # sample up to 4000 points
    if weights is None:
        idx = np.random.choice(len(x), size=n_resample, replace=True)
    else:
        p = np.array(weights, dtype=float)
        p = p / p.sum()
        idx = np.random.choice(len(x), size=n_resample, replace=True, p=p)

    xs = np.asarray(x)[idx]
    if y is None:
        sns.kdeplot(x=xs, ax=ax, label=label, color=color, bw_adjust=bw_adjust, fill=fill, alpha = alpha)
    else:
        ys = np.asarray(y)[idx]
        sns.kdeplot(x=xs, y=ys, ax=ax, label=label, color=color, levels=levels, bw_adjust=bw_adjust, fill=fill, alpha = alpha)

# compute global axis limits for nicer consistent plotting
def global_limits(col, pad_frac=0.05):
    vals = []
    for m in samples:
        ser = samples[m]["df"].get(col)
        if ser is None: continue
        vals.append(ser.values)
    if len(vals) == 0:
        return None
    arr = np.concatenate(vals)
    lo, hi = np.nanpercentile(arr, [0.5, 99.5])
    rng = hi - lo
    return (max(0, lo - pad_frac*rng), hi + pad_frac*rng)
    
from model_colors import MODEL_COLORS_BY_ID

# list of model identifiers (your db_path values)
models = [
    "nonbehav",
    "otj_nonbehav",
    "otj_disc_no_rw",
    "otj_cyclical_e_disc_no_rw",
    "otj_disc",
    "otj_cyclical_e_disc",
    #"otj_disc_strict_rw",
    #"otj_cyclical_e_disc_strict_rw"
]

if any("cyclical_e" in m for m in models):
    yes_theta = True
else:
    yes_theta = False

name_map = {
        "nonbehav": "Non-behavioural",
        "otj_nonbehav": "Non-behavioural w. OTJ",
        "otj_cyclical_e_disc": "Behavioural w. Cyc. OTJ w. RW",
        "otj_disc": "Behavioural w.o. Cyc. OTJ w. RW",
       # "otj_disc_strict_rw": "Behavioural w.o. Cyc. OTJ w. Strict RW",
       # "otj_cyclical_e_disc_strict_rw": "Behavioural w. Cyc. OTJ w. Strict RW",
        "otj_cyclical_e_disc_no_rw": "Behavioural w. Cyc. OTJ w.o RW",
        "otj_disc_no_rw": "Behavioural w.o. Cyc. OTJ w.o RW"
    }

colors = {m: MODEL_COLORS_BY_ID[m] for m in models if m in MODEL_COLORS_BY_ID}

for which_params in ["full_omn" , "single_node", "onet", "onet_wage_asym"]: 
    if which_params == "onet":
        nw_label = "ONET"
    elif which_params == "onet_wage_asym":
        nw_label = "ONET w. Wage Asymmetry Links"
    elif which_params == "full_omn":
        
        nw_label = "Occupational Mobility Network"
    elif which_params == "omn_soc_minor":
        nw_label = "OMN - SOC Minor"
    elif which_params == "single_node":
        nw_label = "Single Node Network"

        
    # Collect all model samples
    samples = {}
    for name in models:
        df, w = load_model_distribution(name)

        # ###################################################################
        samples[name] = {"df": df, "w": np.asarray(w)}

    du_lim = global_limits("d_u")
    gu_lim = global_limits("gamma_u")
    if yes_theta:
        theta_lim = global_limits("theta")

    limit_buffer = 0
    du_min, du_max = global_limits("d_u")
    gu_min, gu_max = global_limits("gamma_u")
    if yes_theta:
        theta_min, theta_max = global_limits("theta")

    du_min -= limit_buffer
    du_max += limit_buffer
    gu_min -= limit_buffer
    gu_max += limit_buffer
    if yes_theta:
        theta_min -= limit_buffer
        theta_max += limit_buffer
    
    if dropbox:
        out_dir = os.path.expanduser(f"~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/{prefix}{which_params}/{vac_rate_setting}")
    else:
        out_dir = f'output/{prefix}{which_params}/figures/calib_figures/'
        os.makedirs(out_dir, exist_ok=True)

    # %%
    # ------------------- 1) Joint: d_u vs gamma_u (all models) -------------------
    _fig_w, _fig_h = TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO
    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
    _legend_colors = {}
    for name, meta in samples.items():
        df = meta["df"]
        w = meta["w"]
        print(name)
        if "d_u" not in df.columns or "gamma_u" not in df.columns:
            continue
        plot_kde_on_ax(ax, df["d_u"], df["gamma_u"],  weights=w, label=name_map.get(name, name), color=colors[name], levels=6, fill = True, alpha = 0.5)
        _legend_colors[name_map.get(name, name)] = colors[name]
        # weighted mean marker
        my = np.average(df["gamma_u"].values, weights=w)
        mx = np.average(df["d_u"].values, weights=w)
        #ax.scatter(mx, my, s=50, marker="o", linewidth=0.8, color=colors[name], label=name_map.get(name, name))


    ax.set_ylabel(r"$\delta_u$")
    ax.set_xlabel(r"$\gamma_u$")
    if du_lim: ax.set_xlim(max(0, du_min), max(0.03, du_max))
    if gu_lim: ax.set_ylim(max(0, gu_min), gu_max)
    ax.set_title(f"KDE: $\\delta_u$ vs. $\\gamma_u$\nNetwork: {nw_label}")
    plt.tight_layout()
    add_shared_legend(fig, _legend_colors, ncol=3)
    save_plot_standard(fig, os.path.join(out_dir, "all_models_du_vs_gu.png"), aspect_ratio=_fig_h / _fig_w)
    #plt.show()
    plt.close(fig)

    # %%
    # ------------------- 2) Joint: d_u vs theta (only models with theta) -------------------
    _fig_w, _fig_h = TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO
    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
    any_theta = False
    _legend_colors = {}
    for name, meta in samples.items():
        df = meta["df"]
        w = meta["w"]
        if "theta" not in df.columns:
            continue
        any_theta = True
        plot_kde_on_ax(ax, df["d_u"], df["theta"], weights=w,  color=colors[name], levels=6, fill =True, alpha = 0.5)
        ax.scatter(np.average(df["d_u"].values, weights=w), np.average(df["theta"].values, weights=w),
                color=colors[name], s=50, marker="o", edgecolor="k", linewidth=0.8)
        _legend_colors[name_map.get(name, name)] = colors[name]

    if any_theta:
        ax.set_ylabel(r"$\delta_u$")
        ax.set_xlabel(r"$\beta_C$")
        if du_lim: ax.set_xlim(du_lim)
        if theta_lim: ax.set_ylim(theta_lim)
        ax.set_title(f"KDE: $\\delta_u$ vs. $\\beta_C$ (models with $\\beta_C$)\nNetwork: {nw_label}")
        plt.tight_layout()
        add_shared_legend(fig, _legend_colors, ncol=1)
        save_plot_standard(fig, os.path.join(out_dir, "all_models_du_vs_theta.png"), aspect_ratio=_fig_h / _fig_w)
        #plt.show()
        plt.close(fig)
    else:
        print("No models had theta -> skipping d_u vs theta plot.")



    # %%

    # ------------------- 3) Joint: gamma_u vs theta (only models with theta) -------------------
    _fig_w, _fig_h = TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO
    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
    any_theta = False
    _legend_colors = {}
    for name, meta in samples.items():
        df = meta["df"]
        w = meta["w"]
        if "theta" not in df.columns:
            continue
        any_theta = True
        plot_kde_on_ax(ax, df["gamma_u"], df["theta"], weights=w, color=colors[name], levels=6, fill = True, alpha = 0.5)
        ax.scatter(np.average(df["gamma_u"].values, weights=w), np.average(df["theta"].values, weights=w),
                color=colors[name], s=50, marker="o", edgecolor="k", linewidth=0.8)
        _legend_colors[name_map.get(name, name)] = colors[name]

    if any_theta:
        ax.set_xlabel(r"$\gamma_u$")
        ax.set_ylabel(r"$\beta_C$")
        if gu_lim: ax.set_xlim(gu_lim)
        if theta_lim: ax.set_ylim(theta_lim)
        ax.set_title(f"KDE: $\\gamma_u$ vs. $\\beta_C$ (models with $\\beta_C$)\nNetwork: {nw_label}")
        plt.tight_layout()
        add_shared_legend(fig, _legend_colors, ncol=1)
        save_plot_standard(fig, os.path.join(out_dir, "all_models_gu_vs_theta.png"), aspect_ratio=_fig_h / _fig_w)
        #plt.show()
        plt.close(fig)
    else:
        print("No models had theta -> skipping gamma_u vs theta plot.")



    # %%
    # ------------------- 4) "Diagonal KDE matrix": 1D marginals overlaid -------------------
    if yes_theta:
        vars_to_plot = ["d_u", "gamma_u", "theta"]
    else:
        vars_to_plot = ["d_u", "gamma_u"]

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT_3 * len(vars_to_plot)
    fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(_fig_w, _fig_h))

    _legend_colors = {}
    for idx, var in enumerate(vars_to_plot):
        ax = axes[idx]
        for name, meta in samples.items():
            df = meta["df"]
            w = meta["w"]
            if var not in df.columns:
                continue
            plot_kde_on_ax(ax, df[var], weights=w, color=colors[name], fill=True, alpha = 0.3)
            # weighted mean vertical line
            ax.axvline(np.average(df[var].values, weights=w), color=colors[name], linestyle='dashed', linewidth=1)
            _legend_colors[name_map.get(name, name)] = colors[name]
        if var == "d_u":
            ax.set_title(r"$\delta_u$")
        elif var == "gamma_u":
            ax.set_title(r"$\gamma_u$")
        elif var == "theta":
            print(np.average(df[var].values, weights=w))
            ax.set_title(r"$\beta_C$")
        ax.set_xlabel(None)
        ax.set_xlim()

    fig.suptitle(f"KDE\nNetwork: {nw_label}", fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, _legend_colors, ncol=3)
    save_plot_standard(fig, os.path.join(out_dir, "all_models_marginals.png"), aspect_ratio=_fig_h / _fig_w)
    #plt.show()
    plt.close(fig)

    print("Saved combined plots to", out_dir)

