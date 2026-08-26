# Provides different options for plotting the LTUER distributions
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# ============================================================================
# CDF PLOT 
# ============================================================================
def plot_ltuer_cdf(net_dict, names, save=False, path="./", colors=None):
    """
    Plot Cumulative Distribution Functions for unemployment duration.
    
    CDFs are MUCH better than histograms for skewed data because:
    - No binning artifacts
    - Easy to compare distributions
    - Can read percentiles directly from y-axis
    """
    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(net_dict.keys(), colors))
    
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (name, net) in enumerate(net_dict.items()):
        # Collect all unemployment durations
        total_unemp = []
        for occ in net:
            total_unemp.extend(wrkr.time_unemployed for wrkr in occ.list_of_unemployed)
        
        # Sort for CDF
        sorted_data = np.sort(total_unemp)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot full CDF
        ax_full.plot(sorted_data, y, label=names[idx], color=colors[name], linewidth=2)
        
        # Plot zoomed CDF (0-36 months)
        mask = sorted_data <= 36
        ax_zoom.plot(sorted_data[mask], y[mask], label=names[idx], 
                    color=colors[name], linewidth=2)
        
        # Add median line
        median = np.median(total_unemp)
        ax_full.axvline(median, color=colors[name], linestyle='--', alpha=0.3, linewidth=1)
        if median <= 36:
            ax_zoom.axvline(median, color=colors[name], linestyle='--', alpha=0.3, linewidth=1)
    
    # Formatting
    ax_full.set_xlabel("Time Unemployed (months)", fontsize=12)
    ax_full.set_ylabel("Cumulative Probability", fontsize=12)
    ax_full.set_title("CDF - Full Distribution", fontweight='bold')
    ax_full.grid(True, alpha=0.3)
    ax_full.set_ylim(0, 1)
    
    ax_zoom.set_xlabel("Time Unemployed (months)", fontsize=12)
    ax_zoom.set_ylabel("Cumulative Probability", fontsize=12)
    ax_zoom.set_title("CDF - Zoom ≤ 3 years", fontweight='bold')
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_xlim(0, 36)
    ax_zoom.set_ylim(0, 1)
    
    # Add reference lines for quartiles
    for prob, label in [(0.25, '25%'), (0.5, '50%'), (0.75, '75%')]:
        ax_full.axhline(prob, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax_zoom.axhline(prob, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Legend
    handles, labels_leg = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=min(len(labels_leg), 4), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save:
        fig.savefig(f'{path}ltuer_cdf.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


# ============================================================================
# KDE PLOT (Smooth density curves)
# ============================================================================
def plot_ltuer_kde(net_dict, names, save=False, path="./", colors=None):
    """
    Plot Kernel Density Estimates - smoother than histograms.
    """
    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(net_dict.keys(), colors))
    
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for idx, (name, net) in enumerate(net_dict.items()):
        total_unemp = []
        for occ in net:
            total_unemp.extend(wrkr.time_unemployed for wrkr in occ.list_of_unemployed)
        
        # KDE for full range
        kde = stats.gaussian_kde(total_unemp)
        x_full = np.linspace(0, max(total_unemp), 500)
        ax_full.plot(x_full, kde(x_full), label=names[idx], 
                    color=colors[name], linewidth=2)
        ax_full.axvline(np.mean(total_unemp), color=colors[name], 
                       linestyle='--', alpha=0.5, linewidth=1)
        
        # KDE for zoomed range
        x_zoom = np.linspace(0, 36, 500)
        ax_zoom.plot(x_zoom, kde(x_zoom), label=names[idx], 
                    color=colors[name], linewidth=2)
        mean_val = np.mean(total_unemp)
        if mean_val <= 36:
            ax_zoom.axvline(mean_val, color=colors[name], 
                           linestyle='--', alpha=0.5, linewidth=1)
    
    ax_full.set_xlabel("Time Unemployed (months)", fontsize=12)
    ax_full.set_ylabel("Density", fontsize=12)
    ax_full.set_title("KDE - Full Distribution", fontweight='bold')
    
    ax_zoom.set_xlabel("Time Unemployed (≤ 36 months)", fontsize=12)
    ax_zoom.set_title("KDE - Zoom ≤ 3 years", fontweight='bold')
    ax_zoom.set_xlim(0, 36)
    
    handles, labels_leg = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=min(len(labels_leg), 4), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save:
        fig.savefig(f'{path}ltuer_kde.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


# ============================================================================
# COMBINED HISTOGRAM + KDE
# ============================================================================
def plot_ltuer_hist_kde(net_dict, names, save=False, path="./", colors=None):
    """
    Histogram with KDE overlay - best of both worlds.
    """
    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(net_dict.keys(), colors))
    
    n_bins = 30
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (name, net) in enumerate(net_dict.items()):
        total_unemp = []
        for occ in net:
            total_unemp.extend(wrkr.time_unemployed for wrkr in occ.list_of_unemployed)
        
        # Histogram (transparent)
        ax_full.hist(total_unemp, bins=n_bins, alpha=0.3, 
                    color=colors[name], density=True, label=f'{names[idx]} (hist)')
        ax_zoom.hist(total_unemp, bins=n_bins, range=(0, 36), alpha=0.3,
                    color=colors[name], density=True)
        
        # KDE overlay (solid line)
        kde = stats.gaussian_kde(total_unemp)
        x_full = np.linspace(0, max(total_unemp), 500)
        ax_full.plot(x_full, kde(x_full), color=colors[name], 
                    linewidth=2.5, label=f'{names[idx]} (KDE)')
        
        x_zoom = np.linspace(0, 36, 500)
        ax_zoom.plot(x_zoom, kde(x_zoom), color=colors[name], linewidth=2.5)
    
    ax_full.set_xlabel("Time Unemployed (months)", fontsize=12)
    ax_full.set_ylabel("Density", fontsize=12)
    ax_full.set_title("Histogram + KDE - Full", fontweight='bold')
    
    ax_zoom.set_xlabel("Time Unemployed (≤ 36 months)", fontsize=12)
    ax_zoom.set_ylabel("Density", fontsize=12)
    ax_zoom.set_title("Histogram + KDE - Zoom", fontweight='bold')
    ax_zoom.set_xlim(0, 36)
    
    handles, labels_leg = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=2, frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    if save:
        fig.savefig(f'{path}ltuer_hist_kde.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


# ============================================================================
# BOX PLOT / VIOLIN PLOT
# ============================================================================
def plot_ltuer_boxplot(net_dict, names, save=False, path="./", colors=None):
    """
    Box plots showing quartiles and outliers.
    Optionally can use violin plots instead.
    """
    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(net_dict.keys(), colors))
    
    fig, (ax_box, ax_violin) = plt.subplots(1, 2, figsize=(14, 6))
    
    data_list = []
    color_list = []
    
    for idx, (name, net) in enumerate(net_dict.items()):
        total_unemp = []
        for occ in net:
            total_unemp.extend(wrkr.time_unemployed for wrkr in occ.list_of_unemployed)
        data_list.append(total_unemp)
        color_list.append(colors[name])
    
    # Box plot
    bp = ax_box.boxplot(data_list, labels=names, patch_artist=True,
                        showfliers=False)  # Hide outliers for clarity
    for patch, color in zip(bp['boxes'], color_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax_box.set_ylabel("Time Unemployed (months)", fontsize=12)
    ax_box.set_title("Box Plot (without outliers)", fontweight='bold')
    ax_box.tick_params(axis='x', rotation=45)
    ax_box.grid(axis='y', alpha=0.3)
    
    # Violin plot
    parts = ax_violin.violinplot(data_list, positions=range(1, len(data_list) + 1),
                                  showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color_list[i])
        pc.set_alpha(0.6)
    
    ax_violin.set_xticks(range(1, len(names) + 1))
    ax_violin.set_xticklabels(names, rotation=45, ha='right')
    ax_violin.set_ylabel("Time Unemployed (months)", fontsize=12)
    ax_violin.set_title("Violin Plot", fontweight='bold')
    ax_violin.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        fig.savefig(f'{path}ltuer_boxplot.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


# ============================================================================
# LOG SCALE HISTOGRAM
# ============================================================================
def plot_ltuer_log(net_dict, names, save=False, path="./", colors=None):
    """
    Histogram with log scale on x-axis to handle skew.
    """
    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(net_dict.keys(), colors))
    
    n_bins = 30
    fig, ax = plt.subplots(figsize=(12, 6))
    
    handles = []
    labels_list = []
    
    for idx, (name, net) in enumerate(net_dict.items()):
        total_unemp = []
        for occ in net:
            total_unemp.extend(wrkr.time_unemployed for wrkr in occ.list_of_unemployed)
        
        # Add small epsilon to avoid log(0)
        total_unemp_log = [x + 0.1 for x in total_unemp]
        
        _, _, patches = ax.hist(total_unemp_log, bins=n_bins, alpha=0.5,
                               color=colors[name], label=names[idx])
        
        mean_val = np.mean(total_unemp)
        ax.axvline(mean_val + 0.1, color=colors[name], linestyle='--', linewidth=1)
        
        handles.append(patches[0])
        labels_list.append(names[idx])
    
    ax.set_xscale('log')
    ax.set_xlabel("Time Unemployed (months, log scale)", fontsize=12)
    ax.set_ylabel("Number of Workers", fontsize=12)
    ax.set_title("Distribution with Log Scale", fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    fig.legend(handles, labels_list, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=min(len(labels_list), 4), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save:
        fig.savefig(f'{path}ltuer_log.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


# FOR PLOTTING WITH OBSERVED DATA
# ============================================================================
# HELPER: Process yearly survey data
# ============================================================================
def prepare_survey_data_yearly(survey_df, year=None):
    """
    Process yearly aggregated survey data for plotting.
    
    Parameters:
    -----------
    survey_df : DataFrame
        Must have columns: YEAR, DURUNEMP_MO, n
    year : int or None
        Specific year to filter for (e.g., 2004)
        If None, aggregates across all years
    
    Returns:
    --------
    durations : array of unemployment durations
    counts : array of counts (weighted)
    """
    df = survey_df.copy()
    
    # Remove NA durations
    df = df[df['DURUNEMP_MO'].notna()]
    
    # Filter by year if specified
    if year is not None:
        df = df[df['YEAR'] == year]
    
    # Aggregate if multiple years
    if len(df) > len(df['DURUNEMP_MO'].unique()):
        df = df.groupby('DURUNEMP_MO')['n'].sum().reset_index()
    
    durations = df['DURUNEMP_MO'].values
    counts = df['n'].values
    
    return durations, counts


def expand_survey_data(durations, counts):
    """
    Expand compressed survey data into individual observations.
    E.g., if duration=3 has count=100, create array with 100 threes.
    
    Use with caution for large datasets (can be memory intensive).
    """
    expanded = []
    for dur, count in zip(durations, counts):
        expanded.extend([dur] * int(count))
    return np.array(expanded)


# ============================================================================
# CDF with observed data overlay - YEARLY VERSION
# ============================================================================
def plot_ltuer_cdf_with_observed(net_dict, names, survey_df,
                                  survey_year=None, save=False,
                                  path="./", colors=None, cap_at_observed_max=True):
    """
    CDF plot with observed survey data overlay.

    Parameters:
    -----------
    survey_year : int or None
        Year to use (e.g., 2004). If None, averages across all years.
    cap_at_observed_max : bool
        CPS DURUNEMP_MO is right-censored/top-coded (durations beyond
        ~29 months pile up at that cap rather than being reported
        exactly), while the model's time_unemployed is uncensored and
        can run far higher (the non-behavioural variants' full
        distribution has real mass out past 100+ months). Plotting both
        as-is makes the model's CDF trail visibly below the observed CDF
        for a long stretch that's a censoring artifact, not a real fit
        difference -- see _compute_duration_wasserstein_by_year in
        plot_funs.py for the same fix applied to the Wasserstein metric.
        If True (default), each model's durations are clipped to the
        observed survey's max before the CDF is computed, so the
        model's CDF reaches 1.0 at the same point the observed CDF
        necessarily does. Set to False to show the raw uncensored tail.
    """
    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(net_dict.keys(), colors))

    fig, ax_full = plt.subplots(1, 1, figsize=(7, 6))

    # Observed data first -- the model loop needs its max for capping.
    durations, counts = prepare_survey_data_yearly(survey_df, year=survey_year)
    obs_cap = durations.max()

    # Plot model results
    for idx, (name, net) in enumerate(net_dict.items()):
        total_unemp = []
        for occ in net:
            total_unemp.extend(wrkr.time_unemployed for wrkr in occ.list_of_unemployed)

        if len(total_unemp) == 0:
            continue

        if cap_at_observed_max:
            total_unemp = np.clip(total_unemp, None, obs_cap)

        # Sort for CDF
        sorted_data = np.sort(total_unemp)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        # Plot full CDF
        ax_full.plot(sorted_data, y, label=names[idx],
                    color=colors[name], linewidth=2, alpha=0.7)

    # Compute CDF from counts
    sorted_idx = np.argsort(durations)
    sorted_dur = durations[sorted_idx]
    sorted_counts = counts[sorted_idx]

    cumulative = np.cumsum(sorted_counts)
    cdf_y = cumulative / cumulative[-1]

    # Plot observed CDF
    year_label = f"Observed ({survey_year})" if survey_year else "Observed (All Years)"
    ax_full.plot(sorted_dur, cdf_y, label=year_label,
                color='black', linewidth=2.5, linestyle='--', alpha=0.9)

    # Formatting
    ax_full.set_xlabel("Time Unemployed (months)", fontsize=12)
    ax_full.set_ylabel("Cumulative Probability", fontsize=12)
    ax_full.set_title("CDF - Full Distribution", fontweight='bold')
    ax_full.grid(True, alpha=0.3)
    ax_full.set_ylim(0, 1)

    # Add quartile reference lines
    for prob in [0.25, 0.5, 0.75]:
        ax_full.axhline(prob, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Mark the CPS top-code threshold that model durations were clipped to
    if cap_at_observed_max:
        ax_full.axvline(obs_cap, color='dimgray', linestyle=':', alpha=0.6, linewidth=1.2)
        ax_full.annotate(f'CPS top-code ({obs_cap:.0f} mo)', xy=(obs_cap, 0.02),
                          xytext=(4, 0), textcoords='offset points',
                          rotation=90, fontsize=7, color='dimgray', va='bottom')

    # Legend
    handles, labels_leg = ax_full.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_leg, loc='lower center',
                  bbox_to_anchor=(0.5, -0.05),
                  ncol=min(len(labels_leg), 4), frameon=True, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save:
        fig.savefig(f'{path}ltuer_cdf_with_observed.jpg',
                   bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


# ============================================================================
# Histogram with observed data overlay - YEARLY VERSION
# ============================================================================
def plot_ltuer_hist_with_observed(net_dict, names, survey_df, 
                                   survey_year=None, save=False, 
                                   path="./", colors=None):
    """
    Histogram with observed survey data overlay.
    
    Parameters:
    -----------
    survey_year : int or None
        Year to use (e.g., 2004). If None, averages across all years.
    """
    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(net_dict.keys(), colors))
    
    n_bins = 30
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot model histograms
    for idx, (name, net) in enumerate(net_dict.items()):
        total_unemp = []
        for occ in net:
            total_unemp.extend(wrkr.time_unemployed for wrkr in occ.list_of_unemployed)
        
        if len(total_unemp) == 0:
            continue
        
        ax_full.hist(total_unemp, bins=n_bins, alpha=0.5, 
                    color=colors[name], label=names[idx], density=True)
        ax_zoom.hist(total_unemp, bins=n_bins, range=(0, 36), 
                    alpha=0.5, color=colors[name], density=True)
    
    # Add observed data as bar chart
    durations, counts = prepare_survey_data_yearly(survey_df, year=survey_year)
    
    # Normalize to density
    total_count = counts.sum()
    bin_width = 1  # Each duration is 1 month
    density = counts / (total_count * bin_width)
    
    # Plot as bars
    year_label = f"Observed ({survey_year})" if survey_year else "Observed (All Years)"
    ax_full.bar(durations, density, width=0.8, alpha=0.7, 
               color='black', label=year_label, 
               edgecolor='black', linewidth=1)
    
    mask_obs = durations <= 36
    if np.any(mask_obs):
        ax_zoom.bar(durations[mask_obs], density[mask_obs], 
                   width=0.8, alpha=0.7, color='black', 
                   edgecolor='black', linewidth=1)
    
    ax_full.set_xlabel("Time Unemployed (months)", fontsize=12)
    ax_full.set_ylabel("Density", fontsize=12)
    ax_full.set_title("Histogram - Full Distribution", fontweight='bold')
    
    ax_zoom.set_xlabel("Time Unemployed (≤ 36 months)", fontsize=12)
    ax_zoom.set_title("Histogram - Zoom ≤ 3 years", fontweight='bold')
    ax_zoom.set_xlim(0, 36)
    
    handles, labels_leg = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', 
              bbox_to_anchor=(0.5, -0.05),
              ncol=min(len(labels_leg), 4), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save:
        fig.savefig(f'{path}ltuer_hist_with_observed.jpg', 
                   bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


# ============================================================================
# KDE with observed data overlay - YEARLY VERSION
# ============================================================================
def plot_ltuer_kde_with_observed(net_dict, names, survey_df, 
                                  survey_year=None, save=False, 
                                  path="./", colors=None):
    """
    KDE plot with observed survey data overlay.
    For observed data, uses weighted KDE.
    
    Parameters:
    -----------
    survey_year : int or None
        Year to use (e.g., 2004). If None, averages across all years.
    """
    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(net_dict.keys(), colors))
    
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Plot model KDEs
    for idx, (name, net) in enumerate(net_dict.items()):
        total_unemp = []
        for occ in net:
            total_unemp.extend(wrkr.time_unemployed for wrkr in occ.list_of_unemployed)
        
        if len(total_unemp) == 0 or np.std(total_unemp) < 1e-10:
            continue
        
        try:
            kde = stats.gaussian_kde(total_unemp)
            
            x_full = np.linspace(0, max(total_unemp), 500)
            ax_full.plot(x_full, kde(x_full), label=names[idx], 
                        color=colors[name], linewidth=2, alpha=0.7)
            
            x_zoom = np.linspace(0, 36, 500)
            ax_zoom.plot(x_zoom, kde(x_zoom), 
                        color=colors[name], linewidth=2, alpha=0.7)
        except:
            continue
    
    # Add observed data KDE (weighted)
    durations, counts = prepare_survey_data_yearly(survey_df, year=survey_year)
    
    # Expand data for weighted KDE
    # Use sampling for large datasets to avoid memory issues
    if counts.sum() > 100000:
        # Sample proportionally
        probs = counts / counts.sum()
        n_samples = 10000
        obs_data = np.random.choice(durations, size=n_samples, p=probs)
    else:
        obs_data = expand_survey_data(durations, counts)
    
    try:
        obs_kde = stats.gaussian_kde(obs_data)
        
        x_full = np.linspace(0, max(durations), 500)
        year_label = f"Observed ({survey_year})" if survey_year else "Observed (All Years)"
        ax_full.plot(x_full, obs_kde(x_full), label=year_label, 
                    color='black', linewidth=2.5, linestyle='--', alpha=0.9)
        
        x_zoom = np.linspace(0, 36, 500)
        ax_zoom.plot(x_zoom, obs_kde(x_zoom), 
                    color='black', linewidth=2.5, linestyle='--', alpha=0.9)
    except Exception as e:
        print(f"Warning: Could not compute KDE for observed data: {e}")
    
    ax_full.set_xlabel("Time Unemployed (months)", fontsize=12)
    ax_full.set_ylabel("Density", fontsize=12)
    ax_full.set_title("KDE - Full Distribution", fontweight='bold')
    
    ax_zoom.set_xlabel("Time Unemployed (≤ 36 months)", fontsize=12)
    ax_zoom.set_title("KDE - Zoom ≤ 3 years", fontweight='bold')
    ax_zoom.set_xlim(0, 36)
    
    handles, labels_leg = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', 
              bbox_to_anchor=(0.5, -0.05),
              ncol=min(len(labels_leg), 4), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save:
        fig.savefig(f'{path}ltuer_kde_with_observed.jpg',
                   bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


# ============================================================================
# TIME-VARYING CDF HEATMAP  (stock-based, matches CPS duration distribution)
# ============================================================================
def plot_ltuer_cdf_heatmap(ue_stock_dict, survey_df, names,
                            calib_date=("2001-01-01", "2024-05-01"),
                            delay=25, max_dur=36,
                            save=False, path="./"):
    """
    Plot time-varying unemployment duration CDFs as heatmaps.

    Uses the stock distribution (ue_stock_df): all currently unemployed workers
    and their UEDuration at each time step, pooled by calendar year. This matches
    the CPS survey data, which also measures the stock of unemployed by duration.

    One panel per model plus one panel for observed CPS data, arranged in a grid.
    X-axis: calendar year.
    Y-axis: unemployment duration (months, 0-max_dur).
    Colour:  CDF value -- P(UEDuration <= d) among currently unemployed in that year.

    Parameters
    ----------
    ue_stock_dict : dict[str, pd.DataFrame]
        Keyed by model name. Each DataFrame has columns:
        'Time Step', 'Occupation', 'UEDuration', 'LongtermUnemp'.
    survey_df : pd.DataFrame
        CPS data with columns 'YEAR', 'DURUNEMP_MO', 'n'.
    names : list[str]
        Display labels matching ue_stock_dict key order.
    calib_date : tuple[str, str]
        Simulation calendar start/end dates.
    delay : int
        Number of burn-in time steps before calib_date[0].
    max_dur : int
        Maximum duration (months) shown on y-axis.
    save : bool
    path : str
    """
    import pandas as pd
    import matplotlib.colors as mcolors

    start_date = pd.to_datetime(calib_date[0])
    end_date   = pd.to_datetime(calib_date[1])
    all_years  = list(range(start_date.year, end_date.year + 1))
    dur_grid   = np.arange(0, max_dur + 1)
    n_years    = len(all_years)
    n_dur      = len(dur_grid)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _model_cdf_matrix(ue_df):
        """Pool all monthly stock snapshots within each calendar year, compute CDF."""
        df = ue_df.copy()
        df['_year'] = df['Time Step'].apply(
            lambda ts: (start_date + pd.DateOffset(months=int(ts - delay))).year)
        mat = np.full((n_dur, n_years), np.nan)
        for j, yr in enumerate(all_years):
            durations = df.loc[df['_year'] == yr, 'UEDuration'].dropna().values
            if len(durations) < 10:
                continue
            mat[:, j] = [(durations <= d).mean() for d in dur_grid]
        return mat

    def _obs_cdf_matrix():
        """Build CDF matrix from CPS survey counts."""
        df = survey_df[survey_df['DURUNEMP_MO'].notna()].copy()
        mat = np.full((n_dur, n_years), np.nan)
        for j, yr in enumerate(all_years):
            yr_df   = df[df['YEAR'] == yr]
            total_n = yr_df['n'].sum()
            if total_n == 0:
                continue
            mat[:, j] = [yr_df.loc[yr_df['DURUNEMP_MO'] <= d, 'n'].sum() / total_n
                         for d in dur_grid]
        return mat

    # ── compute ───────────────────────────────────────────────────────────────
    obs_mat    = _obs_cdf_matrix()
    model_mats = {name: _model_cdf_matrix(df) for name, df in ue_stock_dict.items()}

    panels   = [('Observed (CPS)', obs_mat)] + [(name, model_mats[name]) for name in ue_stock_dict]
    n_panels = len(panels)
    ncols    = min(4, n_panels)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.2),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=1)

    X          = np.arange(n_years)
    Y          = dur_grid
    tick_every = max(1, n_years // 6)
    tick_idx   = list(range(0, n_years, tick_every))

    for ax, (title, mat) in zip(axes, panels):
        ax.pcolormesh(X, Y, mat, cmap=cmap, norm=norm, shading='nearest')

        # Quartile contours
        try:
            cs = ax.contour(X, Y, mat,
                            levels=[0.25, 0.50, 0.75],
                            colors=['white', 'orange', 'white'],
                            linewidths=[0.9, 1.3, 0.9], alpha=0.85)
            ax.clabel(cs, fmt={0.25: '25th', 0.50: '50th', 0.75: '75th'},
                      fontsize=7, inline=True)
        except Exception:
            pass

        # LTU threshold
        ax.axhline(6, color='red', linewidth=1.3, linestyle='--', alpha=0.85)

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_yticks([0, 6, 12, 18, 24, 30, 36])
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(all_years[i]) for i in tick_idx],
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Duration (months)", fontsize=9)
        ax.set_xlabel("Year", fontsize=9)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:n_panels].tolist(), shrink=0.7, pad=0.03)
    cbar.set_label("P(duration <= d | currently unemployed)", fontsize=10)

    fig.suptitle(
        "Time-varying unemployment duration CDF -- stock distribution\n"
        "Contours: 25th / 50th / 75th percentile   |   Red dashed: LTU threshold (>=6 months)",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()

    if save:
        fig.savefig(f'{path}ltuer_cdf_heatmap.png',
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()
