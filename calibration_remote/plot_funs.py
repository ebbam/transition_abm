import numpy as np
import pandas as pd

_FREQ_LABEL = {'Q': 'quarterly', 'A': 'annual', 'M': 'monthly'}
import seaborn as sns
import math as math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.lines import Line2D
from scipy import stats
from plot_ltuer_options import *
from plot_style import TEXTWIDTH, DEFAULT_ASPECT_RATIO, ROW_HEIGHT, ROW_HEIGHT_3, save_plot_standard, apply_minor_gridlines, add_shared_legend, add_suptitle_spacing, add_supylabel_spacing, shade_out_of_sample, shade_out_of_sample_year

# Plotting functions
def plot_records(sim_res, date_start, date_end, save = False, path = None):
    # Modifies simulation results to pass to plotting functions
    # Separate as we do not want to overload the run function used in the calibration
    sim_record = pd.DataFrame(sim_res)
    sim_record.columns =['Time Step', 'Employment', 'Unemployment', 'Workers', 'Vacancies', 'LT Unemployed Persons', 'Current_Demand', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers']

    record1 = sim_record.groupby(['Time Step']).sum().reset_index()  

    ue_vac = record1.loc[:,['Workers', 'Unemployment', 'LT Unemployed Persons', 'Current_Demand', 'Vacancies', 'Target_Demand', 'Employed Seekers', 'Unemployed Seekers']]
    ue_vac['UE Rate'] = ue_vac['Unemployment'] / ue_vac['Workers']
    ue_vac['Vac Rate'] = ue_vac['Vacancies'] / ue_vac['Target_Demand']
    ue_vac['LTUER'] = ue_vac['LT Unemployed Persons'] / ue_vac['Unemployment']
    ue_vac['DATE'] = pd.date_range(start=date_start, end= date_end, periods=len(sim_record))

    return ue_vac

####################################################
############## LTUER ###############################
####################################################
def plot_ltuer(res_dict, macro_obs, sep_strings=None, sep=False, save=False, path=None, colors=None, calib_date=None):
    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), default_colors))

    def plot_models(ax, models, title, observed, add_legend=False):
        # "Observed" goes first, not last: matplotlib fills legend columns
        # column-major, and the leftover remainder (when the entry count
        # doesn't divide evenly by ncol) always lands in the FIRST column —
        # putting Observed last strands it alone in the final column instead
        # of grouped with the first couple of models.
        obs_line, = ax.plot(observed['DATE'], observed['LTUER'], color="black", linestyle="dotted")
        lines = [obs_line]
        labels = ["Observed"]

        for name, res in models.items():
            col = colors.get(name) if colors else None
            line, = ax.plot(res['DATE'], res['LTUER'], color=col)
            lines.append(line)
            labels.append(name)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Long-Term Unemployment Rate (LTUER)")

        if calib_date is not None:
            shade_out_of_sample(ax, calib_date[1])

        # Add legend to THIS subplot if requested
        if add_legend:
            ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5),
                     ncol=3, frameon=True, fontsize=10)

        return lines, labels

    # Prepare observed data once
    observed = macro_obs.dropna(subset=["UNRATE", "VACRATE"]).reset_index()

    if sep and sep_strings:
        # Handle titles via tuples: (match_string, title)
        categorized = {match: {} for match, _ in sep_strings}
        titles = {match: title for match, title in sep_strings}
        unmatched = {}

        for name, res in res_dict.items():
            matched = False
            for match_str, _ in sep_strings:
                if match_str in name:
                    categorized[match_str][name] = res
                    matched = True
                    break
            if not matched:
                unmatched[name] = res

        n_subplots = len(sep_strings) + (1 if unmatched else 0)
        # Width fixed at TEXTWIDTH; height set from row count (ROW_HEIGHT
        # in/row) -- a single row of panels here, independent of n_subplots
        # (the column count).
        _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT
        fig, axes = plt.subplots(1, n_subplots, figsize=(_fig_w, _fig_h), sharey=True)

        if n_subplots == 1:
            axes = [axes]

        # Each panel shows a different subset of models, but they share one
        # consistent color mapping across the whole figure — so one legend
        # below the whole figure (not per-panel) is enough to identify any
        # model in any panel, and avoids reserving/overlap issues for three
        # independent per-axes legends.
        for i, (match_str, title) in enumerate(sep_strings):
            plot_models(axes[i], categorized[match_str], f"LTUER - {title}", observed, add_legend=False)

        if unmatched:
            plot_models(axes[-1], unmatched, "LTUER - Other Models", observed, add_legend=False)

        plt.tight_layout()
        add_shared_legend(fig, colors,
                           extra_handles=[Line2D([0], [0], color='black', linestyle='dotted', label='Observed')],
                           ncol=3)

    else:
        # Single combined plot — true single-panel figure, so it uses the same
        # default size as every other single-panel figure (matches the R
        # Rmarkdown chunk default). Shared legend (not the per-axes one) so
        # column layout and bottom-margin reservation are handled consistently
        # with every other multi-model figure in this file.
        _fig_w, _fig_h = TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO
        fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
        plot_models(ax, res_dict, "Simulated and Observed Long-Term Unemployment Rates", observed, add_legend=False)

        plt.tight_layout()
        add_shared_legend(fig, colors,
                           extra_handles=[Line2D([0], [0], color='black', linestyle='dotted', label='Observed')],
                           ncol=3)

    # Save or show plot
    if save and sep:
        save_plot_standard(plt.gcf(), f'{path}ltuer_line.jpg', aspect_ratio=_fig_h / _fig_w, width_frac=max(1.0, _fig_w / TEXTWIDTH))
        plt.close()
    if save and not sep:
        save_plot_standard(plt.gcf(), f'{path}ltuer_line_single_panel.jpg', aspect_ratio=_fig_h / _fig_w, width_frac=max(1.0, _fig_w / TEXTWIDTH))
        plt.close()
    else:
        plt.show()


####################################################
############## BEV CURVES ##########################
####################################################
def plot_bev_curve(res_dict, macro_obs, sep_strings=None, sep=False, save=False, path=None):
    def plot_single_model(ax, name, res):
        ax.plot(res['UER'], res['VACRATE'])
        ax.scatter(res['UER'], res['VACRATE'], c=res.index, s=50, lw=0)
        ax.plot(macro_obs['UER'], macro_obs['VACRATE'], label="Real Values", color="grey")
        ax.set_title(name)
        ax.set_xlabel('UE Rate')
        ax.set_ylabel('Vac Rate')
        ax.legend()

    if sep and sep_strings:
        # Step 1: Categorize models
        categorized = {match: {} for match, _ in sep_strings}
        titles = {match: title for match, title in sep_strings}
        unmatched = {}

        for name, res in res_dict.items():
            matched = False
            for match_str, _ in sep_strings:
                if match_str in name:
                    categorized[match_str][name] = res
                    matched = True
                    break
            if not matched:
                unmatched[name] = res

        # Add unmatched to end if needed
        if unmatched:
            categorized["__unmatched__"] = unmatched
            titles["__unmatched__"] = "Other Models"

        num_cols = len(categorized)
        max_rows = max(len(models) for models in categorized.values())

        fig, axes = plt.subplots(max_rows, num_cols, figsize=(5 * num_cols, 4 * max_rows), squeeze=False)

        for col, (match_str, models) in enumerate(categorized.items()):
            for row, (name, res) in enumerate(models.items()):
                ax = axes[row][col]
                plot_single_model(ax, name, res)

            # Remove empty subplots in this column
            for r in range(len(models), max_rows):
                fig.delaxes(axes[r][col])

            # Set column title at the top row
            axes[0][col].set_title(titles[match_str], loc='center', fontsize=12, fontweight='bold')

        plt.tight_layout()

    else:
        # Non-separated grid layout
        n = len(res_dict)
        max_cols = 3
        cols = min(n, max_cols)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axes = axes.flatten()

        for i, (name, res) in enumerate(res_dict.items()):
            plot_single_model(axes[i], name, res)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
    

    if save:
        plt.savefig(f'{path}bev_curves.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:
        plt.show()

def plot_bev_curve_color(res_dict, macro_obs, sep_strings=None, sep=False, smooth=1, save=False, path=None, colors=None):
    """
    Plot Beveridge curves with optional smoothing.

    Each model's curve uses its own assigned colour (from `colors`) rather
    than a shared colormap, at full opacity throughout -- a light/pale
    palette entry (e.g. the lighter half of each colour family in
    model_colors.py) would fade to near-invisible against a white background
    if also alpha-faded by time, so direction is carried entirely by the
    arrows between consecutive points instead of a colour/alpha gradient.
    Arrows show the direction of increasing time directly (a Beveridge curve
    loops back on itself in UER-VACRATE space, so position alone doesn't
    indicate which way time runs).

    Parameters:
    -----------
    smooth : int
        Window size for moving average smoothing. Default is 1 (no smoothing).
        Higher values (e.g., 3, 6, 12) create smoother curves.
    colors : dict {model_name: color}, optional
        Falls back to 'steelblue' for any model not in the dict (or if
        colors=None).
    """
    def plot_single_model(ax, name, res, smooth_window=1, color=None):
        import matplotlib.dates as mdates
        import numpy as np
        import pandas as pd

        # Apply smoothing if requested
        if smooth_window > 1:
            x = res['UER'].rolling(window=smooth_window, center=True).mean().values
            y = res['VACRATE'].rolling(window=smooth_window, center=True).mean().values
            dates = res['DATE'].values

            # Remove NaN values from edges
            valid_idx = ~(np.isnan(x) | np.isnan(y))
            x = x[valid_idx]
            y = y[valid_idx]
            dates = dates[valid_idx]
        else:
            x = res['UER'].values
            y = res['VACRATE'].values
            dates = res['DATE'].values

        model_color = color if color is not None else 'steelblue'

        # Arrows between consecutive points indicate the direction of
        # increasing time -- position alone can't, since a Beveridge curve
        # loops back over similar (UER, VACRATE) regions across different
        # periods.
        ax.quiver(x[:-1], y[:-1], np.diff(x), np.diff(y),
                  angles='xy', scale_units='xy', scale=1,
                  color=model_color, width=0.006,
                  headwidth=4, headlength=5, headaxislength=4.5,
                  zorder=3)

        # Plot observed data (optionally smoothed) -- no label, described in the
        # shared legend instead.
        if smooth_window > 1:
            obs_x = macro_obs['UER'].rolling(window=smooth_window, center=True).mean()
            obs_y = macro_obs['VACRATE'].rolling(window=smooth_window, center=True).mean()
            ax.plot(obs_x, obs_y, color="grey", alpha=0.7, zorder=1)
        else:
            ax.plot(macro_obs['UER'], macro_obs['VACRATE'], color="grey", alpha=0.7, zorder=1)

        # No per-panel model-name title -- each panel's own colour (matched to
        # the shared legend below) identifies the model instead, same
        # convention as the dumbbell panels in plot_wage_ue_ee_comparison.
        ax.set_xlabel('UE Rate')
        ax.set_ylabel('Vac. Rate')

        # Set axis limits to ensure all data is visible
        ax.set_xlim(x.min() * 0.95, x.max() * 1.05)
        ax.set_ylim(y.min() * 0.95, y.max() * 1.05)

    if sep and sep_strings:
        # Step 1: Categorize models
        categorized = {match: {} for match, _ in sep_strings}
        titles = {match: title for match, title in sep_strings}
        unmatched = {}

        for name, res in res_dict.items():
            matched = False
            for match_str, _ in sep_strings:
                if match_str in name:
                    categorized[match_str][name] = res
                    matched = True
                    break
            if not matched:
                unmatched[name] = res

        # Add unmatched to end if needed
        if unmatched:
            categorized["__unmatched__"] = unmatched
            titles["__unmatched__"] = "Other Models"

        num_cols = len(categorized)
        max_rows = max(len(models) for models in categorized.values())

        # Width fixed at TEXTWIDTH; height set from row count only (ROW_HEIGHT
        # in/row), independent of num_cols (the sep_strings groups) -- shared
        # convention across every multi-row figure so N rows is always the
        # same height regardless of column count.
        _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * max_rows
        fig = plt.figure(figsize=(_fig_w, _fig_h))

        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(max_rows, num_cols, figure=fig, wspace=0.3, hspace=0.3)

        axes = np.empty((max_rows, num_cols), dtype=object)
        for row in range(max_rows):
            for col in range(num_cols):
                axes[row, col] = fig.add_subplot(gs[row, col])

        for col, (match_str, models) in enumerate(categorized.items()):
            for row, (name, res) in enumerate(models.items()):
                ax = axes[row, col]
                model_color = colors.get(name) if colors else None
                plot_single_model(ax, name, res, smooth_window=smooth, color=model_color)

            # Remove empty subplots in this column
            for r in range(len(models), max_rows):
                axes[r, col].axis('off')

            # Set column title at the top row
            axes[0, col].set_title(titles[match_str], loc='center', fontweight='bold')

    else:
        # Non-separated grid layout
        n = len(res_dict)
        max_cols = 2
        cols = min(n, max_cols)
        rows = math.ceil(n / cols)

        _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * rows
        fig = plt.figure(figsize=(_fig_w, _fig_h))

        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.3, hspace=0.3)

        axes = []
        for row in range(rows):
            for col in range(cols):
                if len(axes) < n:
                    axes.append(fig.add_subplot(gs[row, col]))

        for i, (name, res) in enumerate(res_dict.items()):
            model_color = colors.get(name) if colors else None
            plot_single_model(axes[i], name, res, smooth_window=smooth, color=model_color)

    _legend_colors = {name: (colors.get(name, 'steelblue') if colors else 'steelblue')
                      for name in res_dict}
    add_shared_legend(
        fig, _legend_colors,
        extra_handles=[Line2D([0], [0], color='black', linestyle='none',
                               marker=r'$\rightarrow$', markersize=12,
                               label='Arrows show the direction of increasing time')],
        ncol=3,
    )

    if save:
        save_plot_standard(fig, f'{path}bev_curves.jpg', aspect_ratio=_fig_h / _fig_w, width_frac=max(1.0, _fig_w / TEXTWIDTH))
        plt.close()
    else:
        plt.show()



####################################################
############## AVERAGE WAGE BY OCCUPATION ##########
####################################################
def plot_avg_wages(mod_results_dict, save=False, path=None):
    """
    Creates scatterplots of mean wage by occupation for each element in the dictionary.
    
    Parameters:
    - sim_results_dict: Dictionary containing simulation results DataFrames
    - save: Boolean indicating whether to save the plots
    - path: Path to save the plots if save=True
    """
    colors = ["orange", "blue", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    # First, calculate global y-axis limits
    plt.figure(figsize=(12, 6))
    for i, (name, res) in enumerate(mod_results_dict.items()):
        # Create the scatterplot
        
        plt.plot(res['DATE'], res['AVGWAGE'], color = colors[i], label = name)
        
        # Customize the plot
        plt.title(f'Total Wages', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Total Wages')
        
        # Set x-axis ticks to show occupation IDs in the sorted order
        #plt.xticks(mean_wage['x_pos'], mean_wage['Occupation'], rotation=45, ha='right')
        
    # Adjust layout
    plt.tight_layout()
    plt.legend()

    # Save or show
    if save:
        plt.savefig(f'{path}avg_wage_overall.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:    
        plt.show()


####################################################
############## RELATIVE WAGES ##########
####################################################
def plot_rel_wages(mod_results_dict, save=False, path=None, freq='M', common_y_axis=False, 
                   colors=None, unemp_only=False, lowess=False, lowess_frac=0.2, suffix = None):
    """
    Creates plots of relative wages with optional smoothing:
    - Left: Unemployed relative wages
    - Right: Employed relative wages
    Includes shading to show deviations above/below 1.0
    
    Parameters:
    - mod_results_dict: Dictionary containing simulation results DataFrames
    - save: Boolean indicating whether to save the plots
    - path: Path to save the plots if save=True
    - freq: Frequency for resampling ('M' for monthly, 'Y' for annual, etc.)
    - common_y_axis: Boolean to use common y-axis limits
    - colors: Dictionary mapping model names to colors
    - unemp_only: Boolean to plot only unemployed wages
    - lowess: Boolean to apply LOWESS smoothing
    - lowess_frac: Fraction of data used for LOWESS smoothing (0.05-0.3, default 0.1)
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_smooth

    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(mod_results_dict.keys(), colors))

    def apply_smoothing(values, use_lowess=False, frac=0.1):
        """Apply LOWESS smoothing if requested"""
        if use_lowess:
            smoothed = lowess_smooth(values, np.arange(len(values)), frac=frac)
            return smoothed[:, 1]
        else:
            return values

    # Create figure with two subplots
    if not unemp_only:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 6 / 18))
        
        for name, res in mod_results_dict.items():
            # Resample to annual frequency
            if freq != 'M':
                annual_data = res.set_index('DATE').resample(freq).mean()
            else:
                annual_data = res

            # Create arrays for x-axis values
            dates_array = np.array([d for d in annual_data.index])

            # Plot unemployed relative wages (with optional smoothing)
            u_wages_raw = annual_data['U_REL_WAGE_MEAN'].values
            u_wages = apply_smoothing(u_wages_raw, use_lowess=lowess, frac=lowess_frac)

            ax1.plot(dates_array, u_wages,
                    color=colors[name], zorder=3)

            # Optional: show raw data as faint background
            if lowess:
                ax1.plot(dates_array, u_wages_raw,
                        color=colors[name], alpha=0.15, linewidth=0.5, zorder=1)

            # Add shading
            ax1.fill_between(dates_array, u_wages, 1,
                            color=colors[name], alpha=0.2, zorder=2)

            # Plot employed relative wages (with optional smoothing)
            e_wages_raw = annual_data['E_REL_WAGE_MEAN'].values
            e_wages = apply_smoothing(e_wages_raw, use_lowess=lowess, frac=lowess_frac)

            ax2.plot(dates_array, e_wages,
                    color=colors[name], zorder=3)

            # Optional: show raw data as faint background
            if lowess:
                ax2.plot(dates_array, e_wages_raw,
                        color=colors[name], alpha=0.15, linewidth=0.5, zorder=1)

            # Add shading
            ax2.fill_between(dates_array, e_wages, 1,
                            color=colors[name], alpha=0.2, zorder=2)
        
        # Add horizontal line at y=1
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, zorder=2)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, zorder=2)
        
        # Customize unemployed plot
        title_suffix = " (LOWESS Smoothed)" if lowess else ""
        ax1.set_title(f'Unemployed Relative Wages{title_suffix}', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Relative Wages')
        
        # Customize employed plot  
        ax2.set_title(f'Employed Relative Wages{title_suffix}', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Relative Wages')

        if common_y_axis:
            # Get the y-axis limits from both plots
            y1_min, y1_max = ax1.get_ylim()
            y2_min, y2_max = ax2.get_ylim()
            
            # Find the common min and max
            common_min = min(y1_min, y2_min)
            common_max = max(y1_max, y2_max)
            
            # Apply the same limits to both axes
            ax1.set_ylim(common_min, common_max)
            ax2.set_ylim(common_min, common_max)

        # Adjust layout to make room for legend
        plt.tight_layout()
        add_shared_legend(fig, colors, ncol=3)

    else:
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO))

        for name, res in mod_results_dict.items():
            # Resample to annual frequency
            if freq != 'M':
                annual_data = res.set_index('DATE').resample(freq).mean()
                dates_array = np.array([d for d in annual_data.index])
            else:
                annual_data = res
                dates_array = annual_data['DATE'].values  # Use DATE column

            # Plot unemployed relative wages (with optional smoothing)
            u_wages_raw = annual_data['U_REL_WAGE_MEAN'].values
            u_wages = apply_smoothing(u_wages_raw, use_lowess=lowess, frac=lowess_frac)

            plt.plot(dates_array, u_wages,
                    color=colors[name], zorder=3)

            # Optional: show raw data as faint background
            if lowess:
                plt.plot(dates_array, u_wages_raw,
                        color=colors[name], alpha=0.15, linewidth=0.5, zorder=1)

            # Add shading
            plt.fill_between(dates_array, u_wages, 1,
                            color=colors[name], alpha=0.2, zorder=2)

        # Add horizontal line at y=1
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, zorder=2)

        # Customize unemployed plot
        title_suffix = " (LOWESS Smoothed)" if lowess else ""
        plt.title(f'Unemployed Relative Wages{title_suffix}', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Relative Wages')

        # Adjust layout to make room for legend
        plt.tight_layout()
        add_shared_legend(fig, colors, ncol=3)
    
    # Save or show
    if save:
        if not unemp_only:
            save_plot_standard(fig, f'{path}relative_wages_{suffix}.jpg', aspect_ratio=6 / 18)
        else:
            save_plot_standard(fig, f'{path}relative_wages_{suffix}.jpg')
        plt.close()
    else:
        plt.show()

def hires_seps_rate(mod_results_dict, jolts, save=False, path=None, colors=None, calib_v_line=None):

    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(mod_results_dict.keys(), colors))

    # One column: all 6 models overlaid per panel (identified by colour via
    # the shared legend), one panel per rate type -- no more Non-behavioral
    # vs Behavioral column split.
    # (sim_column, obs_column, sim_label, obs_label) -- sim_column/obs_column
    # are the actual DataFrame keys used to look up the data (do not rename
    # these); sim_label/obs_label are the clean display text for the title.
    rate_types = [
        ("Hires Rate",       "HIRESRATE",  "Simulated Hires Rate",          "Observed Hires Rate"),
        ("Separations Rate", "SEPSRATE",   "Simulated Separations Rate",    "Observed Separations Rate"),
        ("EE_Trans_Rate",    "QUITSRATE",  "Simulated E-E Transition Rate", "Observed Quits Rate"),
        ("E-U Rate",         "LAYOFFRATE", "Simulated E-U Transition Rate", "Observed Layoff Rate"),
    ]

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT_3 * len(rate_types)
    fig, axes = plt.subplots(len(rate_types), 1, figsize=(_fig_w, _fig_h), sharex=True)

    for row_idx, (rate_col_sim, rate_col_obs, sim_label, obs_label) in enumerate(rate_types):
        ax = axes[row_idx]

        # Plot observed JOLTS rate
        ax.plot(jolts['DATE'], jolts[rate_col_obs], color='black', linestyle='--')

        # Plot each model
        for model_name, df in mod_results_dict.items():
            ax.plot(df['DATE'], df[rate_col_sim], color = colors[model_name])

        ax.set_title(f'{sim_label} vs. {obs_label}', fontweight='bold')
        ax.set_ylabel('Rate')
        ax.set_xlim(df['DATE'].min(), df['DATE'].max())
        if row_idx == len(rate_types) - 1:
            ax.set_xlabel('Date')
        if calib_v_line is not None:
            shade_out_of_sample(ax, calib_v_line[1])

    plt.tight_layout()
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='black', linestyle='--', label='Observed')],
                       ncol=3)

    if save:
        save_plot_standard(fig, f'{path}hires_seps_rate_grid.jpg', aspect_ratio=_fig_h / _fig_w)
        plt.close()
    else:
        plt.show()
    

####################################################
############## UER & VAC ###########################
####################################################
# def plot_uer_vac(res_dict, macro_obs, recessions, sep = False, save = False, path = None):
#     n = len(res_dict)
#     if sep:
#         cols = 2
#     else:
#         max_cols = 3
#         cols = min(n, max_cols)
#     rows = math.ceil(n / cols)

#     fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
#     axes = axes.flatten()  # Flatten to make indexing easy

#     for i, (name, res) in enumerate(res_dict.items()):
#         ax = axes[i]
#         ax.plot(res['DATE'], res['Vac Rate'], label=f'Sim. VR', color = "red")
#         ax.plot(res['DATE'], res['UE Rate'], label=f'Sim. UER', color = "blue")
#         ax.plot(macro_obs['DATE'], macro_obs['UER'], color="blue", linestyle="dotted", label="Obs. UER")
#         ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="red", linestyle="dotted", label="Obs. VR")
#         #recessions.plot.area(ax=ax, 'DATE', color="grey", alpha=0.2)

#         ax.set_title(name)
#         ax.set_xlabel('DATE')
#         ax.set_ylabel('Rate')
#         ax.legend()

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     if save:
#         plt.savefig(f'{path}uer_vac.jpg', dpi = 300)
#     plt.show()

# def plot_uer_vac(res_dict, macro_obs, recessions=None, sep_strings=None, sep=False, save=False, path=None):
#     colors = ["skyblue", "orange",]# "brown", "pink"]
#     def plot_uer(ax, models, title):
#         for i, (name, res) in enumerate(models.items()):
#             ax.plot(res['DATE'], res['UER'], label=f'Sim. UER: {name}', color=colors[i % len(colors)])
#         ax.plot(macro_obs['DATE'], macro_obs['UER'], color="grey", linestyle="dotted", label="Obs. UER")
#         if recessions is not None:
#             for _, row in recessions.iterrows():
#                 ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
#         ax.set_title(title)
#         ax.set_xlabel("Date")
#         ax.set_ylabel("Unemployment Rate")
#         ax.legend()

#     def plot_vac(ax, models, title):
#         for i, (name, res) in enumerate(models.items()):
#             ax.plot(res['DATE'], res['VACRATE'], label=f'Sim. Vac Rate: {name}', color=colors[i % len(colors)])
#         ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="grey", linestyle="dotted", label="Obs. VR")
#         if recessions is not None:
#             for _, row in recessions.iterrows():
#                 ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
#         ax.set_xlabel("Date")
#         ax.set_ylabel("Vacancy Rate")
#         ax.legend()

#     if sep and sep_strings:
#         # Step 1: Categorize
#         categorized = {match: {} for match, _ in sep_strings}
#         titles = {match: title for match, title in sep_strings}
#         unmatched = {}

#         for name, res in res_dict.items():
#             matched = False
#             for match_str, _ in sep_strings:
#                 if match_str in name:
#                     categorized[match_str][name] = res
#                     matched = True
#                     break
#             if not matched:
#                 unmatched[name] = res

#         # Add unmatched group
#         if unmatched:
#             categorized["__unmatched__"] = unmatched
#             titles["__unmatched__"] = "Other Models"

#         num_plots = len(categorized)
#         fig, axes = plt.subplots(2, num_plots, figsize=(6 * num_plots, 8), sharex='col')
#         if num_plots == 1:
#             axes = np.array(axes).reshape(2, 1)

#         for i, (match_str, models) in enumerate(categorized.items()):
#             plot_uer(axes[0, i], models, titles[match_str])
#             plot_vac(axes[1, i], models, titles[match_str])

#         plt.tight_layout()

#     else:
#         # All in separate plots (grid layout)
#         n = len(res_dict)
#         fig, axes = plt.subplots(2, n, figsize=(6 * n, 8), sharex='col')
#         if n == 1:
#             axes = np.array(axes).reshape(2, 1)

#         for i, (name, res) in enumerate(res_dict.items()):
#             plot_uer(axes[0, i], {name: res}, name)
#             plot_vac(axes[1, i], {name: res}, name)

#         plt.tight_layout()

#     # Save or show
#     if save:
#         plt.savefig(f'{path}uer_vac.jpg', dpi=300)
#     plt.show()

def plot_uer_vac_steady_state(res_dict, macro_obs, calib_date, plot_colors=None, sep_strings=None, sep=False, save=False, path=None, free_date_scale=False, delay_ref=None, smooth=None, suffix=None):
            # Default colors if plot_colors not provided
            default_colors = ["cornflowerblue", "darkorange", "green", "red", "purple", "brown"]
            start = pd.to_datetime(calib_date[0])
            
            # Calculate model start date (delay_ref months BEFORE calib_date[0])
            if delay_ref is not None:
                model_start_date = start - pd.DateOffset(months=delay_ref)
            else:
                model_start_date = start

            def plot_uer(ax, models, title, show_legend=False):
                lines = []
                labels = []
                for i, (name, res) in enumerate(models.items()):
                    if free_date_scale:
                        # Start the model dates from model_start_date (not calib_date[0])
                        res['DATE'] = pd.date_range(
                            start=model_start_date,
                            periods=len(res),
                            freq="ME"
                        )
                    
                    # Get color from plot_colors dict if available, otherwise use default
                    if plot_colors is not None and name in plot_colors:
                        color = plot_colors[name]
                    else:
                        color = default_colors[i % len(default_colors)]
                    
                    # Apply smoothing if requested
                    if smooth is not None and smooth > 1:
                        uer_smoothed = res['UER'].rolling(window=smooth, center=True).mean()
                    else:
                        uer_smoothed = res['UER']
                    
                    line, = ax.plot(res['DATE'], uer_smoothed, color=color, label=name)
                    if show_legend:
                        lines.append(line)
                        labels.append(name)
                
                # Plot observed data (no offset needed)
                obs_line, = ax.plot(macro_obs['DATE'], macro_obs['UER'], color="grey", linestyle="dotted", label="Observed")
                if show_legend:
                    lines.append(obs_line)
                    labels.append("Observed")
                
                # Add vertical line at calib_date[0] (where calibration starts)
                if delay_ref is not None:
                    ax.axvline(x=start, color='black', linestyle='--', linewidth=1, label='Calibration Start')

                # Set title (bold)
                ax.set_title(title, fontweight="bold", pad=15)

                # Add subtitle if smoothing is applied (italic, non-bold)
                if smooth is not None and smooth > 1:
                    ax.text(0.5, 1.01, f"Smoothing over {smooth} points",
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=8, style='italic', color='black')

                ax.set_xlabel("Date")
                ax.set_ylabel("Unemployment Rate")

                return lines, labels

            # Check if we're separating plots or plotting all together
            if sep and sep_strings:
                # Step 1: Categorize
                categorized = {match: {} for match, _ in sep_strings}
                titles = {match: title for match, title in sep_strings}
                unmatched = {}

                for name, res in res_dict.items():
                    matched = False
                    for match_str, _ in sep_strings:
                        if match_str in name:
                            categorized[match_str][name] = res
                            matched = True
                            break
                    if not matched:
                        unmatched[name] = res

                # Add unmatched group
                if unmatched:
                    categorized["__unmatched__"] = unmatched
                    titles["__unmatched__"] = "Other Models"

                num_plots = len(categorized)
                # Width fixed at TEXTWIDTH; height set from row count
                # (ROW_HEIGHT in/row) -- a single row of panels here,
                # independent of num_plots (the column count), matching
                # plot_ltuer's equivalent 1-row-of-panels case.
                _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT
                fig, axes = plt.subplots(1, num_plots, figsize=(_fig_w, _fig_h), sharex='col')
                if num_plots == 1:
                    axes = np.array(axes).reshape(1, 1)

                for i, (match_str, models) in enumerate(categorized.items()):
                    plot_uer(axes[i], models, titles[match_str], show_legend=False)

            else:
                # Plot all together on single subplot -- true single-panel figure,
                # so it uses the same default size as every other single-panel
                # figure (matches the R Rmarkdown chunk default).
                _fig_w, _fig_h = TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO
                fig, ax = plt.subplots(1, 1, figsize=(_fig_w, _fig_h))

                # Plot all models on the same axes
                plot_uer(ax, res_dict, "Steady State Unemployment Rate", show_legend=False)

            plt.tight_layout()
            _colors = {name: (plot_colors[name] if plot_colors and name in plot_colors
                               else default_colors[i % len(default_colors)])
                       for i, name in enumerate(res_dict.keys())}
            add_shared_legend(fig, _colors,
                               extra_handles=[Line2D([0], [0], color='grey', linestyle='dotted', label='Observed')],
                               ncol=3)

            # Save or show
            if save:
                save_plot_standard(fig, f'{path}uer_vac_{suffix}.jpg', aspect_ratio=_fig_h / _fig_w, width_frac=max(1.0, _fig_w / TEXTWIDTH))
                plt.close()
            else:
                plt.show()

def plot_uer_vac(res_dict, macro_obs, calib_date, sep_strings=None, sep=False, save=False, path=None, free_date_scale=False, plot_colors=None):
    _default_colors = ["cornflowerblue", "darkorange", "green", "red", "purple", "brown"]
    def _col(name, i):
        if plot_colors and name in plot_colors:
            return plot_colors[name]
        return _default_colors[i % len(_default_colors)]

    start = pd.to_datetime(calib_date[0])
    model_start_date = start + pd.DateOffset(months=25)

    def plot_uer(ax, models, title, show_legend=False):
        lines = []
        labels = []
        for i, (name, res) in enumerate(models.items()):
            if free_date_scale:
                res['DATE'] = pd.date_range(
                    start=calib_date[0],
                    periods=len(res),
                    freq="ME"
                )
            line, = ax.plot(res['DATE'], res['UER'], color=_col(name, i))
            if show_legend:
                lines.append(line)
                labels.append(name)

        obs_line, = ax.plot(macro_obs['DATE'], macro_obs['UER'], color="grey", linestyle="dotted")
        if show_legend:
            lines.append(obs_line)
            labels.append("Observed")

        ax.set_title(title, fontweight="bold")
        #ax.set_xlabel("Date")
        ax.set_ylabel("Unemployment Rate")
        shade_out_of_sample(ax, calib_date[1])

        return lines, labels

    def plot_vac(ax, models, title):
        for i, (name, res) in enumerate(models.items()):
            if free_date_scale:
                res['DATE'] = pd.date_range(
                    start=calib_date[0],
                    periods=len(res),
                    freq="ME"
                )
            ax.plot(res['DATE'], res['VACRATE'], color=_col(name, i))
        ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="grey", linestyle="dotted")

        ax.set_xlabel("Date")
        ax.set_ylabel("Vacancy Rate")
        shade_out_of_sample(ax, calib_date[1])

    def plot_uer_error(ax, models, title):
        for i, (name, res) in enumerate(models.items()):
            if free_date_scale:
                res['DATE'] = pd.date_range(
                    start=calib_date[0],
                    periods=len(res),
                    freq="ME"
                )
            if len(res['UER']) == len([macro_obs['UER']]):
                ax.fill_between(res['DATE'], 0, res['UER'] - macro_obs['UER'], color=_col(name, i), alpha=0.5)
            else:
                ax.fill_between(res['DATE'], 0, res['UER'] - macro_obs['UER'].iloc[:len(res['UER'])], color=_col(name, i), alpha=0.5)

        ax.set_xlabel("Date")
        ax.set_ylabel("UER Error (Simulated-Observed)")
        shade_out_of_sample(ax, calib_date[1])

    if sep and sep_strings:
        # Step 1: Categorize
        categorized = {match: {} for match, _ in sep_strings}
        titles = {match: title for match, title in sep_strings}
        unmatched = {}

        for name, res in res_dict.items():
            matched = False
            for match_str, _ in sep_strings:
                if match_str in name:
                    categorized[match_str][name] = res
                    matched = True
                    break
            if not matched:
                unmatched[name] = res

        # Add unmatched group
        if unmatched:
            categorized["__unmatched__"] = unmatched
            titles["__unmatched__"] = "Other Models"

        num_plots = len(categorized)
        # Width fixed at TEXTWIDTH; height set from row count (ROW_HEIGHT
        # in/row) -- 2 stacked panels (UER + vacancy rate), independent of
        # num_plots (the column count). Matches the sep=False branch below.
        _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * 2
        fig, axes = plt.subplots(2, num_plots, figsize=(_fig_w, _fig_h), sharex='col')
        if num_plots == 1:
            axes = np.array(axes).reshape(2, 1)

        for i, (match_str, models) in enumerate(categorized.items()):
            plot_uer(axes[0, i], models, titles[match_str], show_legend=False)

            # Plot middle and bottom rows
            #plot_uer_error(axes[1, i], models, titles[match_str])
            plot_vac(axes[1, i], models, titles[match_str])

        # # Ensure same y-axis limits for the UER error row (2nd row)
        # y_mins, y_maxs = [], []
        # for ax in axes[1, :]:
        #     y_min, y_max = ax.get_ylim()
        #     y_mins.append(y_min)
        #     y_maxs.append(y_max)
        # common_min = min(y_mins)
        # common_max = max(y_maxs)
        # for ax in axes[1, :]:
        #     ax.set_ylim(common_min, common_max)

    else:
        # All models on two shared panels: UER (top) and vacancy rate (bottom)
        #_fig_w, _fig_h = TEXTWIDTH, TEXTWIDTH * 8 / 12
        _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * 2
        fig, (ax_uer, ax_vac) = plt.subplots(2, 1, figsize=(_fig_w, _fig_h), sharex=True)

        plot_uer(ax_uer, res_dict, "Simulated and Observed Unemployment and Vacancy Rates", show_legend=False)
        plot_vac(ax_vac, res_dict, "Vacancy Rate")

    plt.tight_layout()
    _uer_vac_colors = {name: _col(name, i) for i, name in enumerate(res_dict.keys())}
    add_shared_legend(fig, _uer_vac_colors,
                       extra_handles=[Line2D([0], [0], color='grey', linestyle='dotted', label='Observed')],
                       ncol=3)

    # Save or show
    if save and sep:
        save_plot_standard(fig, f'{path}uer_vac.jpg', aspect_ratio=_fig_h / _fig_w, width_frac=max(1.0, _fig_w / TEXTWIDTH))
        plt.close()
    if save and not sep:
        save_plot_standard(fig, f'{path}uer_vac_single_panel.jpg', aspect_ratio=_fig_h / _fig_w, width_frac=max(1.0, _fig_w / TEXTWIDTH))
        plt.close()
    else:
        plt.show()



def plot_uer_vac_single_row(res_dict, macro_obs, recessions=None, sep_strings=None, sep=False, save=False, path=None):
    colors = ["skyblue", "lightcoral", "purple", "green", "orange", "brown", "pink"]
    def plot_uer(ax, models, title):
        for i, (name, res) in enumerate(models.items()):
            ax.plot(res['DATE'], res['UER'], label=f'Sim. UER: {name}', color=colors[i % len(colors)])
        ax.plot(macro_obs['DATE'], macro_obs['UER'], color="blue", linestyle="dotted", label="Obs. UER")
        if recessions is not None:
            for _, row in recessions.iterrows():
                ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Unemployment Rate")
        ax.legend()

    def plot_vac(ax, models, title):
        for i, (name, res) in enumerate(models.items()):
            ax.plot(res['DATE'], res['VACRATE'], label=f'Sim. Vac Rate: {name}', color=colors[i % len(colors)])
        ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="red", linestyle="dotted", label="Obs. VR")
        if recessions is not None:
            for _, row in recessions.iterrows():
                ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Vacancy Rate")
        ax.legend()

    if sep and sep_strings:
        # Step 1: Categorize
        categorized = {match: {} for match, _ in sep_strings}
        titles = {match: title for match, title in sep_strings}
        unmatched = {}

        for name, res in res_dict.items():
            matched = False
            for match_str, _ in sep_strings:
                if match_str in name:
                    categorized[match_str][name] = res
                    matched = True
                    break
            if not matched:
                unmatched[name] = res

        # Add unmatched group
        if unmatched:
            categorized["__unmatched__"] = unmatched
            titles["__unmatched__"] = "Other Models"

        num_plots = len(categorized)
        fig, axes = plt.subplots(2, num_plots, figsize=(6 * num_plots, 8), sharex='col')
        if num_plots == 1:
            axes = np.array(axes).reshape(2, 1)

        for i, (match_str, models) in enumerate(categorized.items()):
            plot_uer(axes[0, i], models, titles[match_str])
            plot_vac(axes[1, i], models, titles[match_str])

        plt.tight_layout()

    else:
        # All in separate plots (grid layout)
        n = len(res_dict)
        fig, axes = plt.subplots(2, n, figsize=(6 * n, 8), sharex='col')
        if n == 1:
            axes = np.array(axes).reshape(2, 1)

        for i, (name, res) in enumerate(res_dict.items()):
            plot_uer(axes[0, i], {name: res}, name)
            plot_vac(axes[1, i], {name: res}, name)

        plt.tight_layout()

    # Save or show
    if save:
        plt.savefig(f'{path}uer_vac.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:
        plt.show()



####################################################
############## SEEKER COMPOSITION ##################
####################################################
####################################################
############## SEEKER COMPOSITION ##################
####################################################
def plot_seeker_comp(res_dict, observation, share=False, sep=False, save=False, path=None, colors=None, calib_v_line=None):
    import matplotlib.patches as mpatches
    
    n = len(res_dict)
    if sep:
        cols = 2
    else:
        max_cols = 2
        cols = min(n, max_cols)

    rows = math.ceil(n / cols)

    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), colors))

    # Sized to print at exactly TEXTWIDTH, ROW_HEIGHT in/row.
    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig, axes = plt.subplots(rows, cols, figsize=(_fig_w, _fig_h), squeeze=False)
    axes = axes.flatten()  # Flatten to make indexing easy

    _multisim = any('_ci_lower' in c for c in next(iter(res_dict.values())).columns) if res_dict else False

    for i, (name, res) in enumerate(res_dict.items()):
        ax = axes[i]

        if not share:
            ax.stackplot(res['DATE'], res['Employed Seekers'], res['Unemployed Seekers'],
                         colors=[colors[name], 'lightgrey'])
            if _multisim and 'Employed Seekers_ci_lower' in res.columns:
                ax.fill_between(res['DATE'],
                                res['Employed Seekers_ci_lower'],
                                res['Employed Seekers_ci_upper'],
                                color=colors[name], alpha=0.20)
        elif share:
            total     = res["Employed Seekers"] + res["Unemployed Seekers"]
            emp_share = res["Employed Seekers"] / total
            unemp_share = res["Unemployed Seekers"] / total
            ax.stackplot(res['DATE'], emp_share, unemp_share,
                         colors=[colors[name], "lightgrey"])
            if _multisim and 'Employed Seekers_ci_lower' in res.columns:
                total_lo  = res["Employed Seekers_ci_lower"] + res["Unemployed Seekers"]
                total_hi  = res["Employed Seekers_ci_upper"] + res["Unemployed Seekers"]
                share_lo  = res["Employed Seekers_ci_lower"] / total_lo
                share_hi  = res["Employed Seekers_ci_upper"] / total_hi
                ax.fill_between(res['DATE'], share_lo, share_hi,
                                color=colors[name], alpha=0.25)

        # Plot observed line
        obs_line, = ax.plot(observation['DATE'], observation['Seeker Composition'],
                            color="black", linestyle="dotted")

        if calib_v_line is not None:
            shade_out_of_sample(ax, calib_v_line[1])

    # Set common axis labels and title
    fig.suptitle("Monthly Composition of Job Seekers", y=0.98, fontweight='bold')
    fig.supxlabel("Date")
    fig.supylabel("Composition of Job Seekers")

    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors,
                       extra_handles=[mpatches.Patch(color='lightgrey', label='Unemp Seekers'),
                                      Line2D([0], [0], color='black', linestyle='dotted', label='Observed')],
                       ncol=3)

    # Show plot
    if save:
        save_plot_standard(fig, f'{path}seeker_composition.png',
                            aspect_ratio=_fig_h / _fig_w,
                            width_frac=max(1.0, _fig_w / TEXTWIDTH))
        plt.close()
    else:
        plt.show()

def plot_seeker_comp_line(res_dict, observation, save = False, path = None, colors = None, calib_v_line=None):

    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), colors))

    plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO))
    for name, res in res_dict.items():
        plt.plot(res['DATE'], res["Seeker Composition"], color = colors[name])

    plt.plot(observation['DATE'], observation['Seeker Composition'], color='grey', linestyle = "dotted")
    plt.title("Monthly Composition of Job Seekers")  # Figure-wide title
    plt.xlabel("Date")  # Shared x-axis label
    plt.ylabel("Employed Share of Job Seekers")  # Shared y-axis label

    if calib_v_line is not None:
        shade_out_of_sample(plt.gca(), calib_v_line[1])

    plt.tight_layout()
    fig = plt.gcf()
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='grey', linestyle='dotted', label='Observed')],
                       ncol=3)

    # Show plot
    if save:
        save_plot_standard(fig, f'{path}seeker_composition_line.png')
        plt.close()
    else:
        plt.show()


def plot_applications_and_composition(seekers_recs, comp_dict, comp_observation, save=False, path=None, colors=None, calib_v_line=None):
    """
    Combined 2-row, 1-column figure with one shared legend -- kept alongside,
    not instead of, the separate applications_per_unemployed_seeker.png and
    seeker_composition_line.png figures.

    Top:    Applications per Unemployed Seeker over time (seekers_recs, same
            data as the standalone applications figure; CI ribbon drawn if
            present, i.e. multi-sim mode).
    Bottom: Employed share of job seekers (seeker composition), with the
            observed series as a grey dashed reference.
    """
    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(seekers_recs.keys(), default_colors))

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * 2
    fig, (ax_apps, ax_comp) = plt.subplots(2, 1, figsize=(_fig_w, _fig_h), sharex=True)

    for name, item in seekers_recs.items():
        ax_apps.plot(item['DATE'], item['Application Effort (U)'], color=colors[name])
        if 'Application Effort (U)_ci_lower' in item.columns:
            ax_apps.fill_between(item['DATE'], item['Application Effort (U)_ci_lower'],
                                  item['Application Effort (U)_ci_upper'], color=colors[name], alpha=0.15)
    ax_apps.set_ylabel('Applications Sent per\nUnemployed Seeker')
    ax_apps.set_title('Application Rate Over Time', fontweight='bold')
    ax_apps.set_xlabel('Date')

    for name, res in comp_dict.items():
        ax_comp.plot(res['DATE'], res['Seeker Composition'], color=colors[name])
    ax_comp.plot(comp_observation['DATE'], comp_observation['Seeker Composition'],
                 color='grey', linestyle='dotted')
    ax_comp.set_ylabel('Employed Share\nof Job Seekers')
    ax_comp.set_title('Monthly Composition of Job Seekers', fontweight='bold')
    ax_comp.set_xlabel('Date')

    if calib_v_line is not None:
        shade_out_of_sample(ax_apps, calib_v_line[1])
        shade_out_of_sample(ax_comp, calib_v_line[1])

    plt.tight_layout()
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='grey', linestyle='dotted', label='Observed')],
                       ncol=3)

    if save:
        save_plot_standard(fig, f'{path}applications_and_composition_combined.png', aspect_ratio=_fig_h / _fig_w)
        plt.close()
    else:
        plt.show()


####################################################
############## GENDER WAGE GAPS ####################
####################################################
def plot_gender_gaps(net_dict, sep = False, save = False, path = None):
    """
    Function to plot the gender wage disttribution across models
    """
    n = len(net_dict)
    if sep:
        cols = 2
    else:
        max_cols = 2
        cols = min(n, max_cols)
        
    rows = math.ceil(n / cols)

    # Find the global max wage across all models
    global_max_wage = 0
    for net in net_dict.values():
        for occ in net:
            w_wages = [wrkr.wage for wrkr in occ.list_of_employed if wrkr.female]
            m_wages = [wrkr.wage for wrkr in occ.list_of_employed if not(wrkr.female)]
            max_wage = max(w_wages + m_wages) if (w_wages + m_wages) else 0
            if max_wage > global_max_wage:
                global_max_wage = max_wage

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False, sharey = True)
    axes = axes.flatten()  # Flatten to make indexing easy

    for i, (name, net) in enumerate(net_dict.items()):
        ax = axes[i]
        emp_counter = 0
        women = 0
        men = 0

        w_wages = []
        m_wages = []

        w_wage = 0
        m_wage = 0

        for occ in net:
            emp_counter += len(occ.list_of_employed)
            women += len([wrkr for wrkr in occ.list_of_employed if wrkr.female])
            men += len([wrkr for wrkr in occ.list_of_employed if not(wrkr.female)])
            w_wages.extend([wrkr.wage for wrkr in occ.list_of_employed if wrkr.female])
            m_wages.extend([wrkr.wage for wrkr in occ.list_of_employed if not(wrkr.female)])
            w_wage += np.nansum([wrkr.wage for wrkr in occ.list_of_employed if wrkr.female])
            m_wage += np.nansum([wrkr.wage for wrkr in occ.list_of_employed if not wrkr.female])


        # Count NaNs
        num_nan_women = np.isnan(women_arr).sum()
        num_nan_men = np.isnan(men_arr).sum()

        # Print warnings if any are found
        if num_nan_women > 0:
            print(f"{num_nan_women} NA value(s) found in women's wages for '{name}' — ignored in calculations.")
        if num_nan_men > 0:
            print(f"{num_nan_men} NA value(s) found in men's wages for '{name}' — ignored in calculations.")

        t= " \n" + " \n" +  "Female share of employed: " + str(round((women/emp_counter)*100)) + "% \n" + "Mean Female Wage: $" + str(round(w_wage/women)) + "\n" + "Mean Male Wage: $" + str(round(m_wage/men)) + "\n" + "Gender wage gap: " + str(round(100*(1 - (w_wage/women)/(m_wage/men)))) + "%" + "\n" + "--------------------"

        n_bins = 10
        women_arr = np.array(w_wages)
        men_arr = np.array(m_wages)

        # We can set the number of bins with the *bins* keyword argument.
        ax.hist(women_arr, bins=n_bins, alpha = 0.3, color = 'purple', label = 'Women', fill = True, hatch = '.')
        ax.hist(men_arr, bins=n_bins, alpha = 0.3, label = 'Men', color = 'green', fill = True, hatch = '.')  
        ax.axvline(np.nanmean(women_arr), color='purple', linestyle='dashed', linewidth=1, label = 'Women Avg.')
        ax.axvline(np.nanmean(men_arr), color='green', linestyle='dashed', linewidth=1, label = 'Men Avg.')
        ax.legend(loc='upper right') 
        ax.annotate(
                t,
                xy=(0.5, 0.5),
                xycoords='axes fraction',
                fontsize=7,
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
            )
        ax.set_title(name)
        ax.set_xlim([0, global_max_wage])

    fig.supxlabel("Wage")  # Shared x-axis label
    fig.suptitle('Distribution of Male and Female Wages', fontsize = 15) 
    fig.subplots_adjust(bottom=0.1)

    if save:
        plt.savefig(f'{path}gender_wage_gaps.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:
        plt.show()



####################################################
############## LTUER DISTRIBUTIONS #################
####################################################
def plot_ltuer_dist(net_dict, names,
                    gender=False, sep=False,
                    save=False, path="./", colors = None):

    n_bins  = 30
    max_cols = 3
    if colors is None:
        colors  = ['orange', 'blue', 'brown', 'green',
                'red', 'purple', 'cyan']
        colors = dict(zip(net_dict.keys(), colors))

    # ---------------------------------------------------------------
    #  gender branch (unchanged)
    # ---------------------------------------------------------------
    if gender:

        n = len(net_dict)
        cols = min(n, max_cols)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(5 * cols, 5 * rows),
                                 sharex=True, sharey=True)
        _fig_w, _fig_h = 5 * cols, 5 * rows
        axes = np.array(axes).flatten()

        for i, (name, net) in enumerate(net_dict.items()):
            ax = axes[i]
            w_time_unemp, m_time_unemp = [], []

            for occ in net:
                w_time_unemp.extend([wrkr.time_unemployed
                                     for wrkr in occ.list_of_unemployed
                                     if wrkr.female])
                m_time_unemp.extend([wrkr.time_unemployed
                                     for wrkr in occ.list_of_unemployed
                                     if not wrkr.female])

            ax.hist(w_time_unemp, bins=n_bins, alpha=0.5,
                    label='Women', color=colors[0])
            ax.hist(m_time_unemp, bins=n_bins, alpha=0.5,
                    label='Men',   color=colors[1])
            ax.axvline(np.mean(w_time_unemp), ls='--', lw=1, color=colors[0])
            ax.axvline(np.mean(m_time_unemp), ls='--', lw=1, color=colors[1])

            ax.set_title(name)
            ax.set_xlabel("Time Unemployed (months)")
            ax.set_ylabel("Number of Workers")
            ax.legend()

        # hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle('Distribution of Time Spent Unemployed by Gender per Model')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        add_suptitle_spacing(fig)

    else:
        # two side-by-side panels that share the y-axis -- a single row of
        # panels, sized at ROW_HEIGHT in/row (independent of the fixed
        # 2-column layout).
        _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT
        fig, (ax_full, ax_zoom) = plt.subplots(
            1, 2, figsize=(_fig_w, _fig_h), sharey=True
        )

        for idx, (name, net) in enumerate(net_dict.items()):
            total_unemp = []
            for occ in net:
                total_unemp.extend(
                    wrkr.time_unemployed
                    for wrkr in occ.list_of_unemployed
                )

            # full histogram
            ax_full.hist(total_unemp, bins=n_bins, alpha=0.5, color=colors[name])
            ax_full.axvline(np.mean(total_unemp), ls='--', lw=1, color=colors[name])

            # zoomed histogram (0–36 mos)
            ax_zoom.hist(total_unemp, bins=n_bins, range=(0, 36), alpha=0.5,
                        color=colors[name])
            ax_zoom.axvline(np.mean(total_unemp), ls='--', lw=1, color=colors[name])

        # labels and titles (these describe the panel content, not the model — kept)
        ax_full.set_xlabel("Time Unemployed (months)")
        ax_zoom.set_xlabel("Time Unemployed (≤ 36 months)")
        ax_full.set_ylabel("Number of Workers")
        ax_full.set_title("Full distribution", fontweight='bold')
        ax_zoom.set_title("Zoom ≤ 3 years", fontweight='bold')
        ax_zoom.set_xlim(0, 36)

        plt.tight_layout()
        add_shared_legend(fig, dict(zip(names, [colors[k] for k in net_dict.keys()])), ncol=3)

    if save:
        save_plot_standard(fig, f'{path}ltuer_distributions.jpg',
                            aspect_ratio=_fig_h / _fig_w,
                            width_frac=max(1.0, _fig_w / TEXTWIDTH))
        plt.close()
    else:
        plt.show()



####################################################
############## CURRENT VS TARGET DEMAND ############
####################################################
def plot_cd_vs_td(res_dict, save = False, path = None, colors = None, calib_v_line=None):

    if colors is None:
        colors = ['orange', 'blue', 'purple', 'green', 'pink', 'grey', 'red', 'cyan', 'brown']
        colors = dict(zip(res_dict.keys(), colors))
    for name, res in res_dict.items():
        plt.plot(res['DATE'], res['Current_Demand'], color=colors[name])
        plt.plot(res['DATE'], res['Target_Demand'], color='grey', linestyle='dashed')

    plt.title("Current vs Target Demand", fontweight = "bold")
    plt.xlabel('Date')
    plt.ylabel("Current or Target Labor Demand")

    if calib_v_line is not None:
        shade_out_of_sample(plt.gca(), calib_v_line[1])

    plt.tight_layout()
    fig = plt.gcf()
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='grey', linestyle='dashed', label='Observed Target Demand')],
                       ncol=3)

    # Save the figure to the output folder
    if save:
        save_plot_standard(fig, f'{path}cd_vs_td.png')
        plt.close()
    else:
        plt.show()



def sse_tbl(res_dict, obs):
    for i, (name, res) in enumerate(res_dict.items()):
        uer_sse = np.sum((x_["UER"] - y["UER"])**2) / np.var(y["UER"])
        vacrate_sse = np.sum((x_["VACRATE"] - y["VACRATE"])**2) / np.var(y["VACRATE"])

    return tbl

####################################################
############## AVERAGE WAGE BY OCCUPATION ##########
####################################################
def plot_avg_wage_by_occupation(sim_results_dict, save=False, path=None):
    """
    Creates scatterplots of mean wage by occupation for each element in the dictionary.
    
    Parameters:
    - sim_results_dict: Dictionary containing simulation results DataFrames
    - save: Boolean indicating whether to save the plots
    - path: Path to save the plots if save=True
    """
    # First, calculate global y-axis limits
    all_wages = []
    for sim_results in sim_results_dict.values():
        occ_data = sim_results.loc[:, ['Time Step', 'Occupation', 'AVGWAGE']]
        mean_wage = occ_data.groupby('Occupation')['AVGWAGE'].mean()
        all_wages.extend(mean_wage.values)
    
    # Add some padding to the y-axis limits
    y_min = min(all_wages) * 0.95  # 5% padding below
    y_max = max(all_wages) * 1.05  # 5% padding above
    
    for name, sim_results in sim_results_dict.items():
        # Extract occupation-specific data
        occ_data = sim_results.loc[:, ['Time Step', 'Occupation', 'AVGWAGE']]
        
        # Calculate mean wage for each occupation
        mean_wage = occ_data.groupby('Occupation')['AVGWAGE'].mean().reset_index()
        
        # Sort occupations by mean wage and create ordered x-axis positions
        mean_wage = mean_wage.sort_values('AVGWAGE')
        mean_wage['x_pos'] = range(len(mean_wage))
        
        # Create the scatterplot
        plt.figure(figsize=(12, 6))
        plt.scatter(mean_wage['x_pos'], mean_wage['AVGWAGE'], s=100)
        
        # Customize the plot
        plt.title(f'Mean Wage by Occupation ({name})', fontweight='bold')
        plt.xlabel('Occupation (ordered by mean wage)')
        plt.ylabel('Mean Wage')
        
        # Set x-axis ticks to show occupation IDs in the sorted order
        plt.xticks(mean_wage['x_pos'], mean_wage['Occupation'], rotation=45, ha='right')
        
        # Set consistent y-axis limits
        plt.ylim(y_min, y_max)
        
        # Adjust layout
        plt.tight_layout()

        
        # Save or show
        if save:
            plt.savefig(f'{path}avg_wage_by_occupation_{name}.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
            plt.close()
        else:
            plt.show()


####################################################
############## TRANSITION RATES #####################
####################################################
def plot_trans_rates(mod_results_dict, observation, save=False, path=None, colors = None, calib_v_line=None):
    """
    Creates a two-column plot showing:
    - Left: Mean transition rates with standard deviation error bars for EE and UE
    - Right: Time series of transition rates stacked vertically
    
    Parameters:
    - mod_results_dict: Dictionary containing simulation results DataFrames
    - save: Boolean indicating whether to save the plots
    - path: Path to save the plots if save=True
    """
    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(mod_results_dict.keys(), colors))    
    # Create figure with a 1x2 grid for the main columns
    fig = plt.figure(figsize=(TEXTWIDTH, ROW_HEIGHT * 2))
    
    # Left column: Mean and std dev plot
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    
    # Right column: Time series plots
    ax2 = plt.subplot2grid((2, 2), (0, 1))  # UE transitions
    ax3 = plt.subplot2grid((2, 2), (1, 1))  # EE transitions
    
    # Calculate means and standard deviations for each model
    base_positions = [0.25, 0.75]  # Base positions for EE and UE
    spacing = 0.03  # Space between different models
    n_models = len(mod_results_dict)
    
    for i, (name, res) in enumerate(mod_results_dict.items()):
        # Calculate means and standard deviations
        ee_mean = res['EE_Trans_Rate'].mean()
        ee_std = res['EE_Trans_Rate'].std()
        ue_mean = res['UE_Trans_Rate'].mean()
        ue_std = res['UE_Trans_Rate'].std()
        obs_ee_mean = observation['EE'].mean()
        obs_ee_std = observation['EE'].std()
        obs_ue_mean = observation['UE'].mean()
        obs_ue_std = observation['UE'].std()
        
        # Calculate x-positions for this model with alternating offsets
        if i % 2 == 0:  # Even indices
            ee_x = base_positions[0] + spacing
            ue_x = base_positions[1] + spacing
        else:  # Odd indices
            ee_x = base_positions[0] + 2*spacing
            ue_x = base_positions[1] + 2*spacing
        
        # Plot means and error bars on left panel

        ax1.errorbar(ee_x, ee_mean, yerr=ee_std,
                    fmt='o', color=colors[name],
                    capsize=5, markersize=10)
        ax1.errorbar(ue_x, ue_mean, yerr=ue_std,
                    fmt='o', color=colors[name],
                    capsize=5, markersize=10)

        # Plot time series on right panels
        ax2.plot(res['DATE'], res['UE_Trans_Rate'], color=colors[name])
        ax3.plot(res['DATE'], res['EE_Trans_Rate'], color=colors[name])

    ax1.errorbar(base_positions[0], obs_ee_mean, yerr=obs_ee_std,
                    fmt='o', color="grey",
                    capsize=5, markersize=10)
    ax1.errorbar(base_positions[1], obs_ue_mean, yerr=obs_ue_std,
                    fmt='o', color="grey",
                    capsize=5, markersize=10)
    # Customize left panel
    ax1.set_title('Mean Transition Rates with Standard Deviation', fontweight='bold')
    ax1.set_xticks(base_positions)
    ax1.set_xlim(0, 1)
    ax1.set_xticklabels(['EE', 'UE'])
    ax1.set_ylabel('Transition Rate')

    # Customize right panels
    ax2.set_title('UE Transition Rate Over Time', fontweight='bold')
    ax2.plot(observation['DATE'], observation['UE'], color="grey")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rate')

    ax3.set_title('EE Transition Rate Over Time', fontweight='bold')
    ax3.plot(observation['DATE'], observation['EE'], color="grey")
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Rate')

    if calib_v_line is not None:
        shade_out_of_sample(ax2, calib_v_line[1])
        shade_out_of_sample(ax3, calib_v_line[1])

    # Adjust layout to make room for legend
    plt.tight_layout()
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='grey', marker='o', label='Observed')],
                       ncol=3)

    # Save or show
    if save:
        save_plot_standard(fig, f'{path}transition_rates_comparison.jpg', aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
        plt.close()
    else:
        plt.show()


####################################################
############## OCC-SPECIFIC LTUER ##################
####################################################
         
def plot_occupation_uer_grid(sim_results, observation, save=False, path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    agg_levels = {
        'SOC2010_adj': ('uer_soc2010', 'ltuer_soc2010'), 
        'SOC_broad_adj': ('uer_soc_broad', 'ltuer_soc_broad'),
        'SOC_minor_adj': ('uer_soc_minor', 'ltuer_soc_minor'), 
        'SOC_major_adj': ('uer_soc_major', 'ltuer_soc_major')
    }

    model_names = list(sim_results.keys())
    num_rows = len(agg_levels)
    num_cols = len(model_names)

    fig1, axes1 = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), sharex=False, sharey=False)
    fig1.suptitle("Unemployment Rate (UER): Simulated vs Observed (Sorted by Observed UER)", fontsize=16)

    fig2, axes2 = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), sharex=False, sharey=False)
    fig2.suptitle("Long-Term Unemployment Rate (LTUER): Simulated vs Observed (Sorted by Observed LTUER)", fontsize=16)

    for col_idx, (model_name, sims) in enumerate(sim_results.items()):
        for row_idx, (name_k, (k_uer, k_ltuer)) in enumerate(agg_levels.items()):

            # Merge simulation and observed data
            temp_codes = observation.loc[:, ['acs_occ_code', name_k, k_uer, k_ltuer]]
            occ_data = sims.loc[:, ['Time Step', 'Occupation', 'acs_occ_code', 'Workers', 'Unemployment', 'LT Unemployed Persons']]
            occ_data = occ_data.merge(temp_codes, on='acs_occ_code', how='left')

            # Compute UER and LTUER
            occ_data = occ_data.groupby([name_k, 'Time Step']).sum().reset_index()
            occ_data['UER'] = occ_data['Unemployment'] / occ_data['Workers']
            occ_data['LTUER'] = occ_data['LT Unemployed Persons'] / occ_data['Unemployment']

            # Average over time
            mean_occ_data = occ_data.groupby(name_k)[['UER', 'LTUER']].mean().reset_index()
            obs_values = observation[[name_k, k_uer, k_ltuer]].drop_duplicates(subset=[name_k])
            merged = mean_occ_data.merge(obs_values, on=name_k, how='left')

            # ----- SORT OCCUPATION CODES BY OBSERVED UER -----
            sorted_codes_uer = merged.sort_values(by=k_uer)[name_k].tolist()

            # Re-index values
            sim_vals = merged.set_index(name_k).reindex(sorted_codes_uer)['UER'].tolist()
            obs_vals = merged.set_index(name_k).reindex(sorted_codes_uer)[k_uer].tolist()
            x_ticks = list(range(len(sorted_codes_uer)))

            ax1 = axes1[row_idx, col_idx]
            ax1.scatter(x_ticks, sim_vals, color='blue', label='Simulated', alpha=0.7)
            ax1.scatter(x_ticks, obs_vals, color='orange', label='Observed', alpha=0.7, marker='X')
            ax1.set_title(f"{model_name} - {name_k}")
            ax1.set_xticks(x_ticks)
            ax1.set_xticklabels(sorted_codes_uer, rotation=45, ha='right')
            ax1.set_ylim(0, 0.3)
            if row_idx == num_rows - 1:
                ax1.set_xlabel(name_k)
            if col_idx == 0:
                ax1.set_ylabel('UER')

            # ----- SORT OCCUPATION CODES BY OBSERVED LTUER -----
            sorted_codes_ltuer = merged.sort_values(by=k_ltuer)[name_k].tolist()
            sim_vals_ltuer = merged.set_index(name_k).reindex(sorted_codes_ltuer)['LTUER'].tolist()
            obs_vals_ltuer = merged.set_index(name_k).reindex(sorted_codes_ltuer)[k_ltuer].tolist()
            x_ticks_ltuer = list(range(len(sorted_codes_ltuer)))

            ax2 = axes2[row_idx, col_idx]
            ax2.scatter(x_ticks_ltuer, sim_vals_ltuer, color='blue', label='Simulated', alpha=0.7)
            ax2.scatter(x_ticks_ltuer, obs_vals_ltuer, color='darkorange', label='Observed', alpha=0.7, marker='X')
            ax2.set_title(f"{model_name} - {name_k}")
            ax2.set_xticks(x_ticks_ltuer)
            ax2.set_xticklabels(sorted_codes_ltuer, rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            if row_idx == num_rows - 1:
                ax2.set_xlabel(name_k)
            if col_idx == 0:
                ax2.set_ylabel('LTUER')

            if row_idx == 0 and col_idx == 0:
                ax1.legend()
                ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        fig1.savefig(f'{path}occupation_uer_grid.png', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        fig2.savefig(f'{path}occupation_ltuer_grid.png', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:
        plt.show()

import matplotlib.gridspec as gridspec
def plot_occupation_uer_grid2(sim_results, observation, soc_labs, save=False, path=None):
    agg_levels = {
        'SOC2010_adj': ('uer_soc2010', 'ltuer_soc2010'), 
        'SOC_broad_adj': ('uer_soc_broad', 'ltuer_soc_broad'),
        'SOC_minor_adj': ('uer_soc_minor', 'ltuer_soc_minor'), 
        'SOC_major_adj': ('uer_soc_major', 'ltuer_soc_major')
    }

    model_names = list(sim_results.keys())
    num_rows = len(agg_levels)
    num_cols = len(model_names)

    # Sized to print at exactly TEXTWIDTH, ROW_HEIGHT in/row (independent of
    # num_cols) -- model=column, SOC level=row is a genuine 2D grid, not a
    # wrappable one-panel-per-model list. With 6 models this makes each
    # column quite narrow; bump width_frac manually if it renders too
    # cramped to read.
    _fig1_w, _fig1_h = TEXTWIDTH, ROW_HEIGHT * num_rows
    fig1 = plt.figure(figsize=(_fig1_w, _fig1_h))
    gs1 = gridspec.GridSpec(num_rows * 2, num_cols, height_ratios=[4, 0.7]*num_rows)
    fig1.suptitle("Unemployment Rate (UER): Simulated vs Observed (Sorted by Observed UER)")

    _fig2_w, _fig2_h = _fig1_w, _fig1_h
    fig2 = plt.figure(figsize=(_fig2_w, _fig2_h))
    gs2 = gridspec.GridSpec(num_rows * 2, num_cols, height_ratios=[4, 0.7]*num_rows)
    fig2.suptitle("LT Unemployment Rate (LTUER): Simulated vs Observed (Sorted by Observed LTUER)")

    for col_idx, (model_name, sims) in enumerate(sim_results.items()):
        for row_idx, (name_k, (k_uer, k_ltuer)) in enumerate(agg_levels.items()):

            temp_codes = observation.loc[:, ['acs_occ_code', name_k, k_uer, k_ltuer]]
            occ_data = sims.loc[:, ['Time Step', 'Occupation', 'acs_occ_code', 'Workers', 'Unemployment', 'LT Unemployed Persons']]
            occ_data = occ_data.merge(temp_codes, on='acs_occ_code', how='left')

            occ_data = occ_data.groupby([name_k, 'Time Step']).sum().reset_index()
            occ_data['UER'] = occ_data['Unemployment'] / occ_data['Workers']
            occ_data['LTUER'] = occ_data['LT Unemployed Persons'] / occ_data['Unemployment']

            mean_occ_data = occ_data.groupby(name_k)[['UER', 'LTUER']].mean().reset_index()
            obs_values = observation[[name_k, k_uer, k_ltuer]].drop_duplicates(subset=[name_k])
            merged = mean_occ_data.merge(obs_values, on=name_k, how='left')

            if name_k == "SOC_major_adj":
                merged[name_k] = merged[name_k].astype(str)
                soc_labs["soc_code"] = soc_labs["soc_code"].astype(str)
                merged = merged.merge(soc_labs, left_on=name_k, right_on="soc_code", how="left")
                merged[name_k + "_label"] = merged["label"]
            else:
                merged[name_k + "_label"] = merged[name_k].astype(str)

            sorted_codes_uer = merged.sort_values(by=k_uer)[name_k].tolist()
            sim_vals = merged.set_index(name_k).reindex(sorted_codes_uer)['UER'].tolist()
            obs_vals = merged.set_index(name_k).reindex(sorted_codes_uer)[k_uer].tolist()
            x_ticks = list(range(len(sorted_codes_uer)))
            x_labels = merged.set_index(name_k).reindex(sorted_codes_uer)[name_k + "_label"].tolist()

            # PLOT subplot
            ax1 = fig1.add_subplot(gs1[row_idx * 2, col_idx])
            ax1.scatter(x_ticks, sim_vals, color='blue', label='Simulated', alpha=0.7)
            ax1.scatter(x_ticks, obs_vals, color='orange', label='Observed', alpha=0.7, marker='X')
            ax1.set_title(f"{model_name} - {name_k}")
            ax1.set_ylim(0, 0.35)
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            if col_idx == 0:
                ax1.set_ylabel('UER')

            # --- Best fit line for simulated data (UER) ---
            # Only fit if there are at least 2 points and not all are nan
            sim_vals_np = np.array(sim_vals)
            valid = ~np.isnan(sim_vals_np)
            if np.sum(valid) > 1:
                fit = np.polyfit(np.array(x_ticks)[valid], sim_vals_np[valid], 1)
                fit_line = np.polyval(fit, x_ticks)
                ax1.plot(x_ticks, fit_line, color='blue', linestyle='--', alpha=0.7, label='Best fit (Sim.)')

            sorted_codes_ltuer = merged.sort_values(by=k_ltuer)[name_k].tolist()
            sim_vals_ltuer = merged.set_index(name_k).reindex(sorted_codes_ltuer)['LTUER'].tolist()
            obs_vals_ltuer = merged.set_index(name_k).reindex(sorted_codes_ltuer)[k_ltuer].tolist()
            x_ticks_ltuer = list(range(len(sorted_codes_ltuer)))
            x_labels_ltuer = merged.set_index(name_k).reindex(sorted_codes_ltuer)[name_k + "_label"].tolist()

            # PLOT subplot
            ax2 = fig2.add_subplot(gs2[row_idx * 2, col_idx])
            ax2.scatter(x_ticks_ltuer, sim_vals_ltuer, color='blue', label='Simulated', alpha=0.7)
            ax2.scatter(x_ticks_ltuer, obs_vals_ltuer, color='orange', label='Observed', alpha=0.7, marker='X')
            ax2.set_title(f"{model_name} - {name_k}")
            ax2.set_ylim(0, 1)
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            if col_idx == 0:
                ax2.set_ylabel('LTUER')

        # LABELS subplot (one per row, after all cols)
    
        label_ax1 = fig1.add_subplot(gs1[row_idx * 2 + 1, :])  # spans all columns
        label_ax1.set_xticks(x_ticks)
        label_ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        label_ax1.set_yticks([])
        label_ax1.tick_params(axis='x', which='both', length=0)
        label_ax1.set_frame_on(False)
        label_ax1.set_xlabel(name_k)

        label_ax2 = fig2.add_subplot(gs2[row_idx * 2 + 1, :])  # spans all columns
        label_ax2.set_xticks(x_ticks_ltuer)
        label_ax2.set_xticklabels(x_labels_ltuer, rotation=45, ha='right', fontsize=8)
        label_ax2.set_yticks([])
        label_ax2.tick_params(axis='x', which='both', length=0)
        label_ax2.set_frame_on(False)
        label_ax2.set_xlabel(name_k)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    add_suptitle_spacing(fig1)
    add_suptitle_spacing(fig2)

    _uer_grid_handles = [
        Line2D([0], [0], marker='o', linestyle='', color='blue', label='Simulated', alpha=0.7),
        Line2D([0], [0], marker='X', linestyle='', color='orange', label='Observed', alpha=0.7),
        Line2D([0], [0], color='blue', linestyle='--', alpha=0.7, label='Best fit (Sim.)'),
    ]
    add_shared_legend(fig1, {}, extra_handles=_uer_grid_handles, ncol=3)
    add_shared_legend(fig2, {}, extra_handles=_uer_grid_handles, ncol=3)

    if save:
        save_plot_standard(fig1, f'{path}occupation_uer_grid.png',
                            aspect_ratio=_fig1_h / _fig1_w,
                            width_frac=max(1.0, _fig1_w / TEXTWIDTH))
        save_plot_standard(fig2, f'{path}occupation_ltuer_grid.png',
                            aspect_ratio=_fig2_h / _fig2_w,
                            width_frac=max(1.0, _fig2_w / TEXTWIDTH))
        plt.close()
    else:
        plt.show()

def plot_ltuer_difference_heatmap(sim_results_dict, observed_data, difference_type='absolute', abs_value = True, save=False, path=None):
    """
    Creates a heatmap showing differences between simulated and observed LTUER values.
    
    Parameters:
    - sim_results_dict: Dictionary containing simulation results DataFrames
    - observed_data: DataFrame containing observed LTUER values by occupation
    - difference_type: 'absolute' or 'percentage' to show either absolute or percentage differences
    - save: Boolean indicating whether to save the plots
    - path: Path to save the plots if save=True
    """
    # Calculate differences for each model
    differences = {}
    for name, sim_results in sim_results_dict.items():
        # Calculate mean LTUER for each occupation
        occ_data = sim_results.loc[:, ['acs_occ_code', 'Time Step', 'Occupation', 'Workers', 'Unemployment', 'LT Unemployed Persons']]
        occ_data = occ_data.groupby(['acs_occ_code', 'Time Step']).sum().reset_index()
        occ_data['UER'] = occ_data['Unemployment'] / occ_data['Workers']
        occ_data['LTUER'] = occ_data['LT Unemployed Persons'] / occ_data['Unemployment']
        mean_uer = occ_data.groupby('acs_occ_code')[['UER', 'LTUER']].mean().reset_index()

        # Merge simulation and observed data
        temp_codes = observed_data.loc[:, ['acs_occ_code', 'SOC2010_adj', 'uer_soc2010', 'ltuer_soc2010']]
        mean_occ_data = mean_uer.merge(temp_codes, on='acs_occ_code', how='left')
        
        if difference_type == 'absolute':
            # Calculate absolute differences
            diff = mean_occ_data['ltuer_soc2010'] - mean_occ_data['LTUER']
        else:
            # Calculate percentage differences
            diff = ((mean_occ_data['ltuer_soc2010'] - mean_occ_data['LTUER']) / mean_occ_data['ltuer_soc2010'])*100

        if abs_value:
            diff = np.abs(diff)

        differences[name] = diff
    
    # Create DataFrame for heatmap
    diff_df = pd.DataFrame(differences)
    diff_df.index = [f'O{occ+1}' for occ in range(len(diff_df))]
    
    # Create figure. Width fixed at TEXTWIDTH; height kept at the original 8in
    # (not the original 8/12 aspect) since it's sized for occupation-row count,
    # not for the old 12in-wide canvas.
    plt.figure(figsize=(TEXTWIDTH, 8))

    # Create heatmap
    if difference_type == 'absolute':
        # Use diverging colormap for absolute differences
        cmap = 'RdBu_r'
        center = 0
        vmin = -max(abs(np.nanmin(diff_df.values)), abs(np.nanmax(diff_df.values)))
        vmax = max(abs(np.nanmin(diff_df.values)), abs(np.nanmax(diff_df.values)))
    else:
        # Use diverging colormap for percentage differences
        cmap = 'RdBu_r'
        center = 0
        vmin = -max(abs(np.nanmin(diff_df.values)), abs(np.nanmax(diff_df.values)))
        vmax = max(abs(np.nanmin(diff_df.values)), abs(np.nanmax(diff_df.values)))
    
    if abs_value:
        vmin = 0
    
    sns.heatmap(diff_df, 
                annot=False,  # Show values in cells
                fmt='.2f',   # Format values to 2 decimal places
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': f'{"Absolute" if difference_type == "absolute" else "Percentage"} Difference'})
    
    # Customize plot
    plt.title(f'{"Absolute" if difference_type == "absolute" else "Percentage"} Differences in LTUER\n(Simulated - Observed) [{"Absolute Value" if abs_value else "Signed Value"}]', 
              fontweight='bold', pad=20)
    plt.xlabel('Model')
    plt.ylabel('Occupation')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()

    
    # Save or show
    if save:
        save_plot_standard(plt.gcf(),
                            f'{path}ltuer_difference_heatmap_{difference_type}_absval_{abs_value}.jpg',
                            aspect_ratio=8 / TEXTWIDTH,
                            apply_grid=False)
        plt.close()
    else:
        plt.show()


def plot_avg_wage_offer_diff_over_time(df, save_path=None):
    """
    Plots the average wage offer difference by occupation over time.

    Args:
        df (pd.DataFrame): DataFrame with columns 'Time Step', 'Occupation', 'Avg_Wage_Offer_Diff'
        save_path (str, optional): If provided, saves the plot to this path.
    """
    pivot_df = df.pivot(index="Time Step", columns="Occupation", values="Avg_Wage_Offer_Diff")
    plt.figure(figsize=(14, 7))
    for occ in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[occ])
    plt.title("Average Wage Offer Difference by Occupation Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Average Wage Offer Difference")
    plt.legend(title="Occupation", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_vacancy_vs_wage(record_df, save_path=None):
    """
    Plots the difference between Mean Vacancy Offer and Mean Occupational Wage by occupation.
    
    Args:
        record_df (pd.DataFrame): DataFrame with columns 'Occupation', 'Mean Vacancy Offer', 'Mean Occupational Wage'.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    occ_means = record_df.groupby('Occupation')[["Mean Vacancy Offer", "Mean Occupational Wage"]].mean().reset_index()
    occ_means["Difference"] = occ_means["Mean Vacancy Offer"] - occ_means["Mean Occupational Wage"]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=occ_means.melt(id_vars="Occupation", value_vars=["Mean Vacancy Offer", "Mean Occupational Wage"]),
        x="Occupation", y="value", hue="variable"
    )
    plt.title("Mean Vacancy Offer vs. Mean Occupational Wage by Occupation")
    plt.ylabel("Wage")
    plt.xlabel("Occupation")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=occ_means, x="Occupation", y="Difference", color="gray")
    plt.title("Difference: Mean Vacancy Offer - Mean Occupational Wage")
    plt.ylabel("Difference")
    plt.xlabel("Occupation")
    plt.tight_layout()
    if save_path:
        diff_path = save_path.replace(".png", "_difference.png")
        plt.savefig(diff_path, bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:
        plt.show()

def plot_vacancy_wage_diff_timeseries(record_df, save_path=None):
    """
    Plots the time series of mean vacancy offer, mean occupational wage, and their difference by occupation.

    Args:
        record_df (pd.DataFrame): DataFrame with columns 'Time Step', 'Occupation', 'Mean Vacancy Offer', 'Mean Occupational Wage'.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    record_df = record_df.copy()
    record_df["Difference"] = record_df["Mean Vacancy Offer"] - record_df["Mean Occupational Wage"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    pivot_offer = record_df.pivot(index="Time Step", columns="Occupation", values="Mean Vacancy Offer")
    for occ in pivot_offer.columns:
        axes[0].plot(pivot_offer.index, pivot_offer[occ], label=f"Occ {occ}")
    axes[0].set_title("Mean Vacancy Offer by Occupation")
    axes[0].set_ylabel("Mean Vacancy Offer")

    pivot_wage = record_df.pivot(index="Time Step", columns="Occupation", values="Mean Occupational Wage")
    for occ in pivot_wage.columns:
        axes[1].plot(pivot_wage.index, pivot_wage[occ], label=f"Occ {occ}")
    axes[1].set_title("Mean Occupational Wage by Occupation")
    axes[1].set_ylabel("Mean Occupational Wage")

    pivot_diff = record_df.pivot(index="Time Step", columns="Occupation", values="Difference")
    for occ in pivot_diff.columns:
        axes[2].plot(pivot_diff.index, pivot_diff[occ], label=f"Occ {occ}")
    axes[2].set_title("Difference: Mean Vacancy Offer - Mean Occupational Wage")
    axes[2].set_ylabel("Difference")
    axes[2].set_xlabel("Time Step")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:
        plt.show()


def plot_occupation_vr_grid(sim_results, observation, soc_labs, save=False, path=None):
    agg_levels = {
        'SOC2010_adj': ('uer_soc2010', 'ltuer_soc2010'), 
        'SOC_broad_adj': ('uer_soc_broad', 'ltuer_soc_broad'),
        'SOC_minor_adj': ('uer_soc_minor', 'ltuer_soc_minor'), 
        'SOC_major_adj': ('uer_soc_major', 'ltuer_soc_major')
    }

    model_names = list(sim_results.keys())
    num_rows = len(agg_levels)
    num_cols = len(model_names)

    fig1 = plt.figure(figsize=(5*num_cols, 4*num_rows + 1))
    gs1 = gridspec.GridSpec(num_rows * 2, num_cols, height_ratios=[4, 0.7]*num_rows)
    fig1.suptitle("Vacancy Rate (VR): Simulated vs Observed (Sorted by Observed VR)", fontsize=16)

    for col_idx, (model_name, sims) in enumerate(sim_results.items()):
        for row_idx, (name_k, (k_uer, k_ltuer)) in enumerate(agg_levels.items()):

            temp_codes = observation.loc[:, ['acs_occ_code', name_k, k_uer, k_ltuer]]
            occ_data = sims.loc[:, ['Time Step', 'Occupation', 'acs_occ_code', 'Workers', 'Vacancies', 'Employment']]
            occ_data = occ_data.merge(temp_codes, on='acs_occ_code', how='left')

            occ_data = occ_data.groupby([name_k, 'Time Step']).sum().reset_index()
            occ_data['VACRATE'] = occ_data['Vacancies'] / (occ_data['Vacancies'] + occ_data['Employment'])

            mean_occ_data = occ_data.groupby(name_k)[['VACRATE']].mean().reset_index()
            obs_values = observation[[name_k, k_uer, k_ltuer]].drop_duplicates(subset=[name_k])
            merged = mean_occ_data.merge(obs_values, on=name_k, how='left')

            if name_k == "SOC_major_adj":
                merged[name_k] = merged[name_k].astype(str)
                soc_labs["soc_code"] = soc_labs["soc_code"].astype(str)
                merged = merged.merge(soc_labs, left_on=name_k, right_on="soc_code", how="left")
                merged[name_k + "_label"] = merged["label"]
            else:
                merged[name_k + "_label"] = merged[name_k].astype(str)

            sorted_codes_uer = merged.sort_values(by=k_uer)[name_k].tolist()
            sim_vals = merged.set_index(name_k).reindex(sorted_codes_uer)['VACRATE'].tolist()
            obs_vals = merged.set_index(name_k).reindex(sorted_codes_uer)[k_uer].tolist()
            x_ticks = list(range(len(sorted_codes_uer)))
            x_labels = merged.set_index(name_k).reindex(sorted_codes_uer)[name_k + "_label"].tolist()

            # PLOT subplot
            ax1 = fig1.add_subplot(gs1[row_idx * 2, col_idx])
            ax1.scatter(x_ticks, sim_vals, color='purple', label='Simulated VR', alpha=0.7)
            ax1.scatter(x_ticks, obs_vals, color='orange', label='Observed UER', alpha=0.7, marker='X')
            ax1.set_title(f"{model_name} - {name_k}")
            ax1.set_ylim(0, 0.35)
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            if col_idx == 0:
                ax1.set_ylabel('VACRATE')
            if row_idx == 0 and col_idx == 0:
                ax1.legend()


        # LABELS subplot (one per row, after all cols)
    
        label_ax1 = fig1.add_subplot(gs1[row_idx * 2 + 1, :])  # spans all columns
        label_ax1.set_xticks(x_ticks)
        label_ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        label_ax1.set_yticks([])
        label_ax1.tick_params(axis='x', which='both', length=0)
        label_ax1.set_frame_on(False)
        label_ax1.set_xlabel(name_k)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        fig1.savefig(f'{path}occupation_vr_grid.png', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:
        plt.show()

# %%
# Network graph from adjacency matrix (weighted, colored by node name)
def plot_network_from_matrix(
    A,
    save = False,
    path = None,
    directed=False,
    threshold=0.0,
    abs_threshold=False,
    normalize_edge_widths=True,
    max_edge_width=4.0,
    layout="spring",            # "spring" | "kamada_kawai" | "spectral" | "circular"
    seed=42,
    node_size=220,
    color_key=None,             # function: name -> group string; or dict {name: group}
    palette=None,               # dict {group: color} or None -> auto colormap
    title="Network graph",
):
    """
    A: adjacency as pd.DataFrame (preferred; index=columns=names) or np.ndarray.
    If ndarray, nodes will be named 'N0','N1',... and treated as undirected unless directed=True.
    directed: build DiGraph if True, else Graph.
    threshold: drop edges with weight <= threshold (or <= |threshold| if abs_threshold=True).
    abs_threshold: if True, threshold on absolute value.
    normalize_edge_widths: scale edge widths to [0, max_edge_width].
    color_key: 
        - function taking node name -> group label (str), or
        - dict mapping node name -> group, or
        - None to color all nodes the same.
    palette: dict mapping group -> matplotlib color. If None, auto-assign from tab20.
    """
    # ----- Standardize input -----
    if isinstance(A, pd.DataFrame):
        names = A.index.astype(str).tolist()
        assert list(A.columns.astype(str)) == names, "Adjacency DataFrame must have matching index/columns (same order)."
        M = A.values.astype(float)
    else:
        A = np.asarray(A, dtype=float)
        names = [f"N{i}" for i in range(A.shape[0])]
        M = A

    n = len(names)
    if M.shape[0] != n or M.shape[1] != n:
        raise ValueError("Adjacency matrix must be square.")

    # ----- Build graph -----
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(names)

    # Add edges with weights (sparsify by threshold)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = M[i, j]
            val = abs(w) if abs_threshold else w
            if val > threshold and w != 0:
                # For undirected matrices, avoid double-add: only i<j
                if not directed and j <= i:
                    continue
                G.add_edge(names[i], names[j], weight=float(w))

    if G.number_of_edges() == 0:
        print("No edges after thresholding.")
        return

    # ----- Layout -----
    rng = np.random.default_rng(seed)
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=None)  # k=None lets NX choose based on log(n)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    # ----- Node colors by index name -----
    # Build group per node
    if color_key is None:
        node_groups = {name: "All" for name in names}
    elif callable(color_key):
        node_groups = {name: str(color_key(name)) for name in names}
    elif isinstance(color_key, dict):
        node_groups = {name: str(color_key.get(name, "Other")) for name in names}
    else:
        raise TypeError("color_key must be None, function(name)->group, or dict {name: group}")

    groups = pd.Index(node_groups.values()).unique().tolist()
    # Palette
    if palette is None:
        # auto colors from tab20 (recycle if needed)
        base = plt.get_cmap("tab20").colors
        palette = {g: base[i % len(base)] for i, g in enumerate(groups)}
    else:
        # ensure all groups get a color
        for g in groups:
            palette.setdefault(g, "gray")

    node_colors = [palette[node_groups[name]] for name in names]

    # ----- Edge widths -----
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)], dtype=float)
    if normalize_edge_widths:
        wmin, wmax = np.nanmin(np.abs(weights)), np.nanmax(np.abs(weights))
        # avoid div-by-zero
        if wmax > 0:
            widths = (np.abs(weights) - wmin) / (wmax - wmin + 1e-12) * max_edge_width
            widths = np.clip(widths, 0.5, max_edge_width)
        else:
            widths = np.full_like(weights, 1.5)
    else:
        widths = np.clip(np.abs(weights), 0.5, max_edge_width)

    # ----- Draw -----
    plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, linewidths=0.5, edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6, arrows=directed, arrowstyle="-|>", arrowsize=10)
    # If graph is not too large, label nodes
    if len(names) <= 60:
        nx.draw_networkx_labels(G, pos, font_size=8)

    # Legend for groups
    legend_handles = [Line2D([0], [0], marker='o', color='w',
                            label=str(g), markerfacecolor=palette[g],
                            markersize=8, markeredgecolor="black")
                    for g in groups]
    plt.legend(handles=legend_handles, title="Groups", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    if save:
        save_plot_standard(plt.gcf(), f'{path}network_graph.png', aspect_ratio=0.8, apply_grid=False)
        plt.close()
    else:
        plt.show()


# Net entry plots
# ===== Core: exclude same-occupation moves when counting entrants/exits =====
def _filter_cross_occ(df: pd.DataFrame, ignore_within: bool = True) -> pd.DataFrame:
    """Optionally drop rows where OriginOccupation == DestinationOccupation."""
    if ignore_within:
        return df[df['OriginOccupation'] != df['DestinationOccupation']]
    return df

# ===== Net entry (window summary) =====
def compute_net_entry(ue_spell_origin_df: pd.DataFrame, label_map=None, time_range=None, ignore_within: bool = True):
    """
    NetEntry_i = Entrants_i - Exits_i, excluding self-flows if ignore_within=True.
    ue_spell_origin_df columns: ['Time Step','OriginOccupation','DestinationOccupation','UEDuration']
    """
    df = ue_spell_origin_df.copy()
    if time_range is not None:
        tmin, tmax = time_range
        df = df[(df['Time Step'] >= tmin) & (df['Time Step'] <= tmax)]

    df = _filter_cross_occ(df, ignore_within=ignore_within)

    entrants = df.groupby('DestinationOccupation').size().rename("Entrants")
    exits    = df.groupby('OriginOccupation').size().rename("Exits")

    net_entry = pd.concat([entrants, exits], axis=1).fillna(0)
    net_entry['NetEntry'] = net_entry['Entrants'] - net_entry['Exits']

    if label_map is not None:
        if isinstance(label_map, pd.Series):
            net_entry['Label'] = [label_map.get(i, str(i)) for i in net_entry.index]
        elif isinstance(label_map, dict):
            net_entry['Label'] = [label_map.get(i, str(i)) for i in net_entry.index]
    else:
        net_entry['Label'] = net_entry.index.astype(str)

    return net_entry.sort_values('NetEntry', ascending=False)

def plot_net_entry_highlight(net_entry: pd.DataFrame,
                            trending_occs,
                            topk: int = 40,
                            title: str = "Net entry by occupation (highlighting)"):
    """
    Bars are green by default; bars for occ IDs in trending_occs are red.
    trending_occs: iterable of occupation IDs that match net_entry.index dtype.
    """
    s = net_entry.sort_values('NetEntry', ascending=False)
    if isinstance(topk, int):
        s = s.head(topk)

    # Coerce trending ids to index dtype (robust to str/int mismatch)
    idx_dtype = s.index.dtype
    try:
        trend_set = set(pd.Index(trending_occs).astype(idx_dtype))
        key_series = s.index
    except Exception:
        trend_set = set(map(str, trending_occs))
        key_series = s.index.map(str)

    labels = s['Label'] if 'Label' in s.columns else s.index.astype(str)
    x = np.arange(len(s))
    colors = ['red' if key in trend_set else 'green' for key in key_series]

    plt.figure(figsize=(12, 6))
    plt.bar(x, s['NetEntry'].values, color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Net Entry (Entrants - Exits)")
    plt.xlabel("Occupation")
    plt.title(title)
    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(facecolor='green', edgecolor='black', label='Other occupations'),
        Patch(facecolor='red',   edgecolor='black', label='Highlighted (trending_occs)'),
    ], loc='best')
    plt.tight_layout()
    plt.close()

def build_net_entry_timeseries(ue_spell_origin_df: pd.DataFrame, ignore_within: bool = True) -> pd.DataFrame:
    """
    Panel of net entry per occupation per time step, excluding self-flows if requested.
    Returns columns: ['Time Step','Occupation','Entrants','Exits','NetEntry']
    """
    # 1) Optionally remove within-occupation moves
    df = ue_spell_origin_df.copy()
    if ignore_within:
        df = df[df['OriginOccupation'] != df['DestinationOccupation']]

    # 2) Build entrants and exits as Series with a *consistent* MultiIndex: (Time Step, Occupation)
    entrants_ts = (
        df.groupby(['Time Step', 'DestinationOccupation'])
        .size()
        .rename('Entrants')
    )
    entrants_ts.index = pd.MultiIndex.from_tuples(
        entrants_ts.index, names=['Time Step', 'Occupation']
    )

    exits_ts = (
        df.groupby(['Time Step', 'OriginOccupation'])
        .size()
        .rename('Exits')
    )
    exits_ts.index = pd.MultiIndex.from_tuples(
        exits_ts.index, names=['Time Step', 'Occupation']
    )

    # 3) Create the union index and reindex both series to it (ensures perfect alignment)
    union_idx = entrants_ts.index.union(exits_ts.index)
    entrants_ts = entrants_ts.reindex(union_idx, fill_value=0)
    exits_ts    = exits_ts.reindex(union_idx, fill_value=0)

    # 4) Combine into a single DataFrame and compute NetEntry
    panel = pd.concat([entrants_ts, exits_ts], axis=1)
    panel['NetEntry'] = panel['Entrants'] - panel['Exits']

    # 5) Return as a tidy DataFrame with explicit columns
    panel = panel.reset_index()  # columns: Time Step, Occupation, Entrants, Exits, NetEntry
    return panel[['Time Step', 'Occupation', 'Entrants', 'Exits', 'NetEntry']]

def plot_net_entry_heatmap_grid(models_dict: dict, label_map=None, save=False, path=None, title="Net entry heatmap"):
    """
    Plot heatmaps for all models in a grid layout.
    models_dict: {model_name: dataframe}
    """
    # Prepare models
    plots = [(name, df) for name, df in models_dict.items()]

    # Set up grid. Reshaped from the original 4x2 to 2x3 (same 6-model
    # capacity) so panels stay legible at TEXTWIDTH -- 4 columns at 6.27in
    # total width would give ~1.6in per panel, too narrow for tick labels.
    nrows, ncols = 3, 2
    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(_fig_w, _fig_h))
    axes = axes.flatten()
    
    if len(plots) > len(axes):
        print(f"Warning: {len(plots)} models found but only {len(axes)} subplots available — plotting first {len(axes)}.")
    plots_to_plot = plots[:len(axes)]
    
    for ax, (name, df) in zip(axes, plots_to_plot):
        # Build net entry timeseries for this model
        net_ts = build_net_entry_timeseries(df, ignore_within=True)
        
        # Create pivot table
        piv = net_ts.pivot_table(index="Occupation", columns="Time Step", values="NetEntry", aggfunc="sum")
        order = piv.var(axis=1).sort_values(ascending=False).index  # order by volatility
        piv = piv.loc[order]
        
        # Plot heatmap
        im = ax.imshow(piv.values, aspect="auto", cmap="RdBu", interpolation="nearest")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Net Entry")
        
        # Set labels
        labels = [label_map.get(i, str(i)) if label_map is not None else str(i) for i in piv.index]
        ax.set_yticks(np.arange(len(piv)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Occupation")
        ax.set_title(name, fontweight="bold")
    
    # Turn off unused subplots
    for ax in axes[len(plots_to_plot):]:
        ax.axis('off')
    
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_suptitle_spacing(fig)

    if save:
        save_plot_standard(fig, f'{path}/net_entry_heatmap_grid.png', aspect_ratio=_fig_h/_fig_w, apply_grid=False)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_net_entry_stability_grid(models_dict: dict, label_map=None, topk=100, trending_occs=None, 
                                save=False, path=None, title="Net entry stability"):
    """
    Plot stability scatter plots for all models in a grid layout.
    models_dict: {model_name: dataframe}
    """
    # Prepare models
    plots = [(name, df) for name, df in models_dict.items()]
    
    # Set up grid. Reshaped from the original 4x2 to 2x3 (same 6-model
    # capacity) so panels stay legible at TEXTWIDTH -- 4 columns at 6.27in
    # total width would give ~1.6in per panel, too narrow for tick labels.
    nrows, ncols = 3, 2
    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(_fig_w, _fig_h))
    axes = axes.flatten()

    if len(plots) > len(axes):
        print(f"Warning: {len(plots)} models found but only {len(axes)} subplots available — plotting first {len(axes)}.")
    plots_to_plot = plots[:len(axes)]
    
    for ax, (name, df) in zip(axes, plots_to_plot):
        # Build net entry timeseries for this model
        net_ts = build_net_entry_timeseries(df, ignore_within=True)
        
        # Compute statistics
        stats = net_ts.groupby("Occupation")["NetEntry"].agg(["mean", "std"])
        if isinstance(topk, int):
            stats = stats.sort_values("std", ascending=False).head(topk)

        # Handle trending occupations coloring
        if trending_occs is not None:
            idx_dtype = stats.index.dtype
            try:
                trend_set = set(pd.Index(trending_occs).astype(idx_dtype))
                key_series = stats.index
            except Exception:
                trend_set = set(map(str, trending_occs))
                key_series = stats.index.map(str)
            colors = ["red" if k in trend_set else "blue" for k in key_series]
        else:
            colors = "blue"

        labels = [label_map.get(i, str(i)) if label_map is not None else str(i) 
                for i in stats.index]

        # Plot scatter
        ax.scatter(stats["mean"], stats["std"], c=colors, alpha=0.6)
        
        # Add labels (only for a subset to avoid clutter)
        for x, y, lbl in list(zip(stats["mean"], stats["std"], labels))[:20]:  # Only label top 20
            ax.text(x, y, lbl, fontsize=6, ha="right", va="bottom")
        
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Mean net entry")
        ax.set_ylabel("Std dev of net entry")
        ax.set_title(name, fontweight="bold")
    
    # Turn off unused subplots
    for ax in axes[len(plots_to_plot):]:
        ax.axis('off')
    
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_suptitle_spacing(fig)

    if save:
        save_plot_standard(fig, f'{path}/net_entry_stability_grid.png', aspect_ratio=_fig_h/_fig_w, apply_grid=False)
        plt.close()
    else:
        plt.show()

def plot_market_tightness(res_dict, observation, save = False, path = None, colors = None, adjusted = False, calib_v_line=None):

    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), colors))

    plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO))
    for name, res in res_dict.items():
        if adjusted:
            plt.plot(res['DATE'], res['Vacancies']/(res['Unemployment'] + res['Employed Seekers']), color = colors[name])
        else:
            plt.plot(res['DATE'], res['Vacancies']/res['Unemployment'], color = colors[name])

    if adjusted:
        obs_handle = Line2D([0], [0], color='black', linestyle='dotted', label='Observed (AHR Adjusted)')
        plt.plot(observation['DATE'], observation['LaborTightness_06_AHR'], color='black', linestyle = "dotted")
    else:
        obs_handle = Line2D([0], [0], color='grey', linestyle='dashed', label='Observed (BLS/FRED)')
        plt.plot(observation['DATE'], observation['tightness_jolts'], color='grey', linestyle = "dashed")

    plt.title("Market Tightness (V/U)")  # Figure-wide title
    plt.xlabel("Date")  # Shared x-axis label
    plt.ylabel("Market Tightness")  # Shared y-axis label

    if calib_v_line is not None:
        shade_out_of_sample(plt.gca(), calib_v_line[1])

    plt.tight_layout()
    fig = plt.gcf()
    add_shared_legend(fig, colors, extra_handles=[obs_handle], ncol=3)

    # Show plot
    if save:
        if adjusted:
            save_plot_standard(fig, f'{path}market_tightness_line_adjusted.png')
        else:
            save_plot_standard(fig, f'{path}market_tightness_line.png')
        plt.close()
    else:
        plt.show()


def plot_occ_market_tightness(res_dict, calib_date=None, save=False, path=None):
    """
    For each model, a heatmap of occupation-level market tightness (V/U) averaged
    over simulation years. Occupations are ranked top-to-bottom by mean tightness.
    Colormap centred on V/U = 1 (balanced market): green = employer-tight,
    red = worker-abundant.

    Args:
        res_dict (dict): {model_name: record_df} — use filtered_sim_results, which
                         has columns 'Occupation', 'Time Step', 'Vacancies',
                         'Unemployment' (one row per occupation per month).
        calib_date (list): Optional [start_date, end_date] strings; if provided,
                           x-axis labels show calendar years.
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
    """
    n = len(res_dict)
    cols = min(n, 2)
    rows = math.ceil(n / cols)

    n_occs = next(iter(res_dict.values()))['Occupation'].nunique()
    row_h = max(4, 0.22 * n_occs)

    # Width fixed at TEXTWIDTH; height kept at the original row_h*rows (not
    # rescaled proportionally) since it's sized for occupation-row count, not
    # for the old 7*cols-wide canvas.
    _fig_w, _fig_h = TEXTWIDTH, row_h * rows
    fig, axes = plt.subplots(rows, cols,
                             figsize=(_fig_w, _fig_h),
                             squeeze=False)
    axes = axes.flatten()

    for i, (name, record_df) in enumerate(res_dict.items()):
        ax = axes[i]

        df = record_df[['Time Step', 'Occupation', 'Vacancies', 'Unemployment', 'Employment']].copy()
        df['VU_ratio'] = np.where(df['Unemployment'] > 0,
                                  df['Vacancies'] / df['Unemployment'],
                                  np.nan)

        # Aggregate monthly time steps to annual averages
        t_min = df['Time Step'].min()
        df['Year'] = (df['Time Step'] - t_min) // 12

        occ_year = df.groupby(['Year', 'Occupation']).agg(
            Mean_VU=('VU_ratio', 'mean'),
            Mean_Employment=('Employment', 'mean')
        ).reset_index()

        pivot = occ_year.pivot(index='Occupation', columns='Year', values='Mean_VU')

        # Sort occupations by mean employment size: largest at top
        occ_size = occ_year.groupby('Occupation')['Mean_Employment'].mean()
        pivot = pivot.loc[occ_size.sort_values(ascending=False).index]

        # Label columns with calendar years if calib_date provided
        if calib_date is not None:
            start_year = pd.to_datetime(calib_date[0]).year
            pivot.columns = [start_year + yr for yr in pivot.columns]

        # Cap colour scale at 95th percentile to prevent outliers dominating
        raw_vals = pivot.values[~np.isnan(pivot.values)]
        vmax = np.percentile(raw_vals, 95) if len(raw_vals) > 0 else 2.0

        sns.heatmap(pivot, ax=ax,
                    cmap='RdYlGn', center=1, vmin=0, vmax=vmax,
                    cbar_kws={'label': 'V/U Ratio', 'shrink': 0.7},
                    linewidths=0, yticklabels=True, xticklabels=True)

        ax.set_title(name, fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Occupation (ranked by size, largest at top)")
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Occupation-Level Market Tightness (V/U)",
                 fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    add_suptitle_spacing(fig)

    if save:
        save_plot_standard(fig, f'{path}occ_market_tightness.png', aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_tightness_wage_corr(res_dict, calib_date=None, save=False, path=None):
    """
    For each model, two panels side by side:

    Left — Structural (levels): mean V/U vs mean wage per occupation, averaged
    over the full simulation. Shows whether structurally tight occupations
    also pay higher wages.

    Right — Dynamic (changes): year-over-year change in V/U vs year-over-year
    % change in mean wage, one point per occupation-year. This is the
    occupation-level labour market Phillips curve: does tightening induce
    upward wage pressure?

    Both panels show a regression line and Pearson r / p-value.

    Args:
        res_dict (dict): {model_name: record_df} — use filtered_sim_results,
                         which has columns 'Occupation', 'Time Step',
                         'Vacancies', 'Unemployment', 'Mean Occupational Wage'.
        calib_date (list): Optional; not used for computation but kept for
                           consistency with other functions.
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
    """
    n = len(res_dict)
    # Two panels per model: levels (left) and dynamics (right). Width fixed
    # at TEXTWIDTH; height set from row count (ROW_HEIGHT in/row, one row
    # per model), independent of the fixed 2-column layout.
    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * n
    fig, axes = plt.subplots(n, 2,
                             figsize=(_fig_w, _fig_h),
                             squeeze=False)

    all_years = []
    for record_df in res_dict.values():
        t_min = record_df['Time Step'].min()
        yrs = ((record_df['Time Step'] - t_min) // 12).unique()
        all_years.extend(yrs.tolist())
    year_min, year_max = min(all_years), max(all_years)
    norm = plt.Normalize(vmin=year_min, vmax=year_max)
    cmap = plt.cm.viridis

    for i, (name, record_df) in enumerate(res_dict.items()):
        ax_lev = axes[i, 0]
        ax_dyn = axes[i, 1]

        df = record_df[['Time Step', 'Occupation',
                         'Vacancies', 'Unemployment',
                         'Mean Occupational Wage']].copy()
        df['VU'] = np.where(df['Unemployment'] > 0,
                            df['Vacancies'] / df['Unemployment'],
                            np.nan)
        t_min = df['Time Step'].min()
        df['Year'] = (df['Time Step'] - t_min) // 12

        # Annual averages per occupation
        ann = (df.groupby(['Year', 'Occupation'])
               .agg(VU=('VU', 'mean'), Wage=('Mean Occupational Wage', 'mean'))
               .reset_index()
               .sort_values(['Occupation', 'Year']))

        # ── LEFT: structural levels ──────────────────────────────────────────
        occ_means = ann.groupby('Occupation').agg(
            Mean_VU=('VU', 'mean'), Mean_Wage=('Wage', 'mean')).reset_index()
        occ_means = occ_means.dropna()

        ax_lev.scatter(occ_means['Mean_VU'], occ_means['Mean_Wage'],
                       color='steelblue', alpha=0.7, s=40)

        if len(occ_means) > 2:
            slope, intercept, r, p, _ = stats.linregress(
                occ_means['Mean_VU'], occ_means['Mean_Wage'])
            x_line = np.array([occ_means['Mean_VU'].min(),
                                occ_means['Mean_VU'].max()])
            ax_lev.plot(x_line, intercept + slope * x_line,
                        color='grey', linestyle='dashed', linewidth=1.2)
            p_str = f'{p:.3f}' if p >= 0.001 else '<0.001'
            ax_lev.text(0.05, 0.95, f'r = {r:.2f}, p = {p_str}',
                        transform=ax_lev.transAxes,
                        verticalalignment='top', fontsize=9)

        ax_lev.set_title(f'{name} — Structural (levels)', fontweight='bold')
        ax_lev.set_xlabel('Mean V/U Ratio')
        ax_lev.set_ylabel('Mean Occupation Wage ($)')

        # ── RIGHT: dynamic changes ───────────────────────────────────────────
        ann['Delta_VU'] = ann.groupby('Occupation')['VU'].diff()
        ann['Wage_Growth_pct'] = (ann.groupby('Occupation')['Wage']
                                     .pct_change() * 100)
        dyn = ann.dropna(subset=['Delta_VU', 'Wage_Growth_pct'])

        sc = ax_dyn.scatter(dyn['Delta_VU'], dyn['Wage_Growth_pct'],
                            c=dyn['Year'], cmap=cmap, norm=norm,
                            alpha=0.5, s=20, zorder=3)

        if len(dyn) > 2:
            slope, intercept, r, p, _ = stats.linregress(
                dyn['Delta_VU'], dyn['Wage_Growth_pct'])
            x_line = np.array([dyn['Delta_VU'].min(), dyn['Delta_VU'].max()])
            ax_dyn.plot(x_line, intercept + slope * x_line,
                        color='grey', linestyle='dashed', linewidth=1.2)
            p_str = f'{p:.3f}' if p >= 0.001 else '<0.001'
            ax_dyn.text(0.05, 0.95, f'r = {r:.2f}, p = {p_str}',
                        transform=ax_dyn.transAxes,
                        verticalalignment='top', fontsize=9)

        ax_dyn.axhline(0, color='lightgrey', linewidth=0.8, linestyle='dotted')
        ax_dyn.axvline(0, color='lightgrey', linewidth=0.8, linestyle='dotted')
        ax_dyn.set_title(f'{name} — Dynamic (YoY Changes)', fontweight='bold')
        ax_dyn.set_xlabel('Δ V/U (year-on-year)')
        ax_dyn.set_ylabel('Wage Growth (%/yr)')

    # Shared colourbar for year on the right panels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, 1].tolist(), shrink=0.6, pad=0.02)
    cbar.set_label('Simulation Year', fontsize=10)

    fig.suptitle('Does Market Tightness Drive Occupation Wages?',
                 fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    add_suptitle_spacing(fig)

    if save:
        save_plot_standard(fig, f'{path}tightness_wage_corr.png', aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()

def plot_wage_comparison_dev(res_dict, observation, calib_date,
        save=False, path=None, colors=None, calib_v_line=None):
    """
    6-panel wage validation figure (2 rows × 3 columns).

    Each panel shows one model's raw employment-weighted average wage (left axis)
    against the observed median usual weekly real earnings annualised (right axis),
    matching the original plot_wage_comparison layout but one model per panel.

    Parameters
    ----------
    res_dict    : dict  model_name -> per-occupation record DataFrame
    observation : DataFrame  with 'DATE' and 'Observed Wages' columns (quarterly)
    calib_date  : [start_str, end_str]
    """
    if colors is None:
        colors = dict(zip(res_dict.keys(),
                        ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']))

    obs_sorted = observation.dropna(subset=['Observed Wages']).sort_values('DATE').copy()
    obs_sorted['wage_ann'] = obs_sorted['Observed Wages'] * 52

    names  = list(res_dict.keys())
    ncols  = 2
    nrows  = -(-len(names) // ncols)
    # Sized to print at exactly TEXTWIDTH, ROW_HEIGHT in/row (independent of
    # ncols) -- the shared convention across every multi-row figure.
    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT_3 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(_fig_w, _fig_h), sharex=True)
    axes_flat  = axes.flatten()

    last_dates = None
    for i, name in enumerate(names):
        ax  = axes_flat[i]
        ax2 = ax.twinx()
        # Twin axis needs its right spine visible — it's the visual anchor
        # for the second y-axis, unlike the global top/right-spine-off default.
        ax2.spines['right'].set_visible(True)
        res = res_dict[name]

        result = (res.dropna(subset=['Mean Occupational Wage'])
                .groupby('Time Step')
                .apply(lambda x: np.average(x['Mean Occupational Wage'],
                                            weights=x['Employment']),
                        include_groups=False)
                .to_frame(name='wage').reset_index())
        result['DATE'] = pd.date_range(start=calib_date[0],
                                    periods=len(result), freq='ME')
        last_dates = result['DATE']

        sim_line, = ax.plot(result['DATE'], result['wage'],
                            color=colors[name], label=name)
        obs_line, = ax2.plot(obs_sorted['DATE'], obs_sorted['wage_ann'],
                            color='black', linestyle='dotted',
                            label='Observed')

        # Shade gap between the two lines using the twin-axis linear transform.
        # After plotting, both axes auto-scale independently; the observed dotted
        # line's visual position maps ax2 → ax via the same linear transform
        # matplotlib uses internally. Applying that transform lets fill_between
        # shade exactly the visual gap.
        ax_lo,  ax_hi  = ax.get_ylim()
        ax2_lo, ax2_hi = ax2.get_ylim()
        _obs_ts      = obs_sorted['DATE'].values.astype('int64')
        _sim_ts      = result['DATE'].values.astype('int64')
        _obs_monthly = np.interp(_sim_ts, _obs_ts, obs_sorted['wage_ann'].values)
        _obs_on_ax   = ax_lo + (_obs_monthly - ax2_lo) / (ax2_hi - ax2_lo) * (ax_hi - ax_lo)
        ax.fill_between(result['DATE'], result['wage'], _obs_on_ax,
                        alpha=0.15, color=colors[name])


        ax.set_xlim(last_dates.min(), last_dates.max())
        ax.grid(axis='y', alpha=0.3)
        if calib_v_line is not None:
            shade_out_of_sample(ax, calib_v_line[1])

        _dollar_fmt = plt.matplotlib.ticker.FuncFormatter(lambda x, pos: f'${x:,.0f}')
        ax.yaxis.set_major_formatter(_dollar_fmt)
        ax2.yaxis.set_major_formatter(_dollar_fmt)
        if not (i % ncols == ncols - 1 or i == len(names) - 1):
            ax2.set_yticks([])
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel('Date')

    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('Simulated and Observed Economy-Wide Wages',
                fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    # One shared y-axis label per side (rather than repeating it on every row) --
    # fig.supylabel handles the left side natively; the right side has no matplotlib
    # equivalent, so it's placed manually with fig.text. Both sit at a fixed
    # figure-fraction x position regardless of subplots_adjust -- growing the
    # left/right margins moves the *tick labels* toward/away from them, so the
    # gap is measured and grown iteratively rather than guessed as a fixed
    # margin (dollar-formatted ticks are wider than plain numbers and vary
    # with the data's magnitude, e.g. "$9,500" vs "$150,000").
    left_label = fig.supylabel('Simulated Average Economy-Wide Wage', fontsize=plt.rcParams['axes.labelsize'])
    right_label = fig.text(0.995, 0.5, 'Observed wages\n(1982–84 CPI-adjusted $)', rotation=270,
                            va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
    left, right = 0.12, 0.88
    # A few px reads as "touching" once rendered, not a visible gap -- same
    # font-size-scaled minimum used for the legend's overlap check (~1/3 of
    # the label font's point size, converted to px at the figure's dpi).
    _min_gap_px = plt.rcParams['axes.labelsize'] * fig.dpi / 72 / 3
    for _ in range(5):
        fig.subplots_adjust(left=left, right=right)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_w_px = fig.get_size_inches()[0] * fig.dpi

        supyl_bbox = left_label.get_window_extent(renderer)
        left_col_axes = [ax for ax in fig.axes
                          if ax.get_label() != '<colorbar>' and not ax.spines['right'].get_visible()
                          and ax.get_subplotspec() is not None and ax.get_subplotspec().colspan.start == 0]
        left_tick_x0 = min(
            (t.get_window_extent(renderer).x0
             for ax in left_col_axes for t in ax.get_yticklabels() if t.get_text()),
            default=None)
        adjusted = False
        if left_tick_x0 is not None:
            gap = left_tick_x0 - supyl_bbox.x1
            if gap < _min_gap_px:
                left += (_min_gap_px - gap) / fig_w_px
                adjusted = True

        right_bbox = right_label.get_window_extent(renderer)
        right_col_axes = [ax for ax in fig.axes if ax.spines['right'].get_visible() and ax.get_label() != '<colorbar>']
        right_tick_x1 = max(
            (t.get_window_extent(renderer).x1
             for ax in right_col_axes for t in ax.get_yticklabels() if t.get_text()),
            default=None)
        if right_tick_x1 is not None:
            gap = right_bbox.x0 - right_tick_x1
            if gap < _min_gap_px:
                right -= (_min_gap_px - gap) / fig_w_px
                adjusted = True

        if not adjusted:
            break
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='black', linestyle='dotted',
                                              label='Observed')],
                       ncol=3)

    if save and path is not None:
        save_plot_standard(fig, os.path.join(path, 'wage_comparison_dev.png'), aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_wage_comparison_dev_error(res_dict, observation, calib_date,
        save=False, path=None, colors=None, calib_v_line=None):
    """
    Deviation version of plot_wage_comparison_dev.

    Each panel shows the standardised error between simulated and observed wages:

        sim_z(t)  = (sim(t)  − mean(sim))  / std(sim)
        obs_z(t)  = (obs(t)  − mean(obs))  / std(obs)
        error(t)  = sim_z(t) − obs_z(t)

    Both series are z-scored to their calibration-period mean and SD, removing
    level and volatility differences. The error reflects phase and shape
    alignment only — positive means sim is above its own mean by more than
    observed is above its mean at that point in time.
    """
    if colors is None:
        colors = dict(zip(res_dict.keys(),
                        ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']))

    obs_sorted = observation.dropna(subset=['Observed Wages']).sort_values('DATE').copy()
    obs_sorted['wage_ann'] = obs_sorted['Observed Wages'] * 52
    obs_ts  = obs_sorted['DATE'].values.astype('int64')
    obs_val = obs_sorted['wage_ann'].values

    names  = list(res_dict.keys())
    ncols  = 2
    nrows  = -(-len(names) // ncols)
    # Sized to print at exactly TEXTWIDTH, ROW_HEIGHT in/row (independent of
    # ncols) -- the shared convention across every multi-row figure.
    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT_3 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(_fig_w, _fig_h),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, name in enumerate(names):
        ax  = axes_flat[i]
        col = colors[name]
        res = res_dict[name]

        result = (res.dropna(subset=['Mean Occupational Wage'])
                  .groupby('Time Step')
                  .apply(lambda x: np.average(x['Mean Occupational Wage'],
                                              weights=x['Employment']),
                         include_groups=False)
                  .to_frame(name='wage').reset_index())
        result['DATE'] = pd.date_range(start=calib_date[0],
                                       periods=len(result), freq='ME')

        sim_dates = result['DATE'].values.astype('int64')
        obs_interp = np.interp(sim_dates, obs_ts, obs_val)

        sim_vals = result['wage'].values
        sim_mean, sim_std = sim_vals.mean(), sim_vals.std()
        obs_mean, obs_std = obs_interp.mean(), obs_interp.std()

        if sim_std == 0 or obs_std == 0:
            continue

        sim_z = (sim_vals   - sim_mean) / sim_std
        obs_z = (obs_interp - obs_mean) / obs_std
        error = sim_z - obs_z

        ax.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
        ax.plot(result['DATE'], error, color=col)
        ax.fill_between(result['DATE'], error, 0,
                        where=(error >= 0), alpha=0.15, color=col)
        ax.fill_between(result['DATE'], error, 0,
                        where=(error < 0), alpha=0.15, color=col)

        ax.set_xlim(result['DATE'].min(), result['DATE'].max())
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        if calib_v_line is not None:
            shade_out_of_sample(ax, calib_v_line[1])

        if i % ncols == 0:
            ax.set_ylabel('Standardised error (sim z − obs z)')
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel('Date')

    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        'Simulated vs observed wage — standardised error\n'
        r'$z_{\mathrm{sim}}(t) - z_{\mathrm{obs}}(t)$',
        fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors, ncol=3)

    if save and path is not None:
        save_plot_standard(fig, os.path.join(path, 'wage_comparison_dev_error.png'), aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_wage_comparison_comoment(res_dict, observation, calib_date,
        save=False, path=None, colors=None, rolling_window=36, calib_v_line=None):
    """
    Co-movement figure for simulated vs observed wages.

    For each model, two stacked panels:
      Top:    Both series z-scored to calibration-period mean/SD, overlaid.
              Sim = solid coloured line; Observed = black dotted line.
      Bottom: Rolling Pearson correlation (window = rolling_window months)
              between the z-scored sim and observed series. Shaded grey band
              marks the ±0.5 corridor for reference.

    Layout: ceil(n_models / 3) pairs of rows × 3 columns.
    """
    if colors is None:
        colors = dict(zip(res_dict.keys(),
                        ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']))

    obs_sorted = observation.dropna(subset=['Observed Wages']).sort_values('DATE').copy()
    obs_sorted['wage_ann'] = obs_sorted['Observed Wages'] * 52
    obs_ts  = obs_sorted['DATE'].values.astype('int64')
    obs_val = obs_sorted['wage_ann'].values

    names  = list(res_dict.keys())
    ncols  = 2
    n_model_rows = -(-len(names) // ncols)   # ceil division
    fig_nrows    = 2 * n_model_rows
    # Sized to print at exactly TEXTWIDTH, ROW_HEIGHT in/model-row (each
    # model row is a top+bottom sub-panel pair, height_ratios=[2,1] splits
    # that row's height between them -- not extra height on top).
    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT_3 * n_model_rows
    fig, axes = plt.subplots(
        fig_nrows, ncols,
        figsize=(_fig_w, _fig_h),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1] * n_model_rows, 'hspace': 0.08})

    for i, name in enumerate(names):
        model_row = i // ncols
        col_i     = i % ncols
        ax_top = axes[2 * model_row,     col_i]
        ax_bot = axes[2 * model_row + 1, col_i]
        col    = colors[name]

        res = res_dict[name]
        result = (res.dropna(subset=['Mean Occupational Wage'])
                  .groupby('Time Step')
                  .apply(lambda x: np.average(x['Mean Occupational Wage'],
                                              weights=x['Employment']),
                         include_groups=False)
                  .to_frame(name='wage').reset_index())
        result['DATE'] = pd.date_range(start=calib_date[0],
                                       periods=len(result), freq='ME')

        sim_dates  = result['DATE'].values.astype('int64')
        obs_interp = np.interp(sim_dates, obs_ts, obs_val)

        sim_vals = result['wage'].values
        sim_std, obs_std = sim_vals.std(), obs_interp.std()
        if sim_std == 0 or obs_std == 0:
            continue

        sim_z = (sim_vals   - sim_vals.mean())   / sim_std
        obs_z = (obs_interp - obs_interp.mean()) / obs_std

        # ── Top panel: z-scored overlay ───────────────────────────────────────
        ax_top.plot(result['DATE'], sim_z, color=col)
        ax_top.plot(result['DATE'], obs_z, color='black', linewidth=1.2,
                    linestyle='dotted')
        ax_top.fill_between(result['DATE'], sim_z, obs_z,
                            alpha=0.15, color=col)
        ax_top.axhline(0, color='grey', linewidth=0.6, linestyle='--', alpha=0.5)
        if col_i == 0:
            ax_top.set_ylabel('Z-Score')
        ax_top.tick_params(labelsize=8, labelbottom=False)
        ax_top.grid(axis='y', alpha=0.3)
        ax_top.spines[['top', 'right']].set_visible(False)

        # ── Bottom panel: rolling correlation ─────────────────────────────────
        sim_s = pd.Series(sim_z, index=result['DATE'])
        obs_s = pd.Series(obs_z, index=result['DATE'])
        roll_corr = sim_s.rolling(rolling_window).corr(obs_s)

        bar_colors = [col if v >= 0 else 'lightgrey'
                      for v in roll_corr.fillna(0)]
        ax_bot.bar(result['DATE'], roll_corr.fillna(0),
                   width=25, color=bar_colors, alpha=0.85, linewidth=0)
        ax_bot.axhline(0, color='black', linewidth=0.6, alpha=0.6)
        ax_bot.axhspan(-0.5, 0.5, color='grey', alpha=0.06)
        ax_bot.set_ylim(-1.05, 1.05)
        if col_i == 0:
            ax_bot.set_ylabel(f'Rolling ({rolling_window}m)\nCorrelation\nCoefficient')
        ax_bot.tick_params(labelsize=7)
        ax_bot.grid(axis='y', alpha=0.3)
        ax_bot.spines[['top', 'right']].set_visible(False)
        if model_row == n_model_rows - 1:
            ax_bot.set_xlabel('Date')

        if calib_v_line is not None:
            shade_out_of_sample(ax_top, calib_v_line[1])
            shade_out_of_sample(ax_bot, calib_v_line[1])

    # Hide unused axes
    for j in range(len(names), n_model_rows * ncols):
        model_row = j // ncols
        col_i     = j % ncols
        axes[2 * model_row,     col_i].set_visible(False)
        axes[2 * model_row + 1, col_i].set_visible(False)

    fig.suptitle(
        f'Simulated and Observed Economy-Wide Wages', fontweight='bold')
    # plt.tight_layout() is explicitly incompatible with this figure's custom
    # gridspec_kw (height_ratios + hspace) -- matplotlib warns about it, and
    # in practice it left ~2.7in of dead space between the title and the
    # plots. add_suptitle_spacing sets `top` from a measured target gap
    # (consistent across figure heights), leaving the gridspec's own
    # hspace/wspace untouched -- a hardcoded top=0.94 here previously gave
    # this figure a ~3x bigger title gap than shorter figures, since 0.94 is
    # a fixed *fraction* of this particular (tall) figure's height.
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='black', linestyle='dotted',
                                              linewidth=1.2, label='Observed')],
                       ncol=3)

    if save and path is not None:
        save_plot_standard(fig, os.path.join(path, 'wage_comparison_comoment.png'), aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_wage_comparison(res_dict, observation, calib_date, save=False, path=None, colors=None, adjusted=False, calib_v_line=None):

    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), colors))

    fig, ax_sim = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO))
    ax_obs = ax_sim.twinx()
    # Twin axis needs its right spine visible — it's the visual anchor
    # for the second y-axis, unlike the global top/right-spine-off default.
    ax_obs.spines['right'].set_visible(True)

    last_dates = None
    for name, res in res_dict.items():
        result = (res.dropna(subset=['Mean Occupational Wage'])
                  .groupby('Time Step')
                  .apply(lambda x: np.average(x['Mean Occupational Wage'], weights=x['Employment']),
                         include_groups=False)
                  .to_frame(name='Average Economy-Wide Wage')
                  .reset_index())
        result['DATE'] = pd.date_range(start=calib_date[0], periods=len(result), freq='ME')
        last_dates = result['DATE']
        ax_sim.plot(result['DATE'], result['Average Economy-Wide Wage'],
                    color=colors[name])

    ax_obs.plot(observation['DATE'], observation['Observed Wages'] * 52,
                color='black', linestyle='dotted')

    ax_sim.set_xlabel('Date')
    ax_sim.set_ylabel('Simulated average economy-wide wage')
    ax_obs.set_ylabel('Observed wages (1982–84 CPI-adj. dollars)')

    if last_dates is not None:
        ax_sim.set_xlim(last_dates.min(), last_dates.max())

    if calib_v_line is not None:
        shade_out_of_sample(ax_sim, calib_v_line[1])

    fig.suptitle('Simulated and Observed Economy-Wide Wages', fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='black', linestyle='dotted',
                                              label='Median usual weekly real earnings (observed)')],
                       ncol=3)

    if save:
        save_plot_standard(fig, f'{path}wage_comparison.png')
        plt.close()
    else:
        plt.show()


def plot_wage_vacancy_grid(res_dict, calib_date, smooth_window=6, save=False, path=None, colors=None, calib_v_line=None):
    """
    Plots a grid of subplots (one per model) comparing the economy-wide average wage
    and average vacancy offer over time.

    Args:
        res_dict (dict): Keys are model names, values are record_df DataFrames with
                         'Time Step', 'Mean Occupational Wage', 'Mean Vacancy Offer',
                         'Employment' columns.
        calib_date (list): [start_date, end_date] strings, e.g. ["2000-12-01", "2019-05-01"].
        smooth_window (int): Rolling-mean window (months) applied to vacancy offer series.
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
        colors (dict): Optional mapping of model name -> colour for the wage line.
                       Vacancy offer is always plotted in grey dashed.
    """
    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), default_colors))

    n = len(res_dict)
    max_cols = 2
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig, axes = plt.subplots(rows, cols, figsize=(_fig_w, _fig_h), squeeze=False)
    axes = axes.flatten()

    for i, (name, record_df) in enumerate(res_dict.items()):
        ax = axes[i]

        result = (record_df
            .dropna(subset=['Mean Occupational Wage'])
            .groupby('Time Step')
            .apply(lambda x: np.average(x['Mean Occupational Wage'], weights=x['Employment']), include_groups=False)
            .to_frame(name='Average Economy-Wide Wage')
            .reset_index())

        result_vacs = (record_df
            .dropna(subset=['Mean Vacancy Offer'])
            .groupby('Time Step')
            .apply(lambda x: np.average(x['Mean Vacancy Offer'], weights=x['Employment']), include_groups=False)
            .to_frame(name='Average Economy-Wide Vacancy Offer')
            .reset_index())

        dates = pd.date_range(start=calib_date[0], periods=len(result), freq="ME")
        vac_dates = pd.date_range(start=calib_date[0], periods=len(result_vacs), freq="ME")

        smoothed_vac = result_vacs['Average Economy-Wide Vacancy Offer'].rolling(
            window=smooth_window, center=True, min_periods=1).mean()

        ax.plot(dates, result['Average Economy-Wide Wage'],
                color=colors[name], label='Avg Wage')
        ax.plot(vac_dates, smoothed_vac,
                color='grey', linestyle='dashed', label=f'Avg Vacancy Offer ({smooth_window}m MA)')

        ax.set_xlabel("Date")
        ax.set_ylabel("Wage ($)")
        if calib_v_line is not None:
            shade_out_of_sample(ax, calib_v_line[1])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Average Wage vs. Average Vacancy Offer", fontweight='bold')

    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='grey', linestyle='dashed',
                                              label=f'Avg Vacancy Offer ({smooth_window}m MA)')],
                       ncol=3)

    if save:
        save_plot_standard(fig, f'{path}wage_vacancy_grid.png',
                           aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_retirement_rate_grid(res_dict, calib_date, save=False, path=None, colors=None, calib_v_line=None,
                               model_results_all_sims=None, confidence_level=0.95):
    """
    Grid of subplots (one per model) showing the retirement rate over time.

    Args:
        res_dict (dict): {model_name: grouped_df} where grouped_df has columns
                         'Retirees', 'Workers', 'Time Step' (from run_single_local).
        calib_date (list): [start_date, end_date] strings, e.g. ["2000-12-01", "2019-05-01"].
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
        colors (dict): Optional mapping of model name -> colour.
        model_results_all_sims (dict): Optional {model_name: [grouped_df, ...]} -- one raw
            grouped DataFrame per simulation run (pre-aggregation), keyed the same as
            res_dict. When given (multi-simulation mode, len >= 2), the retirement rate is
            computed per-simulation first, then averaged with a +-CI ribbon -- computing the
            CI on the ratio directly, rather than from the separately-aggregated Retirees/
            Workers CIs, since a ratio of two CIs isn't the same as a CI of the ratio.
        confidence_level (float): Confidence level for the ribbon (default 0.95).
    """
    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), default_colors))

    n = len(res_dict)
    cols = min(n, 2)
    rows = math.ceil(n / cols)

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig, axes = plt.subplots(rows, cols, figsize=(_fig_w, _fig_h), squeeze=False)
    axes = axes.flatten()

    z = stats.norm.ppf((1 + confidence_level) / 2)

    for i, (name, grouped) in enumerate(res_dict.items()):
        ax = axes[i]
        dates = pd.date_range(start=calib_date[0], periods=len(grouped), freq="ME")

        sim_list = model_results_all_sims.get(name) if model_results_all_sims else None
        if sim_list is not None and len(sim_list) > 1:
            per_sim_rates = np.stack(
                [(sim_df['Retirees'] / sim_df['Workers']).values for sim_df in sim_list], axis=0)
            mean_rate = per_sim_rates.mean(axis=0)
            std_rate = per_sim_rates.std(axis=0)
            ci_margin = z * std_rate / np.sqrt(per_sim_rates.shape[0])
            ax.plot(dates, mean_rate, color=colors[name])
            ax.fill_between(dates, mean_rate - ci_margin, mean_rate + ci_margin,
                            color=colors[name], alpha=0.2)
        else:
            ret_rate = grouped['Retirees'] / grouped['Workers']
            ax.plot(dates, ret_rate, color=colors[name])

        ax.set_xlabel("Date")
        ax.set_ylabel("Retirement Rate")
        if calib_v_line is not None:
            shade_out_of_sample(ax, calib_v_line[1])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Monthly Retirement Rate (Retirees / Workforce)", fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors, ncol=3)

    if save:
        save_plot_standard(fig, f'{path}retirement_rate_grid.png',
                           aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_wage_dist_by_year(res_dict, save=False, path=None, colors=None):
    """
    Grid of subplots (one per model) showing economy-wide wage distribution per year
    as box plots, using annual snapshots of individual worker wages.

    Args:
        res_dict (dict): {model_name: wage_age_df} where wage_age_df has columns
                         ['Time Step', 'Year', 'Occupation', 'Wage', 'Age']
                         (10th element of run_single_local return tuple).
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
        colors (dict): Optional model name -> colour (used for the box fills).
    """
    # Requires raw individual-level data with a 'Wage' column.
    # Skips silently when res_dict is empty or contains only summary-format data.
    _sample = next(iter(res_dict.values()), None)
    if _sample is None or _sample.empty or 'Wage' not in _sample.columns:
        return
    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), default_colors))

    n = len(res_dict)
    cols = min(n, 2)
    rows = math.ceil(n / cols)

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig, axes = plt.subplots(rows, cols, figsize=(_fig_w, _fig_h), squeeze=False)
    axes = axes.flatten()

    for i, (name, wage_age_df) in enumerate(res_dict.items()):
        ax = axes[i]
        years = sorted(wage_age_df['Year'].unique())
        data_by_year = [wage_age_df.loc[wage_age_df['Year'] == y, 'Wage'].values for y in years]
        bp = ax.boxplot(data_by_year, labels=years, sym='', patch_artist=True)
        for box in bp['boxes']:
            box.set_facecolor(colors[name])
            box.set_alpha(0.5)
        for median in bp['medians']:
            median.set_color(colors[name])
        ax.set_xlabel("Simulation Year")
        ax.set_ylabel("Wage ($)")
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Economy-Wide Wage Distribution by Year", fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors, ncol=3)

    if save:
        save_plot_standard(fig, f'{path}wage_dist_by_year.png',
                           aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_wage_percentile_ts(res_dict, calib_date, save=False, path=None, colors=None, calib_v_line=None):
    """
    Two separate figures showing economy-wide wage percentile dynamics per model.

    Figure 1 (wage_percentile_levels.png):
        One subplot per model. p10, p50, p90 indexed to 100 at the first year;
        p10–p90 shaded band; p50 solid line; p10/p90 dashed lines.

    Figure 2 (wage_percentile_ratio.png):
        One subplot per model. p90/p10 inequality ratio over time.

    Args:
        res_dict (dict): {model_name: wage_age_df} from filtered_wage_age_dfs.
        calib_date (list): ["YYYY-MM-DD", "YYYY-MM-DD"] — maps Year index to dates.
        save (bool): save figures.
        path (str): output directory.
        colors (dict): optional {model_name: color}.
    """
    start_year = pd.to_datetime(calib_date[0]).year
    _colors = colors or {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    valid = [(name, wdf) for name, wdf in res_dict.items()
             if wdf is not None and len(wdf) > 0]
    n = len(valid)
    if n == 0:
        print("plot_wage_percentile_ts: no data to plot.")
        return

    cols = min(n, 2)
    rows = math.ceil(n / cols)

    # Average summary percentiles across sims for each model
    # Input format: DataFrame with columns Year, p10, p25, p50, p75, p90, mean, sim_id
    model_pcts = {}
    for name, wdf in valid:
        _cols = ['p10', 'p50', 'p90', 'mean']
        _has_sims = 'sim_id' in wdf.columns and wdf['sim_id'].nunique() > 1
        _grp = wdf.groupby('Year')[_cols]
        avg = _grp.mean()
        sem = _grp.sem() if _has_sims else None
        _dates = pd.to_datetime([f"{start_year + y}-01-01" for y in avg.index])
        avg.index = _dates
        if sem is not None:
            sem.index = _dates

        # per-sim p90/p10 ratio for Figure 2 CIs
        if _has_sims:
            _ps = wdf.groupby(['sim_id', 'Year'])[['p10', 'p90']].mean()
            _ratio_per_sim = (_ps['p90'] / _ps['p10']).reset_index()
            _ratio_per_sim.columns = ['sim_id', 'Year', 'ratio']
            _ratio_mean = _ratio_per_sim.groupby('Year')['ratio'].mean()
            _ratio_sem  = _ratio_per_sim.groupby('Year')['ratio'].sem()
            _ratio_mean.index = _dates
            _ratio_sem.index  = _dates
        else:
            _ratio_mean = avg['p90'] / avg['p10']
            _ratio_sem  = None

        model_pcts[name] = dict(avg=avg, sem=sem,
                                ratio_mean=_ratio_mean, ratio_sem=_ratio_sem)

    z = 1.96
    used_colors = {name: _colors.get(name, color_cycle[i % len(color_cycle)])
                   for i, name in enumerate(model_pcts.keys())}

    # ── Figure 1: indexed levels ──────────────────────────────────────────────
    _fig1_w, _fig1_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig1, axes1 = plt.subplots(rows, cols, figsize=(_fig1_w, _fig1_h),
                               squeeze=False)
    axes1 = axes1.flatten()

    for i, (name, d) in enumerate(model_pcts.items()):
        ax   = axes1[i]
        color = _colors.get(name, color_cycle[i % len(color_cycle)])
        pcts = d['avg']
        sem  = d['sem']

        base  = pcts.iloc[0]
        idx   = pcts / base * 100
        # inequality spread band (p10–p90 of the wage distribution)
        ax.fill_between(idx.index, idx['p10'], idx['p90'],
                        color=color, alpha=0.10, label='p10–p90 spread')

        for col, ls, lbl in [('p50', '-', 'p50'), ('p10', '--', 'p10'), ('p90', '--', 'p90')]:
            ax.plot(idx.index, idx[col], color=color,
                    linewidth=2 if col == 'p50' else 0.9, linestyle=ls, label=lbl)
            if sem is not None:
                sem_idx = sem[col] / base[col] * 100
                ax.fill_between(idx.index,
                                idx[col] - z * sem_idx,
                                idx[col] + z * sem_idx,
                                color=color, alpha=0.20)

        ax.axhline(100, color='grey', linewidth=0.6, linestyle=':')
        ax.set_ylabel("Index (year 0 = 100)")
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=30)
        if calib_v_line is not None:
            shade_out_of_sample(ax, calib_v_line[1])

    for j in range(i + 1, len(axes1)):
        axes1[j].set_visible(False)

    fig1.suptitle("Wage Percentiles by Model (p10 / p50 / p90, indexed)",
                  fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig1)
    add_shared_legend(fig1, used_colors,
                       extra_handles=[Line2D([0], [0], color='black', linewidth=2, label='p50'),
                                      Line2D([0], [0], color='black', linewidth=0.9, linestyle='--', label='p10 / p90'),
                                      mpatches.Patch(color='grey', alpha=0.10, label='p10–p90 spread')],
                       ncol=3)

    if save:
        save_plot_standard(fig1, f'{path}wage_percentile_levels.png',
                           aspect_ratio=_fig1_h/_fig1_w, width_frac=max(1.0, _fig1_w/TEXTWIDTH))
        plt.close(fig1)
    else:
        plt.show()

    # ── Figure 2: p90/p10 ratio ───────────────────────────────────────────────
    _fig2_w, _fig2_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig2, axes2 = plt.subplots(rows, cols, figsize=(_fig2_w, _fig2_h),
                               squeeze=False)
    axes2 = axes2.flatten()

    for i, (name, d) in enumerate(model_pcts.items()):
        ax    = axes2[i]
        color = _colors.get(name, color_cycle[i % len(color_cycle)])
        ratio_mean = d['ratio_mean']
        ratio_sem  = d['ratio_sem']

        ax.plot(ratio_mean.index, ratio_mean, color=color)
        if ratio_sem is not None:
            ax.fill_between(ratio_mean.index,
                            ratio_mean - z * ratio_sem,
                            ratio_mean + z * ratio_sem,
                            color=color, alpha=0.20)

        ax.set_ylabel("p90 / p10")
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=30)
        if calib_v_line is not None:
            shade_out_of_sample(ax, calib_v_line[1])

    for j in range(i + 1, len(axes2)):
        axes2[j].set_visible(False)

    fig2.suptitle("Wage Inequality by Model (p90/p10 Ratio)",
                  fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig2)
    add_shared_legend(fig2, used_colors, ncol=3)

    if save:
        save_plot_standard(fig2, f'{path}wage_percentile_ratio.png',
                           aspect_ratio=_fig2_h/_fig2_w, width_frac=max(1.0, _fig2_w/TEXTWIDTH))
        plt.close(fig2)
    else:
        plt.show()


def plot_wage_age_corr(res_dict, save=False, path=None):
    """
    Grid of subplots (one per model) showing the relationship between mean occupation
    wage and mean occupation worker age. Each point = one occupation in one year;
    colour encodes year.

    Args:
        res_dict (dict): {model_name: wage_age_df} where wage_age_df has columns
                         ['Time Step', 'Year', 'Occupation', 'Wage', 'Age']
                         (10th element of run_single_local return tuple).
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
    """
    n = len(res_dict)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    all_years = []
    for wage_age_df in res_dict.values():
        all_years.extend(wage_age_df['Year'].unique().tolist())
    year_min, year_max = min(all_years), max(all_years)
    norm = plt.Normalize(vmin=year_min, vmax=year_max)
    cmap = plt.cm.viridis

    for i, (name, wage_age_df) in enumerate(res_dict.items()):
        ax = axes[i]
        occ_year = (wage_age_df
            .groupby(['Year', 'Occupation'])
            .agg(Mean_Wage=('Wage', 'mean'), Mean_Age=('Age', 'mean'))
            .reset_index())

        sc = ax.scatter(occ_year['Mean_Age'], occ_year['Mean_Wage'],
                        c=occ_year['Year'], cmap=cmap, norm=norm,
                        alpha=0.6, s=20)

        if len(occ_year) > 1:
            r, _ = stats.pearsonr(occ_year['Mean_Age'], occ_year['Mean_Wage'])
            ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
                    verticalalignment='top', fontsize=10)

        ax.set_title(name, fontweight='bold')
        ax.set_xlabel("Mean Occupation Age")
        ax.set_ylabel("Mean Occupation Wage ($)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Single shared colourbar for year
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:i + 1].tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Simulation Year", fontsize=10)

    fig.suptitle("Wage–Age Correlation by Occupation", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    if save:
        plt.savefig(f'{path}wage_age_corr.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_wage_age_corr_over_time(res_dict, save=False, path=None, colors=None, calib_date=None, calib_v_line=None):
    """
    For each model, plots two Pearson r series over simulation years:
      - Cross-sectional r: correlation between occupation mean wage and mean age
        (across occupations), asking whether high-wage occupations have older workers.
      - Within-occupation r (median): median across occupations of the per-occupation
        Pearson r(worker wage, worker age), asking whether older workers within an
        occupation earn more.

    Args:
        res_dict (dict): {model_name: wage_age_df} with columns
                         ['Time Step', 'Year', 'Occupation', 'Wage', 'Age']
                         (10th element of run_single_local return tuple).
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
        colors (dict): Optional model name -> colour.
    """
    _sample = next(iter(res_dict.values()), None)
    if _sample is None or _sample.empty or 'Wage' not in _sample.columns:
        return
    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), default_colors))

    n = len(res_dict)
    cols = min(n, 2)
    rows = math.ceil(n / cols)

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig, axes = plt.subplots(rows, cols, figsize=(_fig_w, _fig_h), squeeze=False)
    axes = axes.flatten()

    # 'Year' is a simulation-relative index (0 = calib_date[0]'s calendar
    # year), not a calendar year itself -- mirrors plot_wage_percentile_ts's
    # start_year convention, so the calibration boundary is converted to the
    # same simulation-relative index before shading.
    calib_end_year = None
    if calib_date is not None and calib_v_line is not None:
        calib_end_year = pd.to_datetime(calib_v_line[1]).year - pd.to_datetime(calib_date[0]).year

    for i, (name, wage_age_df) in enumerate(res_dict.items()):
        ax = axes[i]
        years = sorted(wage_age_df['Year'].unique())
        cross_r = []
        within_r_median = []

        for yr in years:
            yr_df = wage_age_df[wage_age_df['Year'] == yr]

            # Cross-sectional: one point per occupation (mean wage vs mean age)
            occ_stats = (yr_df.groupby('Occupation')
                         .agg(Mean_Wage=('Wage', 'mean'), Mean_Age=('Age', 'mean'))
                         .reset_index())
            if len(occ_stats) > 2:
                r_cross, _ = stats.pearsonr(occ_stats['Mean_Age'], occ_stats['Mean_Wage'])
            else:
                r_cross = np.nan
            cross_r.append(r_cross)

            # Within-occupation: r(wage, age) per occupation, then median
            occ_r_vals = []
            for _, occ_df in yr_df.groupby('Occupation'):
                if len(occ_df) > 2:
                    r_w, _ = stats.pearsonr(occ_df['Age'], occ_df['Wage'])
                    occ_r_vals.append(r_w)
            within_r_median.append(np.median(occ_r_vals) if occ_r_vals else np.nan)

        ax.plot(years, cross_r, color=colors[name], label='Cross-sectional r')
        ax.plot(years, within_r_median, color=colors[name], linestyle='dashed',
                label='Within-occ r (median)')
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='dotted')
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Simulation Year")
        ax.set_ylabel("Pearson r")
        if calib_end_year is not None:
            shade_out_of_sample_year(ax, calib_end_year)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Wage–Age Correlation Over Simulation Years", fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='black', label='Cross-sectional r (across occupations)'),
                                      Line2D([0], [0], color='black', linestyle='dashed', label='Within-occupation r (median)')],
                       ncol=3)

    if save:
        save_plot_standard(fig, f'{path}wage_age_corr_over_time.png',
                           aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_wage_age_growth_scatter(res_dict, save=False, path=None, colors=None):
    """
    For each model, scatter plot of the change in occupation mean workforce age
    (x-axis) against the change in occupation mean wage (y-axis) between the
    first and last simulation year. Each point is one occupation. A positive
    slope means that occupations whose workforces aged more also saw greater
    wage growth. Annotated with Pearson r and p-value.


    Args:
        res_dict (dict): {model_name: wage_age_df} with columns
                         ['Time Step', 'Year', 'Occupation', 'Wage', 'Age']
                         (10th element of run_single_local return tuple).
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
        colors (dict): Optional model name -> colour.
    """
    _sample = next(iter(res_dict.values()), None)
    if _sample is None or _sample.empty or 'Wage' not in _sample.columns:
        return
    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), default_colors))

    n = len(res_dict)
    cols = min(n, 2)
    rows = math.ceil(n / cols)

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig, axes = plt.subplots(rows, cols, figsize=(_fig_w, _fig_h), squeeze=False)
    axes = axes.flatten()

    for i, (name, wage_age_df) in enumerate(res_dict.items()):
        ax = axes[i]
        years = sorted(wage_age_df['Year'].unique())
        yr0, yr_last = years[0], years[-1]

        occ_start = (wage_age_df[wage_age_df['Year'] == yr0]
                     .groupby('Occupation')
                     .agg(Age0=('Age', 'mean'), Wage0=('Wage', 'mean'))
                     .reset_index())
        occ_end = (wage_age_df[wage_age_df['Year'] == yr_last]
                   .groupby('Occupation')
                   .agg(AgeN=('Age', 'mean'), WageN=('Wage', 'mean'))
                   .reset_index())

        merged = occ_start.merge(occ_end, on='Occupation')
        merged['Delta_Age'] = merged['AgeN'] - merged['Age0']
        merged['Delta_Wage'] = merged['WageN'] - merged['Wage0']

        ax.scatter(merged['Delta_Age'], merged['Delta_Wage'],
                   color=colors[name], alpha=0.7, s=50, zorder=3)

        if len(merged) > 2:
            slope, intercept, r, p, _ = stats.linregress(
                merged['Delta_Age'], merged['Delta_Wage'])
            x_line = np.array([merged['Delta_Age'].min(), merged['Delta_Age'].max()])
            ax.plot(x_line, intercept + slope * x_line,
                    color='grey', linestyle='dashed', linewidth=1.2)
            p_str = f'{p:.3f}' if p >= 0.001 else '<0.001'
            ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p_str}',
                    transform=ax.transAxes, verticalalignment='top', fontsize=9)

        ax.axhline(0, color='lightgrey', linewidth=0.8, linestyle='dotted')
        ax.axvline(0, color='lightgrey', linewidth=0.8, linestyle='dotted')
        ax.set_xlabel(f"Δ Mean Workforce Age (yr {yr0} → yr {yr_last})")
        ax.set_ylabel(f"Δ Mean Occupation Wage ($)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Do Aging Occupation Workforces See Wage Growth?",
                 fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors, ncol=3)

    if save:
        save_plot_standard(fig, f'{path}wage_age_growth_scatter.png',
                           aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_age_dist_by_year(res_dict, save=False, path=None, colors=None):
    """
    For each model, shows the economy-wide age distribution of employed workers
    per simulation year as box plots.

    Args:
        res_dict (dict): {model_name: wage_age_df} with columns
                         ['Time Step', 'Year', 'Occupation', 'Wage', 'Age']
                         (10th element of run_single_local return tuple).
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
        colors (dict): Optional model name -> colour (used for the box fills).
    """
    _sample = next(iter(res_dict.values()), None)
    if _sample is None or _sample.empty or 'Age' not in _sample.columns:
        return
    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), default_colors))

    n = len(res_dict)
    cols = min(n, 2)
    rows = math.ceil(n / cols)

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig, axes = plt.subplots(rows, cols, figsize=(_fig_w, _fig_h), squeeze=False)
    axes = axes.flatten()

    for i, (name, wage_age_df) in enumerate(res_dict.items()):
        ax = axes[i]
        years = sorted(wage_age_df['Year'].unique())
        data_by_year = [wage_age_df.loc[wage_age_df['Year'] == y, 'Age'].values for y in years]
        bp = ax.boxplot(data_by_year, labels=years, sym='', patch_artist=True)
        for box in bp['boxes']:
            box.set_facecolor(colors[name])
            box.set_alpha(0.5)
        for median in bp['medians']:
            median.set_color(colors[name])
        ax.set_xlabel("Simulation Year")
        ax.set_ylabel("Worker Age")
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Economy-Wide Age Distribution by Year", fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors, ncol=3)

    if save:
        save_plot_standard(fig, f'{path}age_dist_by_year.png',
                           aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


def plot_occ_wage_growth(res_dict, save=False, path=None):
    """
    For each model, a heatmap of occupation-level year-over-year wage growth rates
    (%). Occupations are ranked top-to-bottom by their mean growth rate, so rising
    occupations appear at the top and declining/stable ones at the bottom.
    Diverging colormap: green = growing, red = declining, white = stable.

    Args:
        res_dict (dict): {model_name: wage_age_df} with columns
                         ['Time Step', 'Year', 'Occupation', 'Wage', 'Age'].
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
    """
    _sample = next(iter(res_dict.values()), None)
    if _sample is None or _sample.empty or 'Wage' not in _sample.columns:
        return
    n = len(res_dict)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    # Determine height from first model's occupation count
    n_occs = next(iter(res_dict.values()))['Occupation'].nunique()
    row_h = max(4, 0.22 * n_occs)

    _fig_w, _fig_h = 7 * cols, row_h * rows
    fig, axes = plt.subplots(rows, cols,
                             figsize=(_fig_w, _fig_h),
                             squeeze=False)
    axes = axes.flatten()

    for i, (name, wage_age_df) in enumerate(res_dict.items()):
        ax = axes[i]

        occ_year = (wage_age_df
            .groupby(['Year', 'Occupation'])
            .agg(Mean_Wage=('Wage', 'mean'))
            .reset_index()
            .sort_values(['Occupation', 'Year']))

        # Year-over-year % change
        occ_year['Wage_Growth'] = (occ_year
            .groupby('Occupation')['Mean_Wage']
            .pct_change() * 100)

        pivot = (occ_year
            .dropna(subset=['Wage_Growth'])
            .pivot(index='Occupation', columns='Year', values='Wage_Growth'))

        # Sort occupations by mean growth rate: fastest growers at top
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        # Cap colour scale at the 95th percentile of absolute values (max 5%)
        # to prevent outlier years from washing out the signal
        raw_vals = pivot.values[~np.isnan(pivot.values)]
        vmax = min(np.percentile(np.abs(raw_vals), 95), 5)

        sns.heatmap(pivot, ax=ax,
                    cmap='RdYlGn', center=0, vmin=-vmax, vmax=vmax,
                    cbar_kws={'label': 'YoY Wage Growth (%)', 'shrink': 0.7},
                    linewidths=0, yticklabels=True, xticklabels=True)

        ax.set_title(name, fontweight='bold')
        ax.set_xlabel("Simulation Year")
        ax.set_ylabel("Occupation (ranked by mean growth)")
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Occupation Wage Growth Rates (Year-on-Year %)",
                 fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    add_suptitle_spacing(fig)

    if save:
        save_plot_standard(fig, f'{path}occ_wage_growth_heatmap.png',
                           aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH),
                           apply_grid=False)
        plt.close()
    else:
        plt.show()


def plot_wage_vs_age_growth_rates(res_dict, save=False, path=None, colors=None):
    """
    For each model, scatter plot of mean annual workforce age growth (x) against
    mean annual wage growth rate (y), one point per occupation. Shows whether
    occupations that age faster also see faster wage growth.
    Annotated with Pearson r and p-value.

    Args:
        res_dict (dict): {model_name: wage_age_df} with columns
                         ['Time Step', 'Year', 'Occupation', 'Wage', 'Age'].
        save (bool): Whether to save the figure.
        path (str): Directory to save to (requires save=True).
        colors (dict): Optional model name -> colour.
    """
    _sample = next(iter(res_dict.values()), None)
    if _sample is None or _sample.empty or 'Wage' not in _sample.columns:
        return
    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), default_colors))

    n = len(res_dict)
    cols = min(n, 2)
    rows = math.ceil(n / cols)

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * rows
    fig, axes = plt.subplots(rows, cols, figsize=(_fig_w, _fig_h), squeeze=False)
    axes = axes.flatten()

    for i, (name, wage_age_df) in enumerate(res_dict.items()):
        ax = axes[i]

        occ_year = (wage_age_df
            .groupby(['Year', 'Occupation'])
            .agg(Mean_Wage=('Wage', 'mean'), Mean_Age=('Age', 'mean'))
            .reset_index()
            .sort_values(['Occupation', 'Year']))

        # Year-over-year changes
        occ_year['Wage_Growth_pct'] = (occ_year
            .groupby('Occupation')['Mean_Wage']
            .pct_change() * 100)
        occ_year['Age_Growth'] = (occ_year
            .groupby('Occupation')['Mean_Age']
            .diff())

        # Mean annual rate per occupation
        occ_rates = (occ_year
            .dropna(subset=['Wage_Growth_pct', 'Age_Growth'])
            .groupby('Occupation')
            .agg(Mean_Wage_Growth=('Wage_Growth_pct', 'mean'),
                 Mean_Age_Growth=('Age_Growth', 'mean'))
            .reset_index())

        ax.scatter(occ_rates['Mean_Age_Growth'], occ_rates['Mean_Wage_Growth'],
                   color=colors[name], alpha=0.7, s=50, zorder=3)

        if len(occ_rates) > 2:
            slope, intercept, r, p, _ = stats.linregress(
                occ_rates['Mean_Age_Growth'], occ_rates['Mean_Wage_Growth'])
            x_line = np.array([occ_rates['Mean_Age_Growth'].min(),
                                occ_rates['Mean_Age_Growth'].max()])
            ax.plot(x_line, intercept + slope * x_line,
                    color='grey', linestyle='dashed', linewidth=1.2)
            p_str = f'{p:.3f}' if p >= 0.001 else '<0.001'
            ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p_str}',
                    transform=ax.transAxes, verticalalignment='top', fontsize=9)

        ax.axhline(0, color='lightgrey', linewidth=0.8, linestyle='dotted')
        ax.axvline(0, color='lightgrey', linewidth=0.8, linestyle='dotted')
        ax.set_xlabel("Mean Annual Δ Workforce Age (years/yr)")
        ax.set_ylabel("Mean Annual Wage Growth (%/yr)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Occupation Wage Growth vs Age Growth Rates",
                 fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors, ncol=3)

    if save:
        save_plot_standard(fig, f'{path}wage_vs_age_growth_rates.png',
                           aspect_ratio=_fig_h/_fig_w, width_frac=max(1.0, _fig_w/TEXTWIDTH))
        plt.close()
    else:
        plt.show()


####################################################
############## GENDER PAY GAP TIME SERIES ##########
####################################################
def plot_gender_pay_gap_ts(mod_results_dict, save=False, path=None,
                           colors=None, freq='M', lowess=False, lowess_frac=0.2):
    """
    Plots the mean and median gender pay gap as time series for each model run.
    Gap = 1 - (female wage / male wage), expressed as %.
    Solid lines = mean-based gap; dashed lines = median-based gap.

    Parameters
    ----------
    mod_results_dict : dict
        Keys are model names; values are grouped DataFrames with a 'DATE' column
        and 'Gender_Pay_Gap' (and optionally 'Gender_Median_Pay_Gap') columns.
    save : bool
    path : str or None
    colors : dict or None
        Maps model name -> colour string.
    freq : str
        Resample frequency (e.g. 'M', '3M', 'Y'). Use 'M' to skip resampling.
    lowess : bool
        Apply LOWESS smoothing.
    lowess_frac : float
        Fraction of data used for LOWESS smoothing.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_smooth

    if colors is None:
        default_colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(mod_results_dict.keys(), default_colors))

    fig, ax = plt.subplots(figsize=(12, 5))

    for name, res in mod_results_dict.items():
        if 'Gender_Pay_Gap' not in res.columns:
            print(f"Warning: 'Gender_Pay_Gap' not found for '{name}'. Re-run the model.")
            continue

        data = res.set_index('DATE').resample(freq).mean() if freq != 'M' else res.copy()
        dates = np.array(data.index if freq != 'M' else data['DATE'])
        mean_gap = data['Gender_Pay_Gap'] * 100
        has_median = 'Gender_Median_Pay_Gap' in data.columns
        if has_median:
            median_gap = data['Gender_Median_Pay_Gap'] * 100

        if lowess:
            sm = lowess_smooth(mean_gap.values, np.arange(len(mean_gap)), frac=lowess_frac)
            ax.plot(dates, mean_gap.values, color=colors[name], alpha=0.15, linewidth=0.8)
            ax.plot(dates, sm[:, 1], color=colors[name], label=f"{name} (mean)", linewidth=2)
            if has_median:
                sm2 = lowess_smooth(median_gap.values, np.arange(len(median_gap)), frac=lowess_frac)
                ax.plot(dates, median_gap.values, color=colors[name], alpha=0.15, linewidth=0.8, linestyle='--')
                ax.plot(dates, sm2[:, 1], color=colors[name], label=f"{name} (median)", linewidth=2, linestyle='--')
        else:
            ax.plot(dates, mean_gap.values, color=colors[name], label=f"{name} (mean)", linewidth=2)
            if has_median:
                ax.plot(dates, median_gap.values, color=colors[name], label=f"{name} (median)", linewidth=2, linestyle='--')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel("Gender Pay Gap (%)\n(% female wage below male wage)")
    ax.set_xlabel("Date")
    ax.set_title("Gender Pay Gap Over Time (solid = mean, dashed = median)")
    ax.legend()
    fig.tight_layout()

    if save:
        plt.savefig(f'{path}gender_pay_gap_ts.jpg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


####################################################
########## TIME-TO-REEMPLOYMENT ####################
####################################################

def plot_time_to_reemployment_ribbon(
        tte_dict,
        calib_date,
        delay=25,
        phase_boundaries=None,
        max_duration=48,
        colors=None,
        log_x=True,
        empirical_economy=None,
        empirical_occ=None,
        save=False,
        path=None):
    """
    Survival-curve ribbon plot for time-to-reemployment.

    Facets = business cycle phases; one line per model; ribbon = IQR across occupations.

    Parameters
    ----------
    tte_dict : dict
        Keys are model names; values are DataFrames with columns
        ['Time Step', 'OriginOccupation', 'DestinationOccupation', 'UEDuration'].
    calib_date : list[str]
        [start_date, end_date] passed to the model (e.g. ["2001-01-01", "2024-05-01"]).
    delay : int
        Burn-in period in time steps (first valid time step).
    phase_boundaries : dict, optional
        Maps phase label -> (ts_start, ts_end) inclusive. Defaults to GFC phases.
    max_duration : int
        Maximum UE duration (months) shown on x-axis.
    colors : dict, optional
        Maps model name -> colour string. Auto-assigned if None.
    log_x : bool
        If True, use log scale on x-axis with readable tick labels.
    empirical_economy : pd.DataFrame, optional
        Economy-wide CPS survival curve: columns ['PHASE', 'months', 'survival'].
        Overlaid as a dashed black line on each panel.
    empirical_occ : pd.DataFrame, optional
        Occupation-level CPS survival: columns ['PHASE', <occ_col>, 'months', 'survival'].
        IQR across occupations overlaid as a grey ribbon.
    save : bool
    path : str
    """
    import warnings

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                      'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(tte_dict.keys(), default_colors))

    if phase_boundaries is None:
        phase_boundaries = {
            'Pre-GFC (2001\u20132007)':     (25,  108),
            'GFC recession (2008\u201309)': (109, 132),
            'Post-GFC (2010\u20132019)':    (133, 244),
        }

    # Normalise phase labels for empirical data matching (strip em-dashes / en-dashes)
    def _norm(s):
        return s.replace('\u2013', '-').replace('\u2014', '-')

    durations = np.arange(1, max_duration + 1)

    def survival_curve(spells):
        n = len(spells)
        if n == 0:
            return np.zeros(len(durations))
        return np.array([(spells > d).sum() / n for d in durations])

    def occ_survival_matrix(df_phase):
        occs = df_phase['OriginOccupation'].unique()
        curves = []
        for occ in occs:
            spells = df_phase.loc[df_phase['OriginOccupation'] == occ, 'UEDuration'].values
            curves.append(survival_curve(spells))
        return np.array(curves)  # shape (n_occs, n_durations)

    # Pre-process empirical occupation data into IQR ribbons per phase
    emp_occ_ribbons = {}
    if empirical_occ is not None:
        occ_col = [c for c in empirical_occ.columns
                   if c not in ('PHASE', 'months', 'survival')][0]
        for phase_label in phase_boundaries:
            ph_norm = _norm(phase_label)
            emp_ph = empirical_occ[
                empirical_occ['PHASE'].apply(_norm) == ph_norm
            ]
            if emp_ph.empty:
                continue
            # Pivot to (n_occs x n_months) and compute IQR
            pivot = emp_ph.pivot(index=occ_col, columns='months', values='survival')
            pivot = pivot.reindex(columns=range(1, max_duration + 1))
            emp_occ_ribbons[phase_label] = {
                'q25': pivot.quantile(0.25).values,
                'q75': pivot.quantile(0.75).values,
                'months': np.array(pivot.columns),
            }

    n_phases = len(phase_boundaries)
    fig, axes = plt.subplots(1, n_phases,
                             figsize=(5 * n_phases, 5),
                             sharey=True)
    if n_phases == 1:
        axes = [axes]

    for ax, (phase_label, (ts_start, ts_end)) in zip(axes, phase_boundaries.items()):

        # Empirical occupation IQR ribbon (grey, behind model lines)
        if phase_label in emp_occ_ribbons:
            r = emp_occ_ribbons[phase_label]
            mask = r['months'] <= max_duration
            ax.fill_between(r['months'][mask], r['q25'][mask], r['q75'][mask],
                            color='black', alpha=0.08, label='CPS occ. IQR')

        # Model lines + ribbons
        for model_name, df in tte_dict.items():
            df_valid = df[(df['Time Step'] >= delay) & (df['UEDuration'] > 0)]
            ph_df = df_valid[(df_valid['Time Step'] >= ts_start) &
                             (df_valid['Time Step'] <= ts_end)]

            if ph_df.empty:
                warnings.warn(f"No data for {model_name} / {phase_label}")
                continue

            occ_curves  = occ_survival_matrix(ph_df)
            median_surv = np.median(occ_curves, axis=0)
            q25_surv    = np.percentile(occ_curves, 25, axis=0)
            q75_surv    = np.percentile(occ_curves, 75, axis=0)

            col = colors.get(model_name, 'grey')
            ax.fill_between(durations, q25_surv, q75_surv, color=col, alpha=0.15)
            ax.plot(durations, median_surv, color=col, linewidth=2, label=model_name)

        # Economy-wide empirical line (dashed black, on top)
        if empirical_economy is not None:
            ph_norm = _norm(phase_label)
            emp_ph = empirical_economy[
                empirical_economy['PHASE'].apply(_norm) == ph_norm
            ]
            emp_ph = emp_ph[emp_ph['months'] <= max_duration]
            if not emp_ph.empty:
                ax.plot(emp_ph['months'], emp_ph['survival'],
                        color='black', linewidth=1.8, linestyle='--',
                        label='CPS (economy-wide)')

        ax.set_title(phase_label, fontsize=10, fontweight='bold')
        ax.set_xlabel("Months unemployed")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='grey', linewidth=0.7, linestyle=':', alpha=0.6)

        if log_x:
            ax.set_xscale('log')
            ticks = [t for t in [1, 2, 3, 6, 12, 18, 24, 36, 48] if t <= max_duration]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.set_xlim(1, max_duration)
        else:
            ax.set_xlim(1, max_duration)

    axes[0].set_ylabel("Fraction still unemployed\n(median across occupations)")
    axes[-1].legend(fontsize=8, framealpha=0.7, loc='upper right',
                    bbox_to_anchor=(1, 1))

    fig.suptitle("Time-to-Reemployment: Survival Curves by Business Cycle Phase\n"
                 "(ribbon = IQR across occupations)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        plt.savefig(f'{path}time_to_reemployment_ribbon.jpg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_tte_by_soc_major(
        tte_dict,
        occ_crosswalk,
        soc_labs,
        empirical_occ=None,
        delay=25,
        phase_boundaries=None,
        max_duration=24,
        colors=None,
        log_x=True,
        save=False,
        path=None):
    """
    One figure per business cycle phase; 4×6 grid of SOC major group panels.
    Lines = models (coloured); dashed black = CPS empirical benchmark.

    Parameters
    ----------
    tte_dict : dict
        Keys are model names; values are DataFrames with columns
        ['Time Step', 'OriginOccupation', 'UEDuration'].
    occ_crosswalk : pd.DataFrame
        Must contain 'acs_occ_code' (int) and 'SOC_major_adj' (int).
        Used to map OriginOccupation -> SOC major group.
    soc_labs : pd.DataFrame
        Must contain 'soc_code' (int/str) and 'label' (str).
    empirical_occ : pd.DataFrame, optional
        CPS survival by SOC major: columns ['PHASE', 'SOC_major', 'months', 'survival'].
    delay, phase_boundaries, max_duration, colors, log_x, save, path : as in
        plot_time_to_reemployment_ribbon.
    """
    import warnings
    import math

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                      'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(tte_dict.keys(), default_colors))

    if phase_boundaries is None:
        phase_boundaries = {
            'Pre-GFC (2001\u20132007)':     (25,  108),
            'GFC recession (2008\u201309)': (109, 132),
            'Post-GFC (2010\u20132019)':    (133, 244),
        }

    def _norm(s):
        return s.replace('\u2013', '-').replace('\u2014', '-')

    durations = np.arange(1, max_duration + 1)

    def survival_curve(spells):
        n = len(spells)
        if n == 0:
            return np.full(len(durations), np.nan)
        return np.array([(spells > d).sum() / n for d in durations])

    def _set_xaxis(ax):
        if log_x:
            ax.set_xscale('log')
            ticks = [t for t in [1, 2, 3, 6, 12, 18, 24, 36, 48] if t <= max_duration]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], fontsize=6)
            ax.set_xlim(1, max_duration)
        else:
            ax.set_xlim(1, max_duration)

    # Build crosswalk: occupation_id (0-indexed) -> SOC_major_adj
    # occ_crosswalk['X'] is 1-indexed row number; occupation_id = X - 1
    cw = occ_crosswalk[['X', 'SOC_major_adj']].drop_duplicates()
    cw = cw.dropna(subset=['SOC_major_adj'])
    cw['occupation_id'] = cw['X'].astype(int) - 1
    cw['SOC_major_adj'] = cw['SOC_major_adj'].astype(int)

    # SOC major labels
    labs = soc_labs.copy()
    labs['soc_code'] = labs['soc_code'].astype(int)
    labs['label']    = labs['label'].str.strip()
    soc_label_map = dict(zip(labs['soc_code'], labs['label']))

    soc_groups = sorted(cw['SOC_major_adj'].unique())
    n_groups = len(soc_groups)
    ncols = 6
    nrows = math.ceil(n_groups / ncols)

    # Attach SOC major to every tte_dict entry once
    tagged = {}
    for model_name, df in tte_dict.items():
        df_v = df[(df['Time Step'] >= delay) & (df['UEDuration'] > 0)].copy()
        df_v['OriginOccupation'] = pd.to_numeric(df_v['OriginOccupation'], errors='coerce')
        df_v = df_v.dropna(subset=['OriginOccupation'])
        df_v['OriginOccupation'] = df_v['OriginOccupation'].astype(int)
        df_v = df_v.merge(cw[['occupation_id', 'SOC_major_adj']],
                          left_on='OriginOccupation', right_on='occupation_id', how='left')
        tagged[model_name] = df_v
        print(tagged[model_name])

    print(df_v['SOC_major_adj'].value_counts())
    phase_bg = {
        'Pre-GFC (2001\u20132007)':     '#ddeeff',
        'GFC recession (2008\u201309)': '#fde8e8',
        'Post-GFC (2010\u20132019)':    '#e8f5e9',
    }

    # One figure per phase
    for phase_label, (ts_start, ts_end) in phase_boundaries.items():
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3, nrows * 2.8),
                                 sharey=True)
        axes_flat = axes.flatten()

        for idx, soc in enumerate(soc_groups):
            ax = axes_flat[idx]
            soc_name = soc_label_map.get(soc, str(soc))

            # Model lines
            for model_name, df_v in tagged.items():
                ph_df = df_v[(df_v['Time Step'] >= ts_start) &
                             (df_v['Time Step'] <= ts_end) &
                             (df_v['SOC_major_adj'] == soc)]
                if ph_df.empty:
                    continue
                curve = survival_curve(ph_df['UEDuration'].values)
                ax.plot(durations, curve,
                        color=colors.get(model_name, 'grey'),
                        linewidth=1.4, label=model_name)

            # CPS empirical line
            if empirical_occ is not None:
                ph_norm = _norm(phase_label)
                emp_soc = pd.to_numeric(empirical_occ['SOC_major'], errors='coerce')
                emp_ph = empirical_occ[
                    (empirical_occ['PHASE'].apply(_norm) == ph_norm) &
                    (emp_soc == soc) &
                    (empirical_occ['months'] <= max_duration)
                ]
                if not emp_ph.empty:
                    ax.plot(emp_ph['months'], emp_ph['survival'],
                            color='black', linewidth=1.4, linestyle='--',
                            label='CPS')

            ax.set_facecolor(phase_bg.get(phase_label, '#f5f5f5'))
            ax.set_title(soc_name, fontsize=7, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color='grey', linewidth=0.5, linestyle=':', alpha=0.6)
            ax.tick_params(axis='y', labelsize=6)
            _set_xaxis(ax)

        # Hide unused panels
        for idx in range(n_groups, len(axes_flat)):
            axes_flat[idx].axis('off')

        # Shared axis labels
        for ax in axes[:, 0]:
            ax.set_ylabel("S(t)", fontsize=7)
        for ax in axes[-1, :]:
            ax.set_xlabel("Months", fontsize=7)

        # Single legend from first non-empty panel
        handles, labels_leg = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc='lower center',
                   ncol=len(tte_dict) + 1, fontsize=8, framealpha=0.8,
                   bbox_to_anchor=(0.5, 0))

        fig.suptitle(f"Time-to-Reemployment by SOC Major Group — {phase_label}",
                     fontsize=11, fontweight='bold')
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])

        if save:
            fname = f'{path}tte_soc_major_{_norm(phase_label).replace(" ", "_")}.jpg'
            plt.savefig(fname, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


####################################################
########## HOTSPOT QUADRANT ########################
####################################################

def plot_hotspot_quadrant(
        sim_results_dict,
        occ_crosswalk,
        soc_labs,
        metric='UER',
        deviation=False,
        benchmark=None,
        phase_boundaries=None,
        delay=25,
        topk=15,
        save=False,
        path=None):
    """
    Quadrant scatter: structural stress (x) vs cyclical stress (y) per occupation.
    One panel per non-benchmark model.

    Parameters
    ----------
    sim_results_dict : dict
        Keys = model names; values = occupation-level DataFrames.
    occ_crosswalk : pd.DataFrame
        Must contain 'X', 'SOC_major_adj', 'label'.
    soc_labs : pd.DataFrame
        Must contain 'soc_code' and 'label' for SOC major group names.
    metric : str
        'UER' or 'LTUER'.
    deviation : bool
        If True, subtract cross-occupation mean so the quadrant centres at (0,0).
    benchmark : str, optional
        Key in sim_results_dict to use as the reference model. When set, axes
        show (model score − benchmark score) so differences are directly visible.
        The benchmark model itself is not plotted as its own panel.
    phase_boundaries : dict, optional
        Maps label -> (ts_start, ts_end). Defaults to GFC phases.
    delay : int
        Burn-in time steps to exclude.
    topk : int
        Number of extreme occupations to annotate.
    save : bool
    path : str
    """
    if phase_boundaries is None:
        phase_boundaries = {
            'Pre-GFC (2001\u20132007)':     (25,  108),
            'GFC recession (2008\u201309)': (109, 132),
            'Post-GFC (2010\u20132019)':    (133, 244),
        }

    recession_label = 'GFC recession (2008\u201309)'
    rec_start, rec_end = phase_boundaries[recession_label]
    exp_ts = set()
    for lbl, (ts0, ts1) in phase_boundaries.items():
        if lbl != recession_label:
            exp_ts.update(range(ts0, ts1 + 1))

    # Crosswalk: occupation_id (0-indexed, = X-1) -> SOC_major_adj + detailed label
    cw = occ_crosswalk[['X', 'SOC_major_adj', 'label']].drop_duplicates('X')
    cw = cw.dropna(subset=['SOC_major_adj'])
    cw['occupation_id'] = pd.to_numeric(cw['X'], errors='coerce').astype('Int64') - 1
    cw['SOC_major_adj'] = pd.to_numeric(cw['SOC_major_adj'], errors='coerce')
    cw = cw.dropna(subset=['occupation_id', 'SOC_major_adj'])

    labs = soc_labs.copy()
    labs['soc_code'] = pd.to_numeric(labs['soc_code'], errors='coerce')
    labs['label']    = labs['label'].str.strip()
    soc_label_map = dict(zip(labs['soc_code'], labs['label']))

    soc_groups = sorted(cw['SOC_major_adj'].unique())
    cmap = plt.get_cmap('tab20', len(soc_groups))
    soc_color = {soc: cmap(i) for i, soc in enumerate(soc_groups)}

    def _compute_scores(df):
        df_v = df[df['Time Step'] >= delay].copy()
        df_v['Occupation'] = pd.to_numeric(df_v['Occupation'], errors='coerce')
        if metric not in df_v.columns:
            if metric == 'UER':
                df_v['UER'] = df_v['Unemployment'] / df_v['Workers']
            elif metric == 'LTUER':
                df_v['LTUER'] = df_v['LT Unemployed Persons'] / df_v['Unemployment']
        exp_df = df_v[df_v['Time Step'].isin(exp_ts)]
        rec_df = df_v[(df_v['Time Step'] >= rec_start) &
                      (df_v['Time Step'] <= rec_end)]
        struct = exp_df.groupby('Occupation')[metric].mean().rename('structural')
        cyclic = (rec_df.groupby('Occupation')[metric].mean() - struct).rename('cyclical')
        return pd.concat([struct, cyclic], axis=1).dropna()

    # Pre-compute benchmark scores if requested
    bench_scores = None
    if benchmark is not None and benchmark in sim_results_dict:
        bench_scores = _compute_scores(sim_results_dict[benchmark])

    plot_models = {k: v for k, v in sim_results_dict.items()
                   if k != benchmark} if benchmark else sim_results_dict

    n_models = len(plot_models)
    fig, axes = plt.subplots(1, n_models,
                             figsize=(5.5 * n_models, 5),
                             sharey=True, sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, df) in zip(axes, plot_models.items()):
        scores = _compute_scores(df)

        if bench_scores is not None:
            # Delta mode: subtract benchmark scores occupation-by-occupation
            scores = (scores - bench_scores).dropna()
            x_mid, y_mid = 0.0, 0.0
        else:
            if deviation:
                scores['structural'] -= scores['structural'].mean()
                scores['cyclical']   -= scores['cyclical'].mean()
            x_mid = 0.0 if deviation else scores['structural'].median()
            y_mid = 0.0 if deviation else scores['cyclical'].median()

        scores = scores.reset_index()
        scores = scores.merge(cw[['occupation_id', 'SOC_major_adj', 'label']],
                              left_on='Occupation', right_on='occupation_id', how='left')

        # Plot points coloured by SOC major
        for soc in soc_groups:
            sub = scores[scores['SOC_major_adj'] == soc]
            if sub.empty:
                continue
            ax.scatter(sub['structural'], sub['cyclical'],
                       color=soc_color[soc], alpha=0.6, s=20,
                       label=soc_label_map.get(soc, str(soc)))

        # Quadrant lines and reference point
        ax.axvline(x_mid, color='grey', linewidth=0.8, linestyle='--')
        ax.axhline(y_mid, color='grey', linewidth=0.8, linestyle='--')

        # Quadrant corner labels
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        pad = 0.02
        if bench_scores is not None:
            labels_q = [('More cyclical\nvs benchmark', xlim[0], ylim[1], 'top', 'left'),
                        ('More dual\nvs benchmark',     xlim[1], ylim[1], 'top', 'right'),
                        ('Less stressed\nvs benchmark', xlim[0], ylim[0], 'bottom', 'left'),
                        ('More structural\nvs benchmark', xlim[1], ylim[0], 'bottom', 'right')]
        else:
            labels_q = [('Cyclical\nhotspot', xlim[0], ylim[1], 'top', 'left'),
                        ('Dual\nhotspot',     xlim[1], ylim[1], 'top', 'right'),
                        ('Low\nstress',       xlim[0], ylim[0], 'bottom', 'left'),
                        ('Structural\nhotspot', xlim[1], ylim[0], 'bottom', 'right')]
        for txt, x, y, va, ha in labels_q:
            ax.text(x + pad*(xlim[1]-xlim[0]) * (1 if ha == 'left' else -1),
                    y + pad*(ylim[1]-ylim[0]) * (-1 if va == 'top' else 1),
                    txt, fontsize=6, va=va, ha=ha, color='dimgrey')

        # Annotate top-k most extreme occupations
        scores['dist'] = np.sqrt((scores['structural'] - x_mid)**2 +
                                 (scores['cyclical']   - y_mid)**2)
        top = scores.nlargest(topk, 'dist')
        for _, row in top.iterrows():
            ax.annotate(str(row['label'])[:22],
                        xy=(row['structural'], row['cyclical']),
                        fontsize=4.5, ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points')

        if bench_scores is not None:
            xlabel = f"\u0394 Structural ({metric}) vs {benchmark}"
            ylabel = f"\u0394 Cyclical ({metric}) vs {benchmark}"
        elif deviation:
            xlabel = f"Structural deviation ({metric})"
            ylabel = f"Cyclical deviation ({metric})"
        else:
            xlabel = f"Structural ({metric}, expansion mean)"
            ylabel = f"Cyclical ({metric}, recession \u2212 expansion)"

        ax.set_xlabel(xlabel, fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(model_name, fontsize=9, fontweight='bold')

    # Shared legend for SOC major groups
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=soc_color[s], markersize=6,
                          label=soc_label_map.get(s, str(s)))
               for s in soc_groups]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=6, framealpha=0.8,
               bbox_to_anchor=(0.5, 0), title='SOC Major Group')

    mode_str = f'vs {benchmark}' if benchmark else ('deviation' if deviation else 'raw')
    fig.suptitle(f"Occupation Hotspots: Structural vs Cyclical {metric} ({mode_str})",
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.15, 1, 0.96])

    if save:
        plt.savefig(f'{path}hotspot_quadrant_{metric}_{mode_str}.jpg',
                    bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


####################################################
########## SIMULATED VS OBSERVED OCC UER ###########
####################################################

def plot_occ_uer_fit(
        sim_results_dict,
        occ_crosswalk,
        soc_labs,
        metric='UER',
        obs_col=None,
        phase_boundaries=None,
        delay=25,
        topk=15,
        save=False,
        path=None):
    """
    Scatter of observed occupational UER (x) vs model bias (simulated − observed, y).
    One panel per model. Points coloured by SOC major group.

    Parameters
    ----------
    sim_results_dict : dict
        Keys = model names; values = occupation-level DataFrames with
        'Time Step', 'Occupation', and metric columns.
    occ_crosswalk : pd.DataFrame
        Must contain 'X', 'SOC_major_adj', 'label', and observed UER columns.
    soc_labs : pd.DataFrame
        'soc_code' and 'label' for SOC major group names.
    metric : str
        'UER' or 'LTUER'.
    obs_col : str, optional
        Column in occ_crosswalk holding the observed value. Defaults to
        'uer_occ' for UER and 'ltuer_occ' for LTUER.
    phase_boundaries, delay : as in plot_hotspot_quadrant.
    topk : int
        Number of largest-bias occupations to annotate.
    save, path : as elsewhere.
    """
    if phase_boundaries is None:
        phase_boundaries = {
            'Pre-GFC (2001\u20132007)':     (25,  108),
            'GFC recession (2008\u201309)': (109, 132),
            'Post-GFC (2010\u20132019)':    (133, 244),
        }

    if obs_col is None:
        obs_col = 'uer_occ' if metric == 'UER' else 'ltuer_occ'

    # Build crosswalk with observed values
    cw = occ_crosswalk[['X', 'SOC_major_adj', 'label', obs_col]].drop_duplicates('X').copy()
    cw['occupation_id'] = pd.to_numeric(cw['X'], errors='coerce').astype('Int64') - 1
    cw['SOC_major_adj'] = pd.to_numeric(cw['SOC_major_adj'], errors='coerce')
    cw[obs_col]         = pd.to_numeric(cw[obs_col], errors='coerce')
    cw = cw.dropna(subset=['occupation_id', 'SOC_major_adj', obs_col])

    labs = soc_labs.copy()
    labs['soc_code'] = pd.to_numeric(labs['soc_code'], errors='coerce')
    labs['label']    = labs['label'].str.strip()
    soc_label_map = dict(zip(labs['soc_code'], labs['label']))

    soc_groups = sorted(cw['SOC_major_adj'].unique())
    cmap = plt.get_cmap('tab20', len(soc_groups))
    soc_color = {soc: cmap(i) for i, soc in enumerate(soc_groups)}

    n_models = len(sim_results_dict)
    fig, axes = plt.subplots(1, n_models,
                             figsize=(5.5 * n_models, 5),
                             sharey=True, sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, df) in zip(axes, sim_results_dict.items()):
        df_valid = df[df['Time Step'] >= delay].copy()
        df_valid['Occupation'] = pd.to_numeric(df_valid['Occupation'], errors='coerce')

        if metric not in df_valid.columns:
            if metric == 'UER':
                df_valid['UER'] = df_valid['Unemployment'] / df_valid['Workers']
            elif metric == 'LTUER':
                df_valid['LTUER'] = df_valid['LT Unemployed Persons'] / df_valid['Unemployment']

        sim_mean = (df_valid.groupby('Occupation')[metric]
                    .mean()
                    .reset_index()
                    .rename(columns={metric: 'sim_mean'}))

        plot_df = sim_mean.merge(
            cw[['occupation_id', 'SOC_major_adj', 'label', obs_col]],
            left_on='Occupation', right_on='occupation_id', how='inner')
        plot_df['bias'] = plot_df['sim_mean'] - plot_df[obs_col]

        for soc in soc_groups:
            sub = plot_df[plot_df['SOC_major_adj'] == soc]
            if sub.empty:
                continue
            ax.scatter(sub[obs_col], sub['bias'],
                       color=soc_color[soc], alpha=0.6, s=20,
                       label=soc_label_map.get(soc, str(soc)))

        ax.axhline(0, color='black', linewidth=1.0, linestyle='--')

        # Annotate top-k by absolute bias
        top = plot_df.reindex(plot_df['bias'].abs().nlargest(topk).index)
        for _, row in top.iterrows():
            ax.annotate(str(row['label'])[:22],
                        xy=(row[obs_col], row['bias']),
                        fontsize=4.5, ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points')

        ax.set_xlabel(f"Observed {metric} (occupation mean)", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(f"Simulated \u2212 Observed {metric}", fontsize=8)
        ax.set_title(model_name, fontsize=9, fontweight='bold')

    # Reference line annotation
    axes[0].text(0.01, 0.01, 'Model underestimates \u2193',
                 transform=axes[0].transAxes, fontsize=6, va='bottom', color='dimgrey')
    axes[0].text(0.01, 0.99, 'Model overestimates \u2191',
                 transform=axes[0].transAxes, fontsize=6, va='top', color='dimgrey')

    # Shared SOC major legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=soc_color[s], markersize=6,
                          label=soc_label_map.get(s, str(s)))
               for s in soc_groups]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=6, framealpha=0.8,
               bbox_to_anchor=(0.5, 0), title='SOC Major Group')

    fig.suptitle(f"Simulated vs Observed Occupational {metric}: Model Bias",
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.15, 1, 0.96])

    if save:
        plt.savefig(f'{path}occ_{metric.lower()}_fit.jpg',
                    bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_accepted_wages(
        sim_results_dict,
        calib_date=None,
        delay=25,
        recessions=None,
        colors=None,
        lowess_frac=0.15,
        save=False,
        path=None):
    """
    Single-panel time series of U→E accepted wages over the business cycle.

    Plots LOWESS-smoothed mean of U_REL_WAGE_MEAN with a 95% CI band.  When
    the DataFrame carries U_REL_WAGE_MEAN_ci_lower / _ci_upper columns
    (produced by aggregate_simulation_results for multi-sim caches), those
    are used directly for the band.  One line per model.  Reference at
    1.0 = occupation mean wage.

    Parameters
    ----------
    sim_results_dict : dict  {model_name: DataFrame}
        Economy-wide output from aggregate_simulation_results (or a single
        sim).  Must contain 'DATE', 'U_REL_WAGE_MEAN', 'Time Step'.
        CI columns U_REL_WAGE_MEAN_ci_lower / _ci_upper are used when present.
    calib_date : list [start, end], optional
        ISO date strings used to clip the time axis.
    delay : int
        First valid time step (rows with Time Step < delay are excluded).
    recessions : pd.DataFrame, optional
        DataFrame with 'start' and 'end' date columns for recession shading.
    colors : dict, optional
        {model_name: colour} mapping.
    lowess_frac : float
        Fraction of data used for LOWESS smoothing (default 0.15).
    save, path : as elsewhere.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_smooth

    if colors is None:
        _palette = ["steelblue", "darkorange", "forestgreen", "crimson",
                    "purple", "brown", "teal", "grey"]
        colors = {k: c for k, c in zip(sim_results_dict.keys(), _palette)}

    _fig_w, _fig_h = TEXTWIDTH, TEXTWIDTH * DEFAULT_ASPECT_RATIO
    fig, ax = plt.subplots(1, 1, figsize=(_fig_w, _fig_h))

    for model_name, df in sim_results_dict.items():
        col = colors.get(model_name, 'black')
        sub = df[df['Time Step'] >= delay].copy()
        sub = sub.dropna(subset=['DATE', 'U_REL_WAGE_MEAN'])

        if calib_date is not None:
            sub = sub[(sub['DATE'] >= calib_date[0]) & (sub['DATE'] <= calib_date[1])]

        if sub.empty:
            continue

        sub = sub.sort_values('DATE')
        dates     = sub['DATE'].values
        mean_vals = sub['U_REL_WAGE_MEAN'].values

        smoothed = lowess_smooth(mean_vals, np.arange(len(mean_vals)),
                                 frac=lowess_frac)[:, 1]
        ax.plot(dates, smoothed, color=col, label=model_name,
                linewidth=1.8, zorder=3)

        if ('U_REL_WAGE_MEAN_ci_lower' in sub.columns
                and 'U_REL_WAGE_MEAN_ci_upper' in sub.columns):
            ci_lo = lowess_smooth(sub['U_REL_WAGE_MEAN_ci_lower'].values,
                                  np.arange(len(dates)), frac=lowess_frac)[:, 1]
            ci_hi = lowess_smooth(sub['U_REL_WAGE_MEAN_ci_upper'].values,
                                  np.arange(len(dates)), frac=lowess_frac)[:, 1]
            ax.fill_between(dates, ci_lo, ci_hi, color=col, alpha=0.12, zorder=2)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)

    if recessions is not None:
        for _, row in recessions.iterrows():
            ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)

    if calib_date is not None:
        ax.set_xlim(pd.Timestamp(calib_date[0]), pd.Timestamp(calib_date[1]))

    ax.set_xlabel('Date')
    ax.set_ylabel('Ratio of Accepted Wage to Pre-Separation Wage')

    plt.tight_layout()
    add_shared_legend(fig, colors, ncol=3)

    if save:
        save_plot_standard(fig, os.path.join(path, 'accepted_wages_ue.jpg'),
                            aspect_ratio=_fig_h / _fig_w)
        plt.close()
    else:
        plt.show()


def plot_wage_level_decomposition(
    sim_results_dict,
    calib_date=None,
    delay=25,
    recessions=None,
    colors=None,
    lowess_frac=0.15,
    save=False,
    path=None,
):
    """
    Decomposes U_REL_WAGE_MEAN into its numerator and denominator to diagnose
    why models differ in relative re-employment wages.

    U_REL_WAGE_MEAN = (absolute accepted wage at U→E) / (mean employment wage)

    Three-panel time series (LOWESS-smoothed):
      A — Absolute accepted wage at U→E:
              sum_occ(U_Rel_Wage × occ_mean_wage) / sum_occ(UE_Transitions)
          where occ_mean_wage = Total_Wages / Employment per occupation per step.
      B — Mean employment wage (denominator):
              sum_occ(Total_Wages) / sum_occ(Employment)
      C — A / B ≈ U_REL_WAGE_MEAN (sanity-check; should match plot_accepted_wages)

    If the red model has a low ratio (C) due to a higher denominator (B) rather
    than a lower numerator (A), that points to a wage-selection composition
    effect from reservation wages rather than genuine wage scarring.

    Parameters
    ----------
    sim_results_dict : dict {model_name: DataFrame}
        Occupation-level output (filtered_sim_results).
        Must contain: Time Step, Occupation, U_Rel_Wage, UE_Transitions,
                      Total_Wages, Employment.
    calib_date : list [start, end] ISO strings, optional
    delay : int  Burn-in time steps to exclude.
    recessions : pd.DataFrame  Columns 'start', 'end' for shading.
    colors : dict {model_name: colour}
    lowess_frac : float  Smoothing bandwidth.
    save, path : as elsewhere.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

    if colors is None:
        _pal = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
        colors = {k: c for k, c in zip(sim_results_dict, _pal)}

    start_date = pd.to_datetime(calib_date[0]) if calib_date else None
    end_date   = pd.to_datetime(calib_date[1]) if calib_date else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    ax_a, ax_b, ax_c = axes

    panel_meta = [
        (ax_a, 'Absolute accepted wage at U→E\n(wage units)',
         'Mean absolute wage accepted by re-hired workers'),
        (ax_b, 'Mean employment wage\n(wage units)',
         'Mean wage of currently employed workers (denominator)'),
        (ax_c, 'Ratio: accepted / employment wage\n(≈ U_REL_WAGE_MEAN)',
         'Ratio = A / B  (cross-check against W1)'),
    ]

    for model_name, df in sim_results_dict.items():
        col = colors.get(model_name, 'black')
        sub = df[df['Time Step'] >= delay].copy()
        sub['Occupation'] = pd.to_numeric(sub['Occupation'], errors='coerce')

        # occ mean wage at this time step
        sub['occ_mean_wage']     = sub['Total_Wages'] / sub['Employment'].clip(lower=1)
        # absolute wage sum for all U→E transitions in this occ-timestep cell
        sub['accepted_wage_abs'] = sub['U_Rel_Wage'] * sub['occ_mean_wage']

        ts = (sub.groupby('Time Step')
                 .agg(accepted_wage_abs=('accepted_wage_abs', 'sum'),
                      UE_Transitions=('UE_Transitions', 'sum'),
                      Total_Wages=('Total_Wages', 'sum'),
                      Employment=('Employment', 'sum'))
                 .reset_index())

        ts['abs_ue_wage']  = ts['accepted_wage_abs'] / ts['UE_Transitions'].clip(lower=1)
        ts['mean_emp_wage'] = ts['Total_Wages'] / ts['Employment'].clip(lower=1)
        ts['ratio']         = ts['abs_ue_wage'] / ts['mean_emp_wage'].clip(lower=1e-9)

        ts['DATE'] = (pd.to_datetime(calib_date[0])
                      + pd.to_timedelta((ts['Time Step'] - delay) * 30.44, unit='D'))
        if calib_date:
            ts = ts[(ts['DATE'] >= start_date) & (ts['DATE'] <= end_date)]
        ts = ts.sort_values('DATE')

        for ax, series_col in zip(axes, ['abs_ue_wage', 'mean_emp_wage', 'ratio']):
            raw = ts[series_col].values
            if len(raw) < 3:
                continue
            smoothed = _lowess(raw, np.arange(len(raw)), frac=lowess_frac)[:, 1]
            ax.plot(ts['DATE'].values, smoothed,
                    color=col, label=model_name, linewidth=1.8, zorder=3)

    for ax, (_, ylabel, title) in zip(axes, panel_meta):
        if recessions is not None:
            for _, row in recessions.iterrows():
                ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
        if calib_date:
            ax.set_xlim(start_date, end_date)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    ax_c.axhline(1.0, color='black', linestyle='--', linewidth=0.9, alpha=0.55)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 4),
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        'Wage decomposition: absolute accepted wage vs mean employment wage\n'
        'A lower ratio in panel C can reflect a higher denominator (B) rather than a lower accepted wage (A)',
        fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    if save and path:
        fig.savefig(os.path.join(path, 'wage_level_decomposition.jpg'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_wage_decomposition_corrected(
    transition_dfs_dict,
    calib_date=None,
    delay=25,
    recessions=None,
    colors=None,
    lowess_frac=0.15,
    grouping_col=None,
    save=False,
    path=None,
):
    """
    Corrected wage decomposition using individual-level transition records.

    Filters to UE transitions and aggregates per time step to produce:
      Panel A — Mean pre-separation wage  (what workers earned before job loss)
      Panel B — Mean accepted wage at U→E (what they accepted on re-hire)
      Panel C — Mean(AcceptedWage / PreSepWage) per time step
                ≈ U_REL_WAGE_MEAN (sanity-check against W1)

    Unlike plot_wage_level_decomposition (which used Total_Wages/Employment as
    the denominator), Panel A here is the actual mean wage of workers *at the
    moment they separated*, drawn directly from transition_df['PreSepWage'].

    The double-bind hypothesis predicts:
      - RW model: higher Panel A (wage preferences keep employed wages elevated,
        so separating workers have a high pre-sep wage / high reservation anchor)
      - RW model: lower Panel B (long duration → worse eventual match)
      - RW model: lower Panel C (both effects compound)

    Parameters
    ----------
    transition_dfs_dict : dict {model_name: DataFrame}
        Each DataFrame is the per-run transition_df from model_fun.py.
        Must contain: Time Step, TransitionType, PreSepWage, AcceptedWage.
    calib_date : list [start, end] ISO strings, optional
    delay : int  Burn-in time steps to exclude.
    recessions : pd.DataFrame  Columns 'start', 'end' for recession shading.
    colors : dict {model_name: colour}
    lowess_frac : float  LOWESS smoothing bandwidth.
    save, path : as elsewhere.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

    if colors is None:
        _pal = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
        colors = {k: c for k, c in zip(transition_dfs_dict, _pal)}

    start_date = pd.to_datetime(calib_date[0]) if calib_date else None
    end_date   = pd.to_datetime(calib_date[1]) if calib_date else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey = True)
    ax_a, ax_b, ax_c = axes

    titles = [
        'Mean pre-separation wage\n(wage workers held before job loss)',
        'Mean accepted wage at U→E\n(wage workers accepted on re-hire)',
        'Ratio: accepted / pre-separation wage\n(≈ U_REL_WAGE_MEAN — cross-check W1)',
    ]
    ylabels = [
        'Pre-separation wage\n(wage units)',
        'Accepted wage at U→E\n(wage units)',
        'Accepted / pre-separation wage',
    ]

    for model_name, df in transition_dfs_dict.items():
        col = colors.get(model_name, 'black')

        ue = df[df['TransitionType'] == 'UE'].copy()
        ue = ue[ue['Time Step'] >= delay]
        ue = ue.dropna(subset=['PreSepWage', 'AcceptedWage'])

        if ue.empty:
            continue

        ue['ratio'] = ue['AcceptedWage'] / ue['PreSepWage'].clip(lower=1e-9)

        ts = (ue.groupby('Time Step')
                .agg(pre_sep=('PreSepWage',  'mean'),
                     accepted=('AcceptedWage', 'mean'),
                     ratio=('ratio',           'mean'))
                .reset_index()
                .sort_values('Time Step'))

        ts['DATE'] = (start_date + pd.to_timedelta(
            (ts['Time Step'] - delay) * 30.44, unit='D'))
        if calib_date:
            ts = ts[(ts['DATE'] >= start_date) & (ts['DATE'] <= end_date)]
        ts = ts.sort_values('DATE')

        for ax, col_name in zip(axes, ['pre_sep', 'accepted', 'ratio']):
            raw = ts[col_name].values
            if len(raw) < 3:
                continue
            smoothed = _lowess(raw, np.arange(len(raw)), frac=lowess_frac)[:, 1]
            ax.plot(ts['DATE'].values, raw, color=col, alpha=0.10,
                    linewidth=0.6, zorder=1)
            ax.plot(ts['DATE'].values, smoothed,
                    color=col, label=model_name, linewidth=1.8, zorder=3)

    for ax, title, ylabel in zip(axes, titles, ylabels):
        if recessions is not None:
            for _, row in recessions.iterrows():
                ax.axvspan(row['start'], row['end'], color='grey', alpha=0.18)
        if calib_date:
            ax.set_xlim(start_date, end_date)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    ax_c.axhline(1.0, color='black', linestyle='--', linewidth=0.9, alpha=0.55,
                 label='No wage change')

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 4),
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        'Wage decomposition: pre-separation wage vs accepted re-employment wage (UE transitions only)\n'
        'Double-bind: RW model workers fall from higher wages and land lower',
        fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    if save and path:
        suffix = f'_{grouping_col}' if grouping_col else ''
        fig.savefig(os.path.join(path, f'wage_decomposition_corrected{suffix}.jpg'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_wage_ue_ee_comparison(
    transition_dfs_dict,
    calib_date=None,
    delay=25,
    recessions=None,
    colors=None,
    lowess_frac=0.15,
    occ_crosswalk=None,
    grouping_col=None,
    soc_labels=None,
    stress_labels=None,
    latex_ready=False,
    show_ts=True,
    save=False,
    path=None,
):
    """
    Figure comparing UE vs EE wage change ratios.

    Section 1 (show_ts=True) — Time series (2×2 grid): LOWESS mean lines + EE–UE
                gap fill per model.
    Section 2 — Dumbbell (1×n_models): per-occupation-group mean ratios sorted by UE
                ratio. Hollow circle = U→E, filled = E→E, connected by a line.
                Coloured by stress_labels if provided, else by model colour.

    Section 2 requires occ_crosswalk and grouping_col.

    Parameters
    ----------
    occ_crosswalk : DataFrame with columns 'X' (1-based occ id) and grouping_col
    grouping_col  : column in occ_crosswalk to group occupations (e.g. 'SOC_major_adj')
    soc_labels    : dict {group_id: label_string}
    stress_labels : dict {group_id: 'structural'|'cyclical'|'unstressed'}
    latex_ready   : if True, use compact LaTeX-ready sizing and save as PDF
    show_ts       : if False, omit the time-series top section and show only the
                    dumbbell panel
    """
    import matplotlib.gridspec as mgridspec
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.colors as mcolors

    STRESS_COLORS = {
        'structural': '#c0392b',
        'cyclical':   '#e67e22',
        'unstressed': '#7f8c8d',
    }

    if colors is None:
        _pal = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
        colors = {k: c for k, c in zip(transition_dfs_dict, _pal)}

    start_date  = pd.to_datetime(calib_date[0]) if calib_date else None
    end_date    = pd.to_datetime(calib_date[1]) if calib_date else None
    model_names = list(transition_dfs_dict.keys())
    n_models    = len(model_names)
    ncols_ts    = 2
    nrows_ts    = (n_models + 1) // 2 if show_ts else 0

    # ── Per-group ratios for sections 2 & 3 ───────────────────────────────
    show_occ = (occ_crosswalk is not None and grouping_col is not None)
    occ_ratios = {}  # {model: {'UE': Series(group_id→ratio), 'EE': Series}}

    if show_occ:
        cw = (occ_crosswalk[['X', grouping_col]]
              .drop_duplicates('X').copy())
        cw['occ_id']  = cw['X'].astype(int) - 1
        cw['group_id'] = cw[grouping_col]
        cw = cw[['occ_id', 'group_id']].dropna(subset=['group_id'])

        for model_name in model_names:
            df  = transition_dfs_dict[model_name]
            df  = df[df['Time Step'] >= delay].copy()
            df  = df.dropna(subset=['PreSepWage', 'AcceptedWage'])
            df['ratio'] = df['AcceptedWage'] / df['PreSepWage'].clip(lower=1e-9)
            mr = {}
            for ttype, occ_col in [('UE', 'OriginOccupation'),
                                    ('EE', 'OriginOccupation')]:
                sub = (df[df['TransitionType'] == ttype]
                       .rename(columns={occ_col: 'occ_id'})
                       .merge(cw, on='occ_id', how='left')
                       .dropna(subset=['group_id']))
                mr[ttype] = sub.groupby('group_id')['ratio'].mean()
            occ_ratios[model_name] = mr

        # sort order: first model's UE ratio, descending (highest at top)
        first_ue = occ_ratios[model_names[0]].get('UE', pd.Series(dtype=float))
        all_groups = sorted(
            set().union(*[set(r.get('UE', pd.Series()).index) |
                          set(r.get('EE', pd.Series()).index)
                          for r in occ_ratios.values()]),
            key=lambda g: (first_ue.get(g, np.nan)
                           if pd.notna(first_ue.get(g, np.nan)) else -999),
            reverse=True,
        )
        n_groups = len(all_groups)

        import textwrap as _textwrap

        def _label(g):
            if soc_labels:
                raw = str(soc_labels.get(g, g))
            else:
                try:
                    return str(int(g))
                except Exception:
                    raw = str(g)
            return _textwrap.fill(raw, width=30)

        group_labels = [_label(g) for g in all_groups]

    # ── Font sizes ────────────────────────────────────────────────────────
    # FS_GROUP is a dense per-occupation-group tick label (can be dozens of
    # rows) -- kept small deliberately, same exception as dense heatmap tick
    # labels elsewhere. The rest inherit the shared rcParams instead of
    # hardcoding a duplicate size.
    FS_GROUP  = 6 if latex_ready else 6.5

    # ── Figure layout (GridSpec) ───────────────────────────────────────────
    # Width fixed at TEXTWIDTH regardless of latex_ready/model count -- ts_h
    # uses ROW_HEIGHT in/row; db_h is content-density-driven (like a heatmap,
    # scales with occupation-group count for label legibility), not a
    # panel-grid row count, so it keeps its own formula.
    fig_w  = TEXTWIDTH
    ts_h   = ROW_HEIGHT * nrows_ts
    db_h   = max(2.5, (n_groups if show_occ else 0) * 0.3)

    n_sections = (1 if show_ts else 0) + (1 if show_occ else 0)
    heights    = ([ts_h] if show_ts else []) + ([db_h] if show_occ else [])
    fig        = plt.figure(figsize=(fig_w, sum(heights)))
    outer = mgridspec.GridSpec(n_sections, 1, figure=fig,
                               height_ratios=heights, hspace=0.08)

    ts_flat = []
    if show_ts:
        ts_gs   = mgridspec.GridSpecFromSubplotSpec(nrows_ts, ncols_ts,
                                                    subplot_spec=outer[0],
                                                    hspace=0.3, wspace=0.12)
        ts_axes = np.array([[fig.add_subplot(ts_gs[r, c])
                             for c in range(ncols_ts)]
                            for r in range(nrows_ts)])
        for r in range(nrows_ts):
            for c in range(ncols_ts):
                if r == 0 and c == 0:
                    continue
                ts_axes[r, c].sharex(ts_axes[0, 0])
                ts_axes[r, c].sharey(ts_axes[0, 0])
        ts_flat = ts_axes.flatten()

    db_section_idx = 1 if show_ts else 0
    if show_occ:
        db_gs   = mgridspec.GridSpecFromSubplotSpec(1, n_models,
                                                    subplot_spec=outer[db_section_idx],
                                                    wspace=0.08)
        db_axes = [fig.add_subplot(db_gs[0, i]) for i in range(n_models)]
        for i in range(1, n_models):
            db_axes[i].sharex(db_axes[0])
            db_axes[i].sharey(db_axes[0])

    # ── Section 1: Time series ─────────────────────────────────────────────
    if show_ts:
        for ax_i, model_name in enumerate(model_names):
            ax  = ts_flat[ax_i]
            col = colors.get(model_name, 'black')
            df  = transition_dfs_dict[model_name]
            df  = df[df['Time Step'] >= delay].copy()
            df  = df.dropna(subset=['PreSepWage', 'AcceptedWage'])
            df['ratio'] = df['AcceptedWage'] / df['PreSepWage'].clip(lower=1e-9)

            sm, dt = {}, {}
            for ttype in ['UE', 'EE']:
                sub = df[df['TransitionType'] == ttype]
                if sub.empty:
                    continue
                ts = (sub.groupby('Time Step')['ratio']
                         .mean().reset_index().sort_values('Time Step'))
                ts['DATE'] = start_date + pd.to_timedelta(
                    (ts['Time Step'] - delay) * 30.44, unit='D')
                if calib_date:
                    ts = ts[(ts['DATE'] >= start_date) & (ts['DATE'] <= end_date)]
                ts  = ts.sort_values('DATE')
                raw = ts['ratio'].values
                if len(raw) < 3:
                    continue
                sm[ttype] = _lowess(raw, np.arange(len(raw)), frac=lowess_frac)[:, 1]
                dt[ttype] = ts['DATE'].values
                ax.plot(ts['DATE'].values, sm[ttype], color=col,
                        linewidth=1.8, linestyle='--' if ttype == 'UE' else '-',
                        zorder=3)

            if 'UE' in sm and 'EE' in sm:
                comb = (pd.DataFrame({'UE': pd.Series(sm['UE'], index=dt['UE']),
                                      'EE': pd.Series(sm['EE'], index=dt['EE'])})
                        .sort_index().interpolate().dropna())
                ax.fill_between(comb.index, comb['UE'], comb['EE'],
                                color=col, alpha=0.15, zorder=2)

            ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            if recessions is not None:
                for _, row in recessions.iterrows():
                    ax.axvspan(row['start'], row['end'], color='grey', alpha=0.15)
            if calib_date:
                ax.set_xlim(start_date, end_date)
            ax.spines[['top', 'right']].set_visible(False)
            if ax_i % ncols_ts == 0:
                ax.set_ylabel('Accepted / pre-separation wage')

        for ax in ts_flat[n_models:]:
            ax.set_visible(False)

    # ── Section 2: Dumbbell ───────────────────────────────────────────────
    if show_occ:
        all_vals = [v for r in occ_ratios.values()
                    for s in r.values() for v in s.dropna().values]
        x_min = max(0,   np.nanpercentile(all_vals, 2)  - 0.05) if all_vals else 0
        x_max = min(3.0, np.nanpercentile(all_vals, 98) + 0.05) if all_vals else 3

        for col_i, model_name in enumerate(model_names):
            ax  = db_axes[col_i]
            col = colors.get(model_name, 'black')
            ue_s = occ_ratios[model_name].get('UE', pd.Series())
            ee_s = occ_ratios[model_name].get('EE', pd.Series())

            for i, grp in enumerate(all_groups):
                ue_val    = ue_s.get(grp, np.nan)
                ee_val    = ee_s.get(grp, np.nan)
                dot_color = (STRESS_COLORS.get(stress_labels.get(grp), col)
                             if stress_labels else col)
                if pd.notna(ue_val) and pd.notna(ee_val):
                    ls = '--' if ee_val > ue_val else '-'
                    ax.plot([ue_val, ee_val], [i, i],
                            color=dot_color, linewidth=1.0, linestyle=ls,
                            alpha=0.5, zorder=2)
                if pd.notna(ue_val):
                    ax.plot(ue_val, i, 'o', color='white', markersize=6,
                            markeredgecolor=dot_color, markeredgewidth=1.3, zorder=4)
                if pd.notna(ee_val):
                    ax.plot(ee_val, i, 'o', color=dot_color, markersize=6, zorder=4)

            ax.axvline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            #ax.set_xlim(x_min, x_max)
            ax.set_yticks(range(n_groups))
            ax.set_ylim(-0.5, n_groups - 0.5)
            if col_i == 0:
                ax.set_yticklabels(group_labels, fontsize=FS_GROUP)
            else:
                ax.tick_params(labelleft=False)
            #ax.set_xlabel('Ratio')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', alpha=0.25)

    # ── Shared legend ─────────────────────────────────────────────────────
    ts_handles = [
        Line2D([0], [0], color='grey', lw=1.8, ls='-',  label='E→E (mean)'),
        Line2D([0], [0], color='grey', lw=1.8, ls='--', label='U→E (mean)'),
        Patch(facecolor='grey', alpha=0.20, label='EE–UE gap'),
    ] if show_ts else []
    db_handles = []
    if show_occ:
        db_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='grey', markeredgewidth=1.3, markersize=6,
                   label='U→E (Occ. Mean)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
                   markersize=6, label='E→E (Occ. Mean)'),
        ]
        if stress_labels:
            for cat, sc in STRESS_COLORS.items():
                db_handles.append(Patch(facecolor=sc, label=cat.capitalize()))

    style_handles = ts_handles + db_handles

    if not latex_ready:
        if show_ts:
            fig.suptitle(
                'UE vs EE wage change ratios  (solid = E→E, dashed = U→E)\n'
                'Top: time series  |  Bottom: per-occupation-group means '
                '(hollow = U→E, filled = E→E; dashed line = EE > UE)',
                fontweight='bold')
        else:
            fig.suptitle(
                'UE vs EE wage change ratios by occupation group\n'
                '(hollow = U→E, filled = E→E; dashed connector = EE > UE)',
                fontweight='bold')
        add_suptitle_spacing(fig)

    fig.supxlabel('Ratio')
    # The usual case for this figure is exactly 6 models in 3 colour families of
    # 2 (blue/purple/orange, per model_colors.py's fixed dict order) with no
    # time-series handles and the 2 occupation-mean markers as the only extras
    # -- lay that out as one column per colour family (2 models + 1 marker each,
    # third column's marker row left empty) rather than letting the generic
    # column-major fill interleave models and extras arbitrarily.
    _model_handles = [Line2D([0], [0], color=colors[name], linewidth=2, label=name)
                      for name in colors if name in model_names]
    if not show_ts and len(_model_handles) == 6 and len(db_handles) == 2:
        _ordered_handles = (_model_handles[0:2] + [db_handles[0]] +
                            _model_handles[2:4] + [db_handles[1]] +
                            _model_handles[4:6])
        add_shared_legend(fig, {}, extra_handles=_ordered_handles, ncol=3)
    else:
        add_shared_legend(fig, colors, extra_handles=style_handles,
                           ncol=min(len(model_names) + len(style_handles), 5))

    if save and path:
        suffix = f'_{grouping_col}' if grouping_col else ''
        if latex_ready:
            save_plot_standard(fig, os.path.join(path, f'wage_ue_ee_comparison{suffix}.pdf'),
                                aspect_ratio=sum(heights) / fig_w)
        else:
            save_plot_standard(fig, os.path.join(path, f'wage_ue_ee_comparison{suffix}.jpg'),
                                aspect_ratio=sum(heights) / fig_w)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_wage_ratio_by_duration(
    transition_dfs_dict,
    delay=25,
    colors=None,
    latex_ready=False,
    save=False,
    path=None,
):
    """
    Mean AcceptedWage / PreSepWage as a function of unemployment duration.

    Shows the mechanical link between spell length and wage penalty: as workers
    stay unemployed longer, their eventual accepted wage declines relative to their
    pre-separation wage. One line per model, a ribbon covering the 25th–75th
    percentile across workers, and a reference at y=1 (full recovery).

    Multi-sim: if 'sim_id' is present in the transition DataFrames, bin means are
    computed per simulation then aggregated to mean ± 95% CI ribbon.

    x-axis : unemployment duration bin (months)
    y-axis : mean AcceptedWage / PreSepWage  (1 = full recovery)
    """
    if colors is None:
        _pal = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
        colors = {k: c for k, c in zip(transition_dfs_dict, _pal)}

    dur_bins   = [0, 3, 6, 12, 24, 48, np.inf]
    dur_labels = ['0–3m', '3–6m', '6–12m', '12–24m', '24–48m', '48+m']
    bin_mids   = [1.5, 4.5, 9, 18, 36, 54]   # for x-axis positioning
    model_names = list(transition_dfs_dict.keys())
    n_models    = len(model_names)

    fig_w = TEXTWIDTH
    fig_h = ROW_HEIGHT

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    for model_name in model_names:
        df  = transition_dfs_dict[model_name].copy()
        df  = df[(df['TransitionType'] == 'UE') & (df['Time Step'] >= delay)]
        df  = df.dropna(subset=['PreSepWage', 'AcceptedWage', 'UEDuration'])
        df['ratio']   = df['AcceptedWage'] / df['PreSepWage'].clip(lower=1e-9)
        df['dur_bin'] = pd.cut(df['UEDuration'], bins=dur_bins,
                               labels=dur_labels, right=False)
        col = colors.get(model_name, 'steelblue')

        _multisim = 'sim_id' in df.columns

        if _multisim:
            # per-sim bin means → aggregate to mean ± CI
            sim_means = (df.groupby(['sim_id', 'dur_bin'])['ratio']
                         .mean().reset_index())
            agg = (sim_means.groupby('dur_bin')['ratio']
                   .agg(['mean', 'std', 'count']).reset_index())
            agg['ci'] = 1.96 * agg['std'] / np.sqrt(agg['count'].clip(lower=1))
            x    = [bin_mids[dur_labels.index(b)] for b in agg['dur_bin'].tolist()
                    if b in dur_labels]
            y    = agg['mean'].values
            lo   = (agg['mean'] - agg['ci']).values
            hi   = (agg['mean'] + agg['ci']).values
        else:
            agg  = (df.groupby('dur_bin')['ratio']
                    .agg(['mean',
                          lambda s: s.quantile(0.25),
                          lambda s: s.quantile(0.75)])
                    .reset_index())
            agg.columns = ['dur_bin', 'mean', 'q25', 'q75']
            x   = [bin_mids[dur_labels.index(b)] for b in agg['dur_bin'].tolist()
                   if b in dur_labels]
            y   = agg['mean'].values
            lo  = agg['q25'].values
            hi  = agg['q75'].values

        ax.plot(x, y, color=col, linewidth=1.8, marker='o', markersize=4,
                label=model_name, zorder=3)
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, zorder=1)

    ax.axhline(1.0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.set_xticks(bin_mids)
    ax.set_xticklabels(dur_labels)
    ax.set_xlabel('Unemployment Duration')
    ax.set_ylabel('Ratio of Accepted Wage\nto Pre-Separation Wage',
                  ha='center', multialignment='center')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax.set_title('Re-Employment Wage Ratio by Unemployment Duration', fontweight='bold')

    plt.tight_layout()
    # add_shared_legend(fig, colors,
    #                    extra_handles=[Line2D([0], [0], color='black', linewidth=0.9,
    #                                           linestyle='--', alpha=0.6,
    #                                           label='Full recovery (ratio = 1)')],
    #                    ncol=min(len(colors) + 1, 5))
    if save and path:
        save_plot_standard(fig, os.path.join(path, 'wage_ratio_by_duration.jpg'),
                            aspect_ratio=fig_h / fig_w)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_wage_scarring_by_duration(
    transition_dfs_dict,
    calib_date=None,
    delay=25,
    colors=None,
    model_groups=None,
    nrows=1,
    group_labels=None,
    fname_suffix='',
    latex_ready=False,
    save=False,
    path=None,
):
    """
    Wage scarring by separation cohort, plotted on a calendar time axis.

    Each cohort (separation year) contributes one line that starts at its
    layoff year and extends forward in calendar time according to the
    unemployment duration bins:
      bin midpoints (months): 1.5, 4.5, 9, 18, 30  →  added as fractional years

    Lines from different cohorts therefore overlap on the calendar axis:
    a 2009-cohort worker at 0–3 months appears at ~2009.1, while a
    2008-cohort worker at 12–24 months appears at ~2009.5.

      - x-axis : calendar year (separation year + bin midpoint in years)
      - y-axis : mean AcceptedWage / PreSepWage for that cohort × duration bin
      - Reference line at 1.0 (full wage recovery)

    Layout modes
    ------------
    model_groups=None (default):
        One panel per model, stacked vertically (original behaviour).

    model_groups=[[m1, m2], [m3, m4], [m5]] (list of lists):
        Grouped layout. Each inner list is one column (colour family).
        nrows=1 : models within a group overlaid on one panel → 1 × n_groups grid.
        nrows=2 : models within a group stacked → 2 × n_groups grid, one model
                  per cell. Groups with only one model leave the second row empty.

    group_labels : list of str, one per group column (used as column title).
    """
    if colors is None:
        _pal = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
        colors = {k: c for k, c in zip(transition_dfs_dict, _pal)}

    start_date  = pd.to_datetime(calib_date[0]) if calib_date else None
    model_names = list(transition_dfs_dict.keys())
    n_models    = len(model_names)

    dur_bins          = [0, 3, 6, 12, 24, np.inf]
    dur_labels        = ['0–3m', '3–6m', '6–12m', '12–24m', '24+m']
    bin_midpoints_yrs = [1.5/12, 4.5/12, 9/12, 18/12, 30/12]

    # ── Pre-process all models ────────────────────────────────────────────────
    all_years = set()
    prepared  = {}
    for model_name in model_names:
        df = transition_dfs_dict[model_name].copy()
        df = df[(df['TransitionType'] == 'UE') & (df['Time Step'] >= delay)]
        df = df.dropna(subset=['PreSepWage', 'AcceptedWage', 'UEDuration'])
        df['ratio'] = df['AcceptedWage'] / df['PreSepWage'].clip(lower=1e-9)
        if 'OriginOccupation' in df.columns:
            occ_mean = df.groupby('OriginOccupation')['ratio'].transform('mean')
            df['ratio'] = df['ratio'] - occ_mean + 1.0
        # Drop left-censored spells: `Time Step >= delay` above only checks
        # when the worker RE-employed, not when they separated. A worker who
        # became unemployed during burn-in (sep_ts < delay) but happened to
        # find a job shortly after `delay` still passes that filter, creating
        # a biased phantom "cohort" before the true start date (only the
        # burn-in spells that ended early survive -- longer ones are still
        # unemployed at `delay` and get excluded from later years too).
        sep_ts = df['Time Step'] - df['UEDuration']
        df     = df[sep_ts >= delay].copy()
        sep_ts = sep_ts[sep_ts >= delay]
        if start_date is not None:
            sep_date   = start_date + pd.to_timedelta(
                (sep_ts - delay) * 30.44, unit='D')
            df['sep_year'] = sep_date.dt.year
        else:
            df['sep_year'] = 0
        df['dur_bin'] = pd.cut(df['UEDuration'], bins=dur_bins,
                               labels=dur_labels, right=False)
        prepared[model_name] = df
        all_years.update(df['sep_year'].dropna().unique())

    years = sorted(all_years)

    def _draw_model(ax, model_name, col):
        df = prepared[model_name]
        for yr in years:
            sub = df[df['sep_year'] == yr]
            if len(sub) < 10:
                continue
            grp   = sub.groupby('dur_bin', observed=True)['ratio']
            means = grp.mean()
            sems  = grp.sem()
            xs, ys, ses = [], [], []
            for b, mid in zip(dur_labels, bin_midpoints_yrs):
                if b in means.index and pd.notna(means[b]):
                    xs.append(yr + mid)
                    ys.append(means[b])
                    ses.append(sems[b] if pd.notna(sems[b]) else 0.0)
            if len(xs) < len(dur_labels):
                # Incomplete cohort: too close to the end of the simulated
                # window to have reached the longest duration bin yet (those
                # spells are still ongoing or the sim ended first). Drop
                # rather than plot a truncated stub that reads as a
                # rendering glitch instead of real (right-censored) data.
                continue
            lo = [y - 1.96 * se for y, se in zip(ys, ses)]
            hi = [y + 1.96 * se for y, se in zip(ys, ses)]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, zorder=2)
            ax.plot(xs, ys, color=col, linewidth=1.2, alpha=0.75, zorder=3)

    def _style_ax(ax, is_bottom, is_leftmost):
        _recessions = [(2001.25, 2001.92), (2007.92, 2009.5), (2020.08, 2020.33)]
        for _r0, _r1 in _recessions:
            ax.axvspan(_r0, _r1, color='lightgrey', alpha=0.4, zorder=1)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylim(0.5, 1.5)
        if is_bottom:
            ax.set_xlabel('Date')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Width fixed at TEXTWIDTH, height at ROW_HEIGHT in/row -- independent of
    # column count, matching every other multi-row figure.
    if model_groups is None:
        # original: one panel per model, stacked vertically
        fig_w = TEXTWIDTH
        fig_h = ROW_HEIGHT_3 * n_models
        fig, axes = plt.subplots(n_models, 1, figsize=(fig_w, fig_h),
                                 sharey=True, sharex=True, squeeze=False)
        for ax_i, model_name in enumerate(model_names):
            ax  = axes[ax_i, 0]
            col = colors.get(model_name, 'steelblue')
            _draw_model(ax, model_name, col)
            _style_ax(ax,
                      is_bottom=(ax_i == n_models - 1),
                      is_leftmost=True)
        legend_colors = {m: colors.get(m, 'steelblue') for m in model_names if m in prepared}
    else:
        n_cols = len(model_groups)
        n_actual_rows = nrows

        fig_w = TEXTWIDTH
        fig_h = ROW_HEIGHT_3 * n_actual_rows
        fig, axes = plt.subplots(n_actual_rows, n_cols,
                                 figsize=(fig_w, fig_h),
                                 sharey=True, sharex=True, squeeze=False)

        all_models_in_groups = [m for grp in model_groups for m in grp]
        for col_i, group in enumerate(model_groups):
            if nrows == 1:
                ax = axes[0, col_i]
                for model_name in group:
                    if model_name not in prepared:
                        continue
                    col = colors.get(model_name, 'steelblue')
                    _draw_model(ax, model_name, col)
                _style_ax(ax, is_bottom=True, is_leftmost=(col_i == 0))

            else:  # nrows == 2
                for row_i in range(n_actual_rows):
                    ax = axes[row_i, col_i]
                    if row_i < len(group):
                        model_name = group[row_i]
                        if model_name in prepared:
                            col = colors.get(model_name, 'steelblue')
                            _draw_model(ax, model_name, col)
                            _style_ax(ax,
                                      is_bottom=(row_i == n_actual_rows - 1),
                                      is_leftmost=(col_i == 0))
                    else:
                        ax.set_visible(False)

        legend_colors = {m: colors.get(m, 'steelblue') for m in all_models_in_groups if m in prepared}

    fig.supylabel('Ratio of Accepted to Pre-Separation Wage\n(Occupation-Demeaned)',
                  multialignment='center')
    fig.suptitle('Re-Employment Wage Ratio by Separation Cohort', fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_supylabel_spacing(fig)
    add_shared_legend(fig, legend_colors, ncol=min(len(legend_colors), 3))

    if save and path:
        fname = f'wage_scarring_by_duration{fname_suffix}'
        if latex_ready:
            save_plot_standard(fig, os.path.join(path, f'{fname}.pdf'), aspect_ratio=fig_h / fig_w)
        else:
            save_plot_standard(fig, os.path.join(path, f'{fname}.jpg'), aspect_ratio=fig_h / fig_w)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_episode_trajectories(
    transition_dfs_dict,
    episode_occ_ids,
    episode_window,
    episode_name="Episode",
    empirical_data=None,
    cps_dur_dist=None,
    occ_crosswalk=None,
    ue_stock_dfs=None,
    n_surv_years=3,
    show_wage_panel=True,
    show_ltuer_panel=True,
    ltuer_threshold=6,
    calib_date=None,
    delay=25,
    colors=None,
    latex_ready=False,
    save=False,
    path=None,
):
    """
    Post-separation wage trajectories for workers in a specific shock episode,
    compared across models.  One figure per episode with up to three panels.

    Layout (top to bottom):
      Panel C (optional, full width) — LTUER time series over the ENTIRE
                simulated period for the episode occupation group.
                LTUER = fraction of currently unemployed workers with elapsed
                duration >= ltuer_threshold months (default 6).  One coloured
                line per model (quarterly, from ue_stock_dfs) + black dashed
                CPS annual reference.  Episode window shaded in grey.

      Panel A (optional, full width) — wage ratio (AcceptedWage/PreSepWage)
                by UE duration bin, restricted to separations during the
                episode window.  95% CI ribbons.

      Panel B (grid: n_surv_years rows × 4 columns) — cross-sectional elapsed
                UE duration survival curves P(elapsed UE > t), quarterly.
                Annual CPS reference identical across all 4 quarters of a year.

    The wage-ratio occupation baseline is from the same occupations OUTSIDE
    the episode window, so y=1 means "normal times for this occupation".

    Parameters
    ----------
    show_ltuer_panel : bool
        If True (default), draw Panel C (LTUER time series).
    show_wage_panel : bool
        If True (default), draw Panel A (wage ratios).
    ltuer_threshold : int
        Elapsed months defining long-term unemployment (default 6).
    n_surv_years : int
        Number of year rows in Panel B (each = 4 quarterly panels; default 3).
    """
    occ_set    = set(episode_occ_ids)
    start_date = pd.to_datetime(calib_date[0]) if calib_date else None
    ep_start   = pd.to_datetime(episode_window[0])
    ep_end     = pd.to_datetime(episode_window[1])

    if colors is None:
        _pal   = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
        colors = {k: c for k, c in zip(transition_dfs_dict, _pal)}

    model_names = list(transition_dfs_dict.keys())
    n_models    = len(model_names)

    dur_bins   = [0, 3, 6, 12, 24]
    dur_labels = ['0–3m', '3–6m', '6–12m', '12–24m']
    bin_mids   = [1.5, 4.5, 9, 18]

    surv_months = np.arange(0, 30)
    surv_years  = [ep_start.year + i for i in range(n_surv_years)]
    Q_LABELS    = ['Q1', 'Q2', 'Q3', 'Q4']

    FS_LABEL = 8
    FS_TICK  = 7

    # ── figure layout: rows = [Panel C?] + [Panel A?] + n_surv_years × Panel B
    N_COLS       = 4
    fig_w        = max(10.0, N_COLS * 2.5) if not latex_ready else max(9.0, N_COLS * 2.2)
    h_ltuer      = 1.8 if show_ltuer_panel else 0
    h_wage       = 1.8 if show_wage_panel  else 0
    h_grid       = n_surv_years * (2.0 if not latex_ready else 1.8)
    fig_h        = (h_ltuer + h_wage + h_grid) * (1 if latex_ready else 1.05)
    fig_h        = max(fig_h, 3.0)

    n_top_rows   = int(show_ltuer_panel) + int(show_wage_panel)
    n_total_rows = n_top_rows + n_surv_years
    top_ratios   = ([1.8] * int(show_ltuer_panel)) + ([1.8] * int(show_wage_panel))
    height_ratios = top_ratios + [1.0] * n_surv_years

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(n_total_rows, N_COLS,
                           height_ratios=height_ratios,
                           hspace=0.55, wspace=0.3)

    _row = 0
    if show_ltuer_panel:
        ax_ltuer = fig.add_subplot(gs[_row, :]);  _row += 1
    else:
        ax_ltuer = None
    if show_wage_panel:
        ax_wage  = fig.add_subplot(gs[_row, :]);  _row += 1
    else:
        ax_wage  = None
    axs_q = [[fig.add_subplot(gs[_row + yr_i, q_i]) for q_i in range(N_COLS)]
              for yr_i in range(n_surv_years)]

    # ── helper: survival curve from elapsed-duration array
    def _survival(durations):
        n = len(durations)
        if n == 0:
            return np.full(len(surv_months), np.nan)
        return np.array([(durations > d).sum() / n for d in surv_months])

    # ── ts → (year, quarter 1-4)
    def _ts_to_yq(ts):
        d = start_date + pd.DateOffset(months=int(ts) - delay - 1)
        return d.year, (d.month - 1) // 3 + 1

    # ── Panel A: wage ratios from transition_dfs (skipped when show_wage_panel=False)
    if show_wage_panel:
        for model_name in model_names:
            col = colors.get(model_name, 'steelblue')
            df  = transition_dfs_dict[model_name].copy()
            df  = df[(df['TransitionType'] == 'UE') & (df['Time Step'] >= delay)]
            df  = df.dropna(subset=['PreSepWage', 'AcceptedWage', 'UEDuration'])

            if 'OriginOccupation' not in df.columns or df.empty:
                continue
            df = df[df['OriginOccupation'].isin(occ_set)]
            if df.empty:
                continue

            df = df.copy()
            df['ratio'] = df['AcceptedWage'] / df['PreSepWage'].clip(lower=1e-9)

            if start_date is not None:
                sep_ts         = df['Time Step'] - df['UEDuration']
                df['sep_date'] = start_date + pd.to_timedelta(
                    (sep_ts - delay) * 30.44, unit='D')
                outside = df[(df['sep_date'] < ep_start) | (df['sep_date'] > ep_end)]
            else:
                outside = df

            occ_baseline = outside.groupby('OriginOccupation')['ratio'].mean()
            df['ratio']  = df['ratio'] - df['OriginOccupation'].map(occ_baseline) + 1.0

            if start_date is not None:
                during = df[(df['sep_date'] >= ep_start) & (df['sep_date'] <= ep_end)]
            else:
                during = df

            if len(during) < 5:
                continue

            during = during.copy()
            during['dur_bin'] = pd.cut(during['UEDuration'], bins=dur_bins,
                                       labels=dur_labels, right=False)

            if 'sim_id' in during.columns:
                # per-sim bin means, then aggregate across sims
                sim_bin_means = (during.groupby(['sim_id', 'dur_bin'], observed=True)['ratio']
                                        .mean().reset_index())
                sim_agg = (sim_bin_means.groupby('dur_bin', observed=True)['ratio']
                                        .agg(['mean', 'sem']))
                means = sim_agg['mean']
                sems  = sim_agg['sem']
            else:
                grp   = during.groupby('dur_bin', observed=True)['ratio']
                means = grp.mean()
                sems  = grp.sem()

            xs, ys, ses = [], [], []
            for b, mid in zip(dur_labels, bin_mids):
                if b in means.index and pd.notna(means[b]):
                    xs.append(mid)
                    ys.append(means[b])
                    ses.append(sems[b] if pd.notna(sems[b]) else 0.0)
            if len(xs) >= 2:
                lo = [y - 1.96 * se for y, se in zip(ys, ses)]
                hi = [y + 1.96 * se for y, se in zip(ys, ses)]
                ax_wage.fill_between(xs, lo, hi, color=col, alpha=0.15, zorder=2)
                ax_wage.plot(xs, ys, color=col, lw=1.6, alpha=0.9,
                             zorder=3, label=model_name)

        if empirical_data is not None:
            e_means = empirical_data['means']
            e_ses   = empirical_data.get('ses', [0] * len(e_means))
            e_xs    = [bin_mids[i] for i, m in enumerate(e_means) if pd.notna(m)]
            e_ys    = [m for m in e_means if pd.notna(m)]
            e_lo    = [y - 1.96 * se for y, se in zip(e_ys, e_ses) if pd.notna(y)]
            e_hi    = [y + 1.96 * se for y, se in zip(e_ys, e_ses) if pd.notna(y)]
            ax_wage.fill_between(e_xs, e_lo, e_hi, color='black', alpha=0.08, zorder=2)
            ax_wage.plot(e_xs, e_ys, color='black', lw=1.8, ls='--', alpha=0.85,
                         zorder=4, label='Empirical')

        ax_wage.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax_wage.set_xticks(bin_mids)
        ax_wage.set_xticklabels(dur_labels, fontsize=FS_TICK, rotation=30, ha='right')
        ax_wage.set_xlim(0, 24)
        ax_wage.set_ylabel('Accepted / pre-sep. wage\n(rel. to occ. baseline)', fontsize=FS_LABEL)
        ax_wage.tick_params(labelsize=FS_TICK)
        ax_wage.spines[['top', 'right']].set_visible(False)
        ax_wage.grid(axis='y', alpha=0.25, lw=0.5)
        ax_wage.legend(fontsize=FS_TICK, frameon=False, loc='lower left',
                       ncol=min(n_models + 1, 3))

    # ── CPS SOC group lookup (once, shared across all panels)
    soc_groups = []
    if cps_dur_dist is not None and occ_crosswalk is not None:
        cw = occ_crosswalk[['X', 'SOC_major_adj']].drop_duplicates('X').copy()
        cw['occ_id'] = cw['X'].astype(int) - 1
        soc_groups = (cw.loc[cw['occ_id'].isin(occ_set), 'SOC_major_adj']
                        .dropna().unique().astype(int).tolist())

    # ── pre-filter ue_stock_dfs to episode occupations + add Year and Quarter
    stock_by_model = {}
    if ue_stock_dfs is not None and start_date is not None:
        for model_name, stock_df in ue_stock_dfs.items():
            s = stock_df[stock_df['Occupation'].isin(occ_set)].copy()
            if s.empty:
                continue
            yq = s['Time Step'].apply(_ts_to_yq)
            s['Year']    = yq.apply(lambda x: x[0])
            s['Quarter'] = yq.apply(lambda x: x[1])
            stock_by_model[model_name] = s

    # ── Panel C: LTUER time series over entire simulated period
    if show_ltuer_panel and ax_ltuer is not None:
        # model LTUER: quarterly, from ue_stock_dfs for episode occupations
        if ue_stock_dfs is not None and start_date is not None:
            for model_name, stock_df in ue_stock_dfs.items():
                col = colors.get(model_name, 'steelblue')
                s   = stock_df[stock_df['Occupation'].isin(occ_set)].copy()
                if s.empty:
                    continue
                s_post = s[s['Time Step'] >= delay].copy()
                # Use the model's longterm_unemp flag when available; fall back to
                # duration threshold for cached data that predates the flag column.
                if 'LongtermUnemp' in s.columns:
                    s_post['_lt'] = s_post['LongtermUnemp'].astype(bool)
                else:
                    s_post['_lt'] = (s_post['UEDuration'] - 1) >= ltuer_threshold
                if 'sim_id' in s.columns:
                    per_sim = (s_post
                               .groupby(['Time Step', 'sim_id'])
                               .apply(lambda x: x['_lt'].sum() / max(len(x), 1))
                               .reset_index(name='ltuer'))
                    ltuer_ts = (per_sim.groupby('Time Step')['ltuer']
                                       .agg(['mean', 'sem']).reset_index())
                    ltuer_ts['date'] = (start_date
                                        + pd.to_timedelta((ltuer_ts['Time Step'] - delay) * 30.44, unit='D'))
                    ci = 1.96 * ltuer_ts['sem'].fillna(0)
                    ax_ltuer.fill_between(ltuer_ts['date'],
                                          ltuer_ts['mean'] - ci, ltuer_ts['mean'] + ci,
                                          color=col, alpha=0.15, zorder=2)
                    ax_ltuer.plot(ltuer_ts['date'], ltuer_ts['mean'],
                                  color=col, lw=1.2, alpha=0.75, label=model_name, zorder=3)
                else:
                    ltuer_ts = (s_post
                                 .groupby('Time Step')
                                 .apply(lambda x: x['_lt'].sum() / max(len(x), 1))
                                 .reset_index(name='ltuer'))
                    ltuer_ts['date'] = (start_date
                                        + pd.to_timedelta((ltuer_ts['Time Step'] - delay) * 30.44, unit='D'))
                    ax_ltuer.plot(ltuer_ts['date'], ltuer_ts['ltuer'],
                                  color=col, lw=1.2, alpha=0.75, label=model_name, zorder=3)

        # CPS LTUER: annual, from cps_dur_dist for matched SOC groups
        if soc_groups and cps_dur_dist is not None:
            cps_all = cps_dur_dist[cps_dur_dist['group_id'].isin(soc_groups)]
            cps_ltuer = (cps_all.groupby('YEAR')
                                .apply(lambda x: (x.loc[x['dur_months'] >= ltuer_threshold,
                                                         'n_weighted'].sum()
                                                   / max(x['n_weighted'].sum(), 1)))
                                .reset_index(name='ltuer'))
            cps_ltuer['date'] = pd.to_datetime(cps_ltuer['YEAR'].astype(str) + '-07-01')
            ax_ltuer.plot(cps_ltuer['date'], cps_ltuer['ltuer'],
                          color='black', lw=1.6, ls='--', alpha=0.85,
                          label='CPS (annual)', zorder=4)

        # shade episode window
        ax_ltuer.axvspan(ep_start, ep_end, color='gray', alpha=0.12, zorder=1,
                         label='Episode window')
        # shade Panel B grid period (n_surv_years from episode start)
        grid_end = pd.Timestamp(year=ep_start.year + n_surv_years, month=1, day=1)
        ax_ltuer.axvspan(ep_start, grid_end, color='steelblue', alpha=0.08, zorder=1,
                         label=f'Grid period ({n_surv_years}y)')
        ax_ltuer.set_ylabel(f'LTUER\n(UE ≥ {ltuer_threshold}m)', fontsize=FS_LABEL)
        ax_ltuer.tick_params(labelsize=FS_TICK)
        ax_ltuer.spines[['top', 'right']].set_visible(False)
        ax_ltuer.grid(axis='y', alpha=0.25, lw=0.5)
        ax_ltuer.legend(fontsize=FS_TICK, frameon=False, loc='upper left',
                        ncol=min(n_models + 2, 4))

    # ── Panel B: n_surv_years rows × 4 quarterly columns
    #   CPS annual survival is computed once per year and drawn identically in all 4 quarters.
    for yr_i, yr in enumerate(surv_years):
        # annual CPS survival for this year (same reference in all 4 quarterly panels)
        sv_emp_yr = None
        if soc_groups and cps_dur_dist is not None:
            cps_yr = cps_dur_dist[
                (cps_dur_dist['group_id'].isin(soc_groups)) &
                (cps_dur_dist['YEAR'] == yr)
            ]
            if not cps_yr.empty:
                agg   = (cps_yr.groupby('dur_months')['n_weighted']
                               .sum().reindex(surv_months, fill_value=0.0))
                total = agg.sum()
                if total > 0:
                    sv_emp_yr = np.array([agg[agg.index > d].sum() / total
                                          for d in surv_months])

        for q_i, q_label in enumerate(Q_LABELS):
            q    = q_i + 1
            ax_s = axs_q[yr_i][q_i]

            for model_name in model_names:
                col = colors.get(model_name, 'steelblue')
                s   = stock_by_model.get(model_name)
                if s is None:
                    continue
                mask = (s['Year'] == yr) & (s['Quarter'] == q)
                lbl  = model_name if (yr_i == 0 and q_i == 0) else '_nolegend_'
                if 'sim_id' in s.columns:
                    curves = []
                    for sid in s['sim_id'].unique():
                        s_sim     = s.loc[s['sim_id'] == sid]
                        mask_sim  = (s_sim['Year'] == yr) & (s_sim['Quarter'] == q)
                        v = s_sim.loc[mask_sim, 'UEDuration'].values - 1
                        v = v[v >= 0]
                        curves.append(_survival(v))
                    if not curves:
                        continue
                    arr     = np.array(curves)
                    mean_sv = np.nanmean(arr, axis=0)
                    sem_sv  = np.nanstd(arr, axis=0, ddof=1) / max(np.sqrt(len(curves)), 1)
                    ci_sv   = 1.96 * sem_sv
                    ax_s.fill_between(surv_months, mean_sv - ci_sv, mean_sv + ci_sv,
                                      color=col, alpha=0.15, zorder=2)
                    ax_s.plot(surv_months, mean_sv, color=col, lw=1.4,
                              alpha=0.85, zorder=3, label=lbl)
                else:
                    vals = s.loc[mask, 'UEDuration'].values - 1
                    vals = vals[vals >= 0]
                    if len(vals) < 5:
                        continue
                    ax_s.plot(surv_months, _survival(vals), color=col, lw=1.4,
                              alpha=0.85, zorder=3, label=lbl)

            if sv_emp_yr is not None:
                ax_s.plot(surv_months, sv_emp_yr, color='black', lw=1.5,
                          ls='--', alpha=0.8, zorder=4,
                          label='CPS (annual)' if (yr_i == 0 and q_i == 0) else '_nolegend_')

            # axis labels: quarter header on top row, year+ylabel on left, x-label on bottom row
            ax_s.set_title(q_label, fontsize=FS_TICK, pad=2)
            if q_i == 0:
                ax_s.set_ylabel(f'{yr}\nP(UE > t)', fontsize=FS_TICK)
            if yr_i == n_surv_years - 1:
                ax_s.set_xlabel('Elapsed UE (months)', fontsize=FS_TICK)
            ax_s.tick_params(labelsize=FS_TICK - 1)
            ax_s.spines[['top', 'right']].set_visible(False)
            ax_s.grid(axis='y', alpha=0.2, lw=0.5)

    # figure-level legend for Panel B (drawn from Q1/Y1 handles)
    handles, labels = axs_q[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=FS_TICK, frameon=False,
                   loc='lower center', ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.02))

    ep_str = f"{ep_start.strftime('%Y')}–{ep_end.strftime('%Y')}"
    fig.suptitle(f"{episode_name}  ({ep_str},  {len(episode_occ_ids)} occs)",
                 fontsize=FS_LABEL + 1, y=1.01)

    if save and path:
        safe = episode_name.lower().replace('/', '_').replace(' ', '_').replace('–', '-')
        ext  = 'pdf' if latex_ready else 'jpg'
        kw   = {} if latex_ready else {'dpi': 200}
        fig.savefig(os.path.join(path, f'episode_traj_{safe}.{ext}'),
                    bbox_inches='tight', **kw)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_shock_episode_wages(
    transition_dfs_dict,
    shocked_occ_ids,
    shock_horizon,
    shock_info_df=None,
    shock_window=None,
    shock_label="Shocked occupations",
    calib_date=None,
    delay=25,
    colors=None,
    latex_ready=False,
    save=False,
    path=None,
):
    """
    Compare wage scarring and UE duration for workers separated from
    demand-shocked occupations vs all other workers in the same period.

    Layout: 2 rows × n_models columns
      Row 1 — wage ratio (AcceptedWage/PreSepWage, occ-demeaned) by duration bin
              shocked=solid line, rest=dashed line, 95% CI ribbons
      Row 2 — UE duration box plots (shocked left, rest right per panel)

    Parameters
    ----------
    shocked_occ_ids : list of int
        0-indexed OriginOccupation ids classified as shocked.
    shock_horizon : (str, str)
        (start, end) calendar dates; only separations within this window
        are included.
    shock_window : (str, str), optional
        The window used to identify shocks — shown in a suptitle annotation.
    shock_label : str
        Short description of the shock episode for annotations.
    """
    shocked_set = set(shocked_occ_ids)
    start_date  = pd.to_datetime(calib_date[0]) if calib_date else None
    h_start     = pd.to_datetime(shock_horizon[0])
    h_end       = pd.to_datetime(shock_horizon[1])

    if colors is None:
        _pal = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
        colors = {k: c for k, c in zip(transition_dfs_dict, _pal)}

    model_names = list(transition_dfs_dict.keys())
    n_models    = len(model_names)

    dur_bins   = [0, 3, 6, 12, 24]
    dur_labels = ['0–3m', '3–6m', '6–12m', '12–24m']
    bin_mids   = [1.5, 4.5, 9, 18]

    FS_TITLE = 9
    FS_LABEL = 8
    FS_TICK  = 7

    fig_w = 8.5 if latex_ready else max(4.0 * n_models, 10)
    fig_h = 6.0 if latex_ready else 7.5

    fig, axes = plt.subplots(
        2, n_models,
        figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': [1.6, 1.0]},
        squeeze=False,
    )

    for ax_i, model_name in enumerate(model_names):
        col = colors.get(model_name, 'steelblue')
        df  = transition_dfs_dict[model_name].copy()
        df  = df[(df['TransitionType'] == 'UE') & (df['Time Step'] >= delay)]
        df  = df.dropna(subset=['PreSepWage', 'AcceptedWage', 'UEDuration'])
        df['ratio'] = df['AcceptedWage'] / df['PreSepWage'].clip(lower=1e-9)

        # compute sep_date for all workers before splitting
        if start_date is not None:
            sep_ts         = df['Time Step'] - df['UEDuration']
            df['sep_date'] = start_date + pd.to_timedelta((sep_ts - delay) * 30.44, unit='D')
            baseline       = df[(df['sep_date'] < h_start) | (df['sep_date'] > h_end)]
        else:
            baseline = df  # fallback: use full sample as baseline

        # occupation baseline: mean ratio outside shock horizon, re-centre at 1
        # y=1 means "same outcome as this occupation achieves in non-shock times"
        if 'OriginOccupation' in df.columns:
            occ_baseline = baseline.groupby('OriginOccupation')['ratio'].mean()
            df['ratio']  = df['ratio'] - df['OriginOccupation'].map(occ_baseline) + 1.0

        # filter to shock horizon
        if start_date is not None:
            df = df[(df['sep_date'] >= h_start) & (df['sep_date'] <= h_end)]

        df['dur_bin'] = pd.cut(df['UEDuration'], bins=dur_bins,
                               labels=dur_labels, right=False)
        df['shocked'] = df['OriginOccupation'].isin(shocked_set)

        # ── Row 1: wage ratio by duration ──────────────────────────────────────
        ax_wage = axes[0, ax_i]
        _multisim_w5 = 'sim_id' in df.columns
        for grp_flag, ls, lw, alpha_line, alpha_fill, label in [
            (True,  '-',  1.5, 0.90, 0.15, shock_label),
            (False, '--', 1.2, 0.70, 0.10, "Rest"),
        ]:
            sub = df[df['shocked'] == grp_flag]
            if len(sub) < 5:
                continue
            if _multisim_w5:
                sim_bin_means = (sub.groupby(['sim_id', 'dur_bin'], observed=True)['ratio']
                                    .mean().reset_index())
                sim_agg = (sim_bin_means.groupby('dur_bin', observed=True)['ratio']
                                        .agg(['mean', 'sem']))
                means = sim_agg['mean']
                sems  = sim_agg['sem']
            else:
                grp   = sub.groupby('dur_bin', observed=True)['ratio']
                means = grp.mean()
                sems  = grp.sem()
            xs, ys, ses = [], [], []
            for b, mid in zip(dur_labels, bin_mids):
                if b in means.index and pd.notna(means[b]):
                    xs.append(mid)
                    ys.append(means[b])
                    ses.append(sems[b] if pd.notna(sems[b]) else 0.0)
            if len(xs) < 2:
                continue
            lo = [y - 1.96 * se for y, se in zip(ys, ses)]
            hi = [y + 1.96 * se for y, se in zip(ys, ses)]
            ax_wage.fill_between(xs, lo, hi, color=col, alpha=alpha_fill, zorder=2)
            ax_wage.plot(xs, ys, color=col, ls=ls, lw=lw, alpha=alpha_line,
                         zorder=3, label=label)

        ax_wage.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax_wage.set_title(model_name, fontsize=FS_TITLE, fontweight='bold', color=col)
        ax_wage.set_xticks(bin_mids)
        ax_wage.set_xticklabels(dur_labels, fontsize=FS_TICK, rotation=30, ha='right')
        ax_wage.set_xlim(0, 24)
        ax_wage.tick_params(labelsize=FS_TICK)
        ax_wage.spines[['top', 'right']].set_visible(False)
        ax_wage.grid(axis='y', alpha=0.25, lw=0.5)
        if ax_i == 0:
            ax_wage.set_ylabel('Accepted / pre-sep. wage\n(rel. to occ. baseline)', fontsize=FS_LABEL)
        if ax_i == 0:
            ax_wage.legend(fontsize=FS_TICK, frameon=False,
                           loc='lower left', handlelength=1.5)

        # ── Row 2: UE duration boxplots ──────────────────────────────────────
        ax_dur = axes[1, ax_i]
        data_shocked = df.loc[df['shocked'],  'UEDuration'].dropna().values
        data_rest    = df.loc[~df['shocked'], 'UEDuration'].dropna().values

        bp = ax_dur.boxplot(
            [data_shocked, data_rest],
            patch_artist=True,
            widths=0.4,
            medianprops={'color': 'black', 'lw': 1.5},
            flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.3, 'color': col},
            whiskerprops={'lw': 0.8},
            capprops={'lw': 0.8},
        )
        bp['boxes'][0].set_facecolor(col)
        bp['boxes'][0].set_alpha(0.75)
        bp['boxes'][1].set_facecolor('lightgray')
        bp['boxes'][1].set_alpha(0.75)
        ax_dur.set_yscale('log')
        ax_dur.set_xticks([1, 2])
        ax_dur.set_xticklabels(['Shocked', 'Rest'], fontsize=FS_TICK)
        ax_dur.tick_params(labelsize=FS_TICK)
        ax_dur.spines[['top', 'right']].set_visible(False)
        ax_dur.grid(axis='y', alpha=0.25, lw=0.5)
        if ax_i == 0:
            ax_dur.set_ylabel('UE duration (log months)', fontsize=FS_LABEL)

    # suptitle annotation
    win_str = (f" | shock window: {shock_window[0][:7]} – {shock_window[1][:7]}"
               if shock_window else "")
    fig.suptitle(f"Demand-shock episode: {shock_label}{win_str}",
                 fontsize=FS_LABEL, y=1.01)

    plt.tight_layout()

    if save and path:
        fname = 'shock_episode_wages'
        if latex_ready:
            fig.savefig(os.path.join(path, f'{fname}.pdf'), bbox_inches='tight')
        else:
            fig.savefig(os.path.join(path, f'{fname}.jpg'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_occ_wage_trajectories(
    sim_results_dict,
    calib_date,
    benchmark,
    topk=15,
    phase_boundaries=None,
    delay=25,
    save=False,
    path=None,
):
    """
    Mean occupation wage over time, coloured by stress category.

    One panel per model.  Each line = one occupation.
      - Structural stress top-k : red, alpha=0.8, lw=1.2
      - Cyclical stress top-k   : orange, alpha=0.8, lw=1.2
      - Unstressed               : grey, alpha=0.2, lw=0.6

    Stress classification is derived from the benchmark model.

    Parameters
    ----------
    sim_results_dict : dict  {model_name: occupation-level record_df}
    calib_date : list  ['YYYY-MM-DD', 'YYYY-MM-DD']
    benchmark : str
    topk : int
    phase_boundaries : dict, optional
    delay : int
    save : bool
    path : str
    """
    if phase_boundaries is None:
        phase_boundaries = {
            'Pre-GFC (2001–2007)':     (25,  108),
            'GFC recession (2008–09)': (109, 132),
            'Post-GFC (2010–2019)':    (133, 244),
        }

    recession_label = 'GFC recession (2008–09)'
    rec_start, rec_end = phase_boundaries[recession_label]
    exp_ts = set()
    for lbl, (ts0, ts1) in phase_boundaries.items():
        if lbl != recession_label:
            exp_ts.update(range(ts0, ts1 + 1))

    # ── stress classification from benchmark ──────────────────────────────────
    bench_df = sim_results_dict[benchmark].copy()
    bench_df['Occupation'] = pd.to_numeric(bench_df['Occupation'], errors='coerce')
    bench_v  = bench_df[bench_df['Time Step'] >= delay].copy()
    bench_v['UER'] = bench_v['Unemployment'] / bench_v['Workers']

    exp_df = bench_v[bench_v['Time Step'].isin(exp_ts)]
    rec_df = bench_v[(bench_v['Time Step'] >= rec_start) &
                     (bench_v['Time Step'] <= rec_end)]

    struct  = exp_df.groupby('Occupation')['UER'].mean().rename('structural')
    cyclic  = (rec_df.groupby('Occupation')['UER'].mean() - struct).rename('cyclical')
    scores  = pd.concat([struct, cyclic], axis=1).dropna()

    struct_stressed = set(scores.nlargest(topk, 'structural').index)
    cyclic_stressed = set(scores.nlargest(topk, 'cyclical').index)

    def _stress_style(occ_id):
        if occ_id in struct_stressed:
            return 'crimson', 0.75, 1.2
        if occ_id in cyclic_stressed:
            return 'darkorange', 0.75, 1.2
        return 'grey', 0.15, 0.5

    start_date = pd.to_datetime(calib_date[0])

    n = len(sim_results_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes.flatten()

    for ax, (name, df) in zip(axes, sim_results_dict.items()):
        df = df.copy()
        df['Occupation'] = pd.to_numeric(df['Occupation'], errors='coerce')
        df = df[df['Time Step'] >= delay]
        df['mean_wage'] = df['Total_Wages'] / df['Employment']
        df['DATE'] = start_date + pd.to_timedelta(
            (df['Time Step'] - delay) * 30.44, unit='D')

        for occ_id, grp in df.groupby('Occupation'):
            col, alpha, lw = _stress_style(occ_id)
            grp = grp.sort_values('DATE')
            ax.plot(grp['DATE'], grp['mean_wage'] / 1000,
                    color=col, alpha=alpha, linewidth=lw)

        ax.set_yscale('log')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Mean occupation wage ($ thousands, log scale)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Legend
    from matplotlib.lines import Line2D as _L2D
    legend_handles = [
        _L2D([0], [0], color='crimson',    linewidth=1.5, label=f'Structural stress (top {topk})'),
        _L2D([0], [0], color='darkorange', linewidth=1.5, label=f'Cyclical stress (top {topk})'),
        _L2D([0], [0], color='grey',       linewidth=1.0, label='Unstressed', alpha=0.5),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f'Occupation Mean Wage Trajectories (stress from: {benchmark})',
        fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save:
        plt.savefig(f'{path}occ_wage_trajectories.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_tte_by_layoff_date(
    tte_dict,
    calib_date,
    delay=25,
    freq='Q',
    cmap=None,
    demean=True,
    benchmark=None,
    recession_spans=None,
    save=False,
    path=None,
):
    """
    Heatmap of mean time-to-reemployment by layoff date and model.

    Parameters
    ----------
    freq : str
        Pandas period alias for binning: 'Q' (quarterly), 'A' (annual),
        'M' (monthly).  Default 'Q' gives ~92 columns over 23 years.
    """
    start_date = pd.to_datetime(calib_date[0])
    end_date   = pd.to_datetime(calib_date[1])
    model_names = list(tte_dict.keys())

    binned   = {}
    all_bins = set()
    for name, df in tte_dict.items():
        sub = df[['Time Step', 'UEDuration']].copy()
        sub['layoff_ts'] = sub['Time Step'] - sub['UEDuration']
        sub = sub[sub['layoff_ts'] >= delay]
        sub['layoff_date'] = start_date + pd.to_timedelta(
            (sub['layoff_ts'] - delay) * 30.44, unit='D')
        sub = sub[(sub['layoff_date'] >= start_date) &
                  (sub['layoff_date'] <= end_date)]
        sub['bin'] = sub['layoff_date'].dt.to_period(freq)
        agg = sub.groupby('bin')['UEDuration'].mean()
        binned[name] = agg
        all_bins.update(agg.index.tolist())

    all_bins = sorted(all_bins)
    n_bins   = len(all_bins)

    matrix = np.full((len(model_names), n_bins), np.nan)
    for r, name in enumerate(model_names):
        agg = binned[name]
        for c, b in enumerate(all_bins):
            if b in agg.index:
                matrix[r, c] = agg[b]

    normalised = False
    if benchmark is not None and benchmark in model_names:
        bench_idx   = model_names.index(benchmark)
        bench_row   = matrix[bench_idx, :].copy()
        matrix      = matrix - bench_row[np.newaxis, :]
        model_names = [n for n in model_names if n != benchmark]
        matrix      = np.delete(matrix, bench_idx, axis=0)
        normalised  = True
        cbar_label  = f'TTE relative to {benchmark} (months)'
    elif demean:
        matrix     = matrix - np.nanmean(matrix, axis=1, keepdims=True)
        normalised = True
        cbar_label = 'Deviation from model mean (months)'
    else:
        cbar_label = 'Mean months to reemployment'

    if cmap is None:
        cmap = 'RdBu_r' if normalised else 'YlOrRd'

    fig_h = max(2.5, len(model_names) * 0.7 + 1.5)
    fig_w = min(14, max(8, n_bins * 0.3))
    fig = plt.figure(figsize=(fig_w + 1.2, fig_h))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.05)
    ax      = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])

    absmax = np.nanmax(np.abs(matrix)) if normalised else None
    im = ax.imshow(matrix, aspect='auto', cmap=cmap,
                   vmin=-absmax if normalised else np.nanmin(matrix),
                   vmax= absmax if normalised else np.nanmax(matrix))

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=9, rotation=270, labelpad=14)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)

    # Year labels at Q1 / January boundaries
    tick_positions, tick_labels = [], []
    for c, b in enumerate(all_bins):
        if b.start_time.month == 1:
            tick_positions.append(c)
            tick_labels.append(str(b.start_time.year))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    if recession_spans is not None:
        rec = recession_spans[
            (recession_spans['end']   >= start_date) &
            (recession_spans['start'] <= end_date)
        ].copy()
        rec['start'] = rec['start'].clip(lower=start_date)
        rec['end']   = rec['end'].clip(upper=end_date)
        for _, row in rec.iterrows():
            sp = row['start'].to_period(freq)
            ep = row['end'].to_period(freq)
            for c, b in enumerate(all_bins):
                if sp <= b <= ep:
                    for r in range(len(model_names)):
                        ax.add_patch(plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            fill=False, edgecolor='steelblue',
                            linewidth=0.6, zorder=3))

    ax.set_xlabel(f'Layoff date ({_FREQ_LABEL.get(freq, freq)})', fontsize=10)
    title_suffix = (f' vs {benchmark}' if benchmark is not None and benchmark in list(tte_dict.keys())
                    else ' (demeaned per model)' if demean else '')
    ax.set_title(f'Time to Reemployment by Layoff Date{title_suffix}\n'
                 'Blue = faster than reference  |  Red = slower  |  Borders = recessions',
                 fontweight='bold', fontsize=11)
    plt.tight_layout()

    if save:
        plt.savefig(f'{path}tte_by_layoff_date.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def _prep_tte_with_uer(tte_dict, sim_results_dict, benchmark, calib_date, delay, freq):
    """
    Shared preprocessing for TTE-by-local-stress plots.

    For every spell, derives layoff_ts, layoff_date, bin, the origin occupation's
    UER at layoff (from the benchmark sim_results), and the cross-occupation
    quartile thresholds (q25, q50, q75) at that time step.

    Returns
    -------
    model_spells : dict {model_name: DataFrame}
        Columns: Time Step, OriginOccupation, UEDuration, layoff_ts,
                 layoff_date, bin, UER, q25, q50, q75
    all_bins : list of Period, sorted
    bench_binned : dict {model_name: Series (bin -> mean UEDuration)}
        Pre-aggregated (no stress split) for benchmark subtraction.
    """
    start_date = pd.to_datetime(calib_date[0])
    end_date   = pd.to_datetime(calib_date[1])

    # UER lookup from benchmark sim_results
    bench_sim = sim_results_dict[benchmark].copy()
    bench_sim['Occupation'] = pd.to_numeric(bench_sim['Occupation'], errors='coerce')
    bench_sim['UER'] = bench_sim['Unemployment'] / bench_sim['Workers']
    uer_lookup = bench_sim[['Time Step', 'Occupation', 'UER']].copy()

    # Cross-occupation quartile thresholds per time step
    period_thresholds = (uer_lookup
        .groupby('Time Step')['UER']
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: 'q25', 0.5: 'q50', 0.75: 'q75'})
        .reset_index())

    uer_at_layoff = uer_lookup.rename(
        columns={'Time Step': 'layoff_ts', 'Occupation': 'OriginOccupation'})

    model_spells = {}
    all_bins     = set()

    for name, df in tte_dict.items():
        sub = df[['Time Step', 'OriginOccupation', 'UEDuration']].copy()
        sub['layoff_ts'] = sub['Time Step'] - sub['UEDuration']
        sub = sub[sub['layoff_ts'] >= delay]

        sub['layoff_date'] = start_date + pd.to_timedelta(
            (sub['layoff_ts'] - delay) * 30.44, unit='D')
        sub = sub[(sub['layoff_date'] >= start_date) &
                  (sub['layoff_date'] <= end_date)]
        sub['bin'] = sub['layoff_date'].dt.to_period(freq)

        sub['OriginOccupation'] = pd.to_numeric(sub['OriginOccupation'], errors='coerce')
        sub = sub.merge(uer_at_layoff, on=['layoff_ts', 'OriginOccupation'], how='left')
        sub = sub.merge(period_thresholds.rename(columns={'Time Step': 'layoff_ts'}),
                        on='layoff_ts', how='left')

        model_spells[name] = sub
        all_bins.update(sub['bin'].dropna().tolist())

    all_bins = sorted(all_bins)
    return model_spells, all_bins


def _build_stress_heatmap(model_spells, model_names, all_bins,
                           groups, benchmark, fig_title,
                           cmap, recession_spans, calib_date,
                           freq, save, path, fname):
    """
    Shared plotting logic for both stress-split TTE heatmaps.

    groups : list of (panel_label, boolean_mask_fn)
        mask_fn takes a DataFrame and returns a boolean Series selecting spells
        that belong to this stress group.
    """
    n_groups = len(groups)
    n_bins   = len(all_bins)
    bin_idx  = {b: i for i, b in enumerate(all_bins)}

    # ── build one matrix per group ────────────────────────────────────────────
    matrices = {}
    for label, mask_fn in groups:
        mat = np.full((len(model_names), n_bins), np.nan)
        for r, name in enumerate(model_names):
            sub  = model_spells[name]
            sub  = sub[mask_fn(sub)]
            agg  = sub.groupby('bin')['UEDuration'].mean()
            for c, b in enumerate(all_bins):
                if b in agg.index:
                    mat[r, c] = agg[b]
        matrices[label] = mat

    # ── benchmark subtraction ─────────────────────────────────────────────────
    bench_idx   = model_names.index(benchmark)
    plot_names  = [n for n in model_names if n != benchmark]
    plot_rows   = [i for i, n in enumerate(model_names) if n != benchmark]

    for label in matrices:
        bench_row       = matrices[label][bench_idx, :].copy()
        matrices[label] = matrices[label][plot_rows, :] - bench_row[np.newaxis, :]

    # ── figure ────────────────────────────────────────────────────────────────
    panel_h = max(2.0, len(plot_names) * 0.6 + 1.0)
    fig_h   = panel_h * n_groups
    fig_w   = min(14, max(8, n_bins * 0.4))
    fig = plt.figure(figsize=(fig_w + 1.2, fig_h))
    gs  = fig.add_gridspec(n_groups, 2, width_ratios=[1, 0.04], wspace=0.05, hspace=0.35)
    axes    = [fig.add_subplot(gs[i, 0]) for i in range(n_groups)]
    cbar_ax = fig.add_subplot(gs[:, 1])
    for i in range(1, n_groups):
        axes[i].sharex(axes[0])

    # Pre-compute recession periods once
    rec_periods = []
    if recession_spans is not None:
        cal_start = pd.to_datetime(calib_date[0])
        cal_end   = pd.to_datetime(calib_date[1])
        rec = recession_spans[
            (recession_spans['end']   >= cal_start) &
            (recession_spans['start'] <= cal_end)
        ].copy()
        rec['start'] = rec['start'].clip(lower=cal_start)
        rec['end']   = rec['end'].clip(upper=cal_end)
        for _, row in rec.iterrows():
            rec_periods.append((
                row['start'].to_period(freq),
                row['end'].to_period(freq),
            ))

    absmax = max(np.nanmax(np.abs(m)) for m in matrices.values())

    for ax, (label, _) in zip(axes, groups):
        mat = matrices[label]
        im  = ax.imshow(mat, aspect='auto', cmap=cmap,
                        vmin=-absmax, vmax=absmax)
        ax.set_title(label, fontweight='bold', fontsize=10)
        ax.set_yticks(range(len(plot_names)))
        ax.set_yticklabels(plot_names, fontsize=8)

        # Recession column borders
        for sp, ep in rec_periods:
            for c, b in enumerate(all_bins):
                if sp <= b <= ep:
                    for r in range(len(plot_names)):
                        ax.add_patch(plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            fill=False, edgecolor='steelblue',
                            linewidth=0.6, zorder=3))

    # x-axis ticks only on the bottom panel (sharex=True)
    tick_pos, tick_lab = [], []
    for c, b in enumerate(all_bins):
        if b.start_time.month == 1:
            tick_pos.append(c)
            tick_lab.append(str(b.start_time.year))
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_lab, rotation=45, ha='right', fontsize=8)
    axes[-1].set_xlabel(f'Layoff date ({_FREQ_LABEL.get(freq, freq)})', fontsize=9)

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'TTE relative to {benchmark} (months)', fontsize=9, rotation=270, labelpad=14)

    fig.suptitle(fig_title, fontweight='bold', fontsize=11, y=1.01)

    if save:
        plt.savefig(f'{path}{fname}', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_tte_by_tightness(
    tte_dict,
    sim_results_dict,
    calib_date,
    benchmark,
    delay=25,
    freq=6,
    cmap='RdBu_r',
    recession_spans=None,
    save=False,
    path=None,
):
    """
    TTE heatmap split by whether the origin occupation's market was slack or
    tight at the time of layoff (Option B).

    Tightness is proxied by UER: occupations above the cross-occupation median
    UER at the layoff time step are classified as 'tight market' (worker faces
    more competition); those below are 'slack market' (easier re-entry).
    Stress classification uses the benchmark model's sim_results so that the
    same spell is classified identically across models.

    Cell colour = TTE relative to benchmark model (blue = faster, red = slower).
    """
    model_names  = list(tte_dict.keys())
    model_spells, all_bins = _prep_tte_with_uer(
        tte_dict, sim_results_dict, benchmark, calib_date, delay, freq)

    groups = [
        ('Slack market (UER ≤ median)',  lambda d: d['UER'] <= d['q50']),
        ('Tight market (UER > median)',  lambda d: d['UER'] >  d['q50']),
    ]

    _build_stress_heatmap(
        model_spells  = model_spells,
        model_names   = model_names,
        all_bins      = all_bins,
        groups        = groups,
        benchmark     = benchmark,
        fig_title     = ('Time to Reemployment by Local Market Tightness at Layoff\n'
                         'Blue = faster than benchmark  |  Red = slower  |  Borders = recessions'),
        cmap          = cmap,
        recession_spans = recession_spans,
        calib_date    = calib_date,
        freq    = freq,
        save          = save,
        path          = path,
        fname         = 'tte_by_tightness.png',
    )


def plot_tte_by_uer_quantile(
    tte_dict,
    sim_results_dict,
    calib_date,
    benchmark,
    delay=25,
    freq=6,
    cmap='RdBu_r',
    recession_spans=None,
    save=False,
    path=None,
):
    """
    TTE heatmap split into four UER quantile groups at the time of layoff
    (Option C).

    Each period, occupations are ranked by UER and assigned to quartiles.
    The four panels — Low, Moderate, High, Very High UER — show whether the
    behavioural model's advantage concentrates among workers displaced from
    the most stressed occupations.

    Cell colour = TTE relative to benchmark model (blue = faster, red = slower).
    """
    model_names  = list(tte_dict.keys())
    model_spells, all_bins = _prep_tte_with_uer(
        tte_dict, sim_results_dict, benchmark, calib_date, delay, freq)

    groups = [
        ('Low UER (Q1)',       lambda d: d['UER'] <= d['q25']),
        ('Moderate UER (Q2)',  lambda d: (d['UER'] > d['q25']) & (d['UER'] <= d['q50'])),
        ('High UER (Q3)',      lambda d: (d['UER'] > d['q50']) & (d['UER'] <= d['q75'])),
        ('Very high UER (Q4)', lambda d: d['UER'] > d['q75']),
    ]

    _build_stress_heatmap(
        model_spells  = model_spells,
        model_names   = model_names,
        all_bins      = all_bins,
        groups        = groups,
        benchmark     = benchmark,
        fig_title     = ('Time to Reemployment by Origin-Occupation UER Quartile at Layoff\n'
                         'Blue = faster than benchmark  |  Red = slower  |  Borders = recessions'),
        cmap          = cmap,
        recession_spans = recession_spans,
        calib_date    = calib_date,
        freq    = freq,
        save          = save,
        path          = path,
        fname         = 'tte_by_uer_quantile.png',
    )


def _prep_wage_with_uer(sim_results_dict, benchmark, calib_date, delay, freq,
                        index_by='rehire', tte_dict=None):
    """
    Shared preprocessing for wage heatmap plots.

    Parameters
    ----------
    index_by : 'rehire' (default) or 'layoff'
        'rehire' bins wages by the time step at which the worker found a job.
        'layoff' bins wages by the estimated layoff time step, computed as
        Time Step minus the average UE duration for that (Time Step, Occupation)
        pair from tte_dict.  Requires tte_dict when index_by='layoff'.
    tte_dict : dict {model_name: DataFrame} or None
        Required when index_by='layoff'.  Each DataFrame must have columns
        ['Time Step', 'OriginOccupation', 'UEDuration'].

    Returns
    -------
    model_dfs : dict {model_name: DataFrame}
    all_bins  : list of Period, sorted
    date_col_label : str  — 'Layoff date' or 'Reemployment date'
    """
    start_date = pd.to_datetime(calib_date[0])
    end_date   = pd.to_datetime(calib_date[1])

    bench = sim_results_dict[benchmark].copy()
    bench = bench[bench['Time Step'] >= delay].copy()
    bench['Occupation'] = pd.to_numeric(bench['Occupation'], errors='coerce')
    bench['UER'] = bench['Unemployment'] / bench['Workers']

    # UER quartile thresholds — indexed by whatever time step we bin on
    if index_by == 'layoff' and tte_dict is not None and benchmark in tte_dict:
        bench_tte = tte_dict[benchmark].copy()
        bench_tte['OriginOccupation'] = pd.to_numeric(
            bench_tte['OriginOccupation'], errors='coerce')
        avg_dur_bench = (bench_tte
            .groupby(['Time Step', 'OriginOccupation'])['UEDuration']
            .mean().reset_index()
            .rename(columns={'OriginOccupation': 'Occupation',
                             'UEDuration': 'avg_duration'}))
        bench_m = bench[['Time Step', 'Occupation', 'UER']].merge(
            avg_dur_bench, on=['Time Step', 'Occupation'], how='left')
        bench_m['ref_ts'] = (
            bench_m['Time Step'] - bench_m['avg_duration'].fillna(0)
        ).round().astype(int).clip(lower=delay)
        period_thresholds = (bench_m.groupby('ref_ts')['UER']
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .rename(columns={0.25: 'q25', 0.5: 'q50', 0.75: 'q75'})
            .reset_index()
            .rename(columns={'ref_ts': 'ref_ts'}))
        threshold_key = 'ref_ts'
        date_col_label = 'Layoff date'
    else:
        period_thresholds = (bench.groupby('Time Step')['UER']
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .rename(columns={0.25: 'q25', 0.5: 'q50', 0.75: 'q75'})
            .reset_index())
        threshold_key = 'Time Step'
        date_col_label = 'Reemployment date'

    model_dfs = {}
    all_bins  = set()

    for name, df in sim_results_dict.items():
        sub = df[df['Time Step'] >= delay].copy()
        sub['Occupation'] = pd.to_numeric(sub['Occupation'], errors='coerce')
        sub['UER'] = sub['Unemployment'] / sub['Workers']

        if index_by == 'layoff' and tte_dict is not None and name in tte_dict:
            tte = tte_dict[name].copy()
            tte['OriginOccupation'] = pd.to_numeric(
                tte['OriginOccupation'], errors='coerce')
            avg_dur = (tte
                .groupby(['Time Step', 'OriginOccupation'])['UEDuration']
                .mean().reset_index()
                .rename(columns={'OriginOccupation': 'Occupation',
                                 'UEDuration': 'avg_duration'}))
            sub = sub.merge(avg_dur, on=['Time Step', 'Occupation'], how='left')
            sub['ref_ts'] = (
                sub['Time Step'] - sub['avg_duration'].fillna(0)
            ).round().astype(int).clip(lower=delay)
            sub['date'] = start_date + pd.to_timedelta(
                (sub['ref_ts'] - delay) * 30.44, unit='D')
            sub = sub.merge(period_thresholds, on='ref_ts', how='left')
        else:
            sub['ref_ts'] = sub['Time Step']
            sub['date'] = start_date + pd.to_timedelta(
                (sub['Time Step'] - delay) * 30.44, unit='D')
            sub = sub.merge(period_thresholds, on=threshold_key, how='left')

        sub = sub[(sub['date'] >= start_date) & (sub['date'] <= end_date)]
        sub['bin'] = sub['date'].dt.to_period(freq)

        model_dfs[name] = sub
        all_bins.update(sub['bin'].dropna().tolist())

    return model_dfs, sorted(all_bins), date_col_label


def _build_wage_matrix(model_dfs, model_names, all_bins, mask_fn=None):
    """
    Build a (n_models × n_bins) matrix of weighted mean wage scar.
    mask_fn : callable(df) -> boolean Series, or None for all rows.
    """
    matrix = np.full((len(model_names), len(all_bins)), np.nan)
    for r, name in enumerate(model_names):
        sub = model_dfs[name]
        if mask_fn is not None:
            sub = sub[mask_fn(sub)]
        agg = (sub.groupby('bin')
               .apply(lambda g: g['U_Rel_Wage'].sum() / g['UE_Transitions'].sum()
                      if g['UE_Transitions'].sum() > 0 else np.nan))
        for c, b in enumerate(all_bins):
            if b in agg.index:
                val = agg[b]
                if not (isinstance(val, float) and np.isnan(val)):
                    matrix[r, c] = val
    return matrix


def _plot_wage_panels(matrices_list, group_labels, plot_names, all_bins,
                      benchmark, cmap, recession_spans, calib_date,
                      freq, fig_title, save, path, fname,
                      date_col_label='Reemployment date'):
    """Shared panel-stacking plotter for wage heatmap functions."""
    n_groups = len(matrices_list)
    n_bins   = len(all_bins)

    absmax = max(np.nanmax(np.abs(m)) for m in matrices_list)

    panel_h = max(2.0, len(plot_names) * 0.6 + 1.0)
    fig_w   = min(14, max(8, n_bins * 0.4))
    fig = plt.figure(figsize=(fig_w + 1.2, panel_h * n_groups))
    gs  = fig.add_gridspec(n_groups, 2,
                           width_ratios=[1, 0.04],
                           wspace=0.05,
                           hspace=0.35)
    axes   = [fig.add_subplot(gs[i, 0]) for i in range(n_groups)]
    cbar_ax = fig.add_subplot(gs[:, 1])

    # Pre-compute recession periods once
    rec_periods = []
    if recession_spans is not None:
        cal_start = pd.to_datetime(calib_date[0])
        cal_end   = pd.to_datetime(calib_date[1])
        rec = recession_spans[
            (recession_spans['end']   >= cal_start) &
            (recession_spans['start'] <= cal_end)
        ].copy()
        rec['start'] = rec['start'].clip(lower=cal_start)
        rec['end']   = rec['end'].clip(upper=cal_end)
        for _, row in rec.iterrows():
            rec_periods.append((
                row['start'].to_period(freq),
                row['end'].to_period(freq),
            ))

    for i, (ax, mat, label) in enumerate(zip(axes, matrices_list, group_labels)):
        if i > 0:
            ax.sharex(axes[0])
        im = ax.imshow(mat, aspect='auto', cmap=cmap,
                       vmin=-absmax, vmax=absmax)
        ax.set_title(label, fontweight='bold', fontsize=10)
        ax.set_yticks(range(len(plot_names)))
        ax.set_yticklabels(plot_names, fontsize=8)

        for sp, ep in rec_periods:
            for c, b in enumerate(all_bins):
                if sp <= b <= ep:
                    for r in range(len(plot_names)):
                        ax.add_patch(plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            fill=False, edgecolor='steelblue',
                            linewidth=0.6, zorder=3))

    tick_pos, tick_lab = [], []
    for c, b in enumerate(all_bins):
        if b.start_time.month == 1:
            tick_pos.append(c)
            tick_lab.append(str(b.start_time.year))
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_lab, rotation=45, ha='right', fontsize=8)
    axes[-1].set_xlabel(f'{date_col_label} ({_FREQ_LABEL.get(freq, freq)})', fontsize=9)

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'Wage scar relative to {benchmark}', fontsize=9, rotation=270, labelpad=14)

    fig.suptitle(fig_title, fontweight='bold', fontsize=11, y=1.01)

    if save:
        plt.savefig(f'{path}{fname}', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_wage_rehire_heatmap(
    sim_results_dict,
    calib_date,
    benchmark,
    delay=25,
    freq=6,
    cmap='RdBu_r',
    recession_spans=None,
    index_by='layoff',
    tte_dict=None,
    save=False,
    path=None,
):
    """
    Economy-wide re-employment wage heatmap relative to benchmark.

    One row per non-benchmark model; columns are quarterly (or freq) bins.
    Cell colour = mean wage scar (U_Rel_Wage / UE_Transitions) minus
    the benchmark model's value for the same period.

    index_by : 'rehire' (default) or 'layoff'
        Whether to bin wages by the date the worker found a job or the
        estimated date they were laid off.  Pass tte_dict when 'layoff'.

    Blue = higher re-employment wage than benchmark.
    Red  = lower re-employment wage than benchmark.
    """
    model_names = list(sim_results_dict.keys())
    model_dfs, all_bins, date_col_label = _prep_wage_with_uer(
        sim_results_dict, benchmark, calib_date, delay, freq,
        index_by=index_by, tte_dict=tte_dict)

    matrix    = _build_wage_matrix(model_dfs, model_names, all_bins)
    bench_idx = model_names.index(benchmark)
    bench_row = matrix[bench_idx, :].copy()
    plot_names = [n for n in model_names if n != benchmark]
    keep_rows  = [i for i, n in enumerate(model_names) if n != benchmark]
    matrix     = matrix[keep_rows, :] - bench_row[np.newaxis, :]

    date_desc = 'Layoff date' if index_by == 'layoff' else 'U→E Rehire date'
    _plot_wage_panels(
        matrices_list   = [matrix],
        group_labels    = ['All occupations'],
        plot_names      = plot_names,
        all_bins        = all_bins,
        benchmark       = benchmark,
        cmap            = cmap,
        recession_spans = recession_spans,
        calib_date      = calib_date,
        freq            = freq,
        fig_title       = (f'Re-employment Wage by {date_desc} vs {benchmark}\n'
                           'Blue = higher wage  |  Red = lower  |  Borders = recessions'),
        save            = save,
        path            = path,
        fname           = f'wage_rehire_heatmap_{index_by}.png',
        date_col_label  = date_col_label,
    )


def plot_wage_rehire_by_uer_quantile(
    sim_results_dict,
    calib_date,
    benchmark,
    delay=25,
    freq=6,
    cmap='RdBu_r',
    recession_spans=None,
    index_by='layoff',
    tte_dict=None,
    save=False,
    path=None,
):
    """
    Re-employment wage heatmap split by origin-occupation UER quartile.

    Four stacked panels — Low, Moderate, High, Very High UER.

    index_by : 'rehire' (default) or 'layoff'
        Whether to bin wages by rehire date or estimated layoff date.
        When 'layoff', UER quartile thresholds are also evaluated at the
        estimated layoff time step.  Pass tte_dict when 'layoff'.
    """
    model_names = list(sim_results_dict.keys())
    model_dfs, all_bins, date_col_label = _prep_wage_with_uer(
        sim_results_dict, benchmark, calib_date, delay, freq,
        index_by=index_by, tte_dict=tte_dict)

    groups = [
        ('Low UER (Q1)',       lambda d: d['UER'] <= d['q25']),
        ('Moderate UER (Q2)',  lambda d: (d['UER'] > d['q25']) & (d['UER'] <= d['q50'])),
        ('High UER (Q3)',      lambda d: (d['UER'] > d['q50']) & (d['UER'] <= d['q75'])),
        ('Very high UER (Q4)', lambda d: d['UER'] > d['q75']),
    ]

    bench_idx  = model_names.index(benchmark)
    plot_names = [n for n in model_names if n != benchmark]
    keep_rows  = [i for i, n in enumerate(model_names) if n != benchmark]

    matrices_list = []
    group_labels  = []
    for label, mask_fn in groups:
        mat       = _build_wage_matrix(model_dfs, model_names, all_bins, mask_fn)
        bench_row = mat[bench_idx, :].copy()
        mat       = mat[keep_rows, :] - bench_row[np.newaxis, :]
        matrices_list.append(mat)
        group_labels.append(label)

    date_desc = 'Layoff date' if index_by == 'layoff' else 'Reemployment date'
    uer_desc  = ('UER at layoff' if index_by == 'layoff' else
                 'Origin-Occupation UER at rehire')
    _plot_wage_panels(
        matrices_list   = matrices_list,
        group_labels    = group_labels,
        plot_names      = plot_names,
        all_bins        = all_bins,
        benchmark       = benchmark,
        cmap            = cmap,
        recession_spans = recession_spans,
        calib_date      = calib_date,
        freq            = freq,
        fig_title       = (f'Re-employment Wage by {uer_desc} Quartile vs {benchmark}\n'
                           f'Blue = higher wage  |  Red = lower  |  Borders = recessions'),
        save            = save,
        path            = path,
        fname           = f'wage_rehire_by_uer_quantile_{index_by}.png',
        date_col_label  = date_col_label,
    )


def plot_wage_scar_by_stress(
    sim_results_dict,
    calib_date,
    benchmark,
    topk=15,
    phase_boundaries=None,
    delay=25,
    lowess_frac=0.15,
    colors=None,
    recession_spans=None,
    occ_crosswalk=None,
    soc_labs=None,
    save=False,
    path=None,
):
    """
    Time series of mean wage scar (U→E accepted wage / pre-unemployment wage)
    separately for structurally stressed, cyclically stressed, and unstressed
    occupations.  One panel per stress category, all models overlaid.

    Stress classification uses the same logic as plot_hotspot_quadrant:
      structural stress  = mean occupation UER during expansion phases
      cyclical stress    = mean UER during GFC recession − structural score
    Top-k occupations by each dimension are labelled 'stressed'; the rest
    are 'unstressed'.

    Parameters
    ----------
    sim_results_dict : dict  {model_name: occupation-level record_df}
        Must contain columns: Time Step, Occupation, U_Rel_Wage, UE_Transitions,
        Unemployment, Workers.
    calib_date : list  ['YYYY-MM-DD', 'YYYY-MM-DD']
    benchmark : str
        Model name used to derive the stress classification (stress labels are
        fixed to this model so comparisons across models are consistent).
    topk : int
        Number of occupations classified as stressed in each dimension.
    phase_boundaries : dict, optional
        {label: (ts_start, ts_end)}.  Defaults to GFC phases.
    delay : int
        Burn-in steps to exclude (should match calibration delay).
    lowess_frac : float
        Smoothing bandwidth for LOWESS trend lines.
    colors : dict, optional  {model_name: colour}
    recession_spans : pd.DataFrame, optional
        Recession shading — must have columns 'start' and 'end' (datetime).
    save : bool
    path : str
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if colors is None:
        defaults = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
        colors = {name: defaults[i % len(defaults)]
                  for i, name in enumerate(sim_results_dict)}

    if phase_boundaries is None:
        phase_boundaries = {
            'Pre-GFC (2001–2007)':     (25,  108),
            'GFC recession (2008–09)': (109, 132),
            'Post-GFC (2010–2019)':    (133, 244),
        }

    recession_label = 'GFC recession (2008–09)'
    rec_start, rec_end = phase_boundaries[recession_label]
    exp_ts = set()
    for lbl, (ts0, ts1) in phase_boundaries.items():
        if lbl != recession_label:
            exp_ts.update(range(ts0, ts1 + 1))

    # ── derive stress classification from benchmark model ──────────────────────
    bench_df = sim_results_dict[benchmark]
    bench_v  = bench_df[bench_df['Time Step'] >= delay].copy()
    bench_v['Occupation'] = pd.to_numeric(bench_v['Occupation'], errors='coerce')
    bench_v['UER'] = bench_v['Unemployment'] / bench_v['Workers']

    exp_df  = bench_v[bench_v['Time Step'].isin(exp_ts)]
    rec_df  = bench_v[(bench_v['Time Step'] >= rec_start) &
                      (bench_v['Time Step'] <= rec_end)]

    struct = exp_df.groupby('Occupation')['UER'].mean().rename('structural')
    cyclic = (rec_df.groupby('Occupation')['UER'].mean() - struct).rename('cyclical')
    scores = pd.concat([struct, cyclic], axis=1).dropna()

    struct_stressed = set(scores.nlargest(topk, 'structural').index)
    cyclic_stressed = set(scores.nlargest(topk, 'cyclical').index)
    all_stressed    = struct_stressed | cyclic_stressed
    unstressed      = set(scores.index) - all_stressed

    stress_groups = {
        f'Structural stress (top {topk})': struct_stressed,
        f'Cyclical stress (top {topk})':   cyclic_stressed,
        f'Unstressed':                     unstressed,
    }

    # ── build occupation label lookup (optional) ──────────────────────────────
    # occupation_id in sim = X - 1 (X is 1-indexed row in crosswalk)
    occ_label_map = {}
    if occ_crosswalk is not None and soc_labs is not None:
        cw = occ_crosswalk[['X', 'label']].drop_duplicates('X').copy()
        cw['occupation_id'] = pd.to_numeric(cw['X'], errors='coerce').astype('Int64') - 1
        cw = cw.dropna(subset=['occupation_id'])
        occ_label_map = dict(zip(cw['occupation_id'], cw['label']))

    show_key = (occ_crosswalk is not None and soc_labs is not None)

    # ── build date axis from Time Step ────────────────────────────────────────
    start_date = pd.to_datetime(calib_date[0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, (group_label, occ_set) in zip(axes, stress_groups.items()):
        for name, df in sim_results_dict.items():
            col = colors.get(name, 'steelblue')

            sub = df[df['Occupation'].isin(occ_set) &
                     (df['Time Step'] >= delay)].copy()

            if sub.empty or sub['UE_Transitions'].sum() == 0:
                continue

            # Weighted mean wage scar per time step
            ts_agg = (sub.groupby('Time Step')
                        .apply(lambda g: g['U_Rel_Wage'].sum() /
                               g['UE_Transitions'].sum()
                               if g['UE_Transitions'].sum() > 0 else np.nan)
                        .reset_index(name='wage_scar'))

            ts_agg['DATE'] = start_date + pd.to_timedelta(
                (ts_agg['Time Step'] - delay) * 30.44, unit='D')

            ts_agg = ts_agg.dropna(subset=['wage_scar'])
            if ts_agg.empty:
                continue

            # Raw scatter (faint)
            ax.plot(ts_agg['DATE'], ts_agg['wage_scar'],
                    color=col, alpha=0.15, linewidth=0.6)

            # LOWESS trend
            trend = lowess(ts_agg['wage_scar'].values,
                           ts_agg['DATE'].astype(np.int64).values,
                           frac=lowess_frac, return_sorted=True)
            trend_dates = pd.to_datetime(trend[:, 0].astype(np.int64))
            ax.plot(trend_dates, trend[:, 1],
                    color=col, linewidth=2, label=name)

        # Recession shading (clipped to calib window so axvspan doesn't expand x-axis)
        if recession_spans is not None:
            cal_start = pd.to_datetime(calib_date[0])
            cal_end   = pd.to_datetime(calib_date[1])
            rec_clipped = recession_spans[
                (recession_spans['end']   >= cal_start) &
                (recession_spans['start'] <= cal_end)
            ].copy()
            rec_clipped['start'] = rec_clipped['start'].clip(lower=cal_start)
            rec_clipped['end']   = rec_clipped['end'].clip(upper=cal_end)
            for _, row in rec_clipped.iterrows():
                ax.axvspan(row['start'], row['end'],
                           alpha=0.12, color='grey', zorder=0)

        ax.axhline(1.0, color='black', linewidth=0.8, linestyle='dotted',
                   label='No wage change')
        ax.set_ylim(0.75, 1.8)
        ax.set_title(group_label, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Ratio of Accepted Wage to Pre-Separation Wage')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    handles, labels = axes[0].get_legend_handles_labels()

    # ── occupation key below corresponding panels ─────────────────────────────
    if show_key:
        def _occ_labels(occ_set):
            return [occ_label_map.get(oid, f'OCC {oid}') for oid in sorted(occ_set)]

        struct_names = _occ_labels(struct_stressed)
        cyclic_names = _occ_labels(cyclic_stressed)
        max_rows = max(len(struct_names), len(cyclic_names))

        struct_str = (f"Structural (top {topk})\n" +
                      '\n'.join(f'• {n}' for n in struct_names))
        cyclic_str = (f"Cyclical (top {topk})\n" +
                      '\n'.join(f'• {n}' for n in cyclic_names))

        axes[0].text(0.5, -0.28, struct_str,
                     transform=axes[0].transAxes,
                     ha='center', va='top', fontsize=6.5,
                     color='crimson', linespacing=1.4, clip_on=False,
                     fontweight='normal',
                     bbox=dict(boxstyle='round,pad=0.3', fc='#fff5f5', ec='crimson',
                               alpha=0.5, lw=0.5))
        axes[0].text(0.5, -0.28, f"Structural (top {topk})",
                     transform=axes[0].transAxes,
                     ha='center', va='top', fontsize=6.5,
                     color='crimson', fontweight='bold', clip_on=False)

        axes[1].text(0.5, -0.28, cyclic_str,
                     transform=axes[1].transAxes,
                     ha='center', va='top', fontsize=6.5,
                     color='darkorange', linespacing=1.4, clip_on=False,
                     bbox=dict(boxstyle='round,pad=0.3', fc='#fff8f0', ec='darkorange',
                               alpha=0.5, lw=0.5))
        axes[1].text(0.5, -0.28, f"Cyclical (top {topk})",
                     transform=axes[1].transAxes,
                     ha='center', va='top', fontsize=6.5,
                     color='darkorange', fontweight='bold', clip_on=False)

        key_rows = max_rows + 1
        tight_bottom = -(key_rows * 0.048 + 0.12)
    else:
        tight_bottom = 0.08

    axes[1].legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15),
                   ncol=min(len(sim_results_dict) + 1, 5),
                   fontsize=9, framealpha=0.8)

    fig.suptitle(
        'Wage Scarring at U→E Rehire by Occupation Stress Category\n'
        f'(stress classification from benchmark: {benchmark})',
        fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, tight_bottom, 1, 0.95])

    if save:
        plt.savefig(f'{path}wage_scar_by_stress.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_wages_by_age(
    wage_age_dict,
    calib_date,
    age_bin_width=5,
    recession_years=None,
    overlay=False,
    colors=None,
    save=False,
    path=None,
):
    """
    Cross-sectional wage-age profiles by business cycle phase.

    Args:
        wage_age_dict (dict): {model_name: wage_age_df} with columns
            ['Time Step', 'Year', 'Occupation', 'Worker_ID', 'Wage', 'Age'].
        calib_date (list): ['YYYY-MM-DD', 'YYYY-MM-DD'].
        age_bin_width (int): Width of age bins in years (default 5).
        recession_years (list[int] | None): 0-based simulation year indices that
            fall in recession.  Derive externally from NBER dates and calib_date.
        overlay (bool): If True, all models are drawn on a single panel (or two
            side-by-side panels when recession_years is provided) so steepness
            can be compared directly.  If False (default), one panel per model.
        colors (dict | None): model name -> colour.
        save (bool): Save figure if True.
        path (str | None): Output directory.
    """
    if colors is None:
        defaults = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
        colors = {name: defaults[i % len(defaults)]
                  for i, name in enumerate(wage_age_dict)}

    bins = np.arange(18, 70, age_bin_width)
    bin_labels = [f'{b}–{b + age_bin_width - 1}' for b in bins]
    x = range(len(bin_labels))

    def _profile(df):
        p = (df.groupby('age_bin', observed=True)['Wage']
               .agg(['mean', 'std', 'count'])
               .reindex(bin_labels)
               .reset_index())
        p['se'] = p['std'] / np.sqrt(p['count'])
        return p

    def _draw(ax, profile, col, label, ls='-', alpha=0.15):
        ax.plot(x, profile['mean'] / 1000, color=col, linestyle=ls,
                linewidth=1.8, label=label)
        ax.fill_between(x,
                        (profile['mean'] - profile['se']) / 1000,
                        (profile['mean'] + profile['se']) / 1000,
                        color=col, alpha=alpha)

    def _style_ax(ax, title):
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Age group')
        ax.set_ylabel('Mean wage ($ thousands)')
        ax.set_title(title, fontweight='bold')

    if overlay:
        n_panels = 2 if recession_years is not None else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
        ax_exp = axes[0, 0]
        ax_rec = axes[0, 1] if n_panels == 2 else None

        for name, df in wage_age_dict.items():
            col = colors.get(name, 'steelblue')
            df = df.copy()
            df['age_bin'] = pd.cut(df['Age'], bins=list(bins) + [70],
                                   labels=bin_labels, right=False)

            if recession_years is not None:
                exp_df = df[~df['Year'].isin(recession_years)]
                rec_df = df[ df['Year'].isin(recession_years)]
                if not exp_df.empty:
                    _draw(ax_exp, _profile(exp_df), col, name)
                if not rec_df.empty and ax_rec is not None:
                    _draw(ax_rec, _profile(rec_df), col, name)
            else:
                _draw(ax_exp, _profile(df), col, name)

        _style_ax(ax_exp, 'Expansion' if recession_years is not None else 'All years')
        ax_exp.legend(fontsize=8, loc='upper left')
        if ax_rec is not None:
            _style_ax(ax_rec, 'Recession')
            ax_rec.legend(fontsize=8, loc='upper left')

    else:
        n = len(wage_age_dict)
        cols_n = min(n, 3)
        rows = math.ceil(n / cols_n)
        fig, axes = plt.subplots(rows, cols_n, figsize=(6 * cols_n, 4 * rows),
                                 squeeze=False)
        axes = axes.flatten()

        for i, (name, df) in enumerate(wage_age_dict.items()):
            ax = axes[i]
            col = colors.get(name, 'steelblue')
            df = df.copy()
            df['age_bin'] = pd.cut(df['Age'], bins=list(bins) + [70],
                                   labels=bin_labels, right=False)

            if recession_years is not None:
                exp_df = df[~df['Year'].isin(recession_years)]
                rec_df = df[ df['Year'].isin(recession_years)]
                for subset, label, ls in [
                    (exp_df, 'Expansion', '-'),
                    (rec_df, 'Recession', '--'),
                ]:
                    if not subset.empty:
                        _draw(ax, _profile(subset), col, label, ls=ls)
                ax.legend(fontsize=8)
            else:
                _draw(ax, _profile(df), col, name, alpha=0.2)

            _style_ax(ax, name)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

    fig.suptitle('Cross-sectional Wage–Age Profile',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig(os.path.join(path, 'wages_by_age.png'), bbox_inches='tight',
                    pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_job_ladders(
    wage_age_dict,
    calib_date,
    n_trajectories=50,
    min_years_observed=5,
    colors=None,
    save=False,
    path=None,
):
    """
    Job ladder visualisation: individual wage trajectories over the career.

    Plots a random sample of workers who appear in at least `min_years_observed`
    annual snapshots, with each trajectory normalised to the worker's first
    observed wage (so lines show proportional growth from entry).  A LOWESS
    trend is overlaid.

    Requires wage_age_df to contain a 'Worker_ID' column (added to model_fun.py).
    DataFrames missing this column are skipped with an informative message.

    Args:
        wage_age_dict (dict): {model_name: wage_age_df}.
        calib_date (list): ['YYYY-MM-DD', 'YYYY-MM-DD'].
        n_trajectories (int): Max trajectories per model (default 50).
        min_years_observed (int): Minimum annual snapshots required per worker.
        colors (dict | None): model name -> colour.
        save (bool): Save figure if True.
        path (str | None): Output directory.
    """
    if colors is None:
        defaults = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
        colors = {name: defaults[i % len(defaults)]
                  for i, name in enumerate(wage_age_dict)}

    n = len(wage_age_dict)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for i, (name, df) in enumerate(wage_age_dict.items()):
        ax = axes[i]
        col = colors.get(name, 'steelblue')

        if 'Worker_ID' not in df.columns:
            ax.text(0.5, 0.5, 'Worker_ID missing\n(re-run model to generate)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(name, fontweight='bold')
            continue

        counts = df.groupby('Worker_ID')['Year'].nunique()
        eligible = counts[counts >= min_years_observed].index.tolist()

        if len(eligible) == 0:
            ax.text(0.5, 0.5, f'No workers with ≥{min_years_observed} years',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(name, fontweight='bold')
            continue

        rng_seed = np.random.default_rng(42)
        sample_ids = rng_seed.choice(eligible,
                                     size=min(n_trajectories, len(eligible)),
                                     replace=False)
        sample_df = df[df['Worker_ID'].isin(sample_ids)].copy()

        first_wage = (sample_df.sort_values('Year')
                      .groupby('Worker_ID')['Wage'].first()
                      .rename('first_wage'))
        sample_df = sample_df.merge(first_wage, on='Worker_ID')
        sample_df['norm_wage'] = sample_df['Wage'] / sample_df['first_wage']

        for wid, traj in sample_df.groupby('Worker_ID'):
            traj = traj.sort_values('Year')
            ax.plot(traj['Year'], traj['norm_wage'],
                    color=col, alpha=0.15, linewidth=0.7)

        from statsmodels.nonparametric.smoothers_lowess import lowess
        all_sorted = sample_df.sort_values('Year')
        trend = lowess(all_sorted['norm_wage'], all_sorted['Year'], frac=0.4)
        ax.plot(trend[:, 0], trend[:, 1], color='black', linewidth=2,
                label='LOWESS trend')

        ax.axhline(1.0, color='grey', linewidth=0.8, linestyle='dotted')
        ax.set_xlabel('Simulation year')
        ax.set_ylabel('Wage / entry wage')
        ax.set_title(name, fontweight='bold')
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Job Ladder: Individual Wage Trajectories (normalised to entry wage)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig(f'{path}job_ladders.png', bbox_inches='tight',
                    pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_wasserstein_dur_dist(
    ue_stock_dfs,
    cps_dur_dist,
    occ_crosswalk,
    soc_labs,
    calib_date,
    delay=25,
    year=2007,
    colors=None,
    save=False,
    path=None,
    cap_at_observed_max=True,
):
    """
    Ranked bar chart of Wasserstein distance between simulated and CPS
    unemployment duration distributions, by SOC major group, for a single year.

    Parameters
    ----------
    ue_stock_dfs : dict
        Keys are model names; values are DataFrames with columns
        ['Time Step', 'Occupation', 'UEDuration'] (UEDuration in months).
    cps_dur_dist : pd.DataFrame
        Columns: ['YEAR', 'SOC_major', 'DURUNEMP', 'dur_months', 'prop'].
    occ_crosswalk : pd.DataFrame
        Must contain 'acs_occ_code' (int) and 'SOC_major_adj' (int/float).
    soc_labs : pd.DataFrame
        Must contain 'soc_code' (numeric) and 'label' (str).
    calib_date : list of str
        ['YYYY-MM-DD', 'YYYY-MM-DD'] start and end of simulation.
    delay : int
        Burn-in time steps to drop from model output.
    year : int
        Calendar year to compare.
    cap_at_observed_max : bool
        cps_dur_dist's dur_months is top-coded (~27 months); the model's
        UEDuration is uncensored. If True (default), model values are
        clipped to that group's observed CPS max before computing
        Wasserstein distance -- see _compute_wasserstein_all_years for
        the same fix applied to the multi-year pipeline.
    """
    from scipy.stats import wasserstein_distance

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                      'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    # ── build time step → calendar year map ───────────────────────────────────
    start_date = pd.Timestamp(calib_date[0])
    def ts_to_year(ts):
        return (start_date + pd.DateOffset(months=int(ts) - delay - 1)).year

    # ── occupation crosswalk: model integer ID → SOC_major ───────────────────
    # occ_crosswalk['X'] is 1-indexed row number; occupation_id = X - 1
    cw = occ_crosswalk[['X', 'SOC_major_adj']].drop_duplicates()
    cw = cw.dropna(subset=['SOC_major_adj']).copy()
    cw['Occupation'] = cw['X'].astype(int) - 1
    cw['SOC_major']  = cw['SOC_major_adj'].astype(int)
    cw = cw[['Occupation', 'SOC_major']]

    # ── SOC labels ────────────────────────────────────────────────────────────
    lab_map = (soc_labs.dropna(subset=['soc_code', 'label'])
               .assign(soc_code=lambda d: d['soc_code'].astype(int))
               .set_index('soc_code')['label']
               .to_dict())

    # ── CPS CDF for the target year ───────────────────────────────────────────
    cps_year = cps_dur_dist[cps_dur_dist['YEAR'] == year].copy()
    cps_soc_groups = cps_year['SOC_major'].dropna().unique()

    # ── compute Wasserstein distances ─────────────────────────────────────────
    records = []
    for model_name, stock_df in ue_stock_dfs.items():
        stock_df = stock_df[stock_df['Time Step'] >= delay].copy()
        stock_df['Year'] = stock_df['Time Step'].apply(ts_to_year)
        stock_year = stock_df[stock_df['Year'] == year].copy()

        # map model occupation → SOC_major
        stock_year = stock_year.merge(cw, on='Occupation', how='left')
        stock_year = stock_year.dropna(subset=['SOC_major'])
        stock_year['SOC_major'] = stock_year['SOC_major'].astype(int)

        for soc in cps_soc_groups:
            model_vals = stock_year.loc[
                stock_year['SOC_major'] == soc, 'UEDuration'].values - 1
            cps_rows = cps_year[cps_year['SOC_major'] == soc]

            if len(model_vals) < 5 or cps_rows.empty:
                continue

            # expand CPS weighted distribution into sample
            cps_vals = np.repeat(
                cps_rows['dur_months'].values,
                np.round(cps_rows['n_weighted'] / cps_rows['n_weighted'].sum() * 1000).astype(int)
            )

            if len(cps_vals) == 0:
                continue

            pct_above_cap = np.nan
            if cap_at_observed_max:
                cap = cps_vals.max()
                pct_above_cap = float((model_vals > cap).mean() * 100)
                model_vals = np.clip(model_vals, None, cap)

            wd = wasserstein_distance(model_vals, cps_vals)
            records.append({
                'Model': model_name,
                'SOC_major': soc,
                'label': lab_map.get(soc, str(soc)),
                'Wasserstein': wd,
                'pct_model_above_cap': pct_above_cap,
            })

    if not records:
        print("No Wasserstein distances computed — check year/occupation overlap.")
        return

    results = pd.DataFrame(records)

    # ── rank occupations by mean Wasserstein across models ────────────────────
    occ_order = (results.groupby('label')['Wasserstein']
                 .mean()
                 .sort_values(ascending=False)
                 .index.tolist())

    model_names = list(ue_stock_dfs.keys())
    n_models = len(model_names)
    n_occs = len(occ_order)
    bar_w = 0.8 / n_models
    x = np.arange(n_occs)

    fig, ax = plt.subplots(figsize=(max(12, n_occs * 0.6), 5))

    for i, model_name in enumerate(model_names):
        sub = results[results['Model'] == model_name].set_index('label')
        vals = [sub.loc[occ, 'Wasserstein'] if occ in sub.index else np.nan
                for occ in occ_order]
        offset = (i - (n_models - 1) / 2) * bar_w
        ax.bar(x + offset, vals, width=bar_w,
               label=model_name, color=colors.get(model_name, None),
               alpha=0.85, edgecolor='white', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(occ_order, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Wasserstein distance (months)')
    ax.set_title(f'Simulated vs CPS unemployment duration distribution — {year}',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-0.5, n_occs - 0.5)
    plt.tight_layout()

    if save and path:
        fig.savefig(os.path.join(path, f'wasserstein_dur_dist_{year}.jpg'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def _wass_box_stats(values, weights):
    """Weighted box-plot statistics (Q1/median/Q3 via weighted CDF, 1.5×IQR whiskers)."""
    if len(values) == 0:
        return None
    sorter = np.argsort(values)
    v = values[sorter].astype(float)
    w = weights[sorter].astype(float)
    cumw = np.cumsum(w) / w.sum()
    q1  = float(np.interp(0.25, cumw, v))
    med = float(np.interp(0.50, cumw, v))
    q3  = float(np.interp(0.75, cumw, v))
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    whislo = float(v[v >= lo_fence].min()) if (v >= lo_fence).any() else q1
    whishi = float(v[v <= hi_fence].max()) if (v <= hi_fence).any() else q3
    fliers = v[(v < lo_fence) | (v > hi_fence)].tolist()
    return dict(med=med, q1=q1, q3=q3, whislo=whislo, whishi=whishi, fliers=fliers)


def _compute_wasserstein_all_years(
    ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay=25,
    sim_results_dict=None, grouping_col='SOC_major_adj', cap_at_observed_max=True
):
    """
    Shared helper: compute Wasserstein distances for all models × years × occupation groups.

    grouping_col controls which crosswalk column maps individual occupations to groups
    (e.g. 'SOC_major_adj', 'SOC_minor_adj', 'SOC_broad_adj', 'SOC2010_adj').
    cps_dur_dist must contain a 'group_id' column whose values match those in grouping_col.

    cps_dur_dist's dur_months is right-censored/top-coded (the
    unemp_duration_dist_yearly_{grouping}.csv files top out at ~27
    months), while the model's UEDuration is uncensored and can run far
    higher. Comparing the two as-is penalizes a model for "missing" CPS
    mass that was never actually observable past the top-code -- a
    censoring artifact, not a real fit difference (same issue as
    _compute_duration_wasserstein_by_year's aggregate-distribution
    version). cap_at_observed_max=True (the default) clips model values
    to that (year, group)'s observed CPS max before computing the
    distance, restoring an apples-to-apples comparison on the same
    support. Set to False to compare the raw uncensored model tail.

    Returns a DataFrame with columns ['Model', 'Year', 'SOC_major', 'label',
    'Wasserstein', 'mean_diff', 'Employment', 'Workers', 'pct_model_above_cap'].
    """
    from scipy.stats import wasserstein_distance

    start_date = pd.Timestamp(calib_date[0])
    def ts_to_year(ts):
        return (start_date + pd.DateOffset(months=int(ts) - delay - 1)).year

    cw = occ_crosswalk[['X', grouping_col]].drop_duplicates()
    cw = cw.dropna(subset=[grouping_col]).copy()
    cw['Occupation'] = cw['X'].astype(int) - 1
    cw['group_id']   = cw[grouping_col]   # preserves original type (int or string)
    cw = cw[['Occupation', 'group_id']]

    # label lookup from *_description column of the crosswalk
    desc_col = grouping_col.replace('_adj', '_description')
    if desc_col in occ_crosswalk.columns:
        lab_map = (occ_crosswalk[[grouping_col, desc_col]]
                   .dropna(subset=[grouping_col, desc_col])
                   .drop_duplicates(grouping_col)
                   .set_index(grouping_col)[desc_col]
                   .to_dict())
    else:
        lab_map = {}

    cps_years = set(cps_dur_dist['YEAR'].dropna().unique().astype(int))

    def _ts_to_year_vec(ts_series):
        """Vectorized: time step → calendar year (no Python-level row loop)."""
        months_offset = ts_series.astype(int) - delay - 1
        total_months  = (start_date.month - 1) + months_offset
        return (start_date.year + total_months // 12).values

    # compute model years from the first model's stock df to define the intersection
    _first_df = next(iter(ue_stock_dfs.values()))
    _first_df = _first_df[_first_df['Time Step'] >= delay].copy()
    _first_df['Year'] = _ts_to_year_vec(_first_df['Time Step'])
    model_years = set(_first_df['Year'].astype(int))
    years = sorted(cps_years & model_years)

    # pre-compute CPS distributions once — they are model-independent
    cps_dist_cache = {}
    for _yr in years:
        _cps_yr = cps_dur_dist[cps_dur_dist['YEAR'] == _yr]
        for _grp in _cps_yr['group_id'].dropna().unique():
            _rows = _cps_yr[_cps_yr['group_id'] == _grp]
            if _rows.empty:
                continue
            _total = _rows['n_weighted'].sum()
            if _total == 0:
                continue
            _vals = np.repeat(
                _rows['dur_months'].values,
                np.round(_rows['n_weighted'].values / _total * 1000).astype(int)
            )
            if len(_vals) > 0:
                cps_dist_cache[(_yr, _grp)] = _vals

    records = []
    for model_name, stock_df in ue_stock_dfs.items():
        stock_df = stock_df[stock_df['Time Step'] >= delay].copy()
        stock_df['Year'] = _ts_to_year_vec(stock_df['Time Step'])
        stock_df = stock_df[stock_df['Year'].isin(years)]
        stock_df = stock_df.merge(cw, on='Occupation', how='left')
        stock_df = stock_df.dropna(subset=['group_id'])

        # build annual mean (Employment, Workers) per group from sim_results_dict
        if sim_results_dict is not None and model_name in sim_results_dict:
            sim_rec = sim_results_dict[model_name].copy()
            sim_rec = sim_rec[sim_rec['Time Step'] >= delay]
            sim_rec['Year'] = _ts_to_year_vec(sim_rec['Time Step'])
            sim_rec = (sim_rec.merge(cw, on='Occupation', how='left')
                              .dropna(subset=['group_id']))
            occ_size = (sim_rec.groupby(['group_id', 'Year'])[['Employment', 'Workers']]
                               .mean().reset_index())
        else:
            occ_size = None

        _multisim = 'sim_id' in stock_df.columns

        for yr in years:
            stock_year = stock_df[stock_df['Year'] == yr]
            if stock_year.empty:
                continue

            for grp in cps_dur_dist.loc[cps_dur_dist['YEAR'] == yr, 'group_id'].dropna().unique():
                grp_mask = stock_year['group_id'] == grp
                if grp_mask.sum() < 5:
                    continue
                cps_vals = cps_dist_cache.get((yr, grp))
                if cps_vals is None or len(cps_vals) == 0:
                    continue

                emp_wt, wrk_wt = np.nan, np.nan
                if occ_size is not None:
                    sz = occ_size[(occ_size['group_id'] == grp) & (occ_size['Year'] == yr)]
                    if not sz.empty:
                        emp_wt = float(sz['Employment'].iloc[0])
                        wrk_wt = float(sz['Workers'].iloc[0])

                try:
                    label = lab_map.get(int(grp), str(grp))
                except (ValueError, TypeError):
                    label = str(grp)

                cap = cps_vals.max() if cap_at_observed_max else None

                if _multisim:
                    # compute per-sim Wasserstein and mean_diff, then aggregate
                    sim_wds, sim_diffs, pct_above_caps = [], [], []
                    for sid in stock_year['sim_id'].unique():
                        sv = stock_year.loc[(stock_year['sim_id'] == sid) & grp_mask, 'UEDuration'].values - 1
                        if len(sv) < 2:
                            continue
                        if cap_at_observed_max:
                            pct_above_caps.append(float((sv > cap).mean() * 100))
                            sv = np.clip(sv, None, cap)
                        sim_wds.append(wasserstein_distance(sv, cps_vals))
                        sim_diffs.append(float(np.mean(sv) - np.mean(cps_vals)))
                    if not sim_wds:
                        continue
                    wd            = float(np.mean(sim_wds))
                    wd_std        = float(np.std(sim_wds, ddof=1)) if len(sim_wds) > 1 else np.nan
                    md            = float(np.mean(sim_diffs))
                    md_std        = float(np.std(sim_diffs, ddof=1)) if len(sim_diffs) > 1 else np.nan
                    pct_above_cap = float(np.mean(pct_above_caps)) if pct_above_caps else np.nan
                else:
                    model_vals = stock_year.loc[grp_mask, 'UEDuration'].values - 1
                    pct_above_cap = np.nan
                    if cap_at_observed_max:
                        pct_above_cap = float((model_vals > cap).mean() * 100)
                        model_vals    = np.clip(model_vals, None, cap)
                    wd         = wasserstein_distance(model_vals, cps_vals)
                    wd_std     = np.nan
                    md         = float(np.mean(model_vals) - np.mean(cps_vals))
                    md_std     = np.nan

                records.append({
                    'Model':               model_name,
                    'Year':                yr,
                    'SOC_major':           grp,
                    'label':               label,
                    'Wasserstein':         wd,
                    'Wasserstein_std':     wd_std,
                    'mean_diff':           md,
                    'mean_diff_std':       md_std,
                    'Employment':          emp_wt,
                    'Workers':             wrk_wt,
                    'pct_model_above_cap': pct_above_cap,
                })

    return pd.DataFrame(records)


def plot_wasserstein_heatmap(
    ue_stock_dfs,
    cps_dur_dist,
    occ_crosswalk,
    calib_date,
    delay=25,
    mode='raw',
    colors=None,
    sim_results_dict=None,
    grouping_col='SOC_major_adj',
    save=False,
    path=None,
    soc_labs=None,
):
    """
    Heatmap of Wasserstein distance — occupation × year, one panel per model.

    mode : str
        'raw'    — colour = raw Wasserstein distance (original behaviour).
        'binary' — best model per cell is highlighted, all others are grey.
        'ranked' — each rank gets a distinct colour:
                   rank 1 (best) = green, rank 2 = steelblue,
                   rank 3 = orange, rank 4 (worst) = red.
    """
    results = _compute_wasserstein_all_years(
        ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay,
        grouping_col=grouping_col)
    if results.empty:
        print("No data for heatmap.")
        return

    model_names = list(ue_stock_dfs.keys())
    n_models    = len(model_names)
    occ_order   = (results.groupby('label')['Wasserstein']
                   .mean().sort_values(ascending=False).index.tolist())
    years = sorted(results['Year'].unique())

    # ── build per-model pivot tables ──────────────────────────────────────────
    pivots = {}
    for model_name in model_names:
        sub = results[results['Model'] == model_name]
        pivots[model_name] = (sub.pivot_table(index='label', columns='Year',
                                              values='Wasserstein', aggfunc='mean')
                                 .reindex(occ_order)
                                 .reindex(columns=years))

    fig, axes = plt.subplots(1, n_models,
                             figsize=(n_models * 5, max(4, len(occ_order) * 0.35)),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    if mode == 'raw':
        vmax = results['Wasserstein'].quantile(0.95)
        for ax, model_name in zip(axes, model_names):
            im = ax.imshow(pivots[model_name].values, aspect='auto',
                           cmap='YlOrRd', vmin=0, vmax=vmax)
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels([str(y) for y in years], rotation=90, fontsize=7)
            ax.set_yticks(range(len(occ_order)))
            ax.set_yticklabels(occ_order if ax == axes[0] else [], fontsize=7)
            ax.set_title(model_name, fontsize=9, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.6, label='Wasserstein (months)')

    else:
        # stack into array: shape (n_models, n_occs, n_years)
        stack = np.stack([pivots[m].values for m in model_names], axis=0)

        # rank per cell: 0 = best (lowest Wasserstein)
        # nan-safe: argsort along model axis
        ranks = np.argsort(np.argsort(
            np.where(np.isnan(stack), np.inf, stack), axis=0), axis=0)

        if mode == 'binary':
            rank_colors = {0: '#2ca02c'}   # green for best; grey for rest
            default_col = '#d3d3d3'
            legend_labels = {0: 'Best fit'}
        elif mode == 'top2':
            rank_colors = {0: '#2ca02c', 1: '#98df8a'}  # dark green, light green
            default_col = '#d3d3d3'
            legend_labels = {0: 'Best fit', 1: '2nd best'}
        else:  # 'ranked'
            rank_palette = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728',
                            '#9467bd', '#8c564b', '#e377c2']
            rank_colors  = {r: rank_palette[r % len(rank_palette)]
                            for r in range(n_models)}
            default_col  = '#d3d3d3'
            rank_names   = ['Best', '2nd', '3rd', '4th', '5th', '6th', '7th']
            legend_labels = {r: rank_names[r] for r in range(n_models)}

        for m_idx, (ax, model_name) in enumerate(zip(axes, model_names)):
            model_ranks = ranks[m_idx]           # (n_occs, n_years)
            rgb = np.zeros((*model_ranks.shape, 3))

            for r, col in rank_colors.items():
                mask = (model_ranks == r)
                rgb[mask] = plt.matplotlib.colors.to_rgb(col)
            if mode in ('binary', 'top2'):
                non_top = ~np.isin(model_ranks, list(rank_colors.keys()))
                rgb[non_top] = plt.matplotlib.colors.to_rgb(default_col)

            # grey out cells with no data
            nan_mask = np.isnan(stack[m_idx])
            rgb[nan_mask] = plt.matplotlib.colors.to_rgb('#f0f0f0')

            ax.imshow(rgb, aspect='auto')
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels([str(y) for y in years], rotation=90, fontsize=7)
            ax.set_yticks(range(len(occ_order)))
            ax.set_yticklabels(occ_order if ax == axes[0] else [], fontsize=7)
            ax.set_title(model_name, fontsize=9, fontweight='bold')

        # shared legend
        handles = [plt.Rectangle((0, 0), 1, 1,
                                  fc=rank_colors[r], label=legend_labels[r])
                   for r in sorted(rank_colors)]
        fig.legend(handles=handles, loc='lower center',
                   ncol=len(handles), fontsize=8,
                   bbox_to_anchor=(0.5, -0.03))

    fig.suptitle('Wasserstein distance: simulated vs CPS duration distribution'
                 f' ({mode})', fontweight='bold', fontsize=11)
    plt.tight_layout()
    if save and path:
        fig.savefig(os.path.join(path, f'wasserstein_heatmap_occ_year_{mode}_{grouping_col}.jpg'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_wasserstein_boxplot(
    ue_stock_dfs,
    cps_dur_dist,
    occ_crosswalk,
    calib_date,
    delay=25,
    colors=None,
    sim_results_dict=None,
    grouping_col='SOC_major_adj',
    save=False,
    path=None,
    soc_labs=None,
):
    """
    Option B: Grouped horizontal box plots — one panel, years on y-axis,
    Wasserstein distance on x-axis, one box per model per year offset vertically.
    Boxes are occupation-size weighted (by Workers) when sim_results_dict is provided.
    """
    results = _compute_wasserstein_all_years(
        ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay,
        sim_results_dict=sim_results_dict, grouping_col=grouping_col)
    if results.empty:
        print("No data for boxplot.")
        return

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                      'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    model_names = list(ue_stock_dfs.keys())
    n_models    = len(model_names)
    years       = sorted(results['Year'].unique())
    n_years     = len(years)

    # horizontal spacing: each year occupies 1 unit; models offset within that
    box_w   = 0.6 / n_models
    fig, ax = plt.subplots(figsize=(max(10, n_years * 0.55), 5))

    legend_handles = []
    for i, model_name in enumerate(model_names):
        sub = results[results['Model'] == model_name]
        col = colors.get(model_name, 'steelblue')
        offset = (i - (n_models - 1) / 2) * box_w

        positions = [yr_idx + offset for yr_idx, _ in enumerate(years)]

        stats_list = []
        for yr in years:
            yr_sub = sub[sub['Year'] == yr]
            v = yr_sub['Wasserstein'].values
            w = yr_sub['Workers'].values
            if results['Workers'].isna().all():
                w = np.ones(len(v))   # fall back to equal weights if no sim_results
            else:
                w = np.where(np.isnan(w), 0.0, w)
                w = w if w.sum() > 0 else np.ones(len(v))
            stats_list.append(_wass_box_stats(v, w))

        valid = [(pos, st) for pos, st in zip(positions, stats_list) if st is not None]
        if not valid:
            continue
        pos_v, stats_v = zip(*valid)
        bp = ax.bxp(list(stats_v), positions=list(pos_v), vert=True,
                    patch_artist=True, widths=box_w * 0.85,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color=col, linewidth=0.8),
                    capprops=dict(color=col, linewidth=0.8),
                    flierprops=dict(marker='o', markersize=2, alpha=0.4,
                                   markerfacecolor=col, markeredgecolor=col))
        for patch in bp['boxes']:
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.75, label=model_name))

    ax.set_xticks(range(n_years))
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Wasserstein distance (months)')
    ax.set_title('Wasserstein distance by year: distribution across occupations',
                 fontweight='bold', fontsize=10)
    ax.legend(handles=legend_handles, fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.6, n_years - 0.4)
    plt.tight_layout()

    if save and path:
        fig.savefig(os.path.join(path, f'wasserstein_boxplot_by_year_{grouping_col}.jpg'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_wasserstein_ribbon(
    ue_stock_dfs,
    cps_dur_dist,
    occ_crosswalk,
    calib_date,
    delay=25,
    colors=None,
    sim_results_dict=None,
    grouping_col='SOC_major_adj',
    save=False,
    path=None,
    soc_labs=None,
):
    """
    Option C: Ribbon plot — mean Wasserstein across occupations over time,
    ±1 SD ribbon, one line per model on a shared axis.
    Mean and SD are occupation-size weighted (by Workers) when sim_results_dict is provided.
    """
    results = _compute_wasserstein_all_years(
        ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay,
        sim_results_dict=sim_results_dict, grouping_col=grouping_col)
    if results.empty:
        print("No data for ribbon.")
        return

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                      'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    fig, ax = plt.subplots(figsize=(12, 4))

    use_weights = not results['Workers'].isna().all()
    for model_name in ue_stock_dfs.keys():
        sub = results[results['Model'] == model_name]
        agg_rows = []
        for yr, grp in sub.groupby('Year'):
            v = grp['Wasserstein'].values
            if use_weights:
                w = grp['Workers'].values
                w = np.where(np.isnan(w), 0.0, w)
                w = w if w.sum() > 0 else np.ones(len(v))
            else:
                w = np.ones(len(v))
            w_norm = w / w.sum()
            mean = float(np.dot(w_norm, v))
            std  = float(np.sqrt(np.dot(w_norm, (v - mean) ** 2)))
            agg_rows.append({'Year': yr, 'mean': mean, 'std': std})
        agg = pd.DataFrame(agg_rows)
        col = colors.get(model_name, 'steelblue')
        ax.plot(agg['Year'], agg['mean'], color=col, label=model_name, linewidth=1.8)
        ax.fill_between(agg['Year'],
                        agg['mean'] - agg['std'],
                        agg['mean'] + agg['std'],
                        color=col, alpha=0.15)

    ax.set_xlabel('Year')
    ax.set_ylabel('Wasserstein distance (months)')
    ax.set_title('Mean Wasserstein distance over time (±1 SD across occupations)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save and path:
        fig.savefig(os.path.join(path, f'wasserstein_ribbon_over_time_{grouping_col}.jpg'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def _compute_duration_wasserstein_by_year(
    ue_stock_dfs, survey_df, calib_date, delay=25, min_obs=10, cap_at_observed_max=True
):
    """
    Shared helper: aggregate (not occupation-disaggregated) Wasserstein
    distance between each model's pooled unemployment-duration stock
    distribution and the CPS aggregate duration distribution, one value
    per model per calendar year.

    This is the quantified version of the gap between curves shown
    informally in plot_ltuer_cdf_with_observed -- for 1D distributions,
    the Wasserstein-1 distance equals the area between the two CDFs --
    computed year by year (rather than once over the full sample) so
    models can be ranked annually, mirroring _compute_wasserstein_all_years'
    per-occupation pipeline but pooling across all occupations instead.

    CPS DURUNEMP_MO is top-coded (observed durations are right-censored
    at ~29 months -- mass beyond that piles up at the cap rather than
    being reported exactly), while the model's UEDuration is uncensored
    and can run far higher (the non-behavioural variants' full-distribution
    CDF shows real mass out past 100+ months). Feeding both as-is into
    wasserstein_distance would penalize models for "missing" CPS mass
    that was never actually observable past the top-code -- a censoring
    artifact, not a real fit difference. cap_at_observed_max=True (the
    default) clips model_vals to that year's observed CPS max before
    computing the distance, so a model worker unemployed beyond the
    top-code is treated the way a real top-coded CPS respondent would
    have been recorded, restoring an apples-to-apples comparison on the
    same support. Set to False to compare the raw uncensored model tail
    (will mechanically inflate Wasserstein distance for any model with a
    heavy right tail, regardless of how well its bulk distribution fits).

    ue_stock_dfs : dict[str, pd.DataFrame], columns include 'Time Step'
        and 'UEDuration' (stock of currently-unemployed workers and their
        elapsed duration at each simulated month). If a 'sim_id' column
        is present (multiple simulations per model), the Wasserstein
        distance is computed per sim_id then averaged, mirroring
        _compute_wasserstein_all_years' multisim handling -- this also
        yields a 'Wasserstein_std' (across-sim std) for a confidence
        ribbon.
    survey_df : pd.DataFrame with columns 'YEAR', 'DURUNEMP_MO', 'n' --
        same CPS aggregate duration schema used by
        plot_ltuer_cdf_with_observed/plot_ltuer_cdf_heatmap. UEDuration is
        compared directly against DURUNEMP_MO with no offset, matching
        plot_ltuer_cdf_heatmap's convention for this same data pairing.

    Returns a DataFrame with columns ['Model', 'Year', 'Wasserstein',
    'Wasserstein_std', 'mean_diff', 'n_model', 'n_obs',
    'pct_model_above_cap']. 'Wasserstein_std' is NaN when ue_stock_dfs
    has no 'sim_id' column (single run per model).
    """
    from scipy.stats import wasserstein_distance

    if not ue_stock_dfs:
        print("_compute_duration_wasserstein_by_year: ue_stock_dfs is empty -- "
              "if loading from a cached pkl, the cache may predate the "
              "'filtered_ue_stock_dfs' key; try FORCE_RERUN = True to rebuild.")
        return pd.DataFrame()

    start_date = pd.Timestamp(calib_date[0])

    def _ts_to_year_vec(ts_series):
        months_offset = ts_series.astype(int) - delay - 1
        total_months  = (start_date.month - 1) + months_offset
        return (start_date.year + total_months // 12).values

    cps = survey_df[survey_df['DURUNEMP_MO'].notna()].copy()
    cps_years = set(cps['YEAR'].dropna().astype(int).unique())

    _first_df = next(iter(ue_stock_dfs.values()))
    _first_df = _first_df[_first_df['Time Step'] >= delay].copy()
    _first_df['Year'] = _ts_to_year_vec(_first_df['Time Step'])
    model_years = set(_first_df['Year'].astype(int))
    years = sorted(cps_years & model_years)

    # pre-compute CPS distributions once per year -- model-independent
    cps_cache = {}
    for yr in years:
        yr_df = cps[cps['YEAR'] == yr]
        total_n = yr_df['n'].sum()
        if total_n == 0:
            continue
        cps_cache[yr] = (yr_df['DURUNEMP_MO'].values, yr_df['n'].values)

    records = []
    for model_name, stock_df in ue_stock_dfs.items():
        s = stock_df[stock_df['Time Step'] >= delay].copy()
        s['Year'] = _ts_to_year_vec(s['Time Step'])
        _multisim = 'sim_id' in s.columns

        for yr in years:
            if yr not in cps_cache:
                continue
            cps_durs, cps_wts = cps_cache[yr]
            cap = cps_durs.max() if cap_at_observed_max else None
            yr_mask = s['Year'] == yr

            if _multisim:
                sim_wds, sim_diffs, pct_above_caps, n_models = [], [], [], []
                for sid in s.loc[yr_mask, 'sim_id'].unique():
                    sv = s.loc[yr_mask & (s['sim_id'] == sid), 'UEDuration'].dropna().values
                    if len(sv) < min_obs:
                        continue
                    if cap_at_observed_max:
                        pct_above_caps.append(float((sv > cap).mean() * 100))
                        sv = np.clip(sv, None, cap)
                    sim_wds.append(wasserstein_distance(sv, cps_durs, v_weights=cps_wts))
                    sim_diffs.append(float(np.mean(sv) - np.average(cps_durs, weights=cps_wts)))
                    n_models.append(len(sv))
                if not sim_wds:
                    continue
                wd               = float(np.mean(sim_wds))
                wd_std           = float(np.std(sim_wds, ddof=1)) if len(sim_wds) > 1 else np.nan
                md               = float(np.mean(sim_diffs))
                pct_above_cap    = float(np.mean(pct_above_caps)) if pct_above_caps else np.nan
                pct_above_cap_std = float(np.std(pct_above_caps, ddof=1)) if len(pct_above_caps) > 1 else np.nan
                n_model          = int(np.sum(n_models))
            else:
                model_vals = s.loc[yr_mask, 'UEDuration'].dropna().values
                if len(model_vals) < min_obs:
                    continue
                pct_above_cap     = np.nan
                pct_above_cap_std = np.nan
                if cap_at_observed_max:
                    pct_above_cap = float((model_vals > cap).mean() * 100)
                    model_vals = np.clip(model_vals, None, cap)
                wd      = wasserstein_distance(model_vals, cps_durs, v_weights=cps_wts)
                wd_std  = np.nan
                md      = float(np.mean(model_vals) - np.average(cps_durs, weights=cps_wts))
                n_model = len(model_vals)

            records.append({
                'Model':                model_name,
                'Year':                 yr,
                'Wasserstein':          wd,
                'Wasserstein_std':      wd_std,
                'mean_diff':            md,
                'n_model':              n_model,
                'n_obs':                int(cps_wts.sum()),
                'pct_model_above_cap':  pct_above_cap,
                'pct_above_cap_std':    pct_above_cap_std,
            })

    return pd.DataFrame(records)


def _compute_duration_wasserstein_by_month(
    ue_stock_dfs, survey_df, calib_date, delay=25, min_obs=10, cap_at_observed_max=True
):
    """
    Monthly analog of _compute_duration_wasserstein_by_year: Wasserstein-1
    distance between each model's pooled unemployment-duration stock and the
    CPS monthly duration distribution, one value per (model, year, month).

    survey_df : pd.DataFrame with columns 'YEAR', 'MONTH', 'DURUNEMP_MO', 'n'
        Same schema as the monthly CPS files used in analyze_ltuer_ts_fit.py
        (MONTH as integer 1–12). CPS durations are top-coded; model values
        are clipped to the per-period observed max before computing distance
        when cap_at_observed_max=True (default), mirroring the annual fix in
        _compute_duration_wasserstein_by_year.

    Returns a DataFrame with columns ['Model', 'Year', 'Month', 'Date',
    'Wasserstein', 'Wasserstein_std', 'mean_diff', 'n_model', 'n_obs',
    'pct_model_above_cap', 'pct_above_cap_std'].
    'Date' is a pd.Timestamp (first day of each month) for convenient plotting.
    'Wasserstein_std' and 'pct_above_cap_std' are NaN when ue_stock_dfs has
    no 'sim_id' column (single run per model).
    """
    from scipy.stats import wasserstein_distance

    if not ue_stock_dfs:
        print("_compute_duration_wasserstein_by_month: ue_stock_dfs is empty -- "
              "if loading from a cached pkl, try FORCE_RERUN = True to rebuild.")
        return pd.DataFrame()

    start_date = pd.Timestamp(calib_date[0])

    def _ts_to_yearmonth_vec(ts_series):
        months_offset = ts_series.astype(int) - delay - 1
        total_months  = (start_date.month - 1) + months_offset
        year  = (start_date.year  + total_months // 12).values
        month = (total_months % 12 + 1).values
        return year, month

    cps = survey_df[survey_df['DURUNEMP_MO'].notna()].copy()
    cps['YEAR']  = cps['YEAR'].astype(int)
    cps['MONTH'] = cps['MONTH'].astype(int)
    cps_periods  = set(zip(cps['YEAR'], cps['MONTH']))

    _first_df = next(iter(ue_stock_dfs.values()))
    _first_df = _first_df[_first_df['Time Step'] >= delay].copy()
    yr_arr, mo_arr = _ts_to_yearmonth_vec(_first_df['Time Step'])
    _first_df['Year']  = yr_arr
    _first_df['Month'] = mo_arr
    model_periods = set(zip(_first_df['Year'].astype(int), _first_df['Month'].astype(int)))
    periods = sorted(cps_periods & model_periods)

    cps_cache = {}
    for (yr, mo) in periods:
        sub = cps[(cps['YEAR'] == yr) & (cps['MONTH'] == mo)]
        if sub['n'].sum() == 0:
            continue
        cps_cache[(yr, mo)] = (sub['DURUNEMP_MO'].values, sub['n'].values)

    records = []
    for model_name, stock_df in ue_stock_dfs.items():
        s = stock_df[stock_df['Time Step'] >= delay].copy()
        yr_arr, mo_arr = _ts_to_yearmonth_vec(s['Time Step'])
        s['Year']  = yr_arr
        s['Month'] = mo_arr
        _multisim  = 'sim_id' in s.columns

        for (yr, mo) in periods:
            if (yr, mo) not in cps_cache:
                continue
            cps_durs, cps_wts = cps_cache[(yr, mo)]
            cap     = cps_durs.max() if cap_at_observed_max else None
            yr_mask = (s['Year'] == yr) & (s['Month'] == mo)

            if _multisim:
                sim_wds, sim_diffs, pct_above_caps, n_models = [], [], [], []
                for sid in s.loc[yr_mask, 'sim_id'].unique():
                    sv = s.loc[yr_mask & (s['sim_id'] == sid), 'UEDuration'].dropna().values
                    if len(sv) < min_obs:
                        continue
                    if cap_at_observed_max:
                        pct_above_caps.append(float((sv > cap).mean() * 100))
                        sv = np.clip(sv, None, cap)
                    sim_wds.append(wasserstein_distance(sv, cps_durs, v_weights=cps_wts))
                    sim_diffs.append(float(np.mean(sv) - np.average(cps_durs, weights=cps_wts)))
                    n_models.append(len(sv))
                if not sim_wds:
                    continue
                wd                = float(np.mean(sim_wds))
                wd_std            = float(np.std(sim_wds, ddof=1)) if len(sim_wds) > 1 else np.nan
                md                = float(np.mean(sim_diffs))
                pct_above_cap     = float(np.mean(pct_above_caps)) if pct_above_caps else np.nan
                pct_above_cap_std = float(np.std(pct_above_caps, ddof=1)) if len(pct_above_caps) > 1 else np.nan
                n_model           = int(np.sum(n_models))
            else:
                model_vals = s.loc[yr_mask, 'UEDuration'].dropna().values
                if len(model_vals) < min_obs:
                    continue
                pct_above_cap     = np.nan
                pct_above_cap_std = np.nan
                if cap_at_observed_max:
                    pct_above_cap = float((model_vals > cap).mean() * 100)
                    model_vals    = np.clip(model_vals, None, cap)
                wd      = wasserstein_distance(model_vals, cps_durs, v_weights=cps_wts)
                wd_std  = np.nan
                md      = float(np.mean(model_vals) - np.average(cps_durs, weights=cps_wts))
                n_model = len(model_vals)

            records.append({
                'Model':                model_name,
                'Year':                 yr,
                'Month':                mo,
                'Date':                 pd.Timestamp(year=yr, month=mo, day=1),
                'Wasserstein':          wd,
                'Wasserstein_std':      wd_std,
                'mean_diff':            md,
                'n_model':              n_model,
                'n_obs':                int(cps_wts.sum()),
                'pct_model_above_cap':  pct_above_cap,
                'pct_above_cap_std':    pct_above_cap_std,
            })

    return pd.DataFrame(records)


def plot_duration_wasserstein_over_time_monthly(
    ue_stock_dfs,
    survey_df,
    calib_date,
    delay=25,
    min_obs=10,
    colors=None,
    save=False,
    path=None,
    dropbox_path=None,
    calib_v_line=None,
):
    """
    Monthly analog of plot_duration_wasserstein_over_time: Wasserstein-1
    distance between each model's unemployment-duration stock and the CPS
    monthly duration distribution, plotted at monthly resolution on a date
    x-axis.

    survey_df : pd.DataFrame with columns 'YEAR', 'MONTH', 'DURUNEMP_MO', 'n'
        Monthly CPS duration distribution in the same schema as the monthly
        UER files used in analyze_ltuer_ts_fit.py (MONTH as integer 1–12).

    Top panel: mean Wasserstein distance per model over time (monthly), with
    ±1 SD ribbon across simulations when ue_stock_dfs carries a 'sim_id'
    column. Scatter markers are coloured by pct_model_above_cap (YlOrRd,
    light = negligible censoring, dark = heavy censoring) to flag months
    where the reported distance understates the true tail divergence.

    Bottom panel: share of simulated workers censored to the CPS top-code
    (percent), monthly, with ±1 SD ribbon across simulations.
    """
    results = _compute_duration_wasserstein_by_month(
        ue_stock_dfs, survey_df, calib_date, delay=delay, min_obs=min_obs)
    if results.empty:
        print("No overlapping months for duration Wasserstein (monthly).")
        return results

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                      'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    has_ci     = results['Wasserstein_std'].notna().any()
    has_pct_ci = results['pct_above_cap_std'].notna().any()
    if not has_ci:
        print("Note: no cross-simulation CI ribbon shown -- ue_stock_dfs has no "
              "'sim_id' column (or only one run per model contributed valid months).")

    fig, (ax, ax_pct) = plt.subplots(
        2, 1, figsize=(TEXTWIDTH, ROW_HEIGHT * 2), sharex=True,
        gridspec_kw={'height_ratios': [3, 1.5], 'hspace': 0.08})

    plotted = []
    for model_name in ue_stock_dfs.keys():
        sub = results[results['Model'] == model_name].sort_values('Date')
        if sub.empty:
            continue
        col = colors.get(model_name, 'steelblue')
        ax.plot(sub['Date'], sub['Wasserstein'], color=col, label=model_name,
                linewidth=1.4, zorder=2)

        std = sub['Wasserstein_std']
        if has_ci and std.notna().any():
            std = std.fillna(0.0)
            ax.fill_between(sub['Date'], sub['Wasserstein'] - std,
                            sub['Wasserstein'] + std, color=col, alpha=0.15, zorder=1)
        plotted.append((sub, col))

    ax.set_ylabel('Model Error\nWasserstein Distance (months)', fontsize=10)
    ax.set_title('Model Fit to Monthly CPS Unemployment-Duration Distributions',
                 fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelbottom=False)

    for sub, col in plotted:
        pct = sub['pct_model_above_cap'].fillna(0.0)
        ax_pct.plot(sub['Date'], pct, color=col, linewidth=1.2)
        if has_pct_ci:
            std = sub['pct_above_cap_std'].fillna(0.0)
            ax_pct.fill_between(sub['Date'],
                                np.maximum(pct - std, 0.0),
                                pct + std,
                                color=col, alpha=0.15)

    ax_pct.set_xlabel('Date')
    ax_pct.set_ylabel('% of Unemployed\nSample Above CPS\nTop-Coding Threshold')
    ax_pct.grid(alpha=0.3)
    ax_pct.set_ylim(bottom=0)
    ax_pct.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax_pct.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.setp(ax_pct.get_xticklabels(), rotation=45, ha='right')

    if calib_v_line is not None:
        shade_out_of_sample(ax, calib_v_line[1])
        shade_out_of_sample(ax_pct, calib_v_line[1])

    # plt.tight_layout() is incompatible with this figure's custom
    # gridspec_kw (height_ratios + hspace) -- see plot_duration_wasserstein_over_time.
    fig.subplots_adjust(left=0.14)
    add_shared_legend(fig, colors, ncol=3)

    if save and path:
        fname = 'duration_wasserstein_over_time_monthly.png'
        save_plot_standard(fig, os.path.join(path, fname), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
        if dropbox_path:
            os.makedirs(dropbox_path, exist_ok=True)
            fname_pdf = os.path.splitext(fname)[0] + '.pdf'
            save_plot_standard(fig, os.path.join(dropbox_path, fname_pdf), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
            print(f"✓ {fname_pdf} saved to Dropbox.")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    return results


def plot_duration_wasserstein_rank_heatmap(
    ue_stock_dfs,
    survey_df,
    calib_date,
    delay=25,
    min_obs=10,
    colors=None,
    save=False,
    path=None,
    dropbox_path=None,
):
    """
    Annual ranking of models by aggregate unemployment-duration fit to
    CPS: one row per model, one column per calendar year, cell colour =
    that model's rank that year (1 = closest to the CPS aggregate
    duration distribution that year, by Wasserstein distance -- see
    _compute_duration_wasserstein_by_year). Cell text shows the rank.

    Reuses the same rank-colour convention as panel (C) of
    plot_wasserstein_summary (RdYlGn_r: green = best, red = worst) so
    the two figures read consistently; this is the aggregate-distribution
    analog of that occupation-disaggregated heatmap, intended to replace
    plot_ltuer_cdf_heatmap's "CDF data pending" visual that ranking
    question was previously implicit in.

    Returns the underlying _compute_duration_wasserstein_by_year DataFrame
    (raw Wasserstein distances, not just ranks) for further inspection.
    """
    results = _compute_duration_wasserstein_by_year(
        ue_stock_dfs, survey_df, calib_date, delay=delay, min_obs=min_obs)
    if results.empty:
        print("No overlapping years for duration Wasserstein ranking.")
        return results

    model_names = list(ue_stock_dfs.keys())
    n_models    = len(model_names)
    years       = sorted(results['Year'].unique())
    n_years     = len(years)

    pivot = (results.pivot_table(index='Model', columns='Year',
                                  values='Wasserstein', aggfunc='mean')
                     .reindex(model_names).reindex(columns=years))

    # rank per year (column-wise): 0 = best (lowest Wasserstein)
    arr   = pivot.values.astype(float)
    ranks = np.argsort(np.argsort(np.where(np.isnan(arr), np.inf, arr), axis=0), axis=0)
    ranks = ranks.astype(float)
    ranks[np.isnan(arr)] = np.nan

    _cmap         = plt.cm.RdYlGn_r   # green (best) → yellow → red (worst)
    rank_colors   = {r: _cmap(r / max(n_models - 1, 1)) for r in range(n_models)}
    rank_names    = ['Best', '2nd', '3rd', '4th', '5th', '6th', '7th']
    legend_labels = {r: rank_names[r] if r < len(rank_names) else f'{r + 1}th'
                      for r in range(n_models)}

    rgb = np.zeros((n_models, n_years, 3))
    for r, col in rank_colors.items():
        rgb[ranks == r] = plt.matplotlib.colors.to_rgb(col)
    rgb[np.isnan(ranks)] = plt.matplotlib.colors.to_rgb('#f0f0f0')

    # Width fixed at TEXTWIDTH; height kept at the original max(3, n_models*0.6)
    # (not rescaled) since it's sized for model-row count. Note: with many
    # year-columns this compresses column (year) tick-label legibility --
    # there's no way to preserve both row and column density at true
    # TEXTWIDTH without a design change (e.g. abbreviating year labels).
    _fig_w, _fig_h = TEXTWIDTH, max(3, n_models * 0.6)
    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
    ax.imshow(rgb, aspect='auto')

    ax.set_xticks(range(n_years))
    ax.set_xticklabels([str(y) for y in years], rotation=90, fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=9)
    if colors:
        for lbl, name in zip(ax.get_yticklabels(), model_names):
            lbl.set_color(colors.get(name, 'black'))
    ax.set_title('Ranking of Aggregate Unemployment-Duration Fit to CPS', fontweight='bold')

    handles = [plt.Rectangle((0, 0), 1, 1, fc=rank_colors[r], label=legend_labels[r])
               for r in sorted(rank_colors) if r < n_models]
    plt.tight_layout()
    #add_shared_legend(fig, {}, extra_handles=handles, ncol=3)

    if save and path:
        fname = 'duration_wasserstein_rank_heatmap.png'
        save_plot_standard(fig, os.path.join(path, fname), aspect_ratio=_fig_h/_fig_w,
                            width_frac=max(1.0, _fig_w/TEXTWIDTH), apply_grid=False)
        if dropbox_path:
            os.makedirs(dropbox_path, exist_ok=True)
            fname_pdf = os.path.splitext(fname)[0] + '.pdf'
            save_plot_standard(fig, os.path.join(dropbox_path, fname_pdf), aspect_ratio=_fig_h/_fig_w,
                                width_frac=max(1.0, _fig_w/TEXTWIDTH), apply_grid=False)
            print(f"✓ {fname_pdf} saved to Dropbox.")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    return results


def plot_duration_wasserstein_rank_heatmap_monthly(
    ue_stock_dfs,
    survey_df,
    calib_date,
    delay=25,
    min_obs=10,
    colors=None,
    save=False,
    path=None,
    dropbox_path=None,
):
    """
    Model × Month rank heatmap of unemployment-duration fit to CPS.
    survey_df must carry a 'MONTH' column (ltuer_cps_data_monthly.csv schema).
    Each column is one calendar month; ranking is computed across models
    for that month (1 = lowest Wasserstein distance that month).
    X-axis tick labels show January of each year; no rank numbers printed
    in cells.  Saves as duration_wasserstein_rank_heatmap_monthly.png.
    """
    results = _compute_duration_wasserstein_by_month(
        ue_stock_dfs, survey_df, calib_date, delay=delay, min_obs=min_obs)
    if results.empty:
        print("No overlapping months for monthly duration Wasserstein ranking.")
        return results

    model_names = list(ue_stock_dfs.keys())
    n_models    = len(model_names)

    # Build date index sorted chronologically
    dates = sorted(results['Date'].unique())
    n_dates = len(dates)

    pivot = (results.pivot_table(index='Model', columns='Date',
                                  values='Wasserstein', aggfunc='mean')
                     .reindex(model_names).reindex(columns=dates))

    arr   = pivot.values.astype(float)
    ranks = np.argsort(np.argsort(np.where(np.isnan(arr), np.inf, arr), axis=0), axis=0)
    ranks = ranks.astype(float)
    ranks[np.isnan(arr)] = np.nan

    _cmap       = plt.cm.RdYlGn_r
    rank_colors = {r: _cmap(r / max(n_models - 1, 1)) for r in range(n_models)}
    rank_names  = ['Best', '2nd', '3rd', '4th', '5th', '6th', '7th']
    legend_labels = {r: rank_names[r] if r < len(rank_names) else f'{r + 1}th'
                     for r in range(n_models)}

    rgb = np.zeros((n_models, n_dates, 3))
    for r, col in rank_colors.items():
        rgb[ranks == r] = plt.matplotlib.colors.to_rgb(col)
    rgb[np.isnan(ranks)] = plt.matplotlib.colors.to_rgb('#f0f0f0')

    # Width fixed at TEXTWIDTH; height kept at the original max(3, n_models*0.6)
    # (not rescaled) since it's sized for model-row count. Note: with many
    # month-columns this compresses column (date) tick-label legibility --
    # there's no way to preserve both row and column density at true
    # TEXTWIDTH without a design change (e.g. showing fewer date ticks).
    _fig_w, _fig_h = TEXTWIDTH, max(3, n_models * 0.6)
    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
    ax.imshow(rgb, aspect='auto')

    # Tick only January of each year to keep the x-axis readable
    import pandas as _pd
    dates_pd = _pd.to_datetime(dates)
    jan_ticks = [i for i, d in enumerate(dates_pd) if d.month == 1]
    jan_labels = [str(dates_pd[i].year) for i in jan_ticks]
    ax.set_xticks(jan_ticks)
    ax.set_xticklabels(jan_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=9)
    if colors:
        for lbl, name in zip(ax.get_yticklabels(), model_names):
            lbl.set_color(colors.get(name, 'black'))
    ax.set_title('Ranking of Aggregate Unemployment-Duration Fit to CPS', fontweight='bold')

    handles = [plt.Rectangle((0, 0), 1, 1, fc=rank_colors[r], label=legend_labels[r])
               for r in sorted(rank_colors) if r < n_models]
    plt.tight_layout()
    #add_shared_legend(fig, {}, extra_handles=handles, ncol=3)

    if save and path:
        fname = 'duration_wasserstein_rank_heatmap_monthly.png'
        save_plot_standard(fig, os.path.join(path, fname), aspect_ratio=_fig_h/_fig_w,
                            width_frac=max(1.0, _fig_w/TEXTWIDTH), apply_grid=False)
        if dropbox_path:
            os.makedirs(dropbox_path, exist_ok=True)
            fname_pdf = os.path.splitext(fname)[0] + '.pdf'
            save_plot_standard(fig, os.path.join(dropbox_path, fname_pdf), aspect_ratio=_fig_h/_fig_w,
                                width_frac=max(1.0, _fig_w/TEXTWIDTH), apply_grid=False)
            print(f"✓ {fname_pdf} saved to Dropbox.")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    return results


def plot_duration_wasserstein_over_time(
    ue_stock_dfs,
    survey_df,
    calib_date,
    delay=25,
    min_obs=10,
    colors=None,
    save=False,
    path=None,
    dropbox_path=None,
    calib_v_line=None,
):
    """
    Mean Wasserstein distance (aggregate duration distribution, sim vs
    CPS) per model over calendar years, with a +-1 SD ribbon across
    simulations when ue_stock_dfs carries a 'sim_id' column (see
    _compute_duration_wasserstein_by_year) -- the magnitude companion to
    plot_duration_wasserstein_rank_heatmap's ordinal ranking (a model can
    rank 1st in a year while still being a poor absolute fit if every
    model is off that year, which the rank heatmap alone can't show).

    Marker fill colour encodes pct_model_above_cap (YlOrRd colormap,
    light=low, dark=high) -- the share of that model-year's simulated
    unemployed that were clipped to the CPS top-code before the
    Wasserstein distance was computed. A darker marker means more of
    that model's tail got clipped, so the plotted distance is more
    likely to understate the true (uncapped) value. The model's own
    line colour is retained as the marker edge so lines and dots stay
    visually linked. A shared colorbar provides the scale.

    A companion lower panel plots pct_model_above_cap directly as line
    graphs (one per model, same colours) so the clipping share can also
    be read off precisely and compared across models over time.
    """
    results = _compute_duration_wasserstein_by_year(
        ue_stock_dfs, survey_df, calib_date, delay=delay, min_obs=min_obs)
    if results.empty:
        print("No overlapping years for duration Wasserstein over time.")
        return results

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                      'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    has_ci = results['Wasserstein_std'].notna().any()
    if not has_ci:
        print("Note: no cross-simulation CI ribbon shown -- ue_stock_dfs has no "
              "'sim_id' column (or only one run per model contributed valid years), "
              "so _compute_duration_wasserstein_by_year fell back to a single point "
              "estimate per model-year instead of per-sim mean +- SD.")

    fig, (ax, ax_pct) = plt.subplots(
        2, 1, figsize=(TEXTWIDTH, ROW_HEIGHT * 2), sharex=True,
        gridspec_kw={'height_ratios': [3, 1.5], 'hspace': 0.08})

    plotted = []
    for model_name in ue_stock_dfs.keys():
        sub = results[results['Model'] == model_name].sort_values('Year')
        if sub.empty:
            continue
        col = colors.get(model_name, 'steelblue')
        ax.plot(sub['Year'], sub['Wasserstein'], color=col, label=model_name,
                linewidth=1.8, marker='o', markersize=3, zorder=2)

        std = sub['Wasserstein_std']
        if has_ci and std.notna().any():
            std = std.fillna(0.0)
            ax.fill_between(sub['Year'], sub['Wasserstein'] - std,
                            sub['Wasserstein'] + std, color=col, alpha=0.15, zorder=1)
        plotted.append((sub, col))

    ax.set_ylabel('Model Error\nWasserstein Distance (months)', fontsize=10)
    ax.set_title('Model Fit to Annual CPS Unemployment-Duration Distributions',
                 fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelbottom=False)

    # lower panel: pct_model_above_cap as line plot with ±1 SD ribbon
    has_pct_ci = 'pct_above_cap_std' in results.columns and results['pct_above_cap_std'].notna().any()
    for sub, col in plotted:
        pct  = sub['pct_model_above_cap'].fillna(0.0)
        ax_pct.plot(sub['Year'], pct, color=col, marker='o', markersize=3)
        if has_pct_ci:
            std = sub['pct_above_cap_std'].fillna(0.0)
            ax_pct.fill_between(sub['Year'],
                                np.maximum(pct - std, 0.0),
                                pct + std,
                                color=col, alpha=0.15)

    ax_pct.set_xlabel('Date')
    ax_pct.set_ylabel('% of Unemployed\nSample Above CPS\nTop-Coding Threshold')
    ax_pct.grid(alpha=0.3)
    ax_pct.set_ylim(bottom=0)
    ax_pct.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    ax_pct.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%d'))

    if calib_v_line is not None:
        calib_end_year = pd.to_datetime(calib_v_line[1]).year
        shade_out_of_sample_year(ax, calib_end_year)
        shade_out_of_sample_year(ax_pct, calib_end_year)

    # plt.tight_layout() is incompatible with this figure's custom
    # gridspec_kw (height_ratios + hspace) -- matplotlib warns about it, and
    # in practice it under-reserves the left margin so the two panels'
    # multi-line y-labels overlap each other. subplots_adjust(left=...)
    # reserves enough room directly, leaving the gridspec's own hspace alone.
    fig.subplots_adjust(left=0.16)
    add_shared_legend(fig, colors, ncol=3)

    if save and path:
        fname = 'duration_wasserstein_over_time.png'
        save_plot_standard(fig, os.path.join(path, fname), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
        if dropbox_path:
            os.makedirs(dropbox_path, exist_ok=True)
            fname_pdf = os.path.splitext(fname)[0] + '.pdf'
            save_plot_standard(fig, os.path.join(dropbox_path, fname_pdf), aspect_ratio=(ROW_HEIGHT * 2) / TEXTWIDTH)
            print(f"✓ {fname_pdf} saved to Dropbox.")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    return results


def plot_wasserstein_summary(
    ue_stock_dfs,
    cps_dur_dist,
    occ_crosswalk,
    calib_date,
    delay=25,
    colors=None,
    sim_results_dict=None,
    grouping_col='SOC_major_adj',
    save=False,
    path=None,
    soc_labs=None,
):
    """
    Two-panel manuscript summary figure for time-to-reemployment model fit.
    Sized at 7 × 3.5 inches (fits \\linewidth in a standard 11pt LaTeX document).
    Save as PDF for vector output suitable for LaTeX inclusion.

    A — Stacked bar chart: share of occupation groups where each model ranks
        1st / 2nd / 3rd / 4th by mean Wasserstein distance (averaged over years).
        Shows which model achieves best fit for most occupations.

    B — Occupation-size-weighted boxplot of Wasserstein distance over time.
        One set of boxes per year; models offset within each year.
        Log y-axis prevents outlier occupations from compressing the main distribution.
        Quantiles use Workers-weighted CDF interpolation (via _wass_box_stats).
    """
    import os
    import matplotlib.gridspec as gridspec

    # Panels A and C use genuinely different colour semantics (model colour
    # vs. rank colour), so each keeps its own per-panel legend rather than
    # a single add_shared_legend -- font sizes all inherit the shared
    # rcParams instead of hardcoding a duplicate hierarchy.
    results = _compute_wasserstein_all_years(
        ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay,
        sim_results_dict=sim_results_dict, grouping_col=grouping_col)
    if results.empty:
        print("No data for summary figure.")
        return

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                      'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    model_names = list(ue_stock_dfs.keys())
    n_models    = len(model_names)
    years       = sorted(results['Year'].unique())
    n_years     = len(years)
    use_weights = not results['Workers'].isna().all()

    short_labels = []
    for m in model_names:
        s = m.replace('Non-behavioural', 'Non-behav.')
        s = s.replace('Behavioural', 'Behav.')
        s = s.replace(' w.o. Cyc. OTJ', '')
        short_labels.append(s.strip())

    # ── figure layout ─────────────────────────────────────────────────────────
    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * 3
    fig = plt.figure(figsize=(_fig_w, _fig_h))
    gs  = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[2.2, 1.8, 1.2], hspace=0.65)
    ax_box    = fig.add_subplot(gs[0, 0])
    ax_violin = fig.add_subplot(gs[1, 0])
    ax_rank   = fig.add_subplot(gs[2, 0])

    for ax in (ax_box, ax_violin, ax_rank):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Panel A (top): Weighted boxplot over time ─────────────────────────────
    box_w = 0.6 / n_models
    legend_handles = []
    for i, model_name in enumerate(model_names):
        sub    = results[results['Model'] == model_name]
        col    = colors.get(model_name, 'steelblue')
        offset = (i - (n_models - 1) / 2) * box_w
        positions = [yr_idx + offset for yr_idx in range(n_years)]

        stats_list = []
        for yr in years:
            yr_sub = sub[sub['Year'] == yr]
            v = yr_sub['Wasserstein'].values
            if use_weights:
                w = yr_sub['Workers'].values
                w = np.where(np.isnan(w), 0.0, w)
                w = w if w.sum() > 0 else np.ones(len(v))
            else:
                w = np.ones(len(v))
            stats_list.append(_wass_box_stats(v, w))

        valid = [(pos, st) for pos, st in zip(positions, stats_list) if st is not None]
        if not valid:
            continue
        pos_v, stats_v = zip(*valid)
        bp = ax_box.bxp(list(stats_v), positions=list(pos_v), vert=True,
                        patch_artist=True, widths=box_w * 0.85,
                        medianprops=dict(color='black', linewidth=1.0),
                        whiskerprops=dict(color=col, linewidth=0.6),
                        capprops=dict(color=col, linewidth=0.6),
                        flierprops=dict(marker='o', markersize=1.5, alpha=0.4,
                                        markerfacecolor=col, markeredgecolor=col))
        for patch in bp['boxes']:
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.75, label=model_name))

    ax_box.set_xticks(range(n_years))
    ax_box.set_xticklabels([str(y) for y in years], rotation=45, ha='right')
    ax_box.set_ylabel('Wasserstein distance (months)')
    ax_box.set_title('(A) Fit over time (distribution across occupations)',
                     fontweight='bold')
    ax_box.legend(handles=legend_handles,
                  loc='upper center', bbox_to_anchor=(0.5, -0.18),
                  ncol=2, borderaxespad=0, frameon=False)
    ax_box.grid(axis='y', alpha=0.25, linewidth=0.5, which='both')
    ax_box.set_xlim(-0.6, n_years - 0.4)

    # ── Panel B (middle): mean_diff violin per model ─────────────────────────
    md_by_occ = (results.groupby(['Model', 'SOC_major'])
                 .agg(mean_diff=('mean_diff', 'mean'),
                      Workers=('Workers', 'mean'))
                 .reset_index())

    violin_data      = []
    violin_positions = []
    violin_colors    = []
    rng = np.random.default_rng(0)
    for i, model_name in enumerate(model_names):
        sub  = md_by_occ[md_by_occ['Model'] == model_name].dropna(subset=['mean_diff'])
        vals = sub['mean_diff'].values
        w    = sub['Workers'].values if use_weights else np.ones(len(vals))
        w    = np.where(np.isnan(w) | (w <= 0), 1.0, w)
        if len(vals) < 2:
            continue
        # build weighted sample: resample proportionally to occupation size
        w_norm  = w / w.sum()
        n_samp  = max(500, len(vals) * 20)
        w_vals  = rng.choice(vals, size=n_samp, replace=True, p=w_norm)
        violin_data.append(w_vals)
        violin_positions.append(i)
        violin_colors.append(colors.get(model_name, 'steelblue'))

    if violin_data:
        parts = ax_violin.violinplot(violin_data, positions=violin_positions,
                                     showmedians=True, showextrema=True, widths=0.6)
        for body, col in zip(parts['bodies'], violin_colors):
            body.set_facecolor(col)
            body.set_alpha(0.70)
            body.set_edgecolor('none')
        for part_name in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if part_name in parts:
                parts[part_name].set_color('black')
                parts[part_name].set_linewidth(0.8)

    ax_violin.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_violin.set_xticks(range(n_models))
    ax_violin.set_xticklabels(short_labels, rotation=30, ha='right')
    ax_violin.set_ylabel('Mean duration difference\n(model − CPS, months)')
    ax_violin.set_title('Directional Bias Across Occupations',
                        fontweight='bold')
    ax_violin.grid(axis='y', alpha=0.25, linewidth=0.5, which='both')

    # ── Panel C (bottom): Rank distribution across occupation groups ──────────
    wass_by_occ = (results.groupby(['Model', 'SOC_major'])['Wasserstein']
                   .mean().reset_index())
    wass_by_occ['rank'] = (wass_by_occ.groupby('SOC_major')['Wasserstein']
                           .rank(method='min').astype(int))
    n_occs = wass_by_occ['SOC_major'].nunique()
    rank_shares = (wass_by_occ.groupby(['Model', 'rank']).size()
                   .reset_index(name='count'))
    rank_shares['share'] = rank_shares['count'] / n_occs * 100

    max_rank     = int(wass_by_occ['rank'].max())
    _cmap        = plt.cm.RdYlGn_r   # green (best) → yellow → red (worst)
    rank_palette = {r: _cmap((r - 1) / max(max_rank - 1, 1))
                    for r in range(1, max_rank + 1)}
    rank_labels  = {1: '1st (best)', max_rank: f'{max_rank}th (worst)'}

    x_pos   = np.arange(n_models)
    bottoms = np.zeros(n_models)
    legend_patches = []
    for rank in range(1, max_rank + 1):
        heights = []
        for model_name in model_names:
            row = rank_shares[(rank_shares['Model'] == model_name) &
                              (rank_shares['rank'] == rank)]
            heights.append(float(row['share'].iloc[0]) if not row.empty else 0.0)
        col = rank_palette.get(rank, '#999999')
        ax_rank.bar(x_pos, heights, bottom=bottoms, color=col, alpha=0.85,
                    edgecolor='white', linewidth=0.4)
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.85,
                                             label=rank_labels.get(rank, f'Rank {rank}')))
        bottoms += np.array(heights)

    ax_rank.set_xticks(x_pos)
    ax_rank.set_xticklabels(short_labels, rotation=30, ha='right')
    ax_rank.set_ylabel('Share of groups (%)')
    ax_rank.set_ylim(0, 105)
    ax_rank.set_title('(C) Best-fit rank across occupation groups',
                      fontweight='bold')
    ax_rank.legend(handles=legend_patches,
                   title='Rank',
                   loc='upper center', bbox_to_anchor=(0.5, -0.30),
                   ncol=4, borderaxespad=0, frameon=False)
    ax_rank.grid(axis='y', alpha=0.25, linewidth=0.5)

    plt.tight_layout()

    if save and path:
        save_plot_standard(fig, os.path.join(path, f'wasserstein_summary_{grouping_col}.pdf'),
                            aspect_ratio=_fig_h / _fig_w)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_wasserstein_summary_panels(
    ue_stock_dfs,
    cps_dur_dist,
    occ_crosswalk,
    calib_date,
    delay=25,
    colors=None,
    sim_results_dict=None,
    grouping_col='SOC_major_adj',
    save=False,
    path=None,
    dropbox_path=None,
    soc_labs=None,
):
    """
    Saves each panel of plot_wasserstein_summary as a separate figure file.

    Output files (PDF):
        wasserstein_panel_A_{grouping_col}.pdf  — Weighted boxplot over time
        wasserstein_panel_B_{grouping_col}.pdf  — Directional-bias violin
        wasserstein_panel_C_{grouping_col}.pdf  — Best-fit rank across occupation groups
    """
    import os

    results = _compute_wasserstein_all_years(
        ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay,
        sim_results_dict=sim_results_dict, grouping_col=grouping_col)
    if results.empty:
        print("No data for summary panels.")
        return

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                      'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    model_names = list(ue_stock_dfs.keys())
    n_models    = len(model_names)
    years       = sorted(results['Year'].unique())
    n_years     = len(years)
    use_weights = not results['Workers'].isna().all()

    short_labels = []
    for m in model_names:
        s = m.replace('Non-behavioural', 'Non-behav.')
        s = s.replace('Behavioural', 'Behav.')
        s = s.replace(' w.o. Cyc. OTJ', '')
        short_labels.append(s.strip())

    _fig_w, _fig_h = TEXTWIDTH, 4

    def _save(fig, fname):
        if save and path:
            save_plot_standard(fig, os.path.join(path, fname), aspect_ratio=_fig_h / _fig_w)
            if dropbox_path:
                os.makedirs(dropbox_path, exist_ok=True)
                save_plot_standard(fig, os.path.join(dropbox_path, fname), aspect_ratio=_fig_h / _fig_w)
                print(f"✓ {fname} saved to Dropbox.")
        else:
            plt.show()
        plt.close(fig)

    # ── Panel A: Weighted boxplot over time ────────────────────────────────────
    fig_a, ax = plt.subplots(figsize=(_fig_w, _fig_h))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    box_w = 0.6 / n_models
    legend_handles = []
    for i, model_name in enumerate(model_names):
        sub    = results[results['Model'] == model_name]
        col    = colors.get(model_name, 'steelblue')
        offset = (i - (n_models - 1) / 2) * box_w
        positions = [yr_idx + offset for yr_idx in range(n_years)]

        stats_list = []
        for yr in years:
            yr_sub = sub[sub['Year'] == yr]
            v = yr_sub['Wasserstein'].values
            if use_weights:
                w = yr_sub['Workers'].values
                w = np.where(np.isnan(w), 0.0, w)
                w = w if w.sum() > 0 else np.ones(len(v))
            else:
                w = np.ones(len(v))
            stats_list.append(_wass_box_stats(v, w))

        valid = [(pos, st) for pos, st in zip(positions, stats_list) if st is not None]
        if not valid:
            continue
        pos_v, stats_v = zip(*valid)
        bp = ax.bxp(list(stats_v), positions=list(pos_v), vert=True,
                    patch_artist=True, widths=box_w * 0.85,
                    medianprops=dict(color='black', linewidth=1.0),
                    whiskerprops=dict(color=col, linewidth=0.6),
                    capprops=dict(color=col, linewidth=0.6),
                    flierprops=dict(marker='o', markersize=1.5, alpha=0.4,
                                   markerfacecolor=col, markeredgecolor=col))
        for patch in bp['boxes']:
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.75, label=model_name))

    ax.set_xticks(range(n_years))
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha='right')
    ax.set_ylabel('Wasserstein distance (months)')
    ax.set_title('(A) Fit over time (distribution across occupations)',
                 fontweight='bold')
    ax.legend(handles=legend_handles,
              loc='upper center', bbox_to_anchor=(0.5, -0.28),
              ncol=2, borderaxespad=0, frameon=False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5, which='both')
    ax.set_xlim(-0.6, n_years - 0.4)
    plt.tight_layout()
    _save(fig_a, f'wasserstein_panel_A_{grouping_col}.pdf')

    # ── Panel B: Directional-bias violin ──────────────────────────────────────
    fig_b, ax = plt.subplots(figsize=(_fig_w, _fig_h))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    md_by_occ = (results.groupby(['Model', 'SOC_major'])
                 .agg(mean_diff=('mean_diff', 'mean'),
                      Workers=('Workers', 'mean'))
                 .reset_index())

    violin_data      = []
    violin_positions = []
    violin_colors    = []
    rng = np.random.default_rng(0)
    for i, model_name in enumerate(model_names):
        sub  = md_by_occ[md_by_occ['Model'] == model_name].dropna(subset=['mean_diff'])
        vals = sub['mean_diff'].values
        w    = sub['Workers'].values if use_weights else np.ones(len(vals))
        w    = np.where(np.isnan(w) | (w <= 0), 1.0, w)
        if len(vals) < 2:
            continue
        w_norm  = w / w.sum()
        n_samp  = max(500, len(vals) * 20)
        w_vals  = rng.choice(vals, size=n_samp, replace=True, p=w_norm)
        violin_data.append(w_vals)
        violin_positions.append(i)
        violin_colors.append(colors.get(model_name, 'steelblue'))

    if violin_data:
        parts = ax.violinplot(violin_data, positions=violin_positions,
                              showmedians=True, showextrema=True, widths=0.6)
        for body, col in zip(parts['bodies'], violin_colors):
            body.set_facecolor(col)
            body.set_alpha(0.70)
            body.set_edgecolor('none')
        for part_name in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if part_name in parts:
                parts[part_name].set_color('black')
                parts[part_name].set_linewidth(0.8)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(short_labels, rotation=30, ha='right')
    ax.set_ylabel('Mean Distance Between\nSimulated and Observed\nDuration Distributions (months)')
    ax.set_title('Directional Bias Across Occupations',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.25, linewidth=0.5, which='both')
    plt.tight_layout()
    _save(fig_b, f'wasserstein_panel_B_{grouping_col}.pdf')

    # ── Panel C: Best-fit rank stacked bar ────────────────────────────────────
    fig_c, ax = plt.subplots(figsize=(_fig_w, _fig_h))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    wass_by_occ = (results.groupby(['Model', 'SOC_major'])['Wasserstein']
                   .mean().reset_index())
    wass_by_occ['rank'] = (wass_by_occ.groupby('SOC_major')['Wasserstein']
                           .rank(method='min').astype(int))
    n_occs = wass_by_occ['SOC_major'].nunique()
    rank_shares = (wass_by_occ.groupby(['Model', 'rank']).size()
                   .reset_index(name='count'))
    rank_shares['share'] = rank_shares['count'] / n_occs * 100

    max_rank     = int(wass_by_occ['rank'].max())
    _cmap        = plt.cm.RdYlGn_r
    rank_palette = {r: _cmap((r - 1) / max(max_rank - 1, 1))
                    for r in range(1, max_rank + 1)}
    rank_labels  = {1: '1st (best)', max_rank: f'{max_rank}th (worst)'}

    x_pos   = np.arange(n_models)
    bottoms = np.zeros(n_models)
    legend_patches = []
    for rank in range(1, max_rank + 1):
        heights = []
        for model_name in model_names:
            row = rank_shares[(rank_shares['Model'] == model_name) &
                              (rank_shares['rank'] == rank)]
            heights.append(float(row['share'].iloc[0]) if not row.empty else 0.0)
        col = rank_palette.get(rank, '#999999')
        ax.bar(x_pos, heights, bottom=bottoms, color=col, alpha=0.85,
               edgecolor='white', linewidth=0.4)
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.85,
                                            label=rank_labels.get(rank, f'Rank {rank}')))
        bottoms += np.array(heights)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, rotation=30, ha='right')
    ax.set_ylabel('Share of groups (%)')
    ax.set_ylim(0, 105)
    ax.set_title('(C) Best-fit rank across occupation groups',
                 fontweight='bold')
    ax.legend(handles=legend_patches,
              title='Rank',
              loc='upper center', bbox_to_anchor=(0.5, -0.30),
              ncol=4, borderaxespad=0, frameon=False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    _save(fig_c, f'wasserstein_panel_C_{grouping_col}.pdf')


def _compute_ltuer_ts_fit(
    ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay=25,
    sim_results_dict=None, grouping_col='SOC_major_adj',
    ltuer_threshold=6, min_years=6
):
    """
    Shared helper: build the annual LTUER series for each (model, occupation
    group) and for CPS, then summarize the fit between the two series with
    NMSE (level fit -- mean squared error of raw annual levels, normalized
    by the group's squared mean observed level so groups with very
    different baseline LTUER aren't compared on different absolute
    scales) and Pearson correlation (co-movement fit, computed on
    year-over-year *changes* in LTUER rather than levels).

    Normalizing by the group's own mean level matters here: a model that's
    off by 2pp on a group with a 2% baseline rate has effectively doubled
    it, while the same 2pp miss on a 15% baseline group is a much smaller
    relative error. Without normalizing, high-baseline groups mechanically
    dominate the SSE distribution (their absolute errors run larger even
    when proportionally well-fit) and low-baseline groups become nearly
    invisible regardless of fit quality.

    Correlating levels is misleading here: both series can share a common
    trend (e.g. the COVID unemployment spike) that produces a high
    correlation even when year-to-year movements are unrelated (classic
    spurious-trend correlation). Differencing first removes that shared
    trend and isolates whether the two series actually move together.
    min_years defaults higher than before (6, i.e. >=5 differenced points)
    because differencing costs one observation and rank correlations on a
    handful of points are too coarse to be informative.

    grouping_col controls which crosswalk column maps individual occupations to
    groups (e.g. 'SOC_major_adj', 'SOC_minor_adj', 'SOC_broad_adj', 'SOC2010_adj').
    cps_dur_dist must contain a 'group_id' column whose values match grouping_col.

    Returns a DataFrame with columns ['Model', 'group_id', 'label', 'n_years',
    'NMSE', 'Pearson', 'Workers'].
    """
    from scipy.stats import pearsonr

    start_date = pd.Timestamp(calib_date[0])

    def _ts_to_year_vec(ts_series):
        months_offset = ts_series.astype(int) - delay - 1
        total_months  = (start_date.month - 1) + months_offset
        return (start_date.year + total_months // 12).values

    cw = occ_crosswalk[['X', grouping_col]].drop_duplicates()
    cw = cw.dropna(subset=[grouping_col]).copy()
    cw['Occupation'] = cw['X'].astype(int) - 1
    cw['group_id']   = cw[grouping_col]
    cw = cw[['Occupation', 'group_id']]

    desc_col = grouping_col.replace('_adj', '_description')
    if desc_col in occ_crosswalk.columns:
        lab_map = (occ_crosswalk[[grouping_col, desc_col]]
                   .dropna(subset=[grouping_col, desc_col])
                   .drop_duplicates(grouping_col)
                   .set_index(grouping_col)[desc_col]
                   .to_dict())
    else:
        lab_map = {}

    # CPS annual LTUER per group (share of unemployed stock with dur_months >= threshold)
    cps_ltuer = (cps_dur_dist.groupby(['YEAR', 'group_id'])
                 .apply(lambda x: (x.loc[x['dur_months'] >= ltuer_threshold, 'n_weighted'].sum()
                                    / max(x['n_weighted'].sum(), 1)))
                 .reset_index(name='ltuer'))

    records = []
    for model_name, stock_df in ue_stock_dfs.items():
        s = stock_df[stock_df['Time Step'] >= delay].copy()
        s['Year'] = _ts_to_year_vec(s['Time Step'])
        s = s.merge(cw, on='Occupation', how='left').dropna(subset=['group_id'])
        if 'LongtermUnemp' in s.columns:
            s['_lt'] = s['LongtermUnemp'].astype(bool)
        else:
            s['_lt'] = (s['UEDuration'] - 1) >= ltuer_threshold

        _multisim = 'sim_id' in s.columns
        group_by  = ['group_id', 'Year', 'sim_id'] if _multisim else ['group_id', 'Year']
        model_ltuer = (s.groupby(group_by)
                        .apply(lambda x: x['_lt'].sum() / max(len(x), 1))
                        .reset_index(name='ltuer'))
        if _multisim:
            model_ltuer = (model_ltuer.groupby(['group_id', 'Year'])['ltuer']
                           .mean().reset_index())

        # mean group size (Workers) over the full period, for point-sizing
        if sim_results_dict is not None and model_name in sim_results_dict:
            sim_rec = sim_results_dict[model_name].copy()
            sim_rec = sim_rec[sim_rec['Time Step'] >= delay]
            sim_rec = (sim_rec.merge(cw, on='Occupation', how='left')
                              .dropna(subset=['group_id']))
            wrk_map = sim_rec.groupby('group_id')['Workers'].mean()
        else:
            wrk_map = pd.Series(dtype=float)

        for grp in model_ltuer['group_id'].unique():
            sim_sub = model_ltuer[model_ltuer['group_id'] == grp][['Year', 'ltuer']]
            obs_sub = (cps_ltuer[cps_ltuer['group_id'] == grp][['YEAR', 'ltuer']]
                       .rename(columns={'YEAR': 'Year'}))
            merged  = (sim_sub.merge(obs_sub, on='Year', suffixes=('_sim', '_obs'))
                       .sort_values('Year'))
            if len(merged) < min_years:
                continue

            obs_mean = merged['ltuer_obs'].mean()
            mse  = float(np.mean((merged['ltuer_sim'] - merged['ltuer_obs']) ** 2))
            nmse = mse / obs_mean ** 2 if obs_mean != 0 else np.nan

            diff_sim = merged['ltuer_sim'].diff().iloc[1:]
            diff_obs = merged['ltuer_obs'].diff().iloc[1:]
            if diff_sim.nunique() < 2 or diff_obs.nunique() < 2:
                rho = np.nan
                conc = np.nan
            else:
                rho = float(pearsonr(diff_sim, diff_obs)[0])
                _ds = diff_sim.values;  _do = diff_obs.values
                _mask = (_ds != 0) & (_do != 0)
                conc = float(np.mean(np.sign(_ds[_mask]) == np.sign(_do[_mask]))) if _mask.sum() > 0 else np.nan

            try:
                label = lab_map.get(int(grp), str(grp))
            except (ValueError, TypeError):
                label = str(grp)

            records.append({
                'Model':       model_name,
                'group_id':    grp,
                'label':       label,
                'n_years':     len(merged),
                'NMSE':        nmse,
                'Pearson':     rho,
                'Concordance': conc,
                'Workers':     float(wrk_map.get(grp, np.nan)),
            })

    return pd.DataFrame(records)


def plot_ltuer_ts_fit_scatter(
    ue_stock_dfs,
    cps_dur_dist,
    occ_crosswalk,
    calib_date,
    delay=25,
    sim_results_dict=None,
    grouping_col='SOC_major_adj',
    ltuer_threshold=6,
    min_years=6,
    colors=None,
    save=False,
    path=None,
):
    """
    Scatter of LTUER time-series fit — one point per (model, occupation group).

    x = NMSE between model and CPS annual LTUER series -- mean squared
        error normalized by the group's squared mean observed level, so
        groups with different baseline LTUER are compared on the same
        relative scale (lower = better level fit)
    y = Pearson correlation between year-over-year *changes* in the two
        series (higher = better co-movement; can be negative). Differencing
        avoids spurious correlation from a shared trend (e.g. COVID) that
        would otherwise inflate the correlation of raw levels.

    One subplot per model, sharing axes so fit quality is comparable across
    models at a glance. Point size scales with the occupation group's average
    Workers count (when sim_results_dict is supplied).
    """
    results = _compute_ltuer_ts_fit(
        ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay,
        sim_results_dict=sim_results_dict, grouping_col=grouping_col,
        ltuer_threshold=ltuer_threshold, min_years=min_years)

    if results.empty:
        print(f"No overlapping years for LTUER ts-fit ({grouping_col}).")
        return results

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                       'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    model_names = list(ue_stock_dfs.keys())
    n_models    = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.2),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for i, model_name in enumerate(model_names):
        ax  = axes[i]
        sub = results[results['Model'] == model_name]
        col = colors.get(model_name, 'steelblue')

        sizes = np.full(len(sub), 25.0)
        if sub['Workers'].notna().any():
            w = sub['Workers'].fillna(sub['Workers'].median())
            sizes = 15 + 60 * (w / w.max())

        ax.scatter(sub['NMSE'], sub['Pearson'], s=sizes, color=col, alpha=0.7,
                  edgecolor='white', linewidth=0.4)
        ax.axhline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax.axhline(1, color='green', ls='--', lw=1.2, alpha=0.8)
        ax.set_title(model_name, fontsize=9)
        ax.set_xlabel('NMSE (Sim vs. CPS)', fontsize=8)
        ax.set_xscale('log')
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel('Pearson Correlation of YoY Changes\n(Sim vs. CPS)', fontsize=8)
    fig.suptitle(f'LTUER time-series fit by occupation group ({grouping_col})',
                fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save and path:
        fig.savefig(os.path.join(path, f'ltuer_ts_fit_scatter_{grouping_col}.png'),
                   dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return results


def _render_ts_fit_density_row(
    ax_sse, ax_rho, results, model_names, colors, metric_label='LTUER', lag_months=12
):
    """
    Shared KDE-density rendering for one row of a ts-fit panel: NMSE
    (level fit -- mean squared error normalized by each group's squared
    mean observed level, log-space) on ax_sse, Pearson correlation of
    lag_months-lag changes (co-movement fit) on ax_rho.

    lag_months must match whatever lag the caller actually differenced by
    (e.g. _compute_ltuer_ts_fit/_compute_uer_ts_fit difference annual
    series, i.e. lag_months=12 / YoY; the monthly variants take an
    explicit lag_months that can be 1) -- it only controls the axis label
    text here, not the underlying computation, so a mismatch would mislabel
    a correctly-computed statistic rather than change it.

    Factored out so plot_ltuer_ts_fit_hist (one row) and
    plot_uer_ltuer_ts_fit_panel (UER row + LTUER row stacked) render
    identically apart from the underlying metric and its label.
    """
    from scipy.stats import gaussian_kde

    lag_label = 'YoY Changes' if lag_months == 12 else f'{lag_months}-month-lag changes'

    def _kde_curve(ax, vals, grid, color, label, log_x=False):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2 or np.ptp(vals) == 0:
            return
        try:
            kde = gaussian_kde(vals)
        except np.linalg.LinAlgError:
            return
        density = kde(grid)
        x = 10 ** grid if log_x else grid
        ax.plot(x, density, lw=1.0, color=color, label=label)

        # Mean marker, pinned to the curve rather than full-height, so it
        # doesn't compete visually with the full-height ideal-reference lines.
        mean_val = vals.mean()
        mean_x   = 10 ** mean_val if log_x else mean_val
        ax.vlines(mean_x, 0, kde(mean_val)[0], color=color, ls='--', lw=1.2, alpha=0.9)

    if results is None or results.empty:
        for ax in (ax_sse, ax_rho):
            ax.text(0.5, 0.5, f'{metric_label} data pending', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray', style='italic')
            ax.set_xticks([])
            ax.set_yticks([])
        return

    sse_pos = results.loc[results['NMSE'] > 0, 'NMSE']
    if len(sse_pos):
        log_sse = np.log10(sse_pos)
        pad = max(0.25, (log_sse.max() - log_sse.min()) * 0.1)
        sse_grid = np.linspace(log_sse.min() - pad, log_sse.max() + pad, 400)
    else:
        sse_grid = np.linspace(-1, 1, 400)
    rho_grid = np.linspace(-1.1, 1.1, 400)

    for model_name in model_names:
        sub = results[results['Model'] == model_name]
        col = colors.get(model_name, 'steelblue')

        sse_vals = sub.loc[sub['NMSE'] > 0, 'NMSE']
        _kde_curve(ax_sse, np.log10(sse_vals) if len(sse_vals) else sse_vals,
                   sse_grid, col, model_name, log_x=True)
        _kde_curve(ax_rho, sub['Pearson'], rho_grid, col, model_name)

    ax_sse.set_xscale('log')
    ax_sse.set_xlabel(f'Normalized MSE\n(Simulated vs. CPS)')
    ax_sse.set_ylabel('Density Across Occupations')
    ax_sse.grid(alpha=0.25, lw=0.5)
    # NMSE's ideal value (0) is unreachable on a log axis, so a line at the
    # ideal point isn't possible -- an arrow indicating direction (axes
    # fraction, not data coordinates, since sse_grid's range is recomputed
    # per-run and a data-anchored arrow could land anywhere/get clipped).
    ax_sse.annotate('Better fit', xy=(0.06, 0.94), xytext=(0.34, 0.94),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                     ha='left', va='center', fontsize=7.5, color='black',
                     style='italic')

    ax_rho.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax_rho.axvline(1, color='black', ls='--', lw=1.4, alpha=0.85)
    ax_rho.set_xlabel(f'Pearson Correlation of {lag_label}\n(Simulated vs. CPS)')
    ax_rho.grid(alpha=0.25, lw=0.5)


def plot_ltuer_ts_fit_hist(
    ue_stock_dfs,
    cps_dur_dist,
    occ_crosswalk,
    calib_date,
    delay=25,
    sim_results_dict=None,
    grouping_col='SOC_major_adj',
    ltuer_threshold=6,
    min_years=6,
    colors=None,
    save=False,
    path=None,
    dropbox_path=None,
):
    """
    Distribution of LTUER time-series fit across occupation groups, one
    overlaid KDE density curve per model — easier to compare across models
    than the per-point scatter when the per-group values themselves look
    noisy, and easier to read than binned histograms.

    Left panel: NMSE -- mean squared error normalized by each group's
    squared mean observed level (KDE fit in log-space, lower = better
    level fit, comparable across occupation groups with different
    baseline LTUER).
    Right panel: Pearson correlation of year-over-year changes in LTUER
    (higher = better co-movement fit; computed on differences, not levels,
    to avoid spurious correlation from a shared trend).
    """
    results = _compute_ltuer_ts_fit(
        ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay,
        sim_results_dict=sim_results_dict, grouping_col=grouping_col,
        ltuer_threshold=ltuer_threshold, min_years=min_years)

    if results.empty:
        print(f"No overlapping years for LTUER ts-fit ({grouping_col}).")
        return results

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                       'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    model_names = list(ue_stock_dfs.keys())

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT
    fig, (ax_sse, ax_rho) = plt.subplots(1, 2, figsize=(_fig_w, _fig_h))
    _render_ts_fit_density_row(ax_sse, ax_rho, results, model_names, colors, metric_label='LTUER', lag_months=12)

    fig.suptitle('Mean and Correlation Fit of Occupational LTUER Time Series',
                fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='black', linestyle='--',
                                              linewidth=1.4, alpha=0.7, label=f'$\\rho$ = 1')],
                       ncol=3)

    if save and path:
        fname = f'ltuer_ts_fit_hist_{grouping_col}.png'
        save_plot_standard(fig, os.path.join(path, fname), aspect_ratio=_fig_h / _fig_w)
        if dropbox_path:
            os.makedirs(dropbox_path, exist_ok=True)
            fname_pdf = os.path.splitext(fname)[0] + '.pdf'
            save_plot_standard(fig, os.path.join(dropbox_path, fname_pdf), aspect_ratio=_fig_h / _fig_w)
            print(f"✓ {fname_pdf} saved to Dropbox.")
        plt.close(fig)
    else:
        plt.show()

    return results


def _compute_uer_ts_fit(
    sim_results_dict, cps_uer_df, occ_crosswalk, calib_date, delay=25,
    grouping_col='SOC_major_adj', min_years=6
):
    """
    UER analog of _compute_ltuer_ts_fit. Builds the annual overall
    unemployment-rate series for each (model, occupation group) directly
    from the sim_results_dict panel (group UER = sum(Unemployment) /
    sum(Workers) across occupations in the group, averaged over the
    months in each year), and compares it to an annual CPS UER series via
    NMSE (level fit -- mean squared error normalized by the group's
    squared mean observed level, see _compute_ltuer_ts_fit for the
    rationale) and Pearson correlation of year-over-year changes
    (co-movement fit) -- identical convention to the LTUER fit.

    cps_uer_df must contain columns ['YEAR', 'group_id', 'uer'], with
    group_id matching the values in grouping_col (e.g. produced by an
    IPUMS-based annual occupation-level UER pipeline, analogous to the
    LTUER duration-distribution CSVs used for _compute_ltuer_ts_fit).
    Pass None/empty until that pipeline exists -- returns an empty frame
    in that case so callers can render a placeholder instead of erroring.

    Returns a DataFrame with columns ['Model', 'group_id', 'label',
    'n_years', 'NMSE', 'Pearson', 'Workers'] -- the same shape as
    _compute_ltuer_ts_fit's output, so both feed _render_ts_fit_density_row.
    """
    from scipy.stats import pearsonr

    _empty = pd.DataFrame(columns=['Model', 'group_id', 'label', 'n_years', 'NMSE', 'Pearson', 'Workers'])
    if cps_uer_df is None or cps_uer_df.empty:
        return _empty

    start_date = pd.Timestamp(calib_date[0])

    def _ts_to_year_vec(ts_series):
        months_offset = ts_series.astype(int) - delay - 1
        total_months  = (start_date.month - 1) + months_offset
        return (start_date.year + total_months // 12).values

    cw = occ_crosswalk[['X', grouping_col]].drop_duplicates()
    cw = cw.dropna(subset=[grouping_col]).copy()
    cw['Occupation'] = cw['X'].astype(int) - 1
    cw['group_id']   = cw[grouping_col]
    cw = cw[['Occupation', 'group_id']]

    desc_col = grouping_col.replace('_adj', '_description')
    if desc_col in occ_crosswalk.columns:
        lab_map = (occ_crosswalk[[grouping_col, desc_col]]
                   .dropna(subset=[grouping_col, desc_col])
                   .drop_duplicates(grouping_col)
                   .set_index(grouping_col)[desc_col]
                   .to_dict())
    else:
        lab_map = {}

    cps_uer = cps_uer_df.rename(columns={'YEAR': 'Year'})[['Year', 'group_id', 'uer']]

    records = []
    for model_name, sim_df in sim_results_dict.items():
        s = sim_df[sim_df['Time Step'] >= delay].copy()
        s['Year'] = _ts_to_year_vec(s['Time Step'])
        s = s.merge(cw, on='Occupation', how='left').dropna(subset=['group_id'])

        grp_ts = (s.groupby(['group_id', 'Time Step', 'Year'])
                    .agg(Unemployment=('Unemployment', 'sum'), Workers=('Workers', 'sum'))
                    .reset_index())
        grp_ts['uer'] = grp_ts['Unemployment'] / grp_ts['Workers']
        model_uer = grp_ts.groupby(['group_id', 'Year'])['uer'].mean().reset_index()
        wrk_map   = grp_ts.groupby('group_id')['Workers'].mean()

        for grp in model_uer['group_id'].unique():
            sim_sub = model_uer[model_uer['group_id'] == grp][['Year', 'uer']]
            obs_sub = cps_uer[cps_uer['group_id'] == grp][['Year', 'uer']]
            merged  = (sim_sub.merge(obs_sub, on='Year', suffixes=('_sim', '_obs'))
                       .sort_values('Year'))
            if len(merged) < min_years:
                continue

            obs_mean = merged['uer_obs'].mean()
            mse  = float(np.mean((merged['uer_sim'] - merged['uer_obs']) ** 2))
            nmse = mse / obs_mean ** 2 if obs_mean != 0 else np.nan

            diff_sim = merged['uer_sim'].diff().iloc[1:]
            diff_obs = merged['uer_obs'].diff().iloc[1:]
            if diff_sim.nunique() < 2 or diff_obs.nunique() < 2:
                rho = np.nan
            else:
                rho = float(pearsonr(diff_sim, diff_obs)[0])

            try:
                label = lab_map.get(int(grp), str(grp))
            except (ValueError, TypeError):
                label = str(grp)

            records.append({
                'Model':    model_name,
                'group_id': grp,
                'label':    label,
                'n_years':  len(merged),
                'NMSE':     nmse,
                'Pearson':  rho,
                'Workers':  float(wrk_map.get(grp, np.nan)),
            })

    return pd.DataFrame(records) if records else _empty


def plot_uer_ltuer_ts_fit_panel(
    ue_stock_dfs,
    cps_dur_dist,
    sim_results_dict,
    cps_uer_df,
    occ_crosswalk,
    calib_date,
    delay=25,
    grouping_col='SOC_major_adj',
    ltuer_threshold=6,
    min_years=6,
    colors=None,
    save=False,
    path=None,
    dropbox_path=None,
):
    """
    Combined 2x2 panel for one occupation-grouping level, intended as the
    final writeup figure: top row is the UER analog (NMSE density, Pearson
    density) of the bottom row, which is exactly plot_ltuer_ts_fit_hist's
    content for LTUER (same rendering code via _render_ts_fit_density_row).

    cps_uer_df is the annual CPS UER series (columns ['YEAR', 'group_id',
    'uer']) from the IPUMS UER pipeline. Pass None until that pipeline
    exists -- the top row then renders a "data pending" placeholder and
    the bottom (LTUER) row is still produced in full, so this can be
    wired in now and will complete automatically once cps_uer_df is ready.

    If dropbox_path is given (and save=True), also saves a second copy of
    the figure there -- e.g. for direct inclusion in an Overleaf project,
    mirroring the SAVE_FIG5_TO_DROPBOX pattern in entry_policy_exercise.py.
    """
    ltuer_results = _compute_ltuer_ts_fit(
        ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay,
        sim_results_dict=sim_results_dict, grouping_col=grouping_col,
        ltuer_threshold=ltuer_threshold, min_years=min_years)

    uer_results = _compute_uer_ts_fit(
        sim_results_dict, cps_uer_df, occ_crosswalk, calib_date, delay,
        grouping_col=grouping_col, min_years=min_years)

    if ltuer_results.empty and uer_results.empty:
        print(f"No overlapping years for either UER or LTUER ts-fit ({grouping_col}).")
        return ltuer_results, uer_results

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                       'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    model_names = list(ue_stock_dfs.keys())

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * 2
    fig, axes = plt.subplots(2, 2, figsize=(_fig_w, _fig_h))
    _render_ts_fit_density_row(axes[0, 0], axes[0, 1], uer_results, model_names, colors, metric_label='UER', lag_months=12)
    _render_ts_fit_density_row(axes[1, 0], axes[1, 1], ltuer_results, model_names, colors, metric_label='LTUER', lag_months=12)

    axes[0, 0].set_title('Unemployment rate (UER)', fontweight='bold', loc='left')
    axes[1, 0].set_title('Long-term unemployment rate (LTUER)', fontweight='bold', loc='left')

    fig.suptitle(f'Time-series fit distribution across occupation groups ({grouping_col})',
                fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='black', linestyle='--',
                                              linewidth=1.4, alpha=0.7, label=f'$\\rho$ = 1')],
                       ncol=3)

    if save and path:
        fname = f'uer_ltuer_ts_fit_panel_{grouping_col}.png'
        save_plot_standard(fig, os.path.join(path, fname), aspect_ratio=_fig_h / _fig_w)
        if dropbox_path:
            os.makedirs(dropbox_path, exist_ok=True)
            fname_pdf = os.path.splitext(fname)[0] + '.pdf'
            save_plot_standard(fig, os.path.join(dropbox_path, fname_pdf), aspect_ratio=_fig_h / _fig_w)
            print(f"✓ {fname_pdf} saved to Dropbox.")
        plt.close(fig)
    else:
        plt.show()

    return ltuer_results, uer_results


def _compute_uer_ts_fit_monthly(
    sim_results_dict, cps_uer_monthly_df, occ_crosswalk, calib_date, delay=25,
    grouping_col='SOC_major_adj', value_col='uer', lag_months=12, min_obs=24
):
    """
    Monthly analog of _compute_uer_ts_fit. sim_results_dict's 'Time Step'
    is already monthly resolution -- each Time Step maps to exactly one
    (Year, Month) via the same delay/calib_date arithmetic used to
    collapse to Year elsewhere, so no aggregation across months is
    needed on the sim side (only across occupations within a group, at
    each Time Step).

    Compares the resulting monthly group UER series to a monthly CPS UER
    series via NMSE (level fit -- mean squared error of raw monthly
    levels, normalized by the group's squared mean observed level, see
    _compute_ltuer_ts_fit for the rationale) and Pearson correlation of
    `lag_months`-lag differences (co-movement fit).
    Lag defaults to 12 months rather than 1: a 1-month diff would leave
    seasonality in the CPS series uncorrected for, and a model without
    matching seasonal dynamics would be unfairly penalized (or
    spuriously rewarded) against it -- a 12-month lag strips trend AND
    seasonality the same way the annual diff stripped trend alone. Going
    monthly also multiplies the number of differenced points per
    occupation group roughly twelvefold relative to the annual version,
    directly addressing the small-N problem that motivated min_years there.
    If value_col points at an already-deseasonalized column (e.g.
    'uer_sa' or 'uer_sa_stl'), pass lag_months=1 instead -- seasonality
    has already been removed at the CSV-build stage, so a 1-month lag
    isolates short-run co-movement rather than re-stripping seasonality
    that's no longer there.

    cps_uer_monthly_df must contain columns ['YEAR', 'MONTH', 'group_id',
    value_col]. Both series are reindexed onto a contiguous Year*12+Month
    integer axis before differencing, so the lag is a true calendar-month
    comparison even where either series has internal gaps.

    Returns a DataFrame with columns ['Model', 'group_id', 'label',
    'n_obs', 'NMSE', 'Pearson', 'Workers'] -- same shape convention as
    _compute_uer_ts_fit, so both feed _render_ts_fit_density_row.
    """
    from scipy.stats import pearsonr

    _empty = pd.DataFrame(columns=['Model', 'group_id', 'label', 'n_obs', 'NMSE', 'Pearson', 'Workers'])
    if (cps_uer_monthly_df is None or cps_uer_monthly_df.empty
            or value_col not in cps_uer_monthly_df.columns):
        return _empty

    start_date = pd.Timestamp(calib_date[0])

    def _ts_to_year_month(ts_series):
        months_offset = ts_series.astype(int) - delay - 1
        total_months  = (start_date.month - 1) + months_offset
        year  = start_date.year + total_months // 12
        month = total_months % 12 + 1
        return year.values, month.values

    cw = occ_crosswalk[['X', grouping_col]].drop_duplicates()
    cw = cw.dropna(subset=[grouping_col]).copy()
    cw['Occupation'] = cw['X'].astype(int) - 1
    cw['group_id']   = cw[grouping_col]
    cw = cw[['Occupation', 'group_id']]

    desc_col = grouping_col.replace('_adj', '_description')
    if desc_col in occ_crosswalk.columns:
        lab_map = (occ_crosswalk[[grouping_col, desc_col]]
                   .dropna(subset=[grouping_col, desc_col])
                   .drop_duplicates(grouping_col)
                   .set_index(grouping_col)[desc_col]
                   .to_dict())
    else:
        lab_map = {}

    cps_uer = cps_uer_monthly_df.rename(columns={'YEAR': 'Year', 'MONTH': 'Month'})[
        ['Year', 'Month', 'group_id', value_col]].rename(columns={value_col: 'uer'}).copy()
    cps_uer = cps_uer.dropna(subset=['uer'])
    cps_uer['month_idx'] = cps_uer['Year'] * 12 + cps_uer['Month']

    records = []
    for model_name, sim_df in sim_results_dict.items():
        s = sim_df[sim_df['Time Step'] >= delay].copy()
        s['Year'], s['Month'] = _ts_to_year_month(s['Time Step'])
        s['month_idx'] = s['Year'] * 12 + s['Month']
        s = s.merge(cw, on='Occupation', how='left').dropna(subset=['group_id'])

        grp_ts = (s.groupby(['group_id', 'month_idx'])
                    .agg(Unemployment=('Unemployment', 'sum'), Workers=('Workers', 'sum'))
                    .reset_index())
        grp_ts['uer'] = grp_ts['Unemployment'] / grp_ts['Workers']
        wrk_map = grp_ts.groupby('group_id')['Workers'].mean()

        for grp in grp_ts['group_id'].unique():
            sim_sub = grp_ts[grp_ts['group_id'] == grp][['month_idx', 'uer']]
            obs_sub = cps_uer[cps_uer['group_id'] == grp][['month_idx', 'uer']]
            merged  = (sim_sub.merge(obs_sub, on='month_idx', suffixes=('_sim', '_obs'))
                       .sort_values('month_idx'))
            if len(merged) < min_obs:
                continue

            # Reindex onto a contiguous monthly axis so .diff(lag_months)
            # below is a true calendar-lag comparison even across gaps.
            full_idx = pd.RangeIndex(merged['month_idx'].min(), merged['month_idx'].max() + 1)
            merged = merged.set_index('month_idx').reindex(full_idx)

            obs_mean = merged['uer_obs'].mean()
            mse  = float(np.nanmean((merged['uer_sim'] - merged['uer_obs']) ** 2))
            nmse = mse / obs_mean ** 2 if obs_mean != 0 else np.nan

            diff_sim = merged['uer_sim'].diff(lag_months)
            diff_obs = merged['uer_obs'].diff(lag_months)
            valid = diff_sim.notna() & diff_obs.notna()
            if valid.sum() < min_obs or diff_sim[valid].nunique() < 2 or diff_obs[valid].nunique() < 2:
                rho = np.nan
            else:
                rho = float(pearsonr(diff_sim[valid], diff_obs[valid])[0])

            try:
                label = lab_map.get(int(grp), str(grp))
            except (ValueError, TypeError):
                label = str(grp)

            records.append({
                'Model':    model_name,
                'group_id': grp,
                'label':    label,
                'n_obs':    int(valid.sum()),
                'NMSE':     nmse,
                'Pearson':  rho,
                'Workers':  float(wrk_map.get(grp, np.nan)),
            })

    return pd.DataFrame(records) if records else _empty


def plot_uer_ts_fit_hist_monthly(
    sim_results_dict,
    cps_uer_monthly_df,
    occ_crosswalk,
    calib_date,
    delay=25,
    grouping_col='SOC_major_adj',
    lag_months=12,
    min_obs=24,
    colors=None,
    save=False,
    path=None,
    dropbox_path=None,
):
    """
    Monthly analog of plot_ltuer_ts_fit_hist, for UER at native monthly
    resolution (12-month-lag differencing instead of annual differencing
    -- see _compute_uer_ts_fit_monthly). Reuses _render_ts_fit_density_row
    so the visual structure matches the annual/LTUER versions exactly.
    """
    results = _compute_uer_ts_fit_monthly(
        sim_results_dict, cps_uer_monthly_df, occ_crosswalk, calib_date, delay,
        grouping_col=grouping_col, lag_months=lag_months, min_obs=min_obs)

    if results.empty:
        print(f"No overlapping months for monthly UER ts-fit ({grouping_col}).")
        return results

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                       'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(sim_results_dict.keys(), default_colors))

    model_names = list(sim_results_dict.keys())

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT
    fig, (ax_sse, ax_rho) = plt.subplots(1, 2, figsize=(_fig_w, _fig_h))
    _render_ts_fit_density_row(ax_sse, ax_rho, results, model_names, colors, metric_label='UER (monthly)', lag_months=lag_months)

    fig.suptitle(f'Monthly UER time-series fit distribution across occupation groups ({grouping_col})',
                fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='black', linestyle='--',
                                              linewidth=1.4, alpha=0.7, label=f'$\\rho$ = 1')],
                       ncol=3)

    if save and path:
        fname = f'uer_ts_fit_hist_monthly_{grouping_col}.png'
        save_plot_standard(fig, os.path.join(path, fname), aspect_ratio=_fig_h / _fig_w)
        if dropbox_path:
            os.makedirs(dropbox_path, exist_ok=True)
            fname_pdf = os.path.splitext(fname)[0] + '.pdf'
            save_plot_standard(fig, os.path.join(dropbox_path, fname_pdf), aspect_ratio=_fig_h / _fig_w)
            print(f"✓ {fname_pdf} saved to Dropbox.")
        plt.close(fig)
    else:
        plt.show()

    return results


def _compute_ltuer_ts_fit_monthly(
    ue_stock_dfs, cps_ltuer_monthly_df, occ_crosswalk, calib_date, delay=25,
    sim_results_dict=None, grouping_col='SOC_major_adj', value_col='ltuer',
    ltuer_threshold=6, lag_months=1, min_obs=24
):
    """
    Monthly analog of _compute_ltuer_ts_fit, mirroring
    _compute_uer_ts_fit_monthly's structure exactly (month_idx = Year*12
    + Month in place of Year, lag_months-diff in place of annual diff).

    value_col selects which column of cps_ltuer_monthly_df to compare
    against -- 'ltuer' (raw), 'ltuer_sa' (decompose()-deseasonalized), or
    'ltuer_sa_stl' (robust-STL-deseasonalized). lag_months should stay at
    12 for the raw column (strips trend AND seasonality together, same
    rationale as _compute_uer_ts_fit_monthly) but drop to 1 for either
    _sa variant, since seasonality has already been removed at the
    CSV-build stage -- a 1-month lag then isolates short-run co-movement
    instead of re-stripping seasonality that's no longer there.

    Returns a DataFrame with columns ['Model', 'group_id', 'label',
    'n_obs', 'NMSE', 'Pearson', 'Workers'] -- same shape convention as
    _compute_uer_ts_fit_monthly, so both feed _render_ts_fit_density_row.
    NMSE is mean squared error normalized by the group's squared mean
    observed level, same rationale as _compute_ltuer_ts_fit.
    """
    from scipy.stats import pearsonr

    _empty = pd.DataFrame(columns=['Model', 'group_id', 'label', 'n_obs', 'NMSE', 'Pearson', 'Workers'])
    if (cps_ltuer_monthly_df is None or cps_ltuer_monthly_df.empty
            or value_col not in cps_ltuer_monthly_df.columns):
        return _empty

    start_date = pd.Timestamp(calib_date[0])

    def _ts_to_year_month(ts_series):
        months_offset = ts_series.astype(int) - delay - 1
        total_months  = (start_date.month - 1) + months_offset
        year  = start_date.year + total_months // 12
        month = total_months % 12 + 1
        return year.values, month.values

    cw = occ_crosswalk[['X', grouping_col]].drop_duplicates()
    cw = cw.dropna(subset=[grouping_col]).copy()
    cw['Occupation'] = cw['X'].astype(int) - 1
    cw['group_id']   = cw[grouping_col]
    cw = cw[['Occupation', 'group_id']]

    desc_col = grouping_col.replace('_adj', '_description')
    if desc_col in occ_crosswalk.columns:
        lab_map = (occ_crosswalk[[grouping_col, desc_col]]
                   .dropna(subset=[grouping_col, desc_col])
                   .drop_duplicates(grouping_col)
                   .set_index(grouping_col)[desc_col]
                   .to_dict())
    else:
        lab_map = {}

    cps_ltuer = cps_ltuer_monthly_df.rename(columns={'YEAR': 'Year', 'MONTH': 'Month'})[
        ['Year', 'Month', 'group_id', value_col]].rename(columns={value_col: 'ltuer'}).copy()
    cps_ltuer = cps_ltuer.dropna(subset=['ltuer'])
    cps_ltuer['month_idx'] = cps_ltuer['Year'] * 12 + cps_ltuer['Month']

    records = []
    for model_name, stock_df in ue_stock_dfs.items():
        s = stock_df[stock_df['Time Step'] >= delay].copy()
        s['Year'], s['Month'] = _ts_to_year_month(s['Time Step'])
        s['month_idx'] = s['Year'] * 12 + s['Month']
        s = s.merge(cw, on='Occupation', how='left').dropna(subset=['group_id'])
        if 'LongtermUnemp' in s.columns:
            s['_lt'] = s['LongtermUnemp'].astype(bool)
        else:
            s['_lt'] = (s['UEDuration'] - 1) >= ltuer_threshold

        _multisim = 'sim_id' in s.columns
        group_by  = ['group_id', 'month_idx', 'sim_id'] if _multisim else ['group_id', 'month_idx']
        grp_ts = (s.groupby(group_by)['_lt']
                    .apply(lambda x: x.sum() / max(len(x), 1))
                    .reset_index(name='ltuer'))
        if _multisim:
            grp_ts = grp_ts.groupby(['group_id', 'month_idx'])['ltuer'].mean().reset_index()

        if sim_results_dict is not None and model_name in sim_results_dict:
            sim_rec = sim_results_dict[model_name].copy()
            sim_rec = sim_rec[sim_rec['Time Step'] >= delay]
            sim_rec = (sim_rec.merge(cw, on='Occupation', how='left')
                              .dropna(subset=['group_id']))
            wrk_map = sim_rec.groupby('group_id')['Workers'].mean()
        else:
            wrk_map = pd.Series(dtype=float)

        for grp in grp_ts['group_id'].unique():
            sim_sub = grp_ts[grp_ts['group_id'] == grp][['month_idx', 'ltuer']]
            obs_sub = cps_ltuer[cps_ltuer['group_id'] == grp][['month_idx', 'ltuer']]
            merged  = (sim_sub.merge(obs_sub, on='month_idx', suffixes=('_sim', '_obs'))
                       .sort_values('month_idx'))
            if len(merged) < min_obs:
                continue

            full_idx = pd.RangeIndex(merged['month_idx'].min(), merged['month_idx'].max() + 1)
            merged = merged.set_index('month_idx').reindex(full_idx)

            obs_mean = merged['ltuer_obs'].mean()
            mse  = float(np.nanmean((merged['ltuer_sim'] - merged['ltuer_obs']) ** 2))
            nmse = mse / obs_mean ** 2 if obs_mean != 0 else np.nan

            diff_sim = merged['ltuer_sim'].diff(lag_months)
            diff_obs = merged['ltuer_obs'].diff(lag_months)
            valid = diff_sim.notna() & diff_obs.notna()
            if valid.sum() < min_obs or diff_sim[valid].nunique() < 2 or diff_obs[valid].nunique() < 2:
                rho = np.nan
            else:
                rho = float(pearsonr(diff_sim[valid], diff_obs[valid])[0])

            try:
                label = lab_map.get(int(grp), str(grp))
            except (ValueError, TypeError):
                label = str(grp)

            records.append({
                'Model':    model_name,
                'group_id': grp,
                'label':    label,
                'n_obs':    int(valid.sum()),
                'NMSE':     nmse,
                'Pearson':  rho,
                'Workers':  float(wrk_map.get(grp, np.nan)),
            })

    return pd.DataFrame(records) if records else _empty


def plot_uer_ltuer_ts_fit_panel_monthly(
    ue_stock_dfs,
    sim_results_dict,
    cps_monthly_df,
    occ_crosswalk,
    calib_date,
    delay=25,
    grouping_col='SOC_major_adj',
    uer_col='uer',
    ltuer_col='ltuer',
    uer_lag_months=12,
    ltuer_lag_months=1,
    ltuer_threshold=6,
    min_obs=24,
    series_label='raw',
    colors=None,
    save=False,
    path=None,
    dropbox_path=None,
):
    """
    Monthly analog of plot_uer_ltuer_ts_fit_panel: top row = UER monthly
    fit, bottom row = LTUER monthly fit, both read from the same
    cps_monthly_df (the unemp_rate_monthly_{level}.csv file, which now
    carries the raw uer/ltuer columns alongside decompose()-deseasonalized
    (uer_sa/ltuer_sa) and robust-STL-deseasonalized (uer_sa_stl/
    ltuer_sa_stl) versions side by side). uer_col/ltuer_col pick which
    pair to use; series_label is purely cosmetic (figure title/filename).

    uer_lag_months/ltuer_lag_months are independent because the two rows
    answer different questions: the UER row defaults to 12 to match
    plot_uer_ts_fit_hist_monthly's standalone 12-month-lag (YoY) UER fit
    exactly, while the LTUER row defaults to 1 since LTUER's own
    deseasonalized variants need lag_months=1 -- see
    _compute_uer_ts_fit_monthly and _compute_ltuer_ts_fit_monthly for the
    rationale.

    If dropbox_path is given (and save=True), also saves a second copy of
    the figure there, mirroring plot_uer_ltuer_ts_fit_panel's convention.
    """
    uer_results = _compute_uer_ts_fit_monthly(
        sim_results_dict, cps_monthly_df, occ_crosswalk, calib_date, delay,
        grouping_col=grouping_col, value_col=uer_col, lag_months=uer_lag_months, min_obs=min_obs)

    ltuer_results = _compute_ltuer_ts_fit_monthly(
        ue_stock_dfs, cps_monthly_df, occ_crosswalk, calib_date, delay,
        sim_results_dict=sim_results_dict, grouping_col=grouping_col,
        value_col=ltuer_col, ltuer_threshold=ltuer_threshold,
        lag_months=ltuer_lag_months, min_obs=min_obs)

    if uer_results.empty and ltuer_results.empty:
        print(f"No overlapping months for monthly {series_label} UER/LTUER ts-fit ({grouping_col}).")
        return uer_results, ltuer_results

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                       'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(sim_results_dict.keys(), default_colors))

    model_names = list(sim_results_dict.keys())

    _fig_w, _fig_h = TEXTWIDTH, ROW_HEIGHT * 2
    fig, axes = plt.subplots(2, 2, figsize=(_fig_w, _fig_h))
    _render_ts_fit_density_row(axes[0, 0], axes[0, 1], uer_results, model_names, colors,
                                metric_label=f'UER ({series_label}, monthly)', lag_months=uer_lag_months)
    _render_ts_fit_density_row(axes[1, 0], axes[1, 1], ltuer_results, model_names, colors,
                                metric_label=f'LTUER ({series_label}, monthly)', lag_months=ltuer_lag_months)

    axes[0, 0].set_title('Unemployment rate (UER)', fontweight='bold', loc='left')
    axes[1, 0].set_title('Long-term unemployment rate (LTUER)', fontweight='bold', loc='left')

    fig.suptitle(f'Monthly time-series fit distribution across occupation groups '
                f'({series_label}, {grouping_col})', fontweight='bold')
    plt.tight_layout()
    add_suptitle_spacing(fig)
    add_shared_legend(fig, colors,
                       extra_handles=[Line2D([0], [0], color='black', linestyle='--',
                                              linewidth=1.4, alpha=0.7, label=f'$\\rho$ = 1')],
                       ncol=3)

    if save and path:
        fname = f'uer_ltuer_ts_fit_panel_monthly_{series_label}_{grouping_col}.png'
        save_plot_standard(fig, os.path.join(path, fname), aspect_ratio=_fig_h / _fig_w)
        if dropbox_path:
            os.makedirs(dropbox_path, exist_ok=True)
            fname_pdf = os.path.splitext(fname)[0] + '.pdf'
            save_plot_standard(fig, os.path.join(dropbox_path, fname_pdf), aspect_ratio=_fig_h / _fig_w)
            print(f"✓ {fname_pdf} saved to Dropbox.")
        plt.close(fig)
    else:
        plt.show()

    return uer_results, ltuer_results

    return results


def _compute_ltuer_xcorr(
    ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay=25,
    grouping_col='SOC_major_adj', ltuer_threshold=6, lags=(-1, 0, 1),
    min_years=6
):
    """
    Cross-correlation (lead/lag) version of the LTUER co-movement check:
    for each (model, occupation group), correlate year-over-year changes in
    the model's LTUER series against year-over-year changes in the CPS
    series shifted by each value in `lags`.

    Convention: lag > 0 means the model's change in a given year is
    compared to CPS's change `lag` years later (the model *leads* CPS by
    `lag` years); lag < 0 compares to CPS's change `lag` years earlier
    (the model *lags* CPS). lag = 0 is the contemporaneous case already
    computed by _compute_ltuer_ts_fit.

    Intended only for the most aggregated grouping levels (SOC_major_adj,
    SOC_minor_adj): shifting costs another observation on top of
    differencing, and annual CPS cells at finer occupation levels are
    already too thin to support that without an even smaller min_years
    floor.

    Returns a DataFrame with columns ['Model', 'group_id', 'label', 'lag',
    'n_obs', 'Pearson'].
    """
    from scipy.stats import pearsonr

    start_date = pd.Timestamp(calib_date[0])

    def _ts_to_year_vec(ts_series):
        months_offset = ts_series.astype(int) - delay - 1
        total_months  = (start_date.month - 1) + months_offset
        return (start_date.year + total_months // 12).values

    cw = occ_crosswalk[['X', grouping_col]].drop_duplicates()
    cw = cw.dropna(subset=[grouping_col]).copy()
    cw['Occupation'] = cw['X'].astype(int) - 1
    cw['group_id']   = cw[grouping_col]
    cw = cw[['Occupation', 'group_id']]

    desc_col = grouping_col.replace('_adj', '_description')
    if desc_col in occ_crosswalk.columns:
        lab_map = (occ_crosswalk[[grouping_col, desc_col]]
                   .dropna(subset=[grouping_col, desc_col])
                   .drop_duplicates(grouping_col)
                   .set_index(grouping_col)[desc_col]
                   .to_dict())
    else:
        lab_map = {}

    cps_ltuer = (cps_dur_dist.groupby(['YEAR', 'group_id'])
                 .apply(lambda x: (x.loc[x['dur_months'] >= ltuer_threshold, 'n_weighted'].sum()
                                    / max(x['n_weighted'].sum(), 1)))
                 .reset_index(name='ltuer'))

    records = []
    for model_name, stock_df in ue_stock_dfs.items():
        s = stock_df[stock_df['Time Step'] >= delay].copy()
        s['Year'] = _ts_to_year_vec(s['Time Step'])
        s = s.merge(cw, on='Occupation', how='left').dropna(subset=['group_id'])
        if 'LongtermUnemp' in s.columns:
            s['_lt'] = s['LongtermUnemp'].astype(bool)
        else:
            s['_lt'] = (s['UEDuration'] - 1) >= ltuer_threshold

        _multisim = 'sim_id' in s.columns
        group_by  = ['group_id', 'Year', 'sim_id'] if _multisim else ['group_id', 'Year']
        model_ltuer = (s.groupby(group_by)
                        .apply(lambda x: x['_lt'].sum() / max(len(x), 1))
                        .reset_index(name='ltuer'))
        if _multisim:
            model_ltuer = (model_ltuer.groupby(['group_id', 'Year'])['ltuer']
                           .mean().reset_index())

        for grp in model_ltuer['group_id'].unique():
            sim_sub = model_ltuer[model_ltuer['group_id'] == grp][['Year', 'ltuer']]
            obs_sub = (cps_ltuer[cps_ltuer['group_id'] == grp][['YEAR', 'ltuer']]
                       .rename(columns={'YEAR': 'Year'}))
            if sim_sub.empty or obs_sub.empty:
                continue

            lo = int(max(sim_sub['Year'].min(), obs_sub['Year'].min()))
            hi = int(min(sim_sub['Year'].max(), obs_sub['Year'].max()))
            if hi - lo + 1 < min_years:
                continue
            full_years = pd.RangeIndex(lo, hi + 1)

            # Reindex to a contiguous year range (rather than relying on an
            # inner merge) so that .shift() below moves by actual years,
            # not just by row position, even if either series has gaps.
            sim_y = sim_sub.drop_duplicates('Year').set_index('Year')['ltuer'].reindex(full_years)
            obs_y = obs_sub.drop_duplicates('Year').set_index('Year')['ltuer'].reindex(full_years)

            diff_sim = sim_y.diff()
            diff_obs = obs_y.diff()

            try:
                label = lab_map.get(int(grp), str(grp))
            except (ValueError, TypeError):
                label = str(grp)

            for lag in lags:
                shifted_obs = diff_obs.shift(-lag)
                paired = pd.concat([diff_sim, shifted_obs], axis=1).dropna()
                n_obs = len(paired)
                if n_obs < max(min_years - abs(lag) - 1, 3):
                    rho = np.nan
                elif paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
                    rho = np.nan
                else:
                    rho = float(pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])[0])

                records.append({
                    'Model':    model_name,
                    'group_id': grp,
                    'label':    label,
                    'lag':      lag,
                    'n_obs':    n_obs,
                    'Pearson': rho,
                })

    return pd.DataFrame(records)


def plot_ltuer_xcorr_hist(
    ue_stock_dfs,
    cps_dur_dist,
    occ_crosswalk,
    calib_date,
    delay=25,
    grouping_col='SOC_major_adj',
    ltuer_threshold=6,
    lags=(-1, 0, 1),
    min_years=6,
    colors=None,
    save=False,
    path=None,
):
    """
    Lead/lag cross-correlation check for LTUER co-movement: one KDE-density
    panel per lag in `lags`, each showing the across-occupation-group
    distribution of Pearson correlation (on YoY Changes) between the
    model and CPS series, one curve per model.

    Meant for the most aggregated grouping levels only (SOC_major_adj,
    SOC_minor_adj) — see _compute_ltuer_xcorr for why finer levels don't
    have enough annual observations to support shifting by a year.
    """
    from scipy.stats import gaussian_kde

    results = _compute_ltuer_xcorr(
        ue_stock_dfs, cps_dur_dist, occ_crosswalk, calib_date, delay,
        grouping_col=grouping_col, ltuer_threshold=ltuer_threshold,
        lags=lags, min_years=min_years)

    if results.empty:
        print(f"No overlapping years for LTUER cross-correlation ({grouping_col}).")
        return results

    default_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                       'purple', 'brown', 'teal']
    if colors is None:
        colors = dict(zip(ue_stock_dfs.keys(), default_colors))

    model_names = list(ue_stock_dfs.keys())
    n_lags = len(lags)

    fig, axes = plt.subplots(1, n_lags, figsize=(4.2 * n_lags, 4.2),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    grid = np.linspace(-1.1, 1.1, 400)

    _lag_titles = {}
    for lag in lags:
        if lag == 0:
            _lag_titles[lag] = 'Contemporaneous (lag = 0)'
        elif lag > 0:
            _lag_titles[lag] = f'Model leads CPS by {lag}yr'
        else:
            _lag_titles[lag] = f'Model lags CPS by {abs(lag)}yr'

    for i, lag in enumerate(lags):
        ax = axes[i]
        sub_lag = results[results['lag'] == lag]
        for model_name in model_names:
            sub = sub_lag[sub_lag['Model'] == model_name]
            vals = sub['Pearson'].dropna().values
            col = colors.get(model_name, 'steelblue')
            if len(vals) < 2 or np.ptp(vals) == 0:
                continue
            try:
                kde = gaussian_kde(vals)
            except np.linalg.LinAlgError:
                continue
            density = kde(grid)
            ax.plot(grid, density, lw=1.8, color=col, label=model_name)
            ax.fill_between(grid, density, alpha=0.12, color=col)

            mean_val = vals.mean()
            ax.vlines(mean_val, 0, kde(mean_val)[0], color=col, ls='--', lw=1.2, alpha=0.9)

        ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax.axvline(1, color='green', ls='--', lw=1.2, alpha=0.8)
        ax.set_title(_lag_titles[lag], fontsize=9)
        ax.set_xlabel('Pearson Correlation\nof YoY Changes', fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel('Density Across Occupations', fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(f'LTUER lead/lag cross-correlation by occupation group ({grouping_col})',
                fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save and path:
        fig.savefig(os.path.join(path, f'ltuer_xcorr_hist_{grouping_col}.png'),
                   dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return results
