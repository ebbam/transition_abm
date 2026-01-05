import numpy as np
import pandas as pd
import seaborn as sns
import math as math
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from scipy import stats
from plot_ltuer_options import *

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
def plot_ltuer(res_dict, macro_obs, sep_strings=None, sep=False, save=False, path=None):
    def plot_models(ax, models, title, observed, add_legend=False):
        lines = []
        labels = []
        
        for name, res in models.items():
            line, = ax.plot(res['DATE'], res['LTUER'])
            lines.append(line)
            labels.append(name)
        
        obs_line, = ax.plot(observed['DATE'], observed['LTUER'], color="black", linestyle="dotted")
        lines.append(obs_line)
        labels.append("Observed")
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("DATE")
        ax.set_ylabel("LTUER")
        
        # Add legend to THIS subplot if requested
        if add_legend:
            ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), 
                     ncol=1, frameon=True, fontsize=10)
        
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
        fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 6), sharey=True)

        if n_subplots == 1:
            axes = [axes]

        # Plot each subplot with its own legend
        for i, (match_str, title) in enumerate(sep_strings):
            plot_models(axes[i], categorized[match_str], f"LTUER - {title}", observed, add_legend=True)

        if unmatched:
            plot_models(axes[-1], unmatched, "LTUER - Other Models", observed, add_legend=True)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for legends

    else:
        # Single combined plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_models(ax, res_dict, "LTUER - All Models", observed, add_legend=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for legend

    # Save or show plot
    if save:
        plt.savefig(f'{path}ltuer_line.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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

def plot_bev_curve_color(res_dict, macro_obs, sep_strings=None, sep=False, smooth=1, save=False, path=None, viridis_color = 'viridis'):
    """
    Plot Beveridge curves with optional smoothing.
    
    Parameters:
    -----------
    smooth : int
        Window size for moving average smoothing. Default is 1 (no smoothing).
        Higher values (e.g., 3, 6, 12) create smoother curves.
    """
    
    def plot_single_model(ax, name, res, smooth_window=1):
        import matplotlib.cm as cm
        from matplotlib.collections import LineCollection
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
        
        # Convert dates to numeric values for coloring
        dates_numeric = mdates.date2num(pd.to_datetime(dates))
        
        # Create points for line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a continuous norm for coloring based on dates
        norm = plt.Normalize(dates_numeric.min(), dates_numeric.max())
        lc = LineCollection(segments, cmap=viridis_color, norm=norm, linewidth=2)
        lc.set_array(dates_numeric[:-1])
        
        # Add the line collection to the plot
        line = ax.add_collection(lc)
        
        # Add scatter points colored by date
        scatter = ax.scatter(x, y, c=dates_numeric, cmap=viridis_color, norm=norm,
                            s=50, zorder=3, edgecolors='white', linewidth=0.5)
        
        # Plot observed data (optionally smoothed)
        if smooth_window > 1:
            obs_x = macro_obs['UER'].rolling(window=smooth_window, center=True).mean()
            obs_y = macro_obs['VACRATE'].rolling(window=smooth_window, center=True).mean()
            ax.plot(obs_x, obs_y, 
                   label=f"Observed (smoothed)", color="grey", alpha=0.7, linewidth=1.5, zorder=1)
        else:
            ax.plot(macro_obs['UER'], macro_obs['VACRATE'], 
                   label="Observed", color="grey", alpha=0.7, linewidth=1.5, zorder=1)
        
        # Set labels and title
        title_suffix = f" (MA={smooth_window})" if smooth_window > 1 else ""
        ax.set_title(name + title_suffix, fontweight='bold')
        ax.set_xlabel('UE Rate')
        ax.set_ylabel('Vac Rate')
        ax.legend(loc='best')
        
        # Set axis limits to ensure all data is visible
        ax.set_xlim(x.min() * 0.95, x.max() * 1.05)
        ax.set_ylim(y.min() * 0.95, y.max() * 1.05)
        
        return scatter, dates_numeric

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

        fig = plt.figure(figsize=(5 * num_cols + 1, 4 * max_rows))
        
        # Create grid spec with space for colorbar
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(max_rows, num_cols + 1, figure=fig, 
                              width_ratios=[1]*num_cols + [0.05], 
                              wspace=0.3, hspace=0.3)
        
        axes = np.empty((max_rows, num_cols), dtype=object)
        for row in range(max_rows):
            for col in range(num_cols):
                axes[row, col] = fig.add_subplot(gs[row, col])

        # Track scatter plot for colorbar
        scatter_ref = None
        all_dates = []
        
        for col, (match_str, models) in enumerate(categorized.items()):
            for row, (name, res) in enumerate(models.items()):
                ax = axes[row, col]
                scatter, dates_numeric = plot_single_model(ax, name, res, smooth_window=smooth)
                all_dates.extend(dates_numeric)
                if scatter_ref is None:
                    scatter_ref = scatter

            # Remove empty subplots in this column
            for r in range(len(models), max_rows):
                axes[r, col].axis('off')

            # Set column title at the top row
            axes[0, col].set_title(titles[match_str], loc='center', fontsize=12, fontweight='bold')

        # Add colorbar to the right of all plots
        if scatter_ref is not None:
            import matplotlib.dates as mdates
            norm = plt.Normalize(min(all_dates), max(all_dates))
            sm = plt.cm.ScalarMappable(cmap=viridis_color, norm=norm)
            sm.set_array([])
            
            # Create colorbar axis on the right
            cbar_ax = fig.add_subplot(gs[:, -1])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label('Date', rotation=270, labelpad=20)
            
            # Format colorbar ticks as dates and reverse order
            cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            cbar.ax.yaxis.set_major_locator(mdates.YearLocator(1))
            cbar.ax.invert_yaxis()  # Reverse the colorbar (earliest at top)

    else:
        # Non-separated grid layout
        n = len(res_dict)
        max_cols = 3
        cols = min(n, max_cols)
        rows = math.ceil(n / cols)

        fig = plt.figure(figsize=(5 * cols + 1, 4 * rows))
        
        # Create grid spec with space for colorbar
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(rows, cols + 1, figure=fig,
                              width_ratios=[1]*cols + [0.05],
                              wspace=0.3, hspace=0.3)
        
        axes = []
        for row in range(rows):
            for col in range(cols):
                if len(axes) < n:
                    axes.append(fig.add_subplot(gs[row, col]))

        scatter_ref = None
        all_dates = []
        for i, (name, res) in enumerate(res_dict.items()):
            scatter, dates_numeric = plot_single_model(axes[i], name, res, smooth_window=smooth)
            all_dates.extend(dates_numeric)
            if scatter_ref is None:
                scatter_ref = scatter

        # Add colorbar with dates
        if scatter_ref is not None:
            import matplotlib.dates as mdates
            norm = plt.Normalize(min(all_dates), max(all_dates))
            sm = plt.cm.ScalarMappable(cmap=viridis_color, norm=norm)
            sm.set_array([])
            
            # Create colorbar axis on the right
            cbar_ax = fig.add_subplot(gs[:, -1])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label('Date', rotation=270, labelpad=20)
            
            # Format colorbar ticks as dates and reverse order
            cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            cbar.ax.yaxis.set_major_locator(mdates.YearLocator(1))
            cbar.ax.invert_yaxis()  # Reverse the colorbar (earliest at top)

    if save:
        plt.savefig(f'{path}bev_curves.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        for i, (name, res) in enumerate(mod_results_dict.items()):
            # Resample to annual frequency
            if freq != 'M':
                annual_data = res.set_index('DATE').resample(freq).mean()
            else:
                annual_data = res

            # Create arrays for x-axis values
            dates_array = np.array([d for d in annual_data.index])
            ones_array = np.ones_like(dates_array)
            
            # Plot unemployed relative wages (with optional smoothing)
            u_wages_raw = annual_data['U_REL_WAGE_MEAN'].values
            u_wages = apply_smoothing(u_wages_raw, use_lowess=lowess, frac=lowess_frac)
            
            ax1.plot(dates_array, u_wages, 
                    color=colors[name], label=name, zorder=3, linewidth=2)
            
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
                    color=colors[name], label=name, zorder=3, linewidth=2)
            
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

        # Capture the lines and labels from one of the axes
        handles, labels = ax1.get_legend_handles_labels()
        # Create single legend below both plots
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                ncol=2, fontsize=10)

        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

    else:
        fig = plt.figure(figsize=(12, 6))
        
        for i, (name, res) in enumerate(mod_results_dict.items()):
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
                    color=colors[name], label=name, zorder=3, linewidth=2)
            
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

        # Create legend below the plot
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                ncol=4, fontsize=9, frameon=True)

        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
    
    # Save or show
    if save:
        plt.savefig(f'{path}relative_wages_{suffix}.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:    
        plt.show()

def hires_seps_rate(mod_results_dict, jolts, save=False, path=None, colors = None):

    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(mod_results_dict.keys(), colors))
    # Step 1: Separate models
    model_groups = {
        "Non-behavioral": {k: v for k, v in mod_results_dict.items() if "Non-behavioural" in k},
        "Behavioral": {k: v for k, v in mod_results_dict.items() if "Non-behavioural" not in k}
    }

    # Step 2: Setup grid: rows = model groups, cols = [Hires Rate, Separations Rate]
    fig, axes = plt.subplots(4, 2, figsize=(20, 12), sharex=True)
    rate_types = [("Hires Rate", "HIRESRATE"), ("Separations Rate", "SEPSRATE"), 
                  ("EE_Trans_Rate", 'QUITSRATE'),
                   ("E-U Rate", 'LAYOFFRATE')]

    # Add overall column titles using group_name
    for col_idx, group_name in enumerate(model_groups.keys()):
        fig.text(
            0.25 + col_idx * 0.5, 0.97, group_name,
            ha='center', va='top', fontsize=18, fontweight='bold'
        )

    # Add row titles spanning both columns
    row_positions = [0.89, 0.69, 0.49, 0.29]  # Adjust these to position titles for each row
    for row_idx, (rate_col_sim, rate_col_obs) in enumerate(rate_types):
        title = f"{rate_col_sim} - {rate_col_obs}"
        fig.text(
            0.5, row_positions[row_idx], title,
            ha='center', va='bottom', fontsize=14, fontweight='bold'
        )

    # Step 3: Loop through rows (model groups) and columns (rate types)
    for col_idx, (group_name, group_models) in enumerate(model_groups.items()):
        for row_idx, (rate_col_sim, rate_col_obs) in enumerate(rate_types):
            ax = axes[row_idx, col_idx]

            # Plot observed JOLTS rate
            ax.plot(jolts['DATE'], jolts[rate_col_obs],
                    label=f'Observed', color='black', linestyle='--')

            # Plot each model in the group
            for model_name, df in group_models.items():
                ax.plot(df['DATE'], df[rate_col_sim], label=model_name, color = colors[model_name])

            # Remove individual subplot titles
            # (Title is now handled by fig.text above)
            
            ax.set_ylabel(rate_col_sim)
            ax.set_xlim(df['DATE'].min(), df['DATE'].max())
            if row_idx == 3:
                ax.set_xlabel('Date')
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, -1), 
               ncol=1, fontsize=10)
            

    plt.subplots_adjust(hspace=0.3)  # hspace adds vertical space
    #plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save:
        plt.savefig(f'{path}hires_seps_rate_grid.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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
#         ax.set_xlabel("DATE")
#         ax.set_ylabel("Unemployment Rate")
#         ax.legend()

#     def plot_vac(ax, models, title):
#         for i, (name, res) in enumerate(models.items()):
#             ax.plot(res['DATE'], res['VACRATE'], label=f'Sim. Vac Rate: {name}', color=colors[i % len(colors)])
#         ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="grey", linestyle="dotted", label="Obs. VR")
#         if recessions is not None:
#             for _, row in recessions.iterrows():
#                 ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
#         ax.set_xlabel("DATE")
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

def plot_uer_vac_steady_state(res_dict, macro_obs, calib_date, plot_colors=None, recessions=None, sep_strings=None, sep=False, save=False, path=None, free_date_scale=False, delay_ref=None, smooth=None, suffix = None):
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
                            freq="M"
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
                
                if recessions is not None:
                    for _, row in recessions.iterrows():
                        ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
                
                # Add vertical line at calib_date[0] (where calibration starts)
                if delay_ref is not None:
                    ax.axvline(x=start, color='black', linestyle='--', linewidth=2, label='Calibration Start')
                
                # Set title (bold)
                ax.set_title(title, fontweight="bold", pad=15)
                
                # Add subtitle if smoothing is applied (italic, non-bold)
                if smooth is not None and smooth > 1:
                    ax.text(0.5, 1.01, f"Smoothing over {smooth} points", 
                        transform=ax.transAxes, ha='center', va='bottom', 
                        fontsize=8, style='italic', color='black')
                
                ax.set_xlabel("DATE")
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
                fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 10), sharex='col')
                if num_plots == 1:
                    axes = np.array(axes).reshape(1, 1)

                for i, (match_str, models) in enumerate(categorized.items()):
                    # Plot top row (UER) and capture legend info
                    lines, labels = plot_uer(axes[i], models, titles[match_str], show_legend=True)
                    
                    # Add legend at the bottom of each column
                    axes[i].legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), 
                                    ncol=1, frameon=True)
                    
            else:
                # Plot all together on single subplot
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Plot all models on the same axes
                lines, labels = plot_uer(ax, res_dict, "Unemployment Rate - All Models", show_legend=True)
                
                # Add legend
                ax.legend(lines, labels, loc='best', frameon=True)
            
            plt.tight_layout()

            # Save or show
            if save:
                plt.savefig(f'{path}uer_vac_{suffix}.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
                plt.close()
            else:
                plt.show()

def plot_uer_vac(res_dict, macro_obs, calib_date, recessions=None, sep_strings=None, sep=False, save=False, path=None, free_date_scale=False):
    colors = ["cornflowerblue", "darkorange"]
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
                    freq="M"
                )
            line, = ax.plot(res['DATE'], res['UER'], color=colors[i % len(colors)])
            if show_legend:
                lines.append(line)
                labels.append(name)
        
        obs_line, = ax.plot(macro_obs['DATE'], macro_obs['UER'], color="grey", linestyle="dotted")
        if show_legend:
            lines.append(obs_line)
            labels.append("Observed")
        
        if recessions is not None:
            for _, row in recessions.iterrows():
                ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
        ax.set_title(title, fontweight = "bold")
        ax.set_xlabel("DATE")
        ax.set_ylabel("Unemployment Rate")
        
        return lines, labels

    def plot_vac(ax, models, title):
        for i, (name, res) in enumerate(models.items()):
            if free_date_scale:
                res['DATE'] = pd.date_range(
                    start=calib_date[0],
                    periods=len(res),
                    freq="M"
                )
            ax.plot(res['DATE'], res['VACRATE'], color=colors[i % len(colors)])
        ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="grey", linestyle="dotted")

        if recessions is not None:
            for _, row in recessions.iterrows():
                ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
        ax.set_xlabel("DATE")
        ax.set_ylabel("Vacancy Rate")

    def plot_uer_error(ax, models, title):
        for i, (name, res) in enumerate(models.items()):
            if free_date_scale:
                res['DATE'] = pd.date_range(
                    start=calib_date[0],
                    periods=len(res),
                    freq="M"
                )
            if len(res['UER']) == len([macro_obs['UER']]):
                ax.fill_between(res['DATE'], 0, res['UER'] - macro_obs['UER'], color=colors[i % len(colors)], alpha=0.5)
            else:
                ax.fill_between(res['DATE'], 0, res['UER'] - macro_obs['UER'].iloc[:len(res['UER'])], color=colors[i % len(colors)], alpha=0.5)

        if recessions is not None:
            for _, row in recessions.iterrows():
                ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
        ax.set_xlabel("DATE")
        ax.set_ylabel("UER Error (Simulated-Observed)")

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
        fig, axes = plt.subplots(3, num_plots, figsize=(8 * num_plots, 10), sharex='col')
        if num_plots == 1:
            axes = np.array(axes).reshape(3, 1)

        for i, (match_str, models) in enumerate(categorized.items()):
            # Plot top row (UER) and capture legend info
            lines, labels = plot_uer(axes[0, i], models, titles[match_str], show_legend=True)
            
            # Plot middle and bottom rows
            plot_uer_error(axes[1, i], models, titles[match_str])
            plot_vac(axes[2, i], models, titles[match_str])
            
            # Add legend at the bottom of each column
            axes[2, i].legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), 
                            ncol=1, frameon=True)

        # Ensure same y-axis limits for the UER error row (2nd row)
        y_mins, y_maxs = [], []
        for ax in axes[1, :]:
            y_min, y_max = ax.get_ylim()
            y_mins.append(y_min)
            y_maxs.append(y_max)
        common_min = min(y_mins)
        common_max = max(y_maxs)
        for ax in axes[1, :]:
            ax.set_ylim(common_min, common_max)

    else:
        # All in separate plots (grid layout)
        n = len(res_dict)
        fig, axes = plt.subplots(2, n, figsize=(6 * n, 8), sharex='col')
        if n == 1:
            axes = np.array(axes).reshape(2, 1)

        for i, (name, res) in enumerate(res_dict.items()):
            lines, labels = plot_uer(axes[0, i], {name: res}, name, show_legend=True)
            plot_vac(axes[1, i], {name: res}, name)
            
            # Add legend at the bottom of each column
            axes[1, i].legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), 
                            ncol=1, frameon=True)

    plt.tight_layout()
    
    # Adjust layout to make room for legends
    plt.subplots_adjust(bottom=0.08)

    # Save or show
    if save:
        plt.savefig(f'{path}uer_vac.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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
        ax.set_xlabel("DATE")
        ax.set_ylabel("Unemployment Rate")
        ax.legend()

    def plot_vac(ax, models, title):
        for i, (name, res) in enumerate(models.items()):
            ax.plot(res['DATE'], res['VACRATE'], label=f'Sim. Vac Rate: {name}', color=colors[i % len(colors)])
        ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="red", linestyle="dotted", label="Obs. VR")
        if recessions is not None:
            for _, row in recessions.iterrows():
                ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)
        ax.set_xlabel("DATE")
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
def plot_seeker_comp(res_dict, observation, share=False, sep=False, save=False, path=None, colors=None):
    import matplotlib.patches as mpatches
    
    n = len(res_dict)
    if sep:
        cols = 2
    else:
        max_cols = 3
        cols = min(n, max_cols)
        
    rows = math.ceil(n / cols)

    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), colors))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()  # Flatten to make indexing easy

    for i, (name, res) in enumerate(res_dict.items()):
        ax = axes[i]

        if not share:
            stack = ax.stackplot(res['DATE'], res['Employed Seekers'], res['Unemployed Seekers'], 
                                 colors=[colors[name], 'lightgrey'])
        elif share:
            stack = ax.stackplot(
                res['DATE'],
                res["Employed Seekers"] / (res["Employed Seekers"] + res["Unemployed Seekers"]),
                res["Unemployed Seekers"] / (res["Employed Seekers"] + res["Unemployed Seekers"]),
                colors=[colors[name], "lightgrey"]
            )

        # Plot observed line
        obs_line, = ax.plot(observation['DATE'], observation['Seeker Composition'], 
                           color="black", linestyle="dotted")
        
        ax.set_title(name)

    # Create custom legend handles
    # Multi-colored patch for Emp Seekers (using gradient or stripes)
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    
    # Create a composite patch showing all employed seeker colors
    # Option 1: Create individual colored rectangles
    emp_colors = [colors[name] for name in res_dict.keys()]
    
    # Create a custom handler for multi-color legend
    class MultiColorPatch:
        def __init__(self, colors):
            self.colors = colors
    
    class MultiColorPatchHandler:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            width, height = handlebox.width, handlebox.height
            n_colors = len(orig_handle.colors)
            patch_width = width / n_colors
            
            patches = []
            for i, color in enumerate(orig_handle.colors):
                patch = Rectangle((i * patch_width, 0), patch_width, height, 
                                 facecolor=color, edgecolor='none', transform=handlebox.get_transform())
                handlebox.add_artist(patch)
                patches.append(patch)
            return patches[0]
    
    # Create legend handles
    emp_handle = MultiColorPatch(emp_colors)
    unemp_handle = mpatches.Patch(color='lightgrey', label='Unemp Seekers')
    obs_handle = mpatches.Patch(edgecolor='black', facecolor='none', linestyle='dotted', label='Observed')
    
    # Alternative: Use Line2D for observed
    from matplotlib.lines import Line2D
    obs_handle = Line2D([0], [0], color='black', linestyle='dotted', label='Observed')
    
    handles = [emp_handle, unemp_handle, obs_handle]
    labels = ['Emp Seekers', 'Unemp Seekers', 'Observed']

    # Set common axis labels and title
    fig.suptitle("Monthly Composition of Job Seekers", fontsize=14, y=0.98, fontweight='bold')
    fig.supxlabel("Date")
    fig.supylabel("Composition of Job Seekers")
    
    # Add single legend with custom handler
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
               ncol=3, frameon=True, fontsize=10,
               handler_map={MultiColorPatch: MultiColorPatchHandler()})

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Show plot
    if save:
        plt.savefig(f'{path}seeker_composition.png', bbox_inches='tight',  
            pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()
    
def plot_seeker_comp_line(res_dict, observation, save = False, path = None, colors = None):

    if colors is None:
        colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'cyan']
        colors = dict(zip(res_dict.keys(), colors))

    plt.figure(figsize=(10, 6))
    for i, (name, res) in enumerate(res_dict.items()):
        plt.plot(res['DATE'], res["Seeker Composition"], label=name, color = colors[name])

        # Set axis label and title
    plt.plot(observation['DATE'], observation['Seeker Composition'], label='Observed', color='grey', linestyle = "dashed")
    plt.title("Monthly Composition of Job Seekers", fontsize=14)  # Figure-wide title
    plt.xlabel("Date")  # Shared x-axis label
    plt.ylabel("Employed Share of Job Seekers")  # Shared y-axis label
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), 
               ncol=2, fontsize=10)

    # Show plot
    if save:
        plt.savefig(f'{path}seeker_composition_line.png', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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

    else:
        # two side-by-side panels that share the y-axis
        fig, (ax_full, ax_zoom) = plt.subplots(
            1, 2, figsize=(12, 6), sharey=True
        )

        # Store handles and labels for legend
        handles = []
        labels = []

        for idx, (name, net) in enumerate(net_dict.items()):
            total_unemp = []
            for occ in net:
                total_unemp.extend(
                    wrkr.time_unemployed
                    for wrkr in occ.list_of_unemployed
                )

            #color = colors[idx % len(colors)]

            # full histogram
            _, _, patches_full = ax_full.hist(total_unemp, bins=n_bins, alpha=0.5,
                                              color=colors[name])
            ax_full.axvline(np.mean(total_unemp), ls='--', lw=1, color=colors[name])

            # zoomed histogram (0–36 mos)
            ax_zoom.hist(total_unemp, bins=n_bins, range=(0, 36), alpha=0.5,
                        color=colors[name])
            ax_zoom.axvline(np.mean(total_unemp), ls='--', lw=1, color=colors[name])

            # Collect handles and labels for legend
            handles.append(patches_full[0])
            labels.append(names[idx])

        # labels and titles
        ax_full.set_xlabel("Time Unemployed (months)")
        ax_zoom.set_xlabel("Time Unemployed (≤ 36 months)")
        ax_full.set_ylabel("Number of Workers")
        ax_full.set_title("Full distribution", fontweight='bold')
        ax_zoom.set_title("Zoom ≤ 3 years", fontweight='bold')
        ax_zoom.set_xlim(0, 36)

        # Single legend below both panels
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                  ncol=min(len(labels), 4), frameon=True, fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

    if save:
        fig.savefig(f'{path}ltuer_distributions.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
        plt.close()
    else:
        plt.show()



####################################################
############## CURRENT VS TARGET DEMAND ############
####################################################
def plot_cd_vs_td(res_dict, save = False, path = None, colors = None):

    if colors is None:
        colors = ['orange', 'blue', 'purple', 'green', 'pink', 'grey', 'red', 'cyan', 'brown']
        colors = dict(zip(res_dict.keys(), colors))
    for i, (name, res) in enumerate(res_dict.items()):
        plt.plot(res['DATE'], res['Current_Demand'], label=f'{name}', color=colors[name])
        if i == 0:
            plt.plot(res['DATE'], res['Target_Demand'], label=f'TD', color='grey', linestyle='dashed')
        else:
            plt.plot(res['DATE'], res['Target_Demand'], color='grey', linestyle='dashed')

    plt.title("Current vs Target Demand", fontweight = "bold")
    plt.xlabel('Date')
    plt.ylabel("Current or Target Labor Demand")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save the figure to the output folder
    if save:
        plt.savefig(f'{path}cd_vs_td.png', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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
def plot_trans_rates(mod_results_dict, observation, save=False, path=None, colors = None):
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
    fig = plt.figure(figsize=(19, 12))
    
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
                    fmt='o', color=colors[name], label=name, 
                    capsize=5, markersize=10)
        ax1.errorbar(ue_x, ue_mean, yerr=ue_std, 
                    fmt='o', color=colors[name], 
                    capsize=5, markersize=10)
        
        # Plot time series on right panels
        ax2.plot(res['DATE'], res['UE_Trans_Rate'], color=colors[name], label=name)
        ax3.plot(res['DATE'], res['EE_Trans_Rate'], color=colors[name], label=name)
    
    ax1.errorbar(base_positions[0], obs_ee_mean, yerr=obs_ee_std, 
                    fmt='o', color="grey", 
                    capsize=5, markersize=10)
    ax1.errorbar(base_positions[1], obs_ue_mean, yerr=obs_ue_std, 
                    fmt='o', color="grey", label="Observed", 
                    capsize=5, markersize=10)
    # Customize left panel
    ax1.set_title('Mean Transition Rates with Standard Deviation', fontweight='bold')
    ax1.set_xticks(base_positions)
    ax1.set_xlim(0, 1)
    ax1.set_xticklabels(['EE', 'UE'])
    ax1.set_ylabel('Transition Rate')
    #ax1.legend()
    
    # Customize right panels
    ax2.set_title('UE Transition Rate Over Time', fontweight='bold')
    ax2.plot(observation['DATE'], observation['UE'], color="grey", label="Observed")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rate')
    #ax2.legend()
    
    ax3.set_title('EE Transition Rate Over Time', fontweight='bold')
    ax3.plot(observation['DATE'], observation['EE'], color="grey", label="Observed")
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Rate')
    #ax3.legend()



    # Capture the lines and labels from one of the axes
    handles, labels = ax1.get_legend_handles_labels()
    # Create single legend below both plots
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),  
            ncol=1, fontsize=10)

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Adjust layout
    plt.tight_layout()

    
    # Save or show
    if save:
        plt.savefig(f'{path}transition_rates_comparison.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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

    fig1 = plt.figure(figsize=(5*num_cols, 4*num_rows + 1))
    gs1 = gridspec.GridSpec(num_rows * 2, num_cols, height_ratios=[4, 0.7]*num_rows)
    fig1.suptitle("Unemployment Rate (UER): Simulated vs Observed (Sorted by Observed UER)", fontsize=16)

    fig2 = plt.figure(figsize=(5*num_cols, 4*num_rows + 1))
    gs2 = gridspec.GridSpec(num_rows * 2, num_cols, height_ratios=[4, 0.7]*num_rows)
    fig2.suptitle("LT Unemployment Rate (LTUER): Simulated vs Observed (Sorted by Observed LTUER)", fontsize=16)

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
            if row_idx == 0 and col_idx == 0:
                ax1.legend()

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
            if row_idx == 0 and col_idx == 0:
                ax2.legend()

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
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
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
        plt.savefig(f'{path}ltuer_difference_heatmap_{difference_type}_absval_{abs_value}.jpg', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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
    plt.figure(figsize=(10, 8))
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
        plt.savefig(f'{path}"network_graph.png', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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
    
    # Set up grid
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 12))
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
    
    fig.suptitle(title, size=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save:
        plt.savefig(f'{path}/net_entry_heatmap_grid.png', bbox_inches='tight',  
            pad_inches=0.1,      
            dpi=300)
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
    
    # Set up grid
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 12))
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
    
    fig.suptitle(title, size=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save:
        plt.savefig(f'{path}/net_entry_stability_grid.png', dpi=300, bbox_inches='tight',  
            pad_inches=0.1)
        plt.close()
    else:
        plt.show()
