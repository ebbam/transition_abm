import numpy as np
import pandas as pd
import seaborn as sns
import math as math
import matplotlib.pyplot as plt
from scipy import stats


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
    ue_vac['LTUE Rate'] = ue_vac['LT Unemployed Persons'] / ue_vac['Unemployment']
    ue_vac['DATE'] = pd.date_range(start=date_start, end= date_end, periods=len(sim_record))

    return ue_vac

####################################################
############## LTUER ###############################
####################################################

def plot_ltuer(res_dict, macro_obs, sep_strings=None, sep=False, save=False, path=None):
    def plot_models(ax, models, title, observed):
        for name, res in models.items():
            ax.plot(res['DATE'], res['LTUE Rate'], label=name)
        ax.plot(observed['DATE'], observed['LTUER'], color="black", linestyle="dotted", label="Observed")
        ax.set_title(title)
        ax.set_xlabel("DATE")
        ax.set_ylabel("LTUER")
        ax.legend(loc="best")

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

        for i, (match_str, title) in enumerate(sep_strings):
            plot_models(axes[i], categorized[match_str], f"LTUER - {title}", observed)

        if unmatched:
            plot_models(axes[-1], unmatched, "LTUER - Other Models", observed)

        plt.tight_layout()

    else:
        # Single combined plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_models(ax, res_dict, "LTUER - All Models", observed)

    # Save or show plot
    if save:
        plt.savefig(f'{path}ltuer_line.jpg', dpi=300)
    
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
        plt.savefig(f'{path}bev_curves.jpg', dpi=300)
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
        plt.savefig(f'{path}avg_wage_overall.jpg', dpi=300)
        
    plt.show()


####################################################
############## RELATIVE WAGES ##########
####################################################
def plot_rel_wages(mod_results_dict, names, save=False, path=None, freq = 'Y'):
    """
    Creates plots of relative wages with annual smoothing:
    - Left: Unemployed relative wages
    - Right: Employed relative wages
    Includes shading to show deviations above/below 1.0
    
    Parameters:
    - mod_results_dict: Dictionary containing simulation results DataFrames
    - save: Boolean indicating whether to save the plots
    - path: Path to save the plots if save=True
    """
    colors = ["orange", "blue", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    
    # Create figure with two subplots
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
        
        # Plot unemployed relative wages (smoothed)
        u_wages = annual_data['U_REL_WAGE_MEAN'].values
        #u_wages = np.clip(u_wages, None, 1.7)
        ax1.plot(dates_array, u_wages, 
                color=colors[i], label=names[i], marker='o', zorder=3)
        # Add shading
        ax1.fill_between(dates_array, u_wages, 1,
                        color=colors[i], alpha=0.2, zorder=2)
        
        # Plot employed relative wages (smoothed)
        e_wages = annual_data['E_REL_WAGE_MEAN'].values
        ax2.plot(dates_array, e_wages, 
                color=colors[i], label=names[i], marker='o', zorder=3)
        # Add shading
        ax2.fill_between(dates_array, e_wages, 1,
                        color=colors[i], alpha=0.2, zorder=2)
        
        # # Add the original data as light lines in the background
        # ax1.plot(res['DATE'], res['U_REL_WAGE_MEAN'], 
        #         color=colors[i], alpha=0.1, linewidth=1, zorder=1)
        # ax2.plot(res['DATE'], res['E_REL_WAGE_MEAN'], 
        #         color=colors[i], alpha=0.1, linewidth=1, zorder=1)
    
    # Add horizontal line at y=1
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, zorder=2)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, zorder=2)
    
    # Customize unemployed plot
    ax1.set_title('Unemployed Relative Wages', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Relative Wages')
    ax1.legend()
    
    # Customize employed plot  
    ax2.set_title('Employed Relative Wages', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Relative Wages')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save:
        plt.savefig(f'{path}relative_wages.jpg', dpi=300)
    
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



def plot_uer_vac(res_dict, macro_obs, recessions=None, sep_strings=None, sep=False, save=False, path=None):
    colors = ["skyblue", "lightcoral", "purple", "green", "orange", "brown", "pink"]
    def plot_group(ax, models, title):
        for i, (name, res) in enumerate(models.items()):
            ax.plot(res['DATE'], res['VACRATE'], label= 'Sim. Vac Rate', color=colors[i])
            ax.plot(res['DATE'], res['UER'], label = "Sim. UER", color=colors[i+1])

        # Observed data
        ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="red", linestyle="dotted", label="Obs. VR")
        ax.plot(macro_obs['DATE'], macro_obs['UER'], color="blue", linestyle="dotted", label="Obs. UER")

        # Recessions shaded area (if provided)
        if recessions is not None:
            for _, row in recessions.iterrows():
                ax.axvspan(row['start'], row['end'], color='grey', alpha=0.2)

        ax.set_title(title)
        ax.set_xlabel("DATE")
        ax.set_ylabel("Rate")
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
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True)

        if num_plots == 1:
            axes = [axes]

        for i, (match_str, models) in enumerate(categorized.items()):
            plot_group(axes[i], models, titles[match_str])

        plt.tight_layout()

    else:
        # All in separate plots (grid layout)
        n = len(res_dict)
        max_cols = 3
        cols = min(n, max_cols)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axes = axes.flatten()

        for i, (name, res) in enumerate(res_dict.items()):
            ax = axes[i]
            plot_group(ax, {name: res}, name)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()


    # Save or show
    if save:
        plt.savefig(f'{path}uer_vac.jpg', dpi=300)
    plt.show()



####################################################
############## SEEKER COMPOSITION ##################
####################################################
def plot_seeker_comp(res_dict, share = False, sep = False, save = False, path = None):
    n = len(res_dict)
    if sep:
        cols = 2
    else:
        max_cols = 3
        cols = min(n, max_cols)
    
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()  # Flatten to make indexing easy

    for i, (name, res) in enumerate(res_dict.items()):
        ax = axes[i]

        if not share:
            ax.stackplot(res['DATE'], res['Employed Seekers'], res['Unemployed Seekers'], 
            labels=['Employed Seekers', 'Unemployed Seekers'], colors=['lightblue', 'lightcoral'])

        elif share:
            ax.stackplot(
                res['DATE'],
                res["Employed Seekers"] / (res["Employed Seekers"] + res["Unemployed Seekers"]),
                res["Unemployed Seekers"] / (res["Employed Seekers"] + res["Unemployed Seekers"]),
                labels=["Emp Seekers", "Unemp Seekers"],
                colors=["lightblue", "lightcoral"]
            )

        # Set individual titles
        ax.set_title(name)
        # Set common axis labels
        # Set common axis labels and title outside the loop
        fig.suptitle("Monthly Composition of Job Seekers", fontsize=14)  # Figure-wide title
        fig.supxlabel("Date")  # Shared x-axis label
        fig.supylabel("Composition of Job Seekers")  # Shared y-axis label


        if ax.has_data():  # Check if the subplot has data before adding a legend
            ax.legend(loc="upper right")  # Apply legend to relevant subplots
        #recessions.plot.area(ax = a,  x= 'DATE', color = "grey", alpha = 0.2)


    # Show plot
    if save:
        plt.savefig(f'{path}seeker_composition.png', dpi = 300)
    plt.show()



####################################################
############## GENDER WAGE GAPS ####################
####################################################
def plot_gender_gaps(net_dict, names, sep = False, save = False, path = None):
    """
    Function to plot the gender wage disttribution across models
    """
    n = len(net_dict)
    if sep:
        cols = 2
    else:
        max_cols = 4
        cols = min(n, max_cols)
        
    rows = math.ceil(n / cols)

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
            w_wage += sum([wrkr.wage for wrkr in occ.list_of_employed if wrkr.female])
            m_wage += sum([wrkr.wage for wrkr in occ.list_of_employed if not(wrkr.female)])
        
            
        t= " \n" + " \n" +  "Female share of employed: " + str(round((women/emp_counter)*100)) + "% \n" + "Mean Female Wage: $" + str(round(w_wage/women)) + "\n" + "Mean Male Wage: $" + str(round(m_wage/men)) + "\n" + "Gender wage gap: " + str(round(100*(1 - (w_wage/women)/(m_wage/men)))) + "%" + "\n" + "--------------------"

        n_bins = 10
        women = np.array(w_wages)
        men = np.array(m_wages)

        # We can set the number of bins with the *bins* keyword argument.
        ax.hist(women, bins=n_bins, alpha = 0.3, color = 'purple', label = 'Women', fill = True, hatch = '.')
        ax.hist(men, bins=n_bins, alpha = 0.3, label = 'Men', color = 'green', fill = True, hatch = '.')  
        ax.axvline(women.mean(), color='purple', linestyle='dashed', linewidth=1, label = 'Women Avg.')
        ax.axvline(men.mean(), color='green', linestyle='dashed', linewidth=1, label = 'Men Avg.')
        ax.legend(loc='upper right') 
        ax.annotate(
                t,
                xy=(0.5, 0.5),
                xycoords='axes fraction',
                fontsize=7,
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
            )
        ax.set_title(names[i])

    fig.supxlabel("Wage")  # Shared x-axis label
    fig.suptitle('Distribution of Male and Female Wages', fontsize = 15) 
    fig.subplots_adjust(bottom=0.1)


    if save:
        plt.savefig(f'{path}gender_wage_gaps.jpg', dpi = 300)
    plt.show()



####################################################
############## LTUER DISTRIBUTIONS #################
####################################################
def plot_ltuer_dist(net_dict, names,
                    gender=False, sep=False,
                    save=False, path="./"):

    n_bins  = 30
    max_cols = 3
    colors  = ['orange', 'blue', 'brown', 'green',
               'blue', 'grey', 'red', 'cyan']

    # ---------------------------------------------------------------
    #  gender branch (unchanged)
    # ---------------------------------------------------------------
    if gender:
        import math, numpy as np, matplotlib.pyplot as plt

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
            ax.set_xlabel("Time Unemployed (sim months)")
            ax.set_ylabel("Number of Workers (Ks)")
            ax.legend()

        # hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle('Distribution of Time Spent Unemployed by Gender per Model')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ---------------------------------------------------------------
    #  combined plot branch  (modified)
    # ---------------------------------------------------------------
    else:
        import numpy as np, matplotlib.pyplot as plt

        # two side-by-side panels that share the y-axis
        fig, (ax_full, ax_zoom) = plt.subplots(
            1, 2, figsize=(12, 6), sharey=True
        )

        for idx, (name, net) in enumerate(net_dict.items()):
            total_unemp = []
            for occ in net:
                total_unemp.extend(
                    wrkr.time_unemployed
                    for wrkr in occ.list_of_unemployed
                )

            color = colors[idx % len(colors)]

            # full histogram
            ax_full.hist(total_unemp, bins=n_bins, alpha=0.5,
                         label=names[idx], color=color)
            ax_full.axvline(np.mean(total_unemp), ls='--', lw=1, color=color)

            # zoomed histogram (0–36 weeks)
            ax_zoom.hist(total_unemp, bins=n_bins, range=(0, 36), alpha=0.5,
                         label=names[idx], color=color)
            ax_zoom.axvline(np.mean(total_unemp), ls='--', lw=1, color=color)

        # labels and titles
        ax_full.set_xlabel("Time Unemployed (weeks)")
        ax_zoom.set_xlabel("Time Unemployed (weeks, zoom ≤ 36)")
        ax_full.set_ylabel("Number of Workers")
        ax_full.set_title("Full distribution")
        ax_zoom.set_title("Zoom ≤ 12 weeks")
        ax_zoom.set_xlim(0, 12)

        # one legend for both panels
        ax_full.legend(loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ---------------------------------------------------------------
    #  save & show
    # ---------------------------------------------------------------
    if save:
        fig.savefig(f'{path}ltuer_distributions.jpg', dpi=300,
                    bbox_inches="tight")
    plt.show()



####################################################
############## CURRENT VS TARGET DEMAND ############
####################################################
def plot_cd_vs_td(res_dict, save = False, path = None):

    colors = ['orange', 'blue', 'purple', 'green', 'pink', 'grey', 'red', 'cyan', 'brown']
    for i, (name, res) in enumerate(res_dict.items()):
        plt.plot(res['DATE'], res['Current_Demand'], label=f'CD {name}', color=colors[i])
        if i == 0:
            plt.plot(res['DATE'], res['Target_Demand'], label=f'TD', color='grey', linestyle='dashed')
        else:
            plt.plot(res['DATE'], res['Target_Demand'], color='grey', linestyle='dashed')

    plt.title("Current vs Target Demand")
    # axes2[1].set_ylim(0.015, 0.055)
    # axes2[1].set_xlim(0.03, 0.125)
    plt.legend(loc = 'upper right')

    plt.tight_layout()

    # Save the figure to the output folder
    if save:
        plt.savefig(f'{path}cd_vs_td.png', dpi=300)
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
            plt.savefig(f'{path}avg_wage_by_occupation_{name}.jpg', dpi=300)
        plt.show()


####################################################
############## TRANSITION RATES #####################
####################################################
def plot_trans_rates(mod_results_dict, observation, names, save=False, path=None):
    """
    Creates a two-column plot showing:
    - Left: Mean transition rates with standard deviation error bars for EE and UE
    - Right: Time series of transition rates stacked vertically
    
    Parameters:
    - mod_results_dict: Dictionary containing simulation results DataFrames
    - save: Boolean indicating whether to save the plots
    - path: Path to save the plots if save=True
    """
    colors = ["orange", "blue", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    
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
                    fmt='o', color=colors[i], label=names[i], 
                    capsize=5, markersize=10)
        ax1.errorbar(ue_x, ue_mean, yerr=ue_std, 
                    fmt='o', color=colors[i], 
                    capsize=5, markersize=10)
        
        # Plot time series on right panels
        ax2.plot(res['DATE'], res['UE_Trans_Rate'], color=colors[i], label=names[i])
        ax3.plot(res['DATE'], res['EE_Trans_Rate'], color=colors[i], label=names[i])
    
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
    ax1.legend()
    
    # Customize right panels
    ax2.set_title('UE Transition Rate Over Time', fontweight='bold')
    ax2.plot(observation['DATE'], observation['UE'], color="grey", label="Observed")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rate')
    ax2.legend()
    
    ax3.set_title('EE Transition Rate Over Time', fontweight='bold')
    ax3.plot(observation['DATE'], observation['EE'], color="grey", label="Observed")
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Rate')
    ax3.legend()
    
    # Adjust layout
    plt.tight_layout()

    
    # Save or show
    if save:
        plt.savefig(f'{path}transition_rates_comparison.jpg', dpi=300)
    plt.show()


####################################################
############## OCC-SPECIFIC LTUER ##################
####################################################
         
def plot_occupation_uer_grid(sim_results, observation, save=False, path=None):
    """
    Creates a grid of scatter plots comparing simulated vs observed UER and LTUER values by occupation.
    
    Parameters:
    - sim_results: Dictionary containing simulation results DataFrames
    - observation: DataFrame containing observed UER and LTUER values by occupation
    - save: Boolean indicating whether to save the plots
    - path: Path to save the plots if save=True
    """
    # Define aggregation levels and corresponding observed columns
    agg_levels = {
        'SOC2010_adj': ('uer_soc2010', 'ltuer_soc2010'), 
        'SOC_broad_adj': ('uer_soc_broad', 'ltuer_soc_broad'),
        'SOC_minor_adj': ('uer_soc_minor', 'ltuer_soc_minor'), 
    'SOC_major_adj': ('uer_soc_major', 'ltuer_soc_major')
    }

    model_names = list(sim_results.keys())
    num_rows = len(agg_levels)
    num_cols = len(model_names)


    # ----------- Plot 1: UER ---------------
    fig1, axes1 = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), sharex=False, sharey=False)
    fig1.suptitle("Unemployment Rate (UER): Simulated vs Observed (Sorted by Observed UER)", fontsize=16)

    # ----------- Plot 2: LTUER ---------------
    fig2, axes2 = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), sharex=False, sharey=False)
    fig2.suptitle("Long-Term Unemployment Rate (LTUER): Simulated vs Observed (Sorted by Observed LTUER)", fontsize=16)

    for col_idx, (model_name, sims) in enumerate(sim_results.items()):
        for row_idx, (name_k, (k_uer, k_ltuer)) in enumerate(agg_levels.items()):
            
            temp_codes = observation.loc[:, ['acs_occ_code', name_k, k_uer, k_ltuer]]
            occ_data = sims.loc[:, ['Time Step', 'Occupation', 'acs_occ_code', 'Workers', 'Unemployment', 'LT Unemployed Persons']]
            occ_data = occ_data.merge(temp_codes, on='acs_occ_code', how='left')
            
            # Group and calculate metrics
            occ_data = occ_data.groupby([name_k, 'Time Step']).sum().reset_index()
            occ_data['UER'] = occ_data['Unemployment'] / occ_data['Workers']
            occ_data['LTUE Rate'] = occ_data['LT Unemployed Persons'] / occ_data['Unemployment']
            
            mean_occ_data = occ_data.groupby(name_k)[['UER', 'LTUE Rate']].mean().reset_index()
            obs_values = observation.loc[:, [name_k, k_uer, k_ltuer]].drop_duplicates(subset=[name_k])
            mean_occ_data = mean_occ_data.merge(obs_values, on=name_k, how='left')

            # -------- UER Plot (sorted by observed UER) --------
            mean_occ_data_uer = mean_occ_data.sort_values(by=k_uer)
            
            ax1 = axes1[row_idx, col_idx]
            ax1.scatter(mean_occ_data_uer[name_k], mean_occ_data_uer['UER'], color='blue', label='Simulated', alpha=0.7)
            ax1.scatter(mean_occ_data_uer[name_k], mean_occ_data_uer[k_uer], color='orange', label='Observed', alpha=0.7, marker='X')
            ax1.set_title(f"{model_name} - {name_k}")
            if row_idx == num_rows - 1:
                ax1.set_xlabel(name_k)
            if col_idx == 0:
                ax1.set_ylabel('UER')
            ax1.tick_params(axis='x', rotation=45)
            ax1.set_ylim(0, 0.3) 

            # -------- LTUER Plot (sorted by observed LTUER) --------
            mean_occ_data_ltuer = mean_occ_data.sort_values(by=k_ltuer)
            ax2 = axes2[row_idx, col_idx]
            ax2.scatter(mean_occ_data_ltuer[name_k], mean_occ_data_ltuer['LTUE Rate'], color='blue', label='Simulated', alpha=0.7)
            ax2.scatter(mean_occ_data_ltuer[name_k], mean_occ_data_ltuer[k_ltuer], color='darkorange', label='Observed', alpha=0.7, marker="X")
            ax2.set_title(f"{model_name} - {name_k}")
            if row_idx == num_rows - 1:
                ax2.set_xlabel(name_k)
            if col_idx == 0:
                ax2.set_ylabel('LTUE Rate')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_ylim(0, 1) 

    # Add legends once
    axes1[0, 0].legend()
    axes2[0, 0].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        fig1.savefig(f'{path}occupation_uer_grid.png', dpi=300)
        fig2.savefig(f'{path}occupation_ltuer_grid.png', dpi=300)
    plt.show()
   

def plot_ltuer_difference_heatmap(sim_results_dict, observed_data, difference_type='absolute', save=False, path=None):
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
        occ_data = sim_results.loc[:, ['Time Step', 'Occupation', 'Workers', 'Unemployment', 'LT Unemployed Persons']]
        occ_data['LTUE Rate'] = occ_data['LT Unemployed Persons'] / occ_data['Unemployment']
        mean_ltuer = occ_data.groupby('Occupation')['LTUE Rate'].mean()
        
        if difference_type == 'absolute':
            # Calculate absolute differences
            diff = mean_ltuer - observed_data['ltuer_occ'].values
        else:
            # Calculate percentage differences
            diff = ((mean_ltuer - observed_data['ltuer_occ'].values) / observed_data['ltue_occ'].values) * 100

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
        vmin = -max(abs(diff_df.values.min()), abs(diff_df.values.max()))
        vmax = max(abs(diff_df.values.min()), abs(diff_df.values.max()))
    else:
        # Use diverging colormap for percentage differences
        cmap = 'RdBu_r'
        center = 0
        vmin = -max(abs(diff_df.values.min()), abs(diff_df.values.max()))
        vmax = max(abs(diff_df.values.min()), abs(diff_df.values.max()))
    
    sns.heatmap(diff_df, 
                annot=True,  # Show values in cells
                fmt='.2f',   # Format values to 2 decimal places
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': f'{"Absolute" if difference_type == "absolute" else "Percentage"} Difference'})
    
    # Customize plot
    plt.title(f'{"Absolute" if difference_type == "absolute" else "Percentage"} Differences in LTUER\n(Simulated - Observed)', 
              fontweight='bold', pad=20)
    plt.xlabel('Model')
    plt.ylabel('Occupation')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()

    
    # Save or show
    if save:
        plt.savefig(f'{path}ltuer_difference_heatmap_{difference_type}.jpg', 
                   dpi=300, bbox_inches='tight')
    plt.show()
