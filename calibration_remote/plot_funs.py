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
    ue_vac['LTUER'] = ue_vac['LT Unemployed Persons'] / ue_vac['Unemployment']
    ue_vac['DATE'] = pd.date_range(start=date_start, end= date_end, periods=len(sim_record))

    return ue_vac

####################################################
############## LTUER ###############################
####################################################

def plot_ltuer(res_dict, macro_obs, sep_strings=None, sep=False, save=False, path=None):
    def plot_models(ax, models, title, observed):
        for name, res in models.items():
            ax.plot(res['DATE'], res['LTUER'], label=name)
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
def plot_rel_wages(mod_results_dict, save=False, path=None, freq = 'Y'):
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
                color=colors[i], label=name, marker='o', zorder=3)
        # Add shading
        ax1.fill_between(dates_array, u_wages, 1,
                        color=colors[i], alpha=0.2, zorder=2)
        
        # Plot employed relative wages (smoothed)
        e_wages = annual_data['E_REL_WAGE_MEAN'].values
        ax2.plot(dates_array, e_wages, 
                color=colors[i], label=name, marker='o', zorder=3)
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

def hires_seps_rate(mod_results_dict, jolts, save=False, path=None):
    # Step 1: Separate models
    model_groups = {
        "Non-behavioral": {k: v for k, v in mod_results_dict.items() if "Non-behavioural" in k},
        "Behavioral/Other": {k: v for k, v in mod_results_dict.items() if "Non-behavioural" not in k}
    }

    # Step 2: Setup grid: rows = model groups, cols = [Hires Rate, Separations Rate]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    rate_types = [("Hires Rate", "HIRESRATE"), ("Separations Rate", "SEPSRATE")]

    # Step 3: Loop through rows (model groups) and columns (rate types)
    for row_idx, (group_name, group_models) in enumerate(model_groups.items()):
        for col_idx, (rate_col_sim, rate_col_obs) in enumerate(rate_types):
            ax = axes[row_idx, col_idx]

            # Plot observed JOLTS rate
            ax.plot(jolts['DATE'], jolts[rate_col_obs],
                    label=f'Observed ({rate_col_obs})', color='black', linestyle='--')

            # Plot each model in the group
            for model_name, df in group_models.items():
                ax.plot(df['DATE'], df[rate_col_sim], label=model_name)

            # Titles and labels
            title = f"{rate_col_sim} ({group_name} Models)"
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(rate_col_sim)
            if row_idx == 1:
                ax.set_xlabel('Date')
            ax.grid(True)
            ax.legend()

    plt.tight_layout()

    if save:
        plt.savefig(f'{path}hires_seps_rate_grid.jpg', dpi=300)
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
    colors = ["skyblue", "orange",]# "brown", "pink"]
    def plot_uer(ax, models, title):
        for i, (name, res) in enumerate(models.items()):
            ax.plot(res['DATE'], res['UER'], label=f'Sim. UER: {name}', color=colors[i % len(colors)])
        ax.plot(macro_obs['DATE'], macro_obs['UER'], color="grey", linestyle="dotted", label="Obs. UER")
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
        ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="grey", linestyle="dotted", label="Obs. VR")
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
        plt.savefig(f'{path}uer_vac.jpg', dpi=300)
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
        plt.savefig(f'{path}uer_vac.jpg', dpi=300)
    plt.show()



####################################################
############## SEEKER COMPOSITION ##################
####################################################
def plot_seeker_comp(res_dict, observation, share = False, sep = False, save = False, path = None):
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
        ax.plot(observation['DATE'], observation['Seeker Composition'], color="black", linestyle="dotted", label="Observed")
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
def plot_seeker_comp_line(res_dict, observation, save = False, path = None):
    plt.figure(figsize=(10, 6))
    for i, (name, res) in enumerate(res_dict.items()):
        plt.plot(res['DATE'], res["Seeker Composition"], label=name)

        # Set axis label and title
    plt.plot(observation['DATE'], observation['Seeker Composition'], label='Observed', color='grey', linestyle = "dashed")
    plt.title("Monthly Composition of Job Seekers", fontsize=14)  # Figure-wide title
    plt.xlabel("Date")  # Shared x-axis label
    plt.ylabel("Employed Share of Job Seekers")  # Shared y-axis label
    plt.legend()

    # Show plot
    if save:
        plt.savefig(f'{path}seeker_composition_line.png', dpi = 300)
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
            ax.set_xlabel("Time Unemployed (months)")
            ax.set_ylabel("Number of Workers")
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

            # zoomed histogram (0–36 mos)
            ax_zoom.hist(total_unemp, bins=n_bins, range=(0, 36), alpha=0.5,
                         label=names[idx], color=color)
            ax_zoom.axvline(np.mean(total_unemp), ls='--', lw=1, color=color)

        # labels and titles
        ax_full.set_xlabel("Time Unemployed (months)")
        ax_zoom.set_xlabel("Time Unemployed (≤ 36 months)")
        ax_full.set_ylabel("Number of Workers")
        ax_full.set_title("Full distribution")
        ax_zoom.set_title("Zoom ≤ 3 years")
        ax_zoom.set_xlim(0, 36)

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
def plot_trans_rates(mod_results_dict, observation, save=False, path=None):
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
                    fmt='o', color=colors[i], label=name, 
                    capsize=5, markersize=10)
        ax1.errorbar(ue_x, ue_mean, yerr=ue_std, 
                    fmt='o', color=colors[i], 
                    capsize=5, markersize=10)
        
        # Plot time series on right panels
        ax2.plot(res['DATE'], res['UE_Trans_Rate'], color=colors[i], label=name)
        ax3.plot(res['DATE'], res['EE_Trans_Rate'], color=colors[i], label=name)
    
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
        fig1.savefig(f'{path}occupation_uer_grid.png', dpi=300)
        fig2.savefig(f'{path}occupation_ltuer_grid.png', dpi=300)
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
        fig1.savefig(f'{path}occupation_uer_grid.png', dpi=300)
        fig2.savefig(f'{path}occupation_ltuer_grid.png', dpi=300)
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
        plt.savefig(f'{path}ltuer_difference_heatmap_{difference_type}_absval_{abs_value}.jpg', 
                   dpi=300, bbox_inches='tight')
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
        plt.savefig(save_path)
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=occ_means, x="Occupation", y="Difference", color="gray")
    plt.title("Difference: Mean Vacancy Offer - Mean Occupational Wage")
    plt.ylabel("Difference")
    plt.xlabel("Occupation")
    plt.tight_layout()
    if save_path:
        diff_path = save_path.replace(".png", "_difference.png")
        plt.savefig(diff_path)
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
        plt.savefig(save_path)
    plt.show()