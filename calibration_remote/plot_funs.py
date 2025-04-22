import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt


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
def plot_ltuer(res_dict, macro_obs, sep=False, save=False, path=None):
    def plot_models(ax, models, title, observed):
        for name, res in models.items():
            ax.plot(res['DATE'], res['LTUE Rate'], label=name)
        ax.plot(observed['DATE'], observed['LTUER'], color="black", linestyle="dotted", label="Observed")
        ax.set_title(title)
        ax.set_xlabel("DATE")
        ax.set_ylabel("LTUER")
        ax.legend()

    # Prepare observed data once
    observed = macro_obs.dropna(subset=["UNRATE", "VACRATE"]).reset_index()

    if sep:
        # Split into two subplots: True and False
        true_models = {k: v for k, v in res_dict.items() if "True" in k}
        false_models = {k: v for k, v in res_dict.items() if "False" in k}
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        plot_models(axes[0], false_models, "LTUER - Behavior=False", observed)
        plot_models(axes[1], true_models, "LTUER - Behavior=True", observed)
        plt.tight_layout()
    else:
        # Single combined plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_models(ax, res_dict, "LTUER - All Models", observed)

    # Save or show plot
    if save:
        filename = f'{path}ltuer_line.jpg'
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()


####################################################
############## BEV CURVES ##########################
####################################################
def plot_bev_curve(res_dict, macro_obs, sep = False, save = False, path = None):
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
        ax.plot(res['UE Rate'], res['Vac Rate'], label=name)
        ax.scatter(res['UE Rate'], res['Vac Rate'], c=res.index, s=50, lw=0)
        ax.plot(macro_obs['UER'], macro_obs['VACRATE'], label = "Real Values", color = "grey")

        ax.set_title(name)
        ax.set_xlabel('UE Rate')
        ax.set_ylabel('Vac Rate')
        ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save:
        plt.savefig(f'{path}bev_curves.jpg', dpi = 300)
        plt.close()
    else:
        plt.show()

####################################################
############## UER & VAC ###########################
####################################################
def plot_uer_vac(res_dict, macro_obs, recessions, sep = False, save = False, path = None):
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
        ax.plot(res['DATE'], res['Vac Rate'], label=f'Sim. VR', color = "red")
        ax.plot(res['DATE'], res['UE Rate'], label=f'Sim. UER', color = "blue")
        ax.plot(macro_obs['DATE'], macro_obs['UER'], color="blue", linestyle="dotted", label="Obs. UER")
        ax.plot(macro_obs['DATE'], macro_obs['VACRATE'], color="red", linestyle="dotted", label="Obs. VR")
        #recessions.plot.area(ax=ax, 'DATE', color="grey", alpha=0.2)

        ax.set_title(name)
        ax.set_xlabel('DATE')
        ax.set_ylabel('Rate')
        ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save:
        plt.savefig(f'{path}uer_vac.jpg', dpi = 300)
        plt.close()
    else:
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
        max_cols = 3
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
        ax.set_title(name)

    fig.supxlabel("Wage")  # Shared x-axis label
    fig.supylabel("Quantity")  # Shared y-axis label

    fig.suptitle('Distribution of Male and Female Wages', fontsize = 15) 
    fig.subplots_adjust(bottom=0.1)

    if save:
        plt.savefig(f'{path}gender_wage_gaps.jpg', dpi = 300)
        plt.close()
    else:
        plt.show()


####################################################
############## LTUER DISTRIBUTIONS #################
####################################################
def plot_ltuer_dist(net_dict, gender=False, sep = False, save=False, path=None):
    """
    Plots the distribution of time unemployed across models.

    Parameters:
    - net_dict: dict of model_name -> list of occupations
    - gender: if True, plots one subplot per model showing men and women distributions
              if False, plots one figure comparing all models (total only)
    - save: if True, saves figure to `path`
    - path: save location (if save=True)
    """
    n_bins = 10
    max_cols = 3
    colors = ['orange', 'blue', 'brown', 'green', 'blue', 'grey', 'red', 'cyan']

    if gender:
        n = len(net_dict)
        cols = min(n, max_cols)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharex=True, sharey=True)
        axes = np.array(axes).flatten()

        for i, (name, net) in enumerate(net_dict.items()):
            ax = axes[i]
            w_time_unemp = []
            m_time_unemp = []

            for occ in net:
                w_time_unemp.extend([wrkr.time_unemployed for wrkr in occ.list_of_unemployed if wrkr.female])
                m_time_unemp.extend([wrkr.time_unemployed for wrkr in occ.list_of_unemployed if not wrkr.female])

            ax.hist(w_time_unemp, bins=n_bins, alpha=0.5, label='Women', color=colors[0])
            ax.hist(m_time_unemp, bins=n_bins, alpha=0.5, label='Men', color=colors[1])
            ax.axvline(np.mean(w_time_unemp), linestyle='dashed', linewidth=1, color=colors[0])
            ax.axvline(np.mean(m_time_unemp), linestyle='dashed', linewidth=1, color=colors[1])

            ax.set_title(name)
            ax.set_xlabel("Time Unemployed (simulated months)")
            ax.set_ylabel("Number of Workers")
            ax.legend()

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle('Distribution of Time Spent Unemployed by Gender per Model')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    else:
        # Split models into two groups
        false_models = {name: net for name, net in net_dict.items() if "False" in name}
        true_models = {name: net for name, net in net_dict.items() if "True" in name}

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        axes = axes.flatten()

        # Plot False models (left)
        for idx, (name, net) in enumerate(false_models.items()):
            total_unemp = []
            for occ in net:
                total_unemp.extend([
                    wrkr.time_unemployed
                    for wrkr in occ.list_of_unemployed
                ])
            axes[0].hist(total_unemp, bins=n_bins, alpha=0.5,
                         label=name, color=colors[idx % len(colors)])
            axes[0].axvline(np.mean(total_unemp), linestyle='dashed',
                            linewidth=1, color=colors[idx % len(colors)])
        axes[0].set_title("Behavior=False")
        axes[0].set_xlabel("Time Unemployed (simulated months)")
        axes[0].set_ylabel("Number of Workers")
        axes[0].legend()

        # Plot True models (right)
        for idx, (name, net) in enumerate(true_models.items()):
            total_unemp = []
            for occ in net:
                total_unemp.extend([
                    wrkr.time_unemployed
                    for wrkr in occ.list_of_unemployed
                ])
            axes[1].hist(total_unemp, bins=n_bins, alpha=0.5,
                         label=name, color=colors[idx % len(colors)])
            axes[1].axvline(np.mean(total_unemp), linestyle='dashed',
                            linewidth=1, color=colors[idx % len(colors)])
        axes[1].set_title("Behavior=True")
        axes[1].set_xlabel("Time Unemployed (simulated months)")
        axes[1].legend()

        fig.suptitle('Distribution of Time Spent Unemployed (Total)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.ylabel("Number of Workers")
        plt.legend()

    if save:
        plt.savefig(f'{path}ltuer_distributions.jpg', dpi=300)
        plt.close()
    else:
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
        plt.savefig(path + f'{path}cd_vs_td.png', dpi=300)
        plt.close() 
    else:
        plt.show()


def sse_tbl(res_dict, obs):
    for i, (name, res) in enumerate(res_dict.items()):
        uer_sse = np.sum((x_["UER"] - y["UER"])**2) / np.var(y["UER"])
        vacrate_sse = np.sum((x_["VACRATE"] - y["VACRATE"])**2) / np.var(y["VACRATE"])

    return tbl