# Initialize the results dictionaries
model_results = {}
net_results = {}
sim_results = {}
sum_stats_list = {}
seekers_recs_list = {}
avg_wage_off_diffs = {}
app_loads = {}
time_to_emps = {}

print(param_df)
# Loop through each model configuration
for name, item in calib_list.items():
    # Create a deep copy of the base parameters
    print(name)
    test_params = deepcopy(params)

    calib_params = {'gamma_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "gamma_u"),"Value"].iloc[0],
                    'd_u': param_df.loc[(param_df['model_cat'] == name) & (param_df["Parameter"] == "d_u"),"Value"].iloc[0]}
                    
    # calib_params = {
    #     'd_u': param_df_new['d_u'][0], # 0.02, 
    #     'gamma_u': param_df_new['gamma_u'][0] #0 # 
    # }
    print(calib_params)
    test_params.update(calib_params)

    # Update with the values from the calib_list
    test_params.update(item)
    test_params.update({"steady_state": False})#,
                        #"time_steps": 300,
                        #"delay": 0})
    
    # Run the model
    sim_record, sim_grouped, sim_net, sum_stats, seekers_rec, avg_wage_off_diff, app_loads_df, time_to_emp = run_single_local(**test_params)

    # Generate plots or metrics (optional step)
    #ue_vac = plot_records(sim_record, calib_date[0], calib_date[1])
    sim_grouped['DATE'] = pd.date_range(start=calib_date[0], end= calib_date[1], periods=len(sim_grouped))
    sim_grouped['LTUER'] = sim_grouped['LT Unemployed Persons'] / sim_grouped['Unemployment']
    sim_grouped['AVGWAGE'] = sim_grouped['Total_Wages'] / sim_grouped['Employment']
    sim_grouped["Seeker Composition"] = sim_grouped["Employed Seekers"] / (sim_grouped["Employed Seekers"] + sim_grouped["Unemployed Seekers"])
    sim_record['LTUER'] = sim_record['LT Unemployed Persons'] / sim_record['Unemployment']
    sim_record['UER'] = sim_record['Unemployment'] / sim_record['Workers']
    sim_record['AVGWAGE'] = sim_record['Total_Wages'] / sim_record['Employment']
    sim_record['VACRATE1'] = sim_record['Vacancies'] / sim_record['Target_Demand']
    sim_record['VACRATE'] = sim_record['Vacancies'] / (sim_record['Vacancies'] + sim_record['Employment'])
    sim_record['U_REL_WAGE_MEAN'] = sim_record['U_Rel_Wage'] / sim_record['UE_Transitions']
    sim_record['E_REL_WAGE_MEAN'] = sim_record['E_Rel_Wage'] / sim_record['EE_Transitions']
    sim_record['UE_Trans_Rate'] = sim_record['UE_Transitions'] / sim_record['Workers']
    sim_record['EE_Trans_Rate'] = sim_record['EE_Transitions'] / sim_record['Workers']
    sim_record = sim_record.merge(occ_ids, left_on='Occupation', right_on="id", how='left')

    seekers_rec['DATE'] = pd.date_range(start=calib_date[0], end= calib_date[1], periods=len(seekers_rec))
    seekers_rec['Application Effort (U)'] = seekers_rec['Applications Sent (Unemployed)'] / seekers_rec['Unemployed Seekers']
    seekers_rec['Application Effort (E)'] = seekers_rec['Applications Sent (Employed)'] / seekers_rec['Employed Seekers']


    # Store the results
    model_results[name] = sim_grouped
    sim_results[name] = sim_record
    net_results[name] = sim_net
    sum_stats_list[name] = sum_stats
    seekers_recs_list[name] = seekers_rec
    avg_wage_off_diffs[name] = avg_wage_off_diff
    app_loads[name] = app_loads_df
    time_to_emps[name] = time_to_emp


output_path = f'output/figures/'

# --- Apply plot labels to models ---
name_map = {
    "nonbehav": "Non-behavioural",
    "otj_nonbehav": "Non-behavioural w. OTJ",
    "otj_cyclical_e_disc": "Behavioural w. Cyc. OTJ",
    "otj_disc": "Behavioural w.o. Cyc. OTJ",
}
desired_order = list(name_map.keys())

def filter_and_relabel(d, mapping, order):
    return {mapping[k]: d[k] for k in order if k in d}

filtered_model_results = filter_and_relabel(model_results, name_map, desired_order)
filtered_net_results   = filter_and_relabel(net_results,   name_map, desired_order)
filtered_sim_results   = filter_and_relabel(sim_results,   name_map, desired_order)
filtered_sum_stats     = filter_and_relabel(sum_stats_list, name_map, desired_order)
seekers_recs    = filter_and_relabel(seekers_recs_list, name_map, desired_order)
filtered_app_loads    = filter_and_relabel(app_loads, name_map, desired_order)
filtered_time_to_emps = filter_and_relabel(time_to_emps, name_map, desired_order)



seekers_comp_obs_full = pd.read_csv('../data/behav_params/Eeckhout_Replication/comp_searchers_s_series_abm_validation.csv')

# Map quarters to first day of quarter
quarter_map = {"Q1": "-01-01", "Q2": "-04-01", "Q3": "-07-01", "Q4": "-10-01"}

# Convert quarterly to monthly time series and interpolate missing values

# Replace and convert
seekers_comp_obs_full['DATE'] = (
    seekers_comp_obs_full['date']
    .replace(quarter_map, regex=True)
    .pipe(pd.to_datetime)
)

# Set DATE as index for resampling
seekers_comp_obs_full = seekers_comp_obs_full.set_index('DATE')

# Resample to monthly frequency, keeping the value at the start of each quarter
monthly = seekers_comp_obs_full.resample('MS').asfreq()

# Interpolate missing values linearly
monthly['comp_searchers_s'] = monthly['comp_searchers_s'].interpolate(method='linear')

# Reset index to restore DATE as a column
seekers_comp_obs_full = monthly.reset_index()


seekers_comp_obs = seekers_comp_obs_full[(seekers_comp_obs_full['DATE'] >= calib_date[0]) & (seekers_comp_obs_full['DATE'] <= calib_date[1])]
seekers_comp_obs = seekers_comp_obs.rename(columns={"comp_searchers_s": "Seeker Composition"})
seekers_comp_obs['DATE'] = pd.to_datetime(seekers_comp_obs['DATE'])

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

    
plot_seeker_comp(filtered_model_results, seekers_comp_obs, sep=True, share=True,
                 save=save_button , path=output_path)
