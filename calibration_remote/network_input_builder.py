import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< Updated upstream
#from torch import norm
from scipy.stats import norm
from abm_funs import *
#calib_date = ["2000-12-01", "2019-05-01"]
calib_date = ["2000-12-01", "2024-05-01"]


path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"


# Network in put file
def network_input_builder(nx, complete):
=======
import os
#from torch import norm
from scipy.stats import norm
from statsmodels.tsa.filters import hp_filter
from abm_funs import *
# calib_date = ["2000-12-01", "2019-05-01"]
# #calib_date = ["2000-12-01", "2024-05-01"]

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"

# Network in put file
def network_input_builder(nx, complete, calib_date):
>>>>>>> Stashed changes
    if nx == "original_omn":
        print("Using Corrected Original OMN")
        #A = pd.read_csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/data/omn_asec_11_19_occ_matched_plus_oldtransitions_2025_normalised.csv", delimiter=",", header = None)
        employment = round(pd.read_csv(path + "dRC_Replication/data/ipums_employment_2016.csv", header=0).iloc[:, [4]] / 10000)

        # Crude approximation using avg unemployment rate of ~5% - should aim for occupation-specific unemployment rates
        unemployment = round(employment * (0.05 / 0.95))

        # Less crude approximation using avg vacancy rate - should still aim for occupation-specific vacancy rates
        vac_rate_base = pd.read_csv(path + "dRC_Replication/data/vacancy_rateDec2000.csv").iloc[:, 2].mean() / 100
        vacancies = round(employment * vac_rate_base / (1 - vac_rate_base))

        # Needs input data...
        demand_target = employment + vacancies
        wages = pd.read_csv(path + "dRC_Replication/data/ipums_variables.csv")[['median_earnings']]
        occ_ids = pd.read_csv(path + "dRC_Replication/data/ipums_variables.csv")[['id', 'acs_occ_code', 'label']]
        assert(all(occ_ids.index.values == occ_ids['id'].values))
        assert(all(range(0, len(occ_ids)) == occ_ids['id'].values))
        gend_share = pd.read_csv(path + "data/ipums_variables_w_gender.csv")[['women_pct']]
        experience_req = pd.read_csv(path + "dRC_Replication/data/ipums_variables_w_exp.csv")
        seps_rates = pd.read_csv(path + "dRC_Replication/data/ipums_variables_w_seps_rate.csv")
        mean_seps_rate = seps_rates['seps_rate'].mean()

        if complete:
            print("Using complete network")
            # Create complete network
            n = A.shape
            complete_network = np.ones(n)
            A = complete_network

        mod_data = {
            "A": A,
            "employment": employment,
            'unemployment': unemployment,
            'vacancies': vacancies,
            'demand_target': demand_target,
            'wages': wages,
            'wage_mu': np.full_like(wages, np.inf),
            'wage_mu': np.full_like(wages, np.inf),
            'gend_share': gend_share,
            'entry_level': experience_req['entry_level'],
            # 'entry_age': experience_req['entry_age'],
            'experience_age': experience_req['experience_age'],
            #'separation_rates': np.full_like(seps_rates['seps_rate'], mean_seps_rate, dtype=float)*10
            'separation_rates': seps_rates['seps_rate']*10
        }


        ###################################
        # Initialise the model
        ##################################
        net_temp, vacs = initialise(
            len(mod_data['A']),
            mod_data['employment'].to_numpy(),
            mod_data['unemployment'].to_numpy(),
            mod_data['vacancies'].to_numpy(),
            mod_data['demand_target'].to_numpy(),
            mod_data['A'],
            mod_data['wages'].to_numpy(),
            mod_data['wage_mu'].to_numpy(),
            mod_data['wage_sigma'].to_numpy(),
            mod_data['gend_share'].to_numpy(),
            7, 1,
            mod_data['entry_level'],
            mod_data['experience_age'],
            mod_data['separation_rates']
        )


        occ_shocks = pd.read_csv(path+"data/occupational_va_shocks.csv", index_col = 0).drop('X', axis = 'columns')
        df_quarterly = occ_shocks.transpose() + 1  # now rows are dates, columns are occupations
        # Convert index to datetime
        df_quarterly.index = pd.to_datetime(df_quarterly.index)

        # Reindex to monthly frequency
        monthly_index = pd.date_range(start=df_quarterly.index.min(), end=df_quarterly.index.max(), freq='MS')  # Month start frequency
        df_monthly = df_quarterly.reindex(monthly_index)

        # Linearly interpolate missing monthly values
        df_monthly = df_monthly.interpolate(method='linear')

        # for col in df_monthly.columns:
        #     # Segment 1: Before calibration window
        #     plt.plot(df_monthly.loc[df_monthly.index < calib_date[0]].index,
        #             df_monthly.loc[df_monthly.index < calib_date[0], col],
        #             color='lavender', linewidth=0.5, alpha=0.2)

        #     # Segment 2: During calibration window
        #     plt.plot(df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].index,
        #             df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1]), col],
        #             color='orchid', linewidth=0.5, alpha = 0.5)

        #     # Segment 3: After calibration window
        #     plt.plot(df_monthly.loc[df_monthly.index > calib_date[1]].index,
        #             df_monthly.loc[df_monthly.index > calib_date[1], col],
        #             color='lavender', linewidth=0.5, alpha=0.2)

        # # Vertical lines for calibration boundaries
        # for cal_date in calib_date:
        #     plt.axvline(x=pd.to_datetime(cal_date), color='steelblue', linestyle='--', alpha=0.6)

        # #plt.plot(monthly_dates, gdp_dat, color='rebeccapurple', label='HP Filtered GDP')

        # plt.xlabel("Date")
        # plt.ylabel("Value")
        # plt.title("Occupation-Specific Shocks Over Time")
        # plt.xticks(rotation=45)
        # plt.grid()
<<<<<<< Updated upstream
        # plt.show()
=======
        # plt.close()
>>>>>>> Stashed changes

        # plt.savefig('output/figures/occ_shocks.png', dpi=300)

        occ_shocks_dat = np.array(df_monthly[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].transpose())

    elif nx == "full_omn":
        print("Using Full Corrected OMN") 
        A = pd.read_csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/data/asec_11_19_occ_alltransitions_2025_normalised.csv", header = None)
        print(A.shape)
        ipums_input = pd.read_csv(path +"dRC_Replication/data/ipums_variables_full_omn_w_exp.csv", delimiter = ",")
        #ipums_input = pd.read_csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/ONET/occ_names_employment_asec_occ_ipums_vars.csv", delimiter=",")

        occ_ids = pd.read_csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/data/occ_names_employment_asec_occ.csv", delimiter=",")[['Code', 'Label']]
        occ_ids = occ_ids.rename(columns={"Label": "label"})
        occ_ids['id'] = occ_ids.index.values
        assert(all(occ_ids['id'].values == range(0,len(occ_ids))))
        assert(all(occ_ids.index.values == occ_ids['id'].values))

        # diagnostics
        missing = occ_ids['label'].isna().sum()
        assert(missing == 0)

        employment = np.round(ipums_input[['emp']]/10000) + 1

        # Crude approximation using avg unemployment rate of ~5% - should aim for occupation-specific unemployment rates
        unemployment = round(employment * (0.05 / 0.95))
        #unemployment = np.round(ipums_input[['unemp']]/10000) + 1

        # Less crude approximation using avg vacancy rate - should still aim for occupation-specific vacancy rates
        vac_rate_base = pd.read_csv(path + "dRC_Replication/data/vacancy_rateDec2000.csv").iloc[:, 2].mean() / 100
        vacancies = round(employment * vac_rate_base / (1 - vac_rate_base))

        # Needs input data...
        demand_target = employment + vacancies

<<<<<<< Updated upstream
        wage_comp = ipums_input[['acs_occ_code', 'median_weekly_earnings']]
=======
        #wage_comp = ipums_input[['acs_occ_code', 'median_weekly_earnings']]
        wage_comp = ipums_input[['acs_occ_code', 'median_weekly_earnings']].copy()
>>>>>>> Stashed changes
        wage_comp['median_annual_earnings'] = wage_comp['median_weekly_earnings'] * 52
        wage_dist = pd.read_csv("~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/data/occ_macro_vars/OEWS/wage_distributions_full_omn.csv", compression='gzip',  delimiter=",", header = 0)
        wage_full = wage_comp.merge(wage_dist, left_on='acs_occ_code', right_on='acs_occ_code', how='inner')

        gend_share = ipums_input[['female_share']]
        experience_req = ipums_input['experience_req']
        entry_level = ipums_input['entry_level']
        experience_age = ipums_input['experience_age']
        seps_rates = pd.read_csv(path + "dRC_Replication/data/ipums_variables_w_seps_rate_full_omn.csv")
        mean_seps_rate = seps_rates['seps_rate'].mean()

        if complete:
            print("Using complete network")
            # Create complete network
            n = A.shape
            complete_network = np.ones(n)
            A = complete_network

        mod_data = {
                "A": A,
                "employment": employment,
                'unemployment': unemployment,
                'vacancies': vacancies,
                'demand_target': demand_target,
                'wages': wage_full['a_median'],
                'wage_mu': wage_full['mu'],
                'wage_sigma': wage_full['sigma'],
                'gend_share': gend_share,
                'entry_level': entry_level,
                #'entry_age': experience_req['entry_age'],
                'experience_age': experience_age,
                'separation_rates': seps_rates['seps_rate']*10
                #'separation_rates': np.full_like(employment, mean_seps_rate, dtype=float)*10
        }
        print("Build mod_data.")
        print(f"Nodes (n): {len(mod_data['A'])}")

        ###################################
        # Initialise the model
        ##################################
        net_temp, vacs = initialise(
                len(mod_data['A']),
                mod_data['employment'].to_numpy(),
                mod_data['unemployment'].to_numpy(),
                mod_data['vacancies'].to_numpy(),
                mod_data['demand_target'].to_numpy(),
                mod_data['A'],
                mod_data['wages'].to_numpy(),
                mod_data['wage_mu'].to_numpy(),
                mod_data['wage_sigma'].to_numpy(),
                mod_data['gend_share'].to_numpy(),
                7, 1,
                mod_data['entry_level'],
                mod_data['experience_age'],
                mod_data['separation_rates']
        )
        print("Initialised network.")
<<<<<<< Updated upstream


        occ_shocks = pd.read_csv(path+"data/occupational_va_shocks_full_omn.csv", index_col = 0).drop('acs_2010_code', axis = 'columns')
        print(occ_shocks.shape)
=======
        
        #################################################################################################
        ####################### INPUT SERIES FOR MODEL ##################################################
        ##################################################################################################
        # Real GDP
        # Source: https://fred.stlouisfed.org/series/GDPC1
        realgdp = pd.read_csv(path+"data/macro_vars/GDPC1.csv", delimiter=',', decimal='.')
        realgdp["DATE"] = pd.to_datetime(realgdp["DATE"])
        realgdp["REALGDP"] = realgdp['GDPC1']
        realgdp['FD_REALGDP'] = pd.Series(realgdp['REALGDP']).diff()
        realgdp = realgdp[["DATE", "REALGDP"]].dropna(subset=["REALGDP"]).reset_index()
        realgdp['log_REALGDP'] = np.log2(realgdp['REALGDP'])

        # GDP Filter
        cycle, trend = hp_filter.hpfilter(realgdp['log_REALGDP'], lamb=129600)
        
        # Adding the trend and cycle to the original DataFrame
        realgdp['log_Trend'] = trend+1
        realgdp['log_Cycle'] = cycle+1
        realgdp['Trend'] = np.exp(trend)
        realgdp['Cycle'] = np.exp(cycle)

        realgdp_no_covid = realgdp[realgdp['DATE'] < "2019-10-1"].copy()
        realgdp['scaled_log_Cycle'] = (realgdp['log_Cycle'] - realgdp['log_Cycle'].min()) / (realgdp['log_Cycle'].max() - realgdp['log_Cycle'].min())
        realgdp_no_covid['scaled_log_Cycle'] = (realgdp_no_covid['log_Cycle'] - realgdp_no_covid['log_Cycle'].min()) / (realgdp_no_covid['log_Cycle'].max() - realgdp_no_covid['log_Cycle'].min())

        # Different calibration windows
        gdp_dat_pd = realgdp.set_index('DATE').sort_index()

        # Interpolate all columns in gdp_dat_pd to monthly frequency
        monthly_dates = pd.date_range(start=gdp_dat_pd.index.min(), end=gdp_dat_pd.index.max(), freq='MS')
        gdp_dat_pd_monthly = gdp_dat_pd.reindex(monthly_dates)

        # Linearly interpolate missing monthly values
        gdp_dat_pd_monthly = gdp_dat_pd_monthly.interpolate(method='linear')
        gdp_dat_pd_monthly = gdp_dat_pd_monthly[(gdp_dat_pd_monthly.index >= calib_date[0]) & (gdp_dat_pd_monthly.index <= calib_date[1])]

        # For backward compatibility with previous code
        gdp_dat = gdp_dat_pd_monthly['log_Cycle'].values + np.random.normal(loc=0, scale=0.001, size=len(gdp_dat_pd_monthly))
        monthly_dates = gdp_dat_pd_monthly.index

        occ_shocks = pd.read_csv(path+"data/occupational_va_shocks_full_omn.csv", index_col = 0).drop('acs_2010_code', axis = 'columns')
        #print(occ_shocks.shape)
>>>>>>> Stashed changes

        df_quarterly = occ_shocks.transpose() + 1  # now rows are dates, columns are occupations
        # Convert index to datetime
        df_quarterly.index = pd.to_datetime(df_quarterly.index)

        # Reindex to monthly frequency
        monthly_index = pd.date_range(start=df_quarterly.index.min(), end=df_quarterly.index.max(), freq='MS')  # Month start frequency
        df_monthly = df_quarterly.reindex(monthly_index)

        # Linearly interpolate missing monthly values
        df_monthly = df_monthly.interpolate(method='linear')

        for col in df_monthly.columns:
            # Segment 1: Before calibration window
            plt.plot(df_monthly.loc[df_monthly.index < calib_date[0]].index,
                    df_monthly.loc[df_monthly.index < calib_date[0], col],
<<<<<<< Updated upstream
                    color='darkseagreen', linewidth=0.5, alpha=0.1)
=======
                    color='lavender', linewidth=0.5, alpha=0.1)
>>>>>>> Stashed changes

            # Segment 2: During calibration window
            plt.plot(df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].index,
                    df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1]), col],
<<<<<<< Updated upstream
                    color='forestgreen', linewidth=0.5, alpha = 0.5)
=======
                    color='orchid', linewidth=0.5, alpha = 0.5)
>>>>>>> Stashed changes

            # Segment 3: After calibration window
            plt.plot(df_monthly.loc[df_monthly.index > calib_date[1]].index,
                    df_monthly.loc[df_monthly.index > calib_date[1], col],
<<<<<<< Updated upstream
                    color='darkseagreen', linewidth=0.5, alpha=0.1)
=======
                    color='lavender', linewidth=0.5, alpha=0.1)
>>>>>>> Stashed changes

        # Vertical lines for calibration boundaries
        for cal_date in calib_date:
            plt.axvline(x=pd.to_datetime(cal_date), color='steelblue', linestyle='--', alpha=0.6)

<<<<<<< Updated upstream
        #plt.plot(monthly_dates, gdp_dat, color='rebeccapurple', label='HP Filtered GDP')

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Occupation-Specific Shocks Over Time (ONET Occs)")
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
=======
        plt.plot(monthly_dates, gdp_dat, color='rebeccapurple', label='HP Filtered GDP')

        plt.xlabel("Date")
        plt.ylabel("De-trended Value")
        plt.title("Occupation-Specific Demand Shocks")
        plt.xticks(rotation=45)
        plt.grid()
        plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/full_omn/figures/occupational_va_shocks.png"), bbox_inches='tight', 
            pad_inches=0.1, dpi=300)
        plt.close()
>>>>>>> Stashed changes

        occ_shocks_dat = np.array(df_monthly[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].transpose())

    elif nx == "onet" or nx == "onet_wage_asym":
        if nx == "onet":
            print("Using ONET Related Occupations Network")
            A = pd.read_csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/ONET/acs_2010_max_index_adjacency_matrix.csv", index_col = 0)
        elif nx == "onet_wage_asym":
            print("Using Reweighted Wage Asymmetric ONET Related Occupations Network")
            A = pd.read_csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/ONET/acs_2010_onet_wage_asym_filled_adjacency_matrix.csv", index_col = 0)
        
        ipums_input = pd.read_csv(path + "dRC_Replication/data/acs_onet_2010_ipums_vars_w_exp.csv", delimiter=",")

        occ_ids = ipums_input[['X.1', 'acs_occ_code']]
        occ_ids = occ_ids.rename(columns={"X.1": "id"})
        occ_ids['id'] = occ_ids['id'] - 1  # change to zero-indexed
        assert(all(occ_ids['id'].values == range(0,len(occ_ids))))
        assert(all(occ_ids.index.values == occ_ids['id'].values))
        labels = pd.read_csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/ONET/acs_onet_2010_soc_cw.csv")

        # merge titles from labels into occ_ids by ACS code and create 'label' column
        labels_map = labels[['acs_2010_code', 'title']].drop_duplicates(subset='acs_2010_code')
        labels_map = labels_map.rename(columns={'acs_2010_code': 'acs_occ_code', 'title': 'label'})

        occ_ids = occ_ids.merge(labels_map, on='acs_occ_code', how='left', sort=False)

        # diagnostics
        missing = occ_ids['label'].isna().sum()
        assert(missing == 0)

        assert(np.array_equal(A.index.values, occ_ids['acs_occ_code'].values))

<<<<<<< Updated upstream
        wage_comp = ipums_input[['acs_occ_code', 'median_weekly_earnings']]
=======
        wage_comp = ipums_input[['acs_occ_code', 'median_weekly_earnings']].copy()
>>>>>>> Stashed changes
        wage_comp['median_annual_earnings'] = wage_comp['median_weekly_earnings'] * 52
        wage_dist = pd.read_csv("~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/data/occ_macro_vars/OEWS/wage_distributions_onet.csv", compression='gzip',  delimiter=",", header = 0)
        wage_full = wage_comp.merge(wage_dist, left_on='acs_occ_code', right_on='acs_occ_code', how='inner')

        A = np.array(A)

        row_sums = A.sum(axis=1, keepdims=True)
        A_norm = np.divide(A, row_sums, where=row_sums != 0)

        # Fill diagonals with 1
        np.fill_diagonal(A_norm, 1.0)

        # # Normalize again so each row sums to 1
        row_sums = A_norm.sum(axis=1, keepdims=True)
        A_norm = np.divide(A_norm, row_sums, where=row_sums != 0)
        A_norm[np.isnan(A_norm)] = 0
        row_sums = A_norm.sum(axis=1, keepdims=True)

        assert(np.allclose(row_sums, 1))
        assert(np.allclose(np.mean(np.diag(A_norm)), 0.5))

        A_norm = pd.DataFrame(A_norm)

        employment = np.round(ipums_input[['emp']]/10000) + 1
        # Crude approximation using avg unemployment rate of ~5% - should aim for occupation-specific unemployment rates
        unemployment = round(employment * (0.05 / 0.95))
        # unemployment = np.round(ipums_input[['unemp']]/10000) + 1

        # Less crude approximation using avg vacancy rate - should still aim for occupation-specific vacancy rates
        vac_rate_base = pd.read_csv(path + "dRC_Replication/data/vacancy_rateDec2000.csv").iloc[:, 2].mean() / 100
        vacancies = round(employment * vac_rate_base / (1 - vac_rate_base))

        # Needs input data...
        demand_target = employment + vacancies
        #wages = ipums_input[['median_weekly_earnings']]*52
        gend_share = ipums_input[['female_share']]
        experience_req = ipums_input['experience_req']
        entry_level = ipums_input['entry_level']
        experience_age = ipums_input['experience_age']
        #seps_rates = np.array(516, 0.02)
        seps_rates = pd.read_csv("/Users/ebbamark/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/dRC_Replication/data/ipums_variables_w_seps_rate_onet.csv")
        mean_seps_rate = seps_rates['seps_rate'].mean()

        if complete:
            print("Using complete network")
            # Create complete network
            n = A_norm.shape
            complete_network = np.ones(n)
            A_norm = complete_network

        mod_data = {
            "A": A_norm,
            "employment": employment,
            'unemployment': unemployment,
            'vacancies': vacancies,
            'demand_target': demand_target,
            'wages': wage_full['a_median'],
            'wage_mu': wage_full['mu'],
            'wage_sigma': wage_full['sigma'],
            'gend_share': gend_share,
            'entry_level': entry_level,
            # 'entry_age': experience_req['entry_age'],
            'experience_age': experience_age,
            'separation_rates': seps_rates['seps_rate']*10
            #'separation_rates': np.full_like(employment, mean_seps_rate, dtype=float)*10
        }
        print("Build mod_data.")
        print(f"Nodes (n): {len(mod_data['A'])}")

        ###################################
        # Initialise the model
        ##################################
        net_temp, vacs = initialise(
            len(mod_data['A']),
            mod_data['employment'].to_numpy(),
            mod_data['unemployment'].to_numpy(),
            mod_data['vacancies'].to_numpy(),
            mod_data['demand_target'].to_numpy(),
            mod_data['A'],
            mod_data['wages'].to_numpy(),
            mod_data['wage_mu'].to_numpy(),
            mod_data['wage_sigma'].to_numpy(),
            mod_data['gend_share'].to_numpy(),
            7, 1,
            mod_data['entry_level'],
            mod_data['experience_age'],
            mod_data['separation_rates']
        )

        print("Initialised network.")

        occ_shocks = pd.read_csv(path+"data/occupational_va_shocks_onet_related_occs.csv", index_col = 0).drop('X', axis = 'columns')
<<<<<<< Updated upstream
        print(occ_shocks.shape)
=======
        #print(occ_shocks.shape)
>>>>>>> Stashed changes

        df_quarterly = occ_shocks.transpose() + 1  # now rows are dates, columns are occupations
        # Convert index to datetime
        df_quarterly.index = pd.to_datetime(df_quarterly.index)

        # Reindex to monthly frequency
        monthly_index = pd.date_range(start=df_quarterly.index.min(), end=df_quarterly.index.max(), freq='MS')  # Month start frequency
        df_monthly = df_quarterly.reindex(monthly_index)

        # Linearly interpolate missing monthly values
        df_monthly = df_monthly.interpolate(method='linear')
        # for col in df_monthly.columns:
        #     # Segment 1: Before calibration window
        #     plt.plot(df_monthly.loc[df_monthly.index < calib_date[0]].index,
        #             df_monthly.loc[df_monthly.index < calib_date[0], col],
        #             color='skyblue', linewidth=0.5, alpha=0.2)

        #     # Segment 2: During calibration window
        #     plt.plot(df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].index,
        #             df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1]), col],
        #             color='steelblue', linewidth=0.5, alpha = 0.5)

        #     # Segment 3: After calibration window
        #     plt.plot(df_monthly.loc[df_monthly.index > calib_date[1]].index,
        #             df_monthly.loc[df_monthly.index > calib_date[1], col],
        #             color='skyblue', linewidth=0.5, alpha=0.2)

        # # Vertical lines for calibration boundaries
        # for cal_date in calib_date:
        #     plt.axvline(x=pd.to_datetime(cal_date), color='steelblue', linestyle='--', alpha=0.6)

        # #plt.plot(monthly_dates, gdp_dat, color='rebeccapurple', label='HP Filtered GDP')

        # plt.xlabel("Date")
        # plt.ylabel("Value")
        # plt.title("Occupation-Specific Shocks Over Time (ONET Occs)")
        # plt.xticks(rotation=45)
        # plt.grid()
<<<<<<< Updated upstream
        # plt.show()
=======
        # plt.close()
>>>>>>> Stashed changes

        occ_shocks_dat = np.array(df_monthly[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].transpose())

    elif nx == "omn_soc_minor":
        print("Using Full Corrected OMN") 
        A = pd.read_csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/data/omn_asec_11_19_soc_2010_minor_alltransitions_2025_normalised.csv", header = None)
        print(A.shape)
        ipums_input = pd.read_csv(path + "dRC_Replication/data/ipums_variables_SOC_minor_w_exp.csv", delimiter = ",")

        occ_ids = pd.read_csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/data/soc_2010_minor_codes_employment_asec.csv", delimiter=",")[['SOC']]
        #occ_ids = occ_ids.rename(columns={"Label": "label"})
        occ_ids['id'] = occ_ids.index.values
        assert(all(occ_ids['id'].values == range(0,len(occ_ids))))
        assert(all(occ_ids.index.values == occ_ids['id'].values))

        # diagnostics
        #missing = occ_ids['label'].isna().sum()
        #assert(missing == 0)

        employment = np.round(ipums_input[['emp']]/10000) + 1

        # Crude approximation using avg unemployment rate of ~5% - should aim for occupation-specific unemployment rates
        unemployment = round(employment * (0.05 / 0.95))
        #unemployment = np.round(ipums_input[['unemp']]/10000) + 1

        # Less crude approximation using avg vacancy rate - should still aim for occupation-specific vacancy rates
        vac_rate_base = pd.read_csv(path + "dRC_Replication/data/vacancy_rateDec2000.csv").iloc[:, 2].mean() / 100
        vacancies = round(employment * vac_rate_base / (1 - vac_rate_base))

        # Needs input data...
        demand_target = employment + vacancies

<<<<<<< Updated upstream
        wage_comp = ipums_input[['SOC_minor', 'median_weekly_earnings']]
=======
        wage_comp = ipums_input[['SOC_minor', 'median_weekly_earnings']].copy()
>>>>>>> Stashed changes
        wage_comp['median_annual_earnings'] = wage_comp['median_weekly_earnings'] * 52
        wage_dist = pd.read_csv("~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/data/occ_macro_vars/OEWS/wage_distributions_omn_soc_minor.csv", delimiter=",", header = 0)
        wage_full = wage_comp.merge(wage_dist, left_on='SOC_minor', right_on='SOC_minor', how='inner')

        gend_share = ipums_input[['female_share']]
        experience_req = ipums_input['experience_req']
        entry_level = ipums_input['entry_level']
        experience_age = ipums_input['experience_age']
        #seps_rates = pd.read_csv(path + "dRC_Replication/data/ipums_variables_w_seps_rate_full_omn.csv")
        #mean_seps_rate = seps_rates['seps_rate'].mean()

        if complete:
            print("Using complete network")
            # Create complete network
            n = A.shape
            complete_network = np.ones(n)
            A = complete_network

        mod_data = {
                "A": A,
                "employment": employment,
                'unemployment': unemployment,
                'vacancies': vacancies,
                'demand_target': demand_target,
                'wages': wage_full['a_median'],
                'wage_mu': wage_full['mu'],
                'wage_sigma': wage_full['sigma'],
                'gend_share': gend_share,
                'entry_level': entry_level,
                #'entry_age': experience_req['entry_age'],
                'experience_age': experience_age,
                #'separation_rates': seps_rates['seps_rate']*10
                'separation_rates': np.full_like(employment, np.nan, dtype=float)
        }
        print("Build mod_data.")
        print(f"Nodes (n): {len(mod_data['A'])}")

        ###################################
        # Initialise the model
        ##################################
        net_temp, vacs = initialise(
                len(mod_data['A']),
                mod_data['employment'].to_numpy(),
                mod_data['unemployment'].to_numpy(),
                mod_data['vacancies'].to_numpy(),
                mod_data['demand_target'].to_numpy(),
                mod_data['A'],
                mod_data['wages'].to_numpy(),
                mod_data['wage_mu'].to_numpy(),
                mod_data['wage_sigma'].to_numpy(),
                mod_data['gend_share'].to_numpy(),
                7, 1,
                mod_data['entry_level'],
                mod_data['experience_age'],
                mod_data['separation_rates']
        )
        print("Initialised network.")


        occ_shocks = pd.read_csv(path+"data/occupational_va_shocks_omn_soc_minor_occs.csv", index_col = 0).drop('occ_code', axis = 'columns')
<<<<<<< Updated upstream
        print(occ_shocks.shape)
=======
        #print(occ_shocks.shape)
>>>>>>> Stashed changes

        df_quarterly = occ_shocks.transpose() + 1  # now rows are dates, columns are occupations
        # Convert index to datetime
        df_quarterly.index = pd.to_datetime(df_quarterly.index)

        # Reindex to monthly frequency
        monthly_index = pd.date_range(start=df_quarterly.index.min(), end=df_quarterly.index.max(), freq='MS')  # Month start frequency
        df_monthly = df_quarterly.reindex(monthly_index)

        # Linearly interpolate missing monthly values
        df_monthly = df_monthly.interpolate(method='linear')

        for col in df_monthly.columns:
            # Segment 1: Before calibration window
            plt.plot(df_monthly.loc[df_monthly.index < calib_date[0]].index,
                    df_monthly.loc[df_monthly.index < calib_date[0], col],
                    color='darkseagreen', linewidth=0.5, alpha=0.1)

            # Segment 2: During calibration window
            plt.plot(df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].index,
                    df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1]), col],
                    color='forestgreen', linewidth=0.5, alpha = 0.5)

            # Segment 3: After calibration window
            plt.plot(df_monthly.loc[df_monthly.index > calib_date[1]].index,
                    df_monthly.loc[df_monthly.index > calib_date[1], col],
                    color='darkseagreen', linewidth=0.5, alpha=0.1)

        # Vertical lines for calibration boundaries
        for cal_date in calib_date:
            plt.axvline(x=pd.to_datetime(cal_date), color='steelblue', linestyle='--', alpha=0.6)

        #plt.plot(monthly_dates, gdp_dat, color='rebeccapurple', label='HP Filtered GDP')

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Occupation-Specific Shocks Over Time (ONET Occs)")
        plt.xticks(rotation=45)
        plt.grid()
<<<<<<< Updated upstream
        plt.show()
=======
        plt.savefig('output/omn_soc_minor/figures/occ_shocks.png', dpi=300)
        plt.close()
>>>>>>> Stashed changes

        occ_shocks_dat = np.array(df_monthly[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].transpose())
    elif nx == "onet_small":
        return
<<<<<<< Updated upstream
    # elif nx == "complete":
    #     print("Using complete network")
    #     # Create complete network
    #     n = A.shape
    #     complete_network = np.ones(n)
    #     mod_data = {
    #         "A": complete_network,
    #         "employment": employment,
    #         'unemployment': unemployment,
    #         'vacancies': vacancies,
    #         'demand_target': demand_target,
    #         'wages': wages,
    #         'gend_share': gend_share,
    #         'entry_level': experience_req['entry_level'],
    #         # 'entry_age': experience_req['entry_age'],
    #         'experience_age': experience_req['experience_age'],
    #         'separation_rates': seps_rates['seps_rate']*10
    #     }

    #     net_temp, vacs = initialise(
    #         len(mod_data['A']),
    #         mod_data['employment'].to_numpy(),
    #         mod_data['unemployment'].to_numpy(),
    #         mod_data['vacancies'].to_numpy(),
    #         mod_data['demand_target'].to_numpy(),
    #         mod_data['A'],
    #         mod_data['wages'].to_numpy(),
    #         mod_data['gend_share'].to_numpy(),
    #         7, 1,
    #         mod_data['entry_level'],
    #         mod_data['experience_age'],
    #         mod_data['separation_rates']
    #     )

=======
    
    elif nx == "single_node":
        print("Using Single Node Network")
        
        A = pd.DataFrame([[1.0]])
        
        employment = pd.DataFrame([10000])
        unemployment = pd.DataFrame([500])
        vacancies = pd.DataFrame([300])
        demand_target = employment + vacancies
        
        wages = pd.DataFrame([50000])
        wage_mu = np.array([10.8])
        wage_sigma = np.array([0.4])
        
        gend_share = pd.DataFrame([0.5])
        entry_level = pd.Series([True])
        experience_age = pd.Series([18])
        seps_rates = pd.DataFrame({'seps_rate': [0.02]})
        
        occ_ids = pd.DataFrame({
            'id': [0],
            'SOC': ['Single-Node'],
            'label': ['Single Occupation']
        })
        
        if complete:
            print("Network already complete (single node)")
        
        mod_data = {
            "A": A,
            "employment": employment,
            'unemployment': unemployment,
            'vacancies': vacancies,
            'demand_target': demand_target,
            'wages': wages.iloc[:, 0],
            'wage_mu': wage_mu,
            'wage_sigma': wage_sigma,
            'gend_share': gend_share,
            'entry_level': entry_level,
            'experience_age': experience_age,
            'separation_rates': seps_rates['seps_rate'] * 10
        }
        
        print("Build mod_data.")
        print(f"Nodes (n): {len(mod_data['A'])}")
        
        net_temp, vacs = initialise(
            len(mod_data['A']),
            mod_data['employment'].to_numpy(),
            mod_data['unemployment'].to_numpy(),
            mod_data['vacancies'].to_numpy(),
            mod_data['demand_target'].to_numpy(),
            mod_data['A'],
            mod_data['wages'].to_numpy() if hasattr(mod_data['wages'], 'to_numpy') else np.array([mod_data['wages']]),
            mod_data['wage_mu'],
            mod_data['wage_sigma'],
            mod_data['gend_share'].to_numpy(),
            7, 1,
            mod_data['entry_level'],
            mod_data['experience_age'],
            mod_data['separation_rates']
        )
        print("Initialised network.")

        #################################################################################################
        ####################### INPUT SERIES FOR MODEL ##################################################
        ##################################################################################################
        # Real GDP
        # Source: https://fred.stlouisfed.org/series/GDPC1
        realgdp = pd.read_csv(path+"data/macro_vars/GDPC1.csv", delimiter=',', decimal='.')
        realgdp["DATE"] = pd.to_datetime(realgdp["DATE"])
        realgdp["REALGDP"] = realgdp['GDPC1']
        realgdp['FD_REALGDP'] = pd.Series(realgdp['REALGDP']).diff()
        realgdp = realgdp[["DATE", "REALGDP"]].dropna(subset=["REALGDP"]).reset_index()
        realgdp['log_REALGDP'] = np.log2(realgdp['REALGDP'])

        # GDP Filter
        cycle, trend = hp_filter.hpfilter(realgdp['log_REALGDP'], lamb=129600)
        
        # Adding the trend and cycle to the original DataFrame
        realgdp['log_Trend'] = trend+1
        realgdp['log_Cycle'] = cycle+1
        realgdp['Trend'] = np.exp(trend)
        realgdp['Cycle'] = np.exp(cycle)

        realgdp_no_covid = realgdp[realgdp['DATE'] < "2019-10-1"].copy()
        realgdp['scaled_log_Cycle'] = (realgdp['log_Cycle'] - realgdp['log_Cycle'].min()) / (realgdp['log_Cycle'].max() - realgdp['log_Cycle'].min())
        realgdp_no_covid['scaled_log_Cycle'] = (realgdp_no_covid['log_Cycle'] - realgdp_no_covid['log_Cycle'].min()) / (realgdp_no_covid['log_Cycle'].max() - realgdp_no_covid['log_Cycle'].min())

        # Different calibration windows
        gdp_dat_pd = realgdp.set_index('DATE').sort_index()

        # Interpolate all columns in gdp_dat_pd to monthly frequency
        monthly_dates = pd.date_range(start=gdp_dat_pd.index.min(), end=gdp_dat_pd.index.max(), freq='MS')
        gdp_dat_pd_monthly = gdp_dat_pd.reindex(monthly_dates)

        # Linearly interpolate missing monthly values
        gdp_dat_pd_monthly = gdp_dat_pd_monthly.interpolate(method='linear')
        gdp_dat_pd_monthly = gdp_dat_pd_monthly[(gdp_dat_pd_monthly.index >= calib_date[0]) & (gdp_dat_pd_monthly.index <= calib_date[1])]

        # For backward compatibility with previous code
        gdp_dat = gdp_dat_pd_monthly['log_Cycle'].values + np.random.normal(loc=0, scale=0.001, size=len(gdp_dat_pd_monthly))
        monthly_dates = gdp_dat_pd_monthly.index
        
        occ_shocks_single = pd.DataFrame({
            0: gdp_dat}, index=monthly_dates)
        
        occ_shocks_dat = np.array(occ_shocks_single.transpose())
        
>>>>>>> Stashed changes
    return mod_data, net_temp, vacs, occ_ids, occ_shocks_dat