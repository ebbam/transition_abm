import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from abm_funs import *
calib_date = ["2000-12-01", "2019-05-01"]

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"


# Network in put file
def network_input_builder(nx):
    if nx == "original_omn":
        print("Using Corrected Original OMN")
        A = pd.read_csv("/Users/ebbamark/Dropbox/GenerateOccMobNets/data/omn_asec_11_19_occ_matched_plus_oldtransitions_2025_normalised.csv", delimiter=",", header = None)
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


        mod_data = {
            "A": A,
            "employment": employment,
            'unemployment': unemployment,
            'vacancies': vacancies,
            'demand_target': demand_target,
            'wages': wages,
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
        # plt.show()

        # plt.savefig('output/figures/occ_shocks.png', dpi=300)

        occ_shocks_dat = np.array(df_monthly[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].transpose())

    elif nx == "full_omn":
        print("Using Full Corrected OMN") 
        A = pd.read_csv("/Users/ebbamark/Dropbox/GenerateOccMobNets/data/asec_11_19_occ_alltransitions_2025_normalised.csv", header = None)
        print(A.shape)
        ipums_input = pd.read_csv(path +"dRC_Replication/data/ipums_variables_full_omn_w_exp.csv", delimiter = ",")
        #ipums_input = pd.read_csv("/Users/ebbamark/Dropbox/GenerateOccMobNets/ONET/occ_names_employment_asec_occ_ipums_vars.csv", delimiter=",")

        occ_ids = pd.read_csv("/Users/ebbamark/Dropbox/GenerateOccMobNets/data/occ_names_employment_asec_occ.csv", delimiter=",")[['Code', 'Label']]
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
        wages = ipums_input[['median_weekly_earnings']]*52
        gend_share = ipums_input[['female_share']]
        experience_req = ipums_input['experience_req']
        entry_level = ipums_input['entry_level']
        experience_age = ipums_input['experience_age']
        seps_rates = pd.read_csv(path + "dRC_Replication/data/ipums_variables_w_seps_rate_full_omn.csv")
        mean_seps_rate = seps_rates['seps_rate'].mean()

        mod_data = {
                "A": A,
                "employment": employment,
                'unemployment': unemployment,
                'vacancies': vacancies,
                'demand_target': demand_target,
                'wages': wages,
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
                mod_data['gend_share'].to_numpy(),
                7, 1,
                mod_data['entry_level'],
                mod_data['experience_age'],
                mod_data['separation_rates']
        )
        print("Initialised network.")


        occ_shocks = pd.read_csv(path+"data/occupational_va_shocks_full_omn.csv", index_col = 0).drop('acs_2010_code', axis = 'columns')
        print(occ_shocks.shape)

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
        plt.show()

        occ_shocks_dat = np.array(df_monthly[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].transpose())

    elif nx == "onet":
        print("Using ONET Related Occupations Network")
        A = pd.read_csv("/Users/ebbamark/Dropbox/GenerateOccMobNets/ONET/acs_2010_max_index_adjacency_matrix.csv", index_col = 0)
        ipums_input = pd.read_csv(path + "dRC_Replication/data/acs_onet_2010_ipums_vars_w_exp.csv", delimiter=",")

        occ_ids = ipums_input[['X.1', 'acs_occ_code']]
        occ_ids = occ_ids.rename(columns={"X.1": "id"})
        occ_ids['id'] = occ_ids['id'] - 1  # change to zero-indexed
        assert(all(occ_ids['id'].values == range(0,len(occ_ids))))
        assert(all(occ_ids.index.values == occ_ids['id'].values))
        labels = pd.read_csv("/Users/ebbamark/Dropbox/GenerateOccMobNets/ONET/acs_onet_2010_soc_cw.csv")

        # merge titles from labels into occ_ids by ACS code and create 'label' column
        labels_map = labels[['acs_2010_code', 'title']].drop_duplicates(subset='acs_2010_code')
        labels_map = labels_map.rename(columns={'acs_2010_code': 'acs_occ_code', 'title': 'label'})

        occ_ids = occ_ids.merge(labels_map, on='acs_occ_code', how='left', sort=False)

        # diagnostics
        missing = occ_ids['label'].isna().sum()
        assert(missing == 0)

        assert(np.array_equal(A.index.values, occ_ids['acs_occ_code'].values))

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
        wages = ipums_input[['median_weekly_earnings']]*52
        gend_share = ipums_input[['female_share']]
        experience_req = ipums_input['experience_req']
        entry_level = ipums_input['entry_level']
        experience_age = ipums_input['experience_age']
        #seps_rates = np.array(516, 0.02)
        seps_rates = pd.read_csv("/Users/ebbamark/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/dRC_Replication/data/ipums_variables_w_seps_rate_onet.csv")
        mean_seps_rate = seps_rates['seps_rate'].mean()

        mod_data = {
            "A": A_norm,
            "employment": employment,
            'unemployment': unemployment,
            'vacancies': vacancies,
            'demand_target': demand_target,
            'wages': wages,
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
            mod_data['gend_share'].to_numpy(),
            7, 1,
            mod_data['entry_level'],
            mod_data['experience_age'],
            mod_data['separation_rates']
        )

        print("Initialised network.")

        occ_shocks = pd.read_csv(path+"data/occupational_va_shocks_onet_related_occs.csv", index_col = 0).drop('X', axis = 'columns')
        print(occ_shocks.shape)

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
        # plt.show()

        occ_shocks_dat = np.array(df_monthly[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].transpose())


    elif nx == "complete":
        print("Using complete network")
        # Create complete network
        n = A.shape
        complete_network = np.ones(n)
        mod_data = {
            "A": complete_network,
            "employment": employment,
            'unemployment': unemployment,
            'vacancies': vacancies,
            'demand_target': demand_target,
            'wages': wages,
            'gend_share': gend_share,
            'entry_level': experience_req['entry_level'],
            # 'entry_age': experience_req['entry_age'],
            'experience_age': experience_req['experience_age'],
            'separation_rates': seps_rates['seps_rate']*10
        }

        net_temp, vacs = initialise(
            len(mod_data['A']),
            mod_data['employment'].to_numpy(),
            mod_data['unemployment'].to_numpy(),
            mod_data['vacancies'].to_numpy(),
            mod_data['demand_target'].to_numpy(),
            mod_data['A'],
            mod_data['wages'].to_numpy(),
            mod_data['gend_share'].to_numpy(),
            7, 1,
            mod_data['entry_level'],
            mod_data['experience_age'],
            mod_data['separation_rates']
        )

    return mod_data, net_temp, vacs, occ_ids, occ_shocks_dat