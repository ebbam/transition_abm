# File to collect all macro variables in one file
import numpy as np
import pandas as pd
from statsmodels.tsa.filters import hp_filter, bk_filter, cf_filter
from quantecon import hamilton_filter
import matplotlib.pyplot as plt
import seaborn as sns
import random as random
import datetime

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"

########################################################################################
############ MACRO_OBSERVATIONS ########################################################
########################################################################################

# Observed unemployment rate
# Monthly, seasonally adjusted
# Source: https://fred.stlouisfed.org/series/UNRATE

unrate = pd.read_csv(path+"data/macro_vars/UNRATE.csv", delimiter=',', decimal='.')
unrate["DATE"] = pd.to_datetime(unrate["DATE"])
unrate["UER"] = unrate['UNRATE']/100
unrate['FD_UNRATE'] = pd.Series(unrate['UER']).diff()

# Monthly, seasonally adjusted job openings rate (total nonfarm)
# Source: https://fred.stlouisfed.org/series/JTSJOR

jorate = pd.read_csv(path+"data/macro_vars/JTSJOR.csv", delimiter=',', decimal='.')
jorate["DATE"] = pd.to_datetime(jorate["DATE"])
jorate["VACRATE"] = jorate['JTSJOR']/100
jorate['FD_VACRATE'] = pd.Series(jorate['VACRATE']).diff()

macro_observations = pd.merge(unrate, jorate, how = 'outer', on = 'DATE')

# Real GDP
# Source: https://fred.stlouisfed.org/series/GDPC1
realgdp = pd.read_csv(path+"data/macro_vars/GDPC1.csv", delimiter=',', decimal='.')
realgdp["DATE"] = pd.to_datetime(realgdp["DATE"])
realgdp["REALGDP"] = realgdp['GDPC1']
realgdp['FD_REALGDP'] = pd.Series(realgdp['REALGDP']).diff()

macro_observations = pd.merge(macro_observations, realgdp, how = 'outer', on = 'DATE')

# Female unemployment rate
# https://fred.stlouisfed.org/series/LNS14000002
f_unrate = pd.read_csv(path+"data/macro_vars/LNS14000002.csv", delimiter=',', decimal='.')
f_unrate["DATE"] = pd.to_datetime(f_unrate["DATE"])
f_unrate["FEMALE_UER"] = f_unrate['LNS14000002']/100
f_unrate['FD_FEMALE_UER'] = pd.Series(f_unrate['FEMALE_UER']).diff()

# Male unemployment rate
# https://fred.stlouisfed.org/series/LNS14000001
m_unrate = pd.read_csv(path+"data/macro_vars/LNS14000001.csv", delimiter=',', decimal='.')
m_unrate["DATE"] = pd.to_datetime(m_unrate["DATE"])
m_unrate["MALE_UER"] = m_unrate['LNS14000001']/100
m_unrate['FD_MALE_UER'] = pd.Series(m_unrate['MALE_UER']).diff()

macro_observations = pd.merge(macro_observations, f_unrate, how = 'outer', on = 'DATE')
macro_observations = pd.merge(macro_observations, m_unrate, how = 'outer', on = 'DATE')

# Monthly, seasonally adjusted Of Total Unemployed, Percent Unemployed 27 Weeks & over (LNS13025703)
# Source: https://fred.stlouisfed.org/series/LNS13025703

ltuer = pd.read_csv(path+"data/macro_vars/LNS13025703.csv", delimiter=',', decimal='.')
ltuer["DATE"] = pd.to_datetime(ltuer["DATE"])
ltuer["LTUER"] = ltuer['LNS13025703']/100
ltuer['FD_LTUER'] = pd.Series(ltuer['LTUER']).diff()

macro_observations = pd.merge(macro_observations, ltuer, how = 'outer', on = 'DATE')

macro_observations.to_csv(path + "data/macro_vars/collated_macro_observations.csv")

########################################################################################
############ RECESSIONS ################################################################
########################################################################################
# Recession dates
# Source: https://fred.stlouisfed.org/series/USREC#:%7E:text=For%20daily%20data%2C%20the%20recession,the%20month%20of%20the%20trough

recessions = pd.read_csv(path+"data/macro_vars/USREC.csv", delimiter=',', decimal='.')
recessions["DATE"] = pd.to_datetime(recessions["DATE"])

recessions.to_csv(path + "data/macro_vars/collated_recessions.csv")

########################################################################################
############ JOLTS #####################################################################
########################################################################################

## JOLTS SURVEY: https://www.bls.gov/charts/job-openings-and-labor-turnover/hire-seps-rates.htm

# Separation rates (Total nonfarm): JOLTS Survey - monthly, seasonally adjusted
# Source: https://fred.stlouisfed.org/series/JTSTSR
seps = pd.read_csv(path+"data/macro_vars/JTSTSR.csv", delimiter=',', decimal='.')
seps["DATE"] = pd.to_datetime(seps["DATE"])
seps["SEPSRATE"] = seps['JTSTSR']/100
seps['FD_SEPSRATE'] = pd.Series(seps['SEPSRATE']).diff()

# Quits rate (Total nonfarm): JOLTS Survey - monthly, seasonally adjusted
# Source: https://fred.stlouisfed.org/series/JTSQUR
quits = pd.read_csv(path+"data/macro_vars/JTSQUR.csv", delimiter=',', decimal='.')
quits["DATE"] = pd.to_datetime(quits["DATE"])
quits["QUITSRATE"] = quits['JTSQUR']/100
quits['FD_QUITSRATE'] = pd.Series(quits['QUITSRATE']).diff()

jolts = pd.merge(quits, seps, how = 'left', on = 'DATE')

# Hires rate (Total nonfarm): JOLTS Survey - monthly, seasonally adjusted
# Source: https://fred.stlouisfed.org/series/JTSHIR
hires = pd.read_csv(path+"data/macro_vars/JTSHIR.csv", delimiter=',', decimal='.')
hires["DATE"] = pd.to_datetime(hires["DATE"])
hires["HIRESRATE"] = hires['JTSHIR']/100
hires['FD_HIRESRATE'] = pd.Series(hires['HIRESRATE']).diff()

jolts = pd.merge(jolts, hires, how = 'left', on = 'DATE')

# Layoffs and Discharges rate (Total nonfarm): JOLTS Survey - monthly, seasonally adjusted
# Source: https://fred.stlouisfed.org/series/JTSLDR
layoffs = pd.read_csv(path+"data/macro_vars/JTSLDR.csv", delimiter=',', decimal='.')
layoffs["DATE"] = pd.to_datetime(layoffs["observation_date"])
layoffs["LAYOFFRATE"] = layoffs['JTSLDR']/100
layoffs['FD_LAYOFFRATE'] = pd.Series(layoffs['LAYOFFRATE']).diff()

jolts = pd.merge(jolts, layoffs, how = 'left', on = 'DATE')

jolts.to_csv(path + "data/macro_vars/collated_jolts.csv")

########################################################################################
############ (T_DEMAND_REAL) REAL TARGET DEMAND ########################################
########################################################################################

# Job openings (levels - thousands)
# Source: https://fred.stlouisfed.org/series/JTSJOL
jobopenings = pd.read_csv(path+"data/macro_vars/JTSJOL.csv", delimiter=',', decimal='.')
jobopenings["DATE"] = pd.to_datetime(jobopenings["DATE"])
jobopenings["JOBOPENINGS"] = jobopenings['JTSJOL']
jobopenings['FD_JOBOPENINGS'] = pd.Series(jobopenings['JOBOPENINGS']).diff()

# Employment (total non-farm) (levels - thousands)
# Source: https://fred.stlouisfed.org/series/PAYEMS
emp_real = pd.read_csv(path+"data/macro_vars/PAYEMS.csv", delimiter=',', decimal='.')
emp_real["DATE"] = pd.to_datetime(emp_real["DATE"])
emp_real["EMPLOYMENT"] = emp_real['PAYEMS']
emp_real['FD_EMPLOYMENT'] = pd.Series(emp_real['EMPLOYMENT']).diff()

t_demand_real = pd.merge(jobopenings, emp_real, how = 'outer', on = 'DATE')
t_demand_real['TARGET_DEMAND'] = t_demand_real['JOBOPENINGS'] + t_demand_real['EMPLOYMENT']

#t_demand_real.to_csv(path + "data/macro_vars/collated_t_demand_real.csv")


#### GDP GROWTH RATE ####
gdp_growth = pd.read_csv(path+"data/macro_vars/GDPC1_pct_ch.csv", delimiter=',', decimal='.')
gdp_growth["DATE"] = pd.to_datetime(gdp_growth["DATE"])
gdp_growth["GDP_GROWTH"] = gdp_growth['GDPC1_PC1']


#################################################################################################
####################### INPUT SERIES FOR MODEL ##################################################
##################################################################################################

realgdp = macro_observations[["DATE", "REALGDP"]].dropna(subset=["REALGDP"]).reset_index()
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

k = 12
bk_cycle = bk_filter.bkfilter(realgdp['log_REALGDP'], low=6, high=32, K=k) + 1
padded_series = pd.Series(
    [np.nan]*k + list(bk_cycle) + [np.nan]*k,
    index=realgdp.index  
)

# Add it to the DataFrame
realgdp['bk_gdp'] = padded_series

k = 12
cf_cycle = cf_filter.cffilter(realgdp['log_REALGDP'], low=6, high=32, drift = True)[0] + 1
# Add it to the DataFrame
realgdp['cf_gdp'] = cf_cycle

# 8 recommended for quarterly data: https://quanteconpy.readthedocs.io/en/latest/tools/filter.html
H = 8
hamilton_cycle = hamilton_filter(realgdp['log_REALGDP'], h = H)[0] + 1

# Add it to the DataFrame
realgdp['hamilton_gdp'] = hamilton_cycle

# Load and clean OECD business confidence data
# Source: https://www.oecd.org/en/data/indicators/business-confidence-index-bci.html
bus_conf = pd.read_csv(path + "data/macro_vars/OECD_bus_conf_usa.csv")
bus_conf = bus_conf.loc[:, bus_conf.nunique() > 1]  # fix: assign back
bus_conf['DATE'] = pd.PeriodIndex(bus_conf['TIME_PERIOD'], freq='Q').start_time
bus_conf = bus_conf.sort_values(by='DATE')

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 8))

# Plot each transformation type
for i, index_type in enumerate(bus_conf['Transformation'].unique()):
    subset = bus_conf[bus_conf['Transformation'] == index_type]
    if index_type == "Index":
        subset['OBS_VALUE'] = hp_filter.hpfilter(subset['OBS_VALUE'], lamb=1600)[0]*.01+1
    ax = axes[i]  # fix: access subplot properly
    ax.plot(subset['DATE'], subset['OBS_VALUE'], marker='o')
    ax.set_title(f'{index_type}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.grid(True)

plt.suptitle('OECD Business Confidence Metrics - USA')
plt.tight_layout()
plt.close()

# Load JOLTS data
plt.figure(figsize=(12, 6))
plt.plot(macro_observations['DATE'], macro_observations['VACRATE'], label='VACRATE')
plt.plot(jolts['DATE'], jolts['HIRESRATE'], label='HIRESRATE')
plt.plot(jolts['DATE'], jolts['SEPSRATE'], label='SEPSRATE')
plt.plot(jolts['DATE'], jolts['QUITSRATE'], label='QUITSRATE')
plt.plot(jolts['DATE'], jolts['LAYOFFRATE'], label='LAYOFFRATE')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.title('JOLTS Rates Over Time')
plt.legend()
plt.grid(True)
plt.close()

#realgdp = macro_observations[["DATE", "REALGDP"]].dropna(subset=["REALGDP"]).reset_index()
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

k = 12
bk_cycle = bk_filter.bkfilter(realgdp['log_REALGDP'], low=6, high=32, K=k) + 1
padded_series = pd.Series(
    [np.nan]*k + list(bk_cycle) + [np.nan]*k,
    index=realgdp.index  
)

# Add it to the DataFrame
realgdp['bk_gdp'] = padded_series

k = 12
cf_cycle = cf_filter.cffilter(realgdp['log_REALGDP'], low=6, high=32, drift = True)[0] + 1
# Add it to the DataFrame
realgdp['cf_gdp'] = cf_cycle

# 8 recommended for quarterly data: https://quanteconpy.readthedocs.io/en/latest/tools/filter.html
H = 8
hamilton_cycle = hamilton_filter(realgdp['log_REALGDP'], h = H)[0] + 1

# Add it to the DataFrame
realgdp['hamilton_gdp'] = hamilton_cycle

# Extract and filter business confidence "Index"
bus_conf_index = bus_conf[bus_conf['Transformation'] == "Index"].copy()
bus_conf_index['OBS_VALUE'] = hp_filter.hpfilter(bus_conf_index['OBS_VALUE'], lamb=1600)[0]

# OPTIONAL: Normalize or scale to GDP cycle range if needed (for better visual comparison)
bus_conf_index['OBS_VALUE_scaled'] = (
    (bus_conf_index['OBS_VALUE'] - bus_conf_index['OBS_VALUE'].min()) /
    (bus_conf_index['OBS_VALUE'].max() - bus_conf_index['OBS_VALUE'].min())
) * (realgdp['log_Cycle'].max() - realgdp['log_Cycle'].min()) + realgdp['log_Cycle'].min()

# Main axis
fig, ax1 = plt.subplots()

# Plot on main axis
line1, = ax1.plot(realgdp['DATE'], realgdp['log_Cycle'], label="GDP (HP) - Current", color="purple")
line2, = ax1.plot(realgdp['DATE'], realgdp['bk_gdp'], label="GDP (BK)", color="blue", linestyle="dashed")
line3, = ax1.plot(realgdp['DATE'], realgdp['cf_gdp'], label="GDP (CF)", color="orange", linestyle="dashed")
line4, = ax1.plot(realgdp['DATE'], realgdp['hamilton_gdp'], label="GDP (Hamilton)", color="darkgreen", linestyle="dashed")
line5, = ax1.plot(bus_conf_index['DATE'], bus_conf_index['OBS_VALUE_scaled'],
    label="Business Confidence Index (HP)",
    color="crimson",
    linestyle="dotted"
)
ax1.set_ylabel("Cyclical GDP (log) and OECD Business Confidence Index")
ax1.set_ylim([0.925, 1.17])

# Set common attributes
ax1.set_xlim([datetime.date(2000, 1, 1), datetime.date(2019, 10, 1)])
ax1.set_title("(log) GDP Filters & Business Confidence Index")
ax1.set_xlabel("Date")

ax1.fill_between(
    recessions['DATE'], 0, 1,
    where=recessions['USREC'] == 1,
    transform=ax1.get_xaxis_transform(),
    color='grey', alpha=0.2, label='Recession'
)
# Combine legends
lines = [line1, line2, line3, line4, line5]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.tight_layout()
plt.close()

# Different calibration windows
# Full time series: "2024-5-1"
# calib_date = ["2004-12-01", "2019-05-01"]
calib_date = ["2000-12-01", "2019-05-01"]
#calib_date = ["2000-12-01", "2024-05-01"]
bus_conf_short = bus_conf_index[(bus_conf_index['DATE'] >= calib_date[0]) & (bus_conf_index['DATE'] <= calib_date[1])]
gdp_dat_pd = realgdp.set_index('DATE').sort_index()

# Interpolate all columns in gdp_dat_pd to monthly frequency
monthly_dates = pd.date_range(start=gdp_dat_pd.index.min(), end=gdp_dat_pd.index.max(), freq='MS')
gdp_dat_pd_monthly = gdp_dat_pd.reindex(monthly_dates)

# Linearly interpolate missing monthly values
gdp_dat_pd_monthly = gdp_dat_pd_monthly.interpolate(method='linear')
gdp_dat_pd_monthly = gdp_dat_pd_monthly[(gdp_dat_pd_monthly.index >= calib_date[0]) & (gdp_dat_pd_monthly.index <= calib_date[1])]

# For backward compatibility with previous code
gdp_dat = gdp_dat_pd_monthly['log_Cycle'].values + np.random.normal(loc=0, scale=0.001, size=len(gdp_dat_pd_monthly))
gdp_dat_bk = gdp_dat_pd_monthly['bk_gdp'].values + np.random.normal(loc=0, scale=0.001, size=len(gdp_dat_pd_monthly))if 'bk_gdp' in gdp_dat_pd_monthly else None
gdp_dat_cf = gdp_dat_pd_monthly['cf_gdp'].values + np.random.normal(loc=0, scale=0.001, size=len(gdp_dat_pd_monthly))if 'cf_gdp' in gdp_dat_pd_monthly else None
gdp_dat_hamilton = gdp_dat_pd_monthly['hamilton_gdp'].values+ np.random.normal(loc=0, scale=0.001, size=len(gdp_dat_pd_monthly)) if 'hamilton_gdp' in gdp_dat_pd_monthly else None
bus_conf_dat = np.array(bus_conf_short['OBS_VALUE_scaled'])

monthly_dates = gdp_dat_pd_monthly.index

# PLOTTING
plt.figure(figsize=(12, 6))

# Plot original data
sns.lineplot(data = realgdp, x = 'DATE', y = 'log_Cycle', color = "purple", label = "Full GDP Time series - UER and VACRATE data available from 2000", linestyle = "dotted")
sns.lineplot(x = monthly_dates, y=gdp_dat, color='purple', label='HP Filtered GDP')
sns.lineplot(x=monthly_dates, y=gdp_dat_bk, color='blue', label='BK Filtered GDP', linestyle = 'dashed')
sns.lineplot(x=monthly_dates, y=gdp_dat_cf, color='orange', label='CF Filtered GDP', linestyle = 'dashed')
sns.lineplot(x=monthly_dates, y=gdp_dat_hamilton, color='darkgreen', label='Hamilton Filtered GDP', linestyle = 'dashed')
sns.lineplot(data = bus_conf_short, x='DATE', y='OBS_VALUE_scaled', color='red', label='Business Confidence Index', linestyle = 'dotted')

# Mark original data boundaries
plt.axvline(x=monthly_dates.min(), color='black', linestyle='--', alpha=0.6)
plt.axvline(x=monthly_dates.max(), color='black', linestyle='--', alpha=0.6)

plt.xlabel('Date')
plt.ylabel('log_Cycle')
plt.title('Calibration Window')
plt.legend()
plt.close()

##################################################################################################
####################### OCCUPATION-SPECIFIC SHOCKS ###############################################
##################################################################################################

# occ_shocks = pd.read_csv(path+"data/occupational_va_shocks.csv", index_col = 0).drop('X', axis = 'columns')
# df_quarterly = occ_shocks.transpose() + 1  # now rows are dates, columns are occupations
# # Convert index to datetime
# df_quarterly.index = pd.to_datetime(df_quarterly.index)

# # Reindex to monthly frequency
# monthly_index = pd.date_range(start=df_quarterly.index.min(), end=df_quarterly.index.max(), freq='MS')  # Month start frequency
# df_monthly = df_quarterly.reindex(monthly_index)

# # Linearly interpolate missing monthly values
# df_monthly = df_monthly.interpolate(method='linear')

# for col in df_monthly.columns:
#     # Segment 1: Before calibration window
#     plt.plot(df_monthly.loc[df_monthly.index < calib_date[0]].index,
#              df_monthly.loc[df_monthly.index < calib_date[0], col],
#              color='lavender', linewidth=0.5, alpha=0.2)

#     # Segment 2: During calibration window
#     plt.plot(df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].index,
#              df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1]), col],
#              color='orchid', linewidth=0.5, alpha = 0.5)

#     # Segment 3: After calibration window
#     plt.plot(df_monthly.loc[df_monthly.index > calib_date[1]].index,
#              df_monthly.loc[df_monthly.index > calib_date[1], col],
#              color='lavender', linewidth=0.5, alpha=0.2)

# # Vertical lines for calibration boundaries
# for cal_date in calib_date:
#     plt.axvline(x=pd.to_datetime(cal_date), color='steelblue', linestyle='--', alpha=0.6)

# plt.plot(monthly_dates, gdp_dat, color='rebeccapurple', label='HP Filtered GDP')

# plt.xlabel("Date")
# plt.ylabel("Value")
# plt.title("Occupation-Specific Shocks Over Time")
# plt.xticks(rotation=45)
# plt.grid()
# plt.close()

# #plt.savefig('output/figures/occ_shocks.png', dpi=300)


# occ_shocks_dat = np.array(df_monthly[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= calib_date[1])].transpose())


# Seeker composition


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

seekers_comp_obs_full = seekers_comp_obs_full.rename(columns={"comp_searchers_s": "Seeker Composition"})
seekers_comp_obs_full['DATE'] = pd.to_datetime(seekers_comp_obs_full['DATE'])
seekers_comp_obs = seekers_comp_obs_full[(seekers_comp_obs_full['DATE'] >= calib_date[0]) & (seekers_comp_obs_full['DATE'] <= calib_date[1])]

#seekers_comp_obs.to_csv(path + "data/macro_vars/collated_seekers_composition.csv", index = False)

plt.plot(seekers_comp_obs_full['DATE'], seekers_comp_obs_full['Seeker Composition'], label='Full Series', color='lightgray', linestyle='dotted')
plt.plot(seekers_comp_obs['DATE'], seekers_comp_obs['Seeker Composition'], label='Calibration Window', color='blue', linestyle='solid')
plt.close()

