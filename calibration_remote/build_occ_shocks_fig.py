import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
#from torch import norm
from scipy.stats import norm
from statsmodels.tsa.filters import hp_filter
from abm_funs import *
from plot_style import TEXTWIDTH, save_plot_standard, apply_minor_gridlines, add_shared_legend

# calib_date = ["2000-12-01", "2019-05-01"]
# #calib_date = ["2000-12-01", "2024-05-01"]
calib_date = ["2000-12-01", "2019-05-01"]

path = "~/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"

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
            color='lavender', linewidth=0.5, alpha=0.1)

    # Segment 2: During calibration window
    plt.plot(df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= "2019-05-01")].index,
            df_monthly.loc[(df_monthly.index >= calib_date[0]) & (df_monthly.index <= "2019-05-01"), col],
            color='orchid', linewidth=0.5, alpha = 0.5)

    # Segment 3: After calibration window
    plt.plot(df_monthly.loc[df_monthly.index > "2019-05-01"].index,
            df_monthly.loc[df_monthly.index > "2019-05-01", col],
            color='lavender', linewidth=0.5, alpha=0.1)

# Vertical lines for calibration boundaries
plt.axvline(x=pd.to_datetime(calib_date[0]), color='steelblue', linestyle='--', alpha=0.6)
plt.axvline(x=pd.to_datetime("2019-05-01"), color='steelblue', linestyle='--', alpha=0.6)

temp_mask = (monthly_dates >= calib_date[0]) & (monthly_dates <= "2019-05-01")
plt.plot(monthly_dates[temp_mask], gdp_dat[temp_mask], color='rebeccapurple', label='HP Filtered GDP')
del temp_mask

plt.xlabel("Date")
plt.ylabel("De-trended Value")
plt.title("Occupation-Specific Demand Shocks")
plt.xticks(rotation=45)
plt.grid()
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/ABM_Transitions/new_figures/calib_final/full_omn/occupational_va_shocks.png"), bbox_inches='tight', 
    pad_inches=0.1, dpi=300)
plt.close()