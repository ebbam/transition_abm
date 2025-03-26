# File to collect all macro variables in one file
import numpy as np
import pandas as pd
import random as random

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
