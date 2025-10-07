import copy
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import os
import networkx as nx

'''Code to reproduce Fig B.1 bottom left panel from
"Reconciling Occupational Mobility in the Current Population Survey." 
Journal of Labor Economics 40(4). DOI: https://doi.org/10.1086/718563.
'''

# Get the current working directory
cwd = os.getcwd()
cwd
# Print the current working directory
print("Current working directory: {0}".format(cwd))

# %matplotlib inline
#defining paths
path_data = "../data/"
path_fig = "../results/fig/"

year_start = 2011
year_end = 2019 + 1 # up to 2019
years = [i for i in range(year_start, year_end)]
# Outdated info on 2014
# print("Remove year 2014 since problems in data")
# years.remove(2014)

# file with occupation names (https://cps.ipums.org/cps/codes/occ_20112019_codes.shtml)
# file edited directly after this post 
# https://forum.ipums.org/t/2010-census-occ-classification-mistmatch-code-for-solar-photovoltaic-installers/4919
file_occ_in_cps2011_2019 = "occ_names_code_2011-2019.csv"
file_asec = "cps_asec_tech_flags.csv"
# read occ dataset
df_occ = pd.read_csv(path_data + file_occ_in_cps2011_2019  )
df_occ.head()

df_occ.loc[df_occ["Code"] == 6540]

df_asec = pd.read_csv(path_data + file_asec)
df_asec.columns
# filter by those with ASECFLAG, AGE and Emp status
df_asec = df_asec[df_asec["ASECFLAG"] == 1]
df_asec = df_asec[df_asec["AGE"] >=18]
employed_codes = [10, 12]
# where unemp codes come from
# a = set(df_asec["OCC"].unique())
# b = set(df_asec["OCCLY"].unique())
# b.difference(a)
unemployed_occ_codes = [0, 9840]
df_asec = df_asec[df_asec["EMPSTAT"].isin(employed_codes)]
df_asec = df_asec[~df_asec["OCCLY"].isin(unemployed_occ_codes)]


### Extra cleaning lines, see effect
df_asec = df_asec[df_asec["ASECWT"].notna()]
# remove CPSIDP unless they are part of the asec over sample
df_asec = df_asec[(df_asec["CPSIDP"] != 0) | (df_asec["ASECOVERP"] == 1)]
# Restrict dataset to years we want
df_asec = df_asec[df_asec["YEAR"].isin(years)]
###

labor_force_average_bf_qf = df_asec["ASECWT"].sum()/(1e6*len(years))
print("Average employment before quality flags", labor_force_average_bf_qf)

def occ_mobility(year):
    # focusing on year 2018 with 1950 codes
    df_asec_year = df_asec[df_asec["YEAR"] == year]
    # Make a column documenting changes in occupations
    df_asec_year["ChangeOcc"] = np.where(df_asec_year['OCC1950']!=df_asec_year['OCC50'+'LY'], True, False)
    # Multiply each change in occupation by asec weight
    df_asec_year["ChangeOccWT"] = df_asec_year["ChangeOcc"] * df_asec_year["ASECWT"]
    # calculate percentage of worker that switched

    print("Occupational mobility rate ", year, \
          np.round(df_asec_year["ChangeOccWT"].sum()/df_asec_year["ASECWT"].sum(), 3))
    return df_asec_year["ChangeOccWT"].sum()/df_asec_year["ASECWT"].sum()

# Print and plot results
mobility_50codes_with_imp = []
for y in range(2011, 2019):
    mobility_50codes_with_imp.append(occ_mobility(y))
    
plt.plot([y for y in range(2011, 2019)], mobility_50codes_with_imp, "o-")
plt.ylabel("occupational mobility rate")
plt.title("occupational mobility using unharmonized 2010 codes")
plt.show()

df_asec = df_asec[df_asec['UH_SUPREC_A2']==1]
df_asec = df_asec[df_asec["QOCC"]==0]
df_asec = df_asec[df_asec["QOCCLY"]==0]
#df_asec = df_asec[df_asec["QIND"]==0]
df_asec.columns
#df_asec = df_asec[df_asec["QOCCLY"]==0]
# Print and plot results
mobility_50codes_wo_imp = []
for y in range(2011, 2019):
    mobility_50codes_wo_imp .append(occ_mobility(y))
    
plt.plot([y for y in range(2011, 2019)], mobility_50codes_wo_imp, "o-")
plt.ylabel("occupational mobility rate")
plt.title("occupational mobility using unharmonized 2010 codes")
plt.show()


plt.plot([y for y in range(2011, 2019)], mobility_50codes_with_imp, "--")
plt.plot([y for y in range(2011, 2019)],  mobility_50codes_wo_imp, "-")
plt.ylabel("Annual Occupational Switching Rate")
plt.title("occupational mobility using unharmonized 2010 codes")
plt.ylim([0.02, 0.14])
plt.xlim([1980, 2020])
plt.savefig(path_fig + "reproduce_plot.png", bbox_inches="tight")
plt.show()

labor_force_average_af_qf = df_asec["ASECWT"].sum()/(1e6*len(years))
print("Average employment before quality flags", labor_force_average_bf_qf)
print("Average employment after quality flags", labor_force_average_af_qf)




# TODO perhaps do everything with OCC codes and narrow it down to 3 digits.
# old codes
df_asec["OCC"] = df_asec["OCC"].astype(str).str[:3]
df_asec["OCCLY"] = df_asec["OCCLY"].astype(int).astype(str).str[:3]

# df_onet['O*NET-SOC Code broad'] = df_onet['O*NET-SOC Code'].str.slice(start=0, stop=6)
# df_onet['O*NET-SOC Code broad'] = df_onet['O*NET-SOC Code broad'] + "0"

def occ_mobility(year):
    # focusing on year 2018 with 1950 codes
    df_asec_year = df_asec[df_asec["YEAR"] == year]
    # Make a column documenting changes in occupations
    df_asec_year["ChangeOcc"] = np.where(df_asec_year['OCC']!=df_asec_year['OCC'+'LY'], True, False)
    # Multiply each change in occupation by asec weight
    df_asec_year["ChangeOccWT"] = df_asec_year["ChangeOcc"] * df_asec_year["ASECWT"]
    # calculate percentage of worker that switched

    print("Occupational mobility rate ", year, \
          np.round(df_asec_year["ChangeOccWT"].sum()/df_asec_year["ASECWT"].sum(), 3))
    return df_asec_year["ChangeOccWT"].sum()/df_asec_year["ASECWT"].sum()


mobility_allcodes = []
for y in range(2011, 2019):
    mobility_allcodes.append(occ_mobility(y))
    
plt.plot([y for y in range(2011, 2019)], mobility_allcodes, "o-")
plt.ylabel("occupational mobility rate")
plt.title("occupational mobility using unharmonized occ codes")
plt.show()

df_asec.columns

df_asec['UH_SUPREC_A2']



len(df_asec_2018["MARBASECIDP"].unique())

# read census file
df_asec = pd.read_pickle(path_data + file_asec_pkl)
df_asec = df_asec[df_asec["ASECWT"].notna()]

# df_asec = df_asec[df_asec["ASECFLAG"]== 1]
df_asec = df_asec[df_asec["ASECOVERP"]== 0]
df_asec_2018 = df_asec[df_asec["YEAR"] == 2018]
df_asec = df_asec[(df_asec["ASECWT"].notna()) & (df_asec["OCC"] != 9999) & (df_asec["OCCLY"] != 9999)\
                 & (df_asec["OCC"] != 9920) & (df_asec["OCCLY"] != 9920)]
# remove CPSIDP unless they are part of the asec over sample
df_asec_2018 = df_asec_2018[(df_asec_2018["CPSIDP"] != 0) | (df_asec_2018["ASECOVERP"] == 1)]
df_asec_2018 = df_asec_2018[df_asec_2018["AGE"] >=18]
df_asec_2018 = df_asec_2018[df_asec_2018["EMPSTAT"].isin([10, 12])]
df_asec_2018["ASECWT"].sum()
#calculating mobility with non harmonized variables
df_asec_2018["ChangeOcc"] = df_asec_2018["OCC"] != df_asec_2018["OCCLY"]
df_asec_2018["ChangeOccWT"] = df_asec_2018["ChangeOcc"] * df_asec_2018["ASECWT"]
df_asec_2018["ChangeOccWT"].sum()/df_asec_2018["ASECWT"].sum()


list(df_asec_2018.columns)

df_asec_2018["ChangeOcc"] = df_asec_2018["OCC2010"] != df_asec_2018["OCC10LY"]
df_asec_2018["ChangeOccWT"] = df_asec_2018["ChangeOcc"] * df_asec_2018["ASECWT"]
df_asec_2018["ChangeOccWT"].sum()/df_asec_2018["ASECWT"].sum()

occ = set(df_asec["OCC"].unique())
occ_2010 = set(df_asec["OCC2010"].unique())

len(occ)
len(occ_2010)

len(occ.difference(occ_2010))

occ_2010
occ



# remove CPSIDP unless they are part of the asec over sample
df_asec = df_asec[(df_asec["CPSIDP"] != 0) | (df_asec["ASECOVERP"] == 1)]
# Restrict dataset to years we want
df_asec = df_asec[df_asec["YEAR"].isin(years)]
# remove nan codes
df_asec = df_asec[(df_asec["ASECWT"].notna()) & (df_asec["OCC"] != 9999) & (df_asec["OCCLY"] != 9999)\
                 & (df_asec["OCC"] != 9920) & (df_asec["OCCLY"] != 9920)]

