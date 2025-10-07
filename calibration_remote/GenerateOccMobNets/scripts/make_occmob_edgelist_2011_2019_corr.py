import copy
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import os
import networkx as nx

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
# adjustment updwards of occ mobility recommended
adjustment = 0.55

# file with occupation names (https://cps.ipums.org/cps/codes/occ_20112019_codes.shtml)
# corrected solar panel switch
file_occ_in_cps2011_2019 = "occ_names_code_2011-2019.csv"
# pkl file for asec, cleaned for fast processing
file_asec = "cps_asec_tech_flags.csv"


# read occ dataset
df_occ = pd.read_csv(path_data + file_occ_in_cps2011_2019)
df_occ.head()

# read census file
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

print("Employed workers average (millions) before quality/imp control", \
df_asec["ASECWT"].sum()/(1e6*len(years)))

### TODO Make csv with employment
df_emp_incimp = df_asec.groupby("OCC")["ASECWT"].sum().reset_index()
df_emp_incimp["ASECWT"].sum()/(1e6*len(years))

df_emp_incimp["ASECWT_average_years"] = df_emp_incimp["ASECWT"]/len(years)
# export
df_emp_incimp.to_csv(path_data + "emp_asec_2011_2019_includeimputations.csv", \
    index=False)
dict_occ_empincimp = dict(zip(df_emp_incimp["OCC"], \
    df_emp_incimp["ASECWT_average_years"]))

# removing imputations and taking quality controls
df_asec = df_asec[df_asec['UH_SUPREC_A2']==1]
df_asec = df_asec[df_asec["QOCC"]==0]
df_asec = df_asec[df_asec["QOCCLY"]==0]
# df_asec = df_asec[df_asec["QIND"]==0]

# Uncomment to print occ mobility broad rate
# df_asec["ChangeOcc"] = df_asec["OCC"] != df_asec["OCCLY"]
# df_asec["ChangeOccWT"] = df_asec["ChangeOcc"] * df_asec["ASECWT"]

# print("Employed workers average (millions)", \
# df_asec["ASECWT"].sum()/(1e6*len(years)))
# print("People that changed occupation (millions) averge all years", \
# df_asec["ChangeOccWT"].sum()/(1e6*len(years)))
# print("Percentage of workforce that switched occupation in a year ", \
# df_asec["ChangeOccWT"].sum()/df_asec["ASECWT"].sum())

# get edgelist
df_edgelist = df_asec.groupby(['OCCLY', 'OCC'])["ASECWT"].sum().reset_index()
df_edgelist = df_asec.groupby(['OCCLY', 'OCC'])["ASECWT"]\
    .agg(ASECWT='sum', Observations='count').reset_index()

# map label
# make dictionary with index
df_occ = pd.read_csv(path_data + file_occ_in_cps2011_2019)
dict_code_name = dict(zip(df_occ["Code"], df_occ["Label"]))
df_edgelist["OCCLY_label"] = df_edgelist["OCCLY"].map(dict_code_name)
df_edgelist["OCC_label"] = df_edgelist["OCC"].map(dict_code_name)

# get employment
# grouping by employment
df_emp = df_asec.groupby("OCCLY")["ASECWT"].sum().reset_index()
# export employment without imputations to csv
df_emp["ASECWT_average_years"] = df_emp["ASECWT"]/len(years)
# export
df_emp.to_csv(path_data + "emp_asec_2011_2019_excludeimputations.csv", \
    index=False)


# dictionary with employment
dict_occly_emp = dict(zip(df_emp["OCCLY"], df_emp["ASECWT_average_years"]))

# now calculate average employment per year
df_emp_occ = df_asec.groupby("OCC")["ASECWT"].sum().reset_index()
df_emp_occ["ASECWT_average_years"] = df_emp_occ["ASECWT"]/len(years)
dict_occ_emp = dict(zip(df_emp_occ["OCC"], df_emp_occ["ASECWT_average_years"]))

df_edgelist["EMPOCCLY_unadj"] = df_edgelist["OCCLY"].map(dict_occly_emp)
df_edgelist["EMPOCC_unadj"] = df_edgelist["OCC"].map(dict_occ_emp)
df_edgelist["EMPOCC_incimp"] = df_edgelist["OCC"].map(dict_occ_empincimp)
# calculate transition prob (occ mobility rate)
df_edgelist["transition_prob"] = df_edgelist["ASECWT"]/df_edgelist["EMPOCCLY_unadj"] 
# adjusting mobility upwards by 55% as recomended
df_edgelist["transition_prob_adj"] = np.where(df_edgelist["OCC"] == df_edgelist["OCCLY"],\
    1 - (1 - df_edgelist["transition_prob"]) * (1 + adjustment), \
    df_edgelist["transition_prob"]* (1 + adjustment)) 
# income probability
df_edgelist["incoming_prob"] =  df_edgelist["ASECWT"]/df_edgelist["EMPOCC_unadj"]

df_edgelist.to_csv(path_data + "edgelist_qualitycontrol_2011_2019.csv", index=False)

# Playing around with the data
df_mobility = df_edgelist[df_edgelist["OCCLY"] != df_edgelist["OCC"]]
df_mobility['ASECWT'].sum()/df_edgelist['ASECWT'].sum()
df_mobility["transition_prob"].mean()


df_edgelist

len(df_edgelist[df_edgelist["transition_prob"] == 1])
df_edgelist[df_edgelist["transition_prob"] == 1]


#TODO count how many observations there are per occupation

#df_edgelist.to_csv(path_data + "edgelist_qualitycontrol_adjustedupwards_2011_2019.csv", index=False)

# Green transition occupations
occ_focus = 6540 # Solar panel installers
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus]
df_edgelist.loc[df_edgelist["OCC"] == occ_focus]

occ_focus = 7440 # 
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus]
df_edgelist.loc[df_edgelist["OCC"] == occ_focus]
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False)

# 1410 Electrical and electronics engineers
occ_focus = 1410 # 
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus]
df_edgelist.loc[df_edgelist["OCC"] == occ_focus]
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False)

# 6355 Electricians
occ_focus = 6355  # 
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus]
df_edgelist.loc[df_edgelist["OCC"] == occ_focus]
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False)

#     s
# 7040
#     Electric motor, power tool, and related repairers
# 7050
#     Electrical and electronics installers and repairers, transportation equipment
# 7100
#     Electrical and electronics repairers, industrial and utility
# 7110
#     Electronic equipment installers and repairers, motor vehicles


occ_focus = 7410 # Electrical power-line installers and repairers
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus]
df_edgelist.loc[df_edgelist["OCC"] == occ_focus]
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False)
df_edgelist.loc[df_edgelist["OCC"] == occ_focus].sort_values("incoming_prob", \
    ascending=False)

# where do they come from
df_edgelist.loc[df_edgelist["OCC"] == occ_focus].sort_values("ASECWT", \
    ascending=False)


# Life occs
# where do they come from
occ_focus = 1200
df_edgelist.loc[df_edgelist["OCC"] == occ_focus].sort_values("ASECWT", \
    ascending=False)
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False)
occ_focus = 1210
df_edgelist.loc[df_edgelist["OCC"] == occ_focus].sort_values("ASECWT", \
    ascending=False)
occ_focus = 1700
df_edgelist.loc[df_edgelist["OCC"] == occ_focus].sort_values("ASECWT", \
    ascending=False)
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False)
occ_focus = 1800
df_edgelist.loc[df_edgelist["OCC"] == occ_focus].sort_values("ASECWT", \
    ascending=False)
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False)

# Athletes and actors occupations
occ_focus = 2720
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus]
df_edgelist.loc[df_edgelist["OCC"] == occ_focus]
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False).to_csv(path_data + "athletes_mobility.csv")
df_edgelist.loc[df_edgelist["OCC"] == occ_focus].sort_values("incoming_prob", \
    ascending=False)

# Actors
occ_focus = 2700
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus]
df_edgelist.loc[df_edgelist["OCC"] == occ_focus]
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False).to_csv(path_data + "actors_mobility.csv")

occ_focus = 2710
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus]
df_edgelist.loc[df_edgelist["OCC"] == occ_focus]
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False)

occ_focus = 2760
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus]
df_edgelist.loc[df_edgelist["OCC"] == occ_focus]
df_edgelist.loc[df_edgelist["OCCLY"] == occ_focus].sort_values("transition_prob", \
    ascending=False)

df_asec["YEAR"].unique()