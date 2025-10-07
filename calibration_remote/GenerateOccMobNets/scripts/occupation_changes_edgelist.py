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
# Outdated info on 2014
# print("Remove year 2014 since problems in data")
# years.remove(2014)

# file with occupation names (https://cps.ipums.org/cps/codes/occ_20112019_codes.shtml)
file_occ_in_cps2011_2019 = "cps_asec_tech.csv"
# pkl file for asec, cleaned for fast processing
file_occ_in_cps2011_2019 = "occ_names_code_2011-2019.csv"

# read dataset with census data
df_asec = pd.read_csv(path_data + "cps_asec_tech_flags.csv")
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
# remove imputed values and quality flags
df_asec = df_asec[df_asec['UH_SUPREC_A2']==1]
df_asec = df_asec[df_asec["QOCC"]==0]
df_asec = df_asec[df_asec["QOCCLY"]==0]
df_asec = df_asec[df_asec["QIND"]==0]
# same broad classification

df_asec.columns


df_edgelist = df_asec[["YEAR", "CPSIDP", "AGE", "SEX", "IND", "INDLY",\
    "ASECWT", "OCCLY", "OCC" ]]

df_edgelist.head()
df_edgelist.to_csv(path_data + "transitions_forclassifier_usa.csv", index=False)