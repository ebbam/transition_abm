import copy
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import os
import networkx as nx

cwd = os.getcwd()
cwd
# Print the current working directory
print("Current working directory: {0}".format(cwd))


path_data = "../data/"
path_fig = "../results/fig/"

year_start = 2011
year_end = 2019 + 1 # up to 2019
years = [i for i in range(year_start, year_end)]
print("Remove year 2014 since problems in data")
years.remove(2014)

# file with occupation names (https://cps.ipums.org/cps/codes/occ_20112019_codes.shtml)
file_occ_in_cps2011_2019 = "occ_names_code_2011-2019.csv"
# pkl file for asec, cleaned for fast processing
file_asec_pkl = "CPS_asec_df_occmob_summary.pkl"
file_xwalk = "xwalk_asec_bls_2011-2019.csv"

T_asec_year_list = []

for y in years:
    A_omn = np.genfromtxt(path_data + "omn_cps_11_19_" + str(y) + "_.csv", delimiter=",")
    T_asec_year_list.append(A_omn)

df_indegree = pd.DataFrame()
for i,y in enumerate(years):
    df_indegree[str(y)] = T_asec_year_list[i].sum(axis=0)
df_indegree = df_indegree.T

df_outdegree = pd.DataFrame()
for i,y in enumerate(years):
    df_outdegree[str(y)] = T_asec_year_list[i].sum(axis=1)
df_outdegree = df_outdegree.T

len(list(zip(*np.where(df_indegree == 0))))/len(years)

df_indegree

np.where(df_indegree == 0)

years

A_omn.shape

codes_ipums = list(pd.read_csv( path_data + "quick_codes.csv")["Unnamed: 0"])

len(codes_ipums)

dict_code_index = {}
for i in range(len(codes_ipums)):
    dict_code_index[codes_ipums[i]] = i
dict_code_index[7740]
df_indegree[1]

np.where(T_asec_cps_year_list[i].sum(axis=0) == 0)

for i in range(len(years)):
    # 7440 in T_asec_cps_year_list[i][T_asec_cps_year_list[i].sum(axis=0) == 0].index
    print(T_asec_cps_year_list[i][T_asec_cps_year_list[i].sum(axis=0) == 0])

T_asec_cps_year_list[0]

# read occ dataset
df_occ = pd.read_csv(path_data + file_occ_in_cps2011_2019  )
df_occ.head()

# read census file
df_asec = pd.read_pickle(path_data + file_asec_pkl)
df_asec = df_asec[df_asec["ASECWT"].notna()]
# remove CPSIDP unless they are part of the asec over sample
df_asec = df_asec[(df_asec["CPSIDP"] != 0) | (df_asec["ASECOVERP"] == 1)]
# Restrict dataset to years we want
df_asec = df_asec[df_asec["YEAR"].isin(years)]
# remove nan codes
df_asec = df_asec[(df_asec["ASECWT"].notna()) & (df_asec["OCC"] != 9999) & (df_asec["OCCLY"] != 9999)\
                 & (df_asec["OCC"] != 9920) & (df_asec["OCCLY"] != 9920)]


print("Workforce estimation (millions) average over years", \
df_asec["ASECWT"].sum()/(1e6*len(years)))

# remove those not in universe
df_asec = df_asec[(df_asec["OCC"] != 0) & (df_asec["OCCLY"] != 0)]

#compute occ mobility
df_asec["ChangeOcc"] = df_asec["OCC"] != df_asec["OCCLY"]
df_asec["ChangeOccWT"] = df_asec["ChangeOcc"] * df_asec["ASECWT"]

print("Labor force with occupation (millions)", \
df_asec["ASECWT"].sum()/(1e6))
print("People that changed occupaiton (millions) all years", \
df_asec["ChangeOccWT"].sum()/1e6)
print("Percentage of workforce that switched occupation in a year ", \
df_asec["ChangeOccWT"].sum()/df_asec["ASECWT"].sum())

# occupation codes in asec
occ_asec = set(df_asec["OCC"].unique())
# occupation codes from external
occ_list = set(df_occ["Code"].unique())
print("Codes in ASEC but not in external list", len(occ_asec.difference(occ_list)))
print("Codes in external list but not ASEC", len(occ_list.difference(occ_asec)))

# remove codes from list not in asec
to_remove_list = occ_list.difference(occ_asec)
print("adding removal dummy in external due to not in ASEC")
df_occ["not_in_asec11-19"] = df_occ["Code"].isin(to_remove_list)
df_occ_clean = copy.deepcopy(df_occ)
# only keep those that are in asec
df_occ_clean = df_occ_clean[~df_occ_clean["not_in_asec11-19"]]
# make dictionary with index
dict_code_name = dict(zip(df_occ_clean["Code"], df_occ_clean["Label"]))
occ_names = list(df_occ_clean["Label"])
occ_codes = list(df_occ_clean["Code"])

#### Get differences in codes from crosswalk and exclude those codes
df_xwalk = pd.read_csv(path_data + file_xwalk)
xwalk_codes = [int(i) for i in list(df_xwalk.columns[1:])]
set(xwalk_codes).difference(set(occ_codes))
to_remove = set(occ_codes).difference(set(xwalk_codes))
len(xwalk_codes)
len(occ_codes)
xwalk_codes

9840 in xwalk_codes

to_remove

for rem_code in to_remove:
    occ_codes.remove(rem_code)

len(occ_codes)

dict_code_index = {}
for i in range(len(occ_codes)):
    dict_code_index[occ_codes[i]] = i

#Employment
dict_occ_emp = {}
for index, row in df_occ.iterrows():
    dict_occ_emp[row["Code"]] = 0

for index, row in df_asec.iterrows():
    dict_occ_emp[row["OCC"]] += row["ASECWT"]/len(years)

df_occ_clean["EMP_2011_2019_avg"] = df_occ_clean["Code"].map(dict_occ_emp)


# Occupation mobility
def occ_mob_asec_occ(df, dict_code_index, year, normalize=True):
    """Makes occupational mobiltiy network from ASEC
    """
    if normalize:
        norm = len(years)
    else:
        norm=1

    df_temp = df[df["YEAR"] == year]
    # make empty matrix
    n = len(dict_code_index)
    A = np.zeros([n, n])
    # fill whenver there is a transition
    for index, row in df_temp .iterrows():
        if row["OCC"] != row["OCCLY"]:
            if row["OCCLY"] in dict_code_index.keys() and row["OCC"] in dict_code_index.keys():
                i, j = dict_code_index[(row["OCCLY"])], dict_code_index[row["OCC"]]
                A[i, j] += row["ASECWT"]/norm # consider ASEC weights

    return A

# T_asec_year_list = []
# for y in years:
#     omn_year = occ_mob_asec_occ(df_asec, dict_code_index, year=y, normalize=False)
#     T_asec_year_list.append(omn_year)
#     np.savetxt(path_data + "omn_cps_11_19_" + str(y) +"_.csv", omn_year, delimiter=",")
