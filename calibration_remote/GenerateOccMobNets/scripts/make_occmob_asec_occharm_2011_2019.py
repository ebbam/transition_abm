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
path_data = "data/"
path_fig = "results/fig/"

year_start = 2011
year_end = 2019 + 1 # up to 2019
years = [i for i in range(year_start, year_end)]
print("Remove year 2014 since problems in data")
years.remove(2014)

# file with occupation names (https://cps.ipums.org/cps/codes/occ_20112019_codes.shtml)
file_occ_in_cps2011_2019 = "occ_names_employment_asec_occ2010.csv"
file_occ_in_cps2011_2019 = "occ2010_in_cps2019.csv"
# pkl file for asec, cleaned for fast processing
file_asec_pkl = "CPS_asec_df_occmob_summary.pkl"

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
df_asec = df_asec[(df_asec["ASECWT"].notna()) & (df_asec["OCC2010"] != 9999) & (df_asec["OCC10LY"] != 9999)\
                 & (df_asec["OCC2010"] != 9920) & (df_asec["OCC10LY"] != 9920)]


print("Workforce estimation (millions) average over years", \
df_asec["ASECWT"].sum()/(1e6*len(years)))

# remove those not in universe
df_asec = df_asec[(df_asec["OCC2010"] != 0) & (df_asec["OCC10LY"] != 0)]

#compute occ mobility
df_asec["ChangeOcc"] = df_asec["OCC2010"] != df_asec["OCC10LY"]
df_asec["ChangeOccWT"] = df_asec["ChangeOcc"] * df_asec["ASECWT"]

print("Labor force with occupation (millions)", \
df_asec["ASECWT"].sum()/(1e6))
print("People that changed occupaiton (millions) all years", \
df_asec["ChangeOccWT"].sum()/1e6)
print("Percentage of workforce that switched occupation in a year ", \
df_asec["ChangeOccWT"].sum()/df_asec["ASECWT"].sum())

# occupation codes in asec
occ_asec = set(df_asec["OCC2010"].unique())
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
dict_code_index = {}
for i in range(len(occ_codes)):
    dict_code_index[occ_codes[i]] = i

#Employment
dict_occ_emp = {}
for index, row in df_occ.iterrows():
    dict_occ_emp[row["Code"]] = 0

for index, row in df_asec.iterrows():
    dict_occ_emp[row["OCC2010"]] += row["ASECWT"]/len(years)

df_occ_clean["EMP_2011_2019_avg"] = df_occ_clean["Code"].map(dict_occ_emp)

len(dict_code_index)

# Occupation mobility

def occ_mob_asec_occ(df, dict_code_index, norm):
    """Makes occupational mobiltiy network from ASEC
    """
    # make empty matrix
    n = len(dict_code_index)
    A = np.zeros([n, n])
    # fill whenver there is a transition
    for index, row in df.iterrows():
        if row["OCC2010"] != row["OCC10LY"]:
            i, j = dict_code_index[(row["OCC10LY"])], dict_code_index[row["OCC2010"]]
            A[i, j] += row["ASECWT"]/norm # consider ASEC weights

    return A



T_asec = occ_mob_asec_occ(df_asec, dict_code_index, norm=1)

T_asec.shape

df_occ_clean["node_id"] = [i for i in range(T_asec.shape[0])]
print("Mean share of population transitioning 2011-2019 ", T_asec.sum()/df_asec["ASECWT"].sum())

print("final size ",T_asec.shape)
print("saving results")
df_occ_clean[["Code", "Label", "EMP_2011_2019_avg"]].to_csv(path_data + "occ_names_employment_asec_occ.csv", index=False)
np.savetxt(path_data + "asec_11_19_occ_alltransitions.csv", T_asec, delimiter=",")

# Getting the strongly connected component
def get_nodes_connected(A):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    largest_component = max(nx.strongly_connected_components(G), key=len)
    return list(largest_component)

sel_nodes = get_nodes_connected(T_asec)
df_occ_clean["connect_comp"] = df_occ_clean["node_id"].isin(sel_nodes)
df_occ_clean = df_occ_clean[df_occ_clean["connect_comp"]]
T_asec = T_asec[:, sel_nodes][sel_nodes, :]

assert(len(df_occ_clean) == T_asec.shape[0])

print("final size ",T_asec.shape)
print("saving results")
df_occ_clean[["Code", "Label", "EMP_2011_2019_avg"]].to_csv(path_data + "occ_names_employment_asec_occ_connected_comp.csv", index=False)
np.savetxt(path_data + "omn_asec_11_19_occ_alltransitions_connected_comp.csv", T_asec, delimiter=",")
