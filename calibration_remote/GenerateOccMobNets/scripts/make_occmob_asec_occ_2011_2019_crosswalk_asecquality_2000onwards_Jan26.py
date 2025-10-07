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

# file with occupation names (https://cps.ipums.org/cps/codes/occ_20112019_codes.shtml)
file_occ_in_cps2011_2019 = "occ_names_code_2011-2019.csv"
# pkl file for asec, cleaned for fast processing
file_asec = "cps_asec_tech_flags.csv"
file_xwalk = "xwalk_asec_bls_2011-2019.csv"

# read occ dataset
df_occ = pd.read_csv(path_data + file_occ_in_cps2011_2019)
df_occ.head()

# read census file
df_asec = pd.read_csv(path_data + file_asec)
df_asec["YEAR"]
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

# removing imputations and taking quality controls
df_asec = df_asec[df_asec['UH_SUPREC_A2']==1]
df_asec = df_asec[df_asec["QOCC"]==0]
df_asec = df_asec[df_asec["QOCCLY"]==0]
# df_asec = df_asec[df_asec["QIND"]==0]

#compute occ mobility
df_asec["ChangeOcc"] = df_asec["OCC"] != df_asec["OCCLY"]
df_asec["ChangeOccWT"] = df_asec["ChangeOcc"] * df_asec["ASECWT"]

print("Employed workers average (millions)", \
df_asec["ASECWT"].sum()/(1e6*len(years)))
print("People that changed occupation (millions) averge all years", \
df_asec["ChangeOccWT"].sum()/(1e6*len(years)))
print("Percentage of workforce that switched occupation in a year ", \
df_asec["ChangeOccWT"].sum()/df_asec["ASECWT"].sum())

# Now choosing occupations that are in the crosswalk with BLS

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
def occ_mob_rate_asec_peryear(df, dict_code_index, year):
    """Makes occupational mobiltiy network from ASEC
    Output is in rate
    """
    df_temp = df[df["YEAR"] == year]
    # make empty matrix
    n = len(dict_code_index)
    A = np.zeros([n, n])
    emp_i = np.zeros(n)
    # fill whenver there is a transition
    for index, row in df_temp .iterrows():
        if row["OCC"] != row["OCCLY"]:
            if row["OCCLY"] in dict_code_index.keys() and row["OCC"] in dict_code_index.keys():
                i, j = dict_code_index[(row["OCCLY"])], dict_code_index[row["OCC"]]
                A[i, j] += row["ASECWT"] # consider ASEC weights
        # get employment through asec weights of i
        if row["OCCLY"] in dict_code_index.keys():
            i = dict_code_index[(row["OCCLY"])]
            emp_i[i] +=  row["ASECWT"]   
    # normalize to make mobility rate     
    for i in range(n):
        if emp_i[i] == 0:
            assert(A[i,j] == 0)
            A[i,j] = np.nan
        else:
            A[i,j] = A[i,j]/emp_i[i]

    return A, emp_i

# Occupation mobility
def occ_mob_asec_occ(df, dict_code_index, year, normalize=True):
    """Makes occupational mobiltiy network from ASEC
    Output is in number of workers (unadjusted for labor force)
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

# mobility networks of transitions (unadjusted for labor force)
T_asec_year_list = []
for y in years:
    omn_year = occ_mob_asec_occ(df_asec, dict_code_index, year=y, normalize=False)
    T_asec_year_list.append(omn_year)
    np.savetxt(path_data + "omn_asec_11_19_unadjusted" + str(y) +"_.csv", omn_year, delimiter=",")

# mobility networks rate 
F_asec_year_list = []
for y in years:
    omn_year = occ_mob_rate_asec_occ(df_asec, dict_code_index, year=y)
    F_asec_year_list.append(omn_year)
    np.savetxt(path_data + "omn_rate_asec_11_19_" + str(y) +"_.csv", omn_year, delimiter=",")


A_t, emp = occ_mob_rate_asec_occ(df_asec, dict_code_index, year=2018)

k = 20
A_t[k][np.where(A_t[k] > 0)]
emp


# how to get average once nan is includes
# https://stackoverflow.com/questions/55064174/how-to-calculate-average-of-n-numpy-arrays

# Now get

plt.imshow(T_asec_year_list[-1])
plt.show()

F_asec_year_list[2]

# From adjacency matrix to edgelist
df = pd.DataFrame(F_average, \
    columns = occ_codes, \
        index = occ_codes)



T_asec_year_list[0].shape

dict_code_index
df_codes = pd.DataFrame.from_dict(dict_code_index, orient='index')
df_codes.to_csv(path_data + "quick_codes.csv")
# Occupation mobility

def occ_mob_asec_occ(df, dict_code_index, norm):
    """Makes occupational mobiltiy network from ASEC
    """
    # make empty matrix
    n = len(dict_code_index)
    A = np.zeros([n, n])
    # fill whenver there is a transition
    for index, row in df.iterrows():
        if row["OCC"] != row["OCCLY"]:
            i, j = dict_code_index[(row["OCCLY"])], dict_code_index[row["OCC"]]
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
