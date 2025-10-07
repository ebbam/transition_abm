import copy
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import os
import networkx as nx

# Get the current working directory
cwd = os.getcwd()
# Print the current working directory
print("Current working directory: {0}".format(cwd))

#defining paths
path_data = "../data/"
path_fig = "../results/fig/"

# file with occupation names (https://cps.ipums.org/cps/codes/occ_20112019_codes.shtml)
file_occ_in_cps2011_2019 = "occ_names_code_2011-2019.csv"
# file cps
file_cps = "CPS_2010_2019.pkl"

# read occ dataset
df_occ = pd.read_csv(path_data + file_occ_in_cps2011_2019  )

df = pd.read_pickle(path_data + file_cps)

year_start = 2011
year_end = 2019 + 1 # up to 2019
years = [i for i in range(year_start, year_end)]
print("Remove year 2014 since problems in data for ASEC")
years.remove(2014)

df = df[df["YEAR"].isin(years)]
df = df[df['CPSIDP'] !=0]

print( "Labor force Feb 2019 using WTFINL (million workers)", df["WTFINL"]\
    [(df["MONTH"] == 2) & (df["YEAR"] == 2019) & (df["LABFORCE"] == 2)].sum()/1e6)
print( "Labor force Feb 2019 using PANLWT (million workers)", df["PANLWT"]\
    [(df["MONTH"] == 2) & (df["YEAR"] == 2019) & (df["LABFORCE"] == 2)].sum()/1e6)
print( "Labor force Feb 2019 using LNKFW1MWT (million workers)", df["LNKFW1MWT"]\
    [(df["MONTH"] == 2) & (df["YEAR"] == 2019) & (df["LABFORCE"] == 2)].sum()/1e6)

person_sample_counts = df['CPSIDP'].value_counts()
sampled_twice = list(person_sample_counts[person_sample_counts >1].index)
# only consider people sampled twice
df = df[df['CPSIDP'].isin(sampled_twice)]

print( "Labor force Feb 2019 using WTFINL only twice or more sampled (million workers)", df["WTFINL"]\
    [(df["MONTH"] == 2) & (df["YEAR"] == 2019) & (df["LABFORCE"] == 2)].sum()/1e6)
print( "Labor force Feb 2019 using PANLWT  only twice or more sampled(million workers)", df["PANLWT"]\
    [(df["MONTH"] == 2) & (df["YEAR"] == 2019) & (df["LABFORCE"] == 2)].sum()/1e6)
print( "Labor force Feb 2019 using LNKFW1MWT  only twice or more sampled(million workers)", df["LNKFW1MWT"]\
    [(df["MONTH"] == 2) & (df["YEAR"] == 2019) & (df["LABFORCE"] == 2)].sum()/1e6)


# this helps speed
df = df[df["LABFORCE"] == 2]
# drop na in PANLTWT
df = df[(df["PANLWT"].notna()) & (df["OCC"] != 9999) & (df["OCC"] != 9920)&\
    (df["OCC"] != 0) ]
df = df[["YEAR", 'SERIAL', "MONTH", "CPSIDP", 'PANLWT', \
                  'OCC', 'OCC2010', 'IND', "LABFORCE"]]



df['Idx'] = df.index
# NOTE sorting is essential for the algorithm to calculate occupational mobility
df = df.sort_values(by = ['CPSIDP', 'Idx'])


def calculate_transitions_cps(df):
    """ Function that calculates occupation\industry transitions
    Args:
        df(DaraFrame): dataframe with cps _sorted_ by cpsid
    NOTE df must be sorted!
    """
    # setting id's to compare when a transition happens
    person_id_old, occ_old, ind_old = 0, 0, 0
    # counters for raw (i.e. not using person weights) transitions
    raw_transitions_occ, raw_transitions_ind, raw_transitions_both = 0, 0, 0
    # coutner for estimated transitions in the use population (using PANLWT)
    transitions_occ, transitions_ind, transitions_both = 0, 0, 0
    # counter for people remaining in same occ or industry
    remained_occ, remained_ind = 0, 0

    # iterate over rows (df is sorted by person id)
    for index, row in df.iterrows():
        person_id_new = row["CPSIDP"]
        # still same person as before
        if person_id_new == person_id_old:
            # check if person changed occupation
            if occ_old != row["OCC"]:
                # add transitions to counters
                raw_transitions_occ +=1
                transitions_occ += 1 * row["PANLWT"]
                occ_old = row["OCC"]
                # check if person also changed industry
                if ind_old != row["IND"]:
                    raw_transitions_both += 1
                    transitions_both += 1 * row["PANLWT"]
            # if did not change occ add in remain
            else:
                remained_occ += 1 * row["PANLWT"]

            # check if person changed industry
            if ind_old != row["IND"]:
                raw_transitions_ind += 1
                transitions_ind += 1 * row["PANLWT"]
            else:
                remained_ind += 1 * row["PANLWT"]
        # if it is not same person a before update id's
        else:
            person_id_old = person_id_new
            occ_old = row["OCC"]
            ind_old = row["IND"]


    return transitions_occ, transitions_ind, transitions_both, \
            raw_transitions_occ, raw_transitions_ind, raw_transitions_both, remained_occ, remained_ind


def occ_mob_cps(df, dict_code_index):
    """ Function that calculates occupation\industry transitions
    Args:
        df(DaraFrame): dataframe with cps SORTED by CPSIDP
    """
    # make empty matrix
    n = len(dict_code_index)
    A = np.zeros([n, n])
    # set id for old occupation
    occ_old = 0
    person_id_old= 0
    # iterate over rows (df is sorted by person id)
    for index, row in df.iterrows():
        person_id_new = row["CPSIDP"]
        # still same person as before
        if person_id_new == person_id_old:
            # check if person changed occupation
            if occ_old != row["OCC"]:
                i, j = dict_code_index[occ_old], dict_code_index[row["OCC"]]
                A[i, j] += row["PANLWT"]
                occ_old = row["OCC"]

        # if it is not same person a before update id's
        else:
            person_id_old = person_id_new
            occ_old = row["OCC"]

    return A

def calculate_transitions_cps_and_omn(df):
    """ Function that calculates occupation\industry transitions
    Args:
        df(DaraFrame): dataframe with cps _sorted_ by cpsid
    NOTE df must be sorted!
    """
    # setting id's to compare when a transition happens
    person_id_old, occ_old, ind_old = 0, 0, 0
    # counters for raw (i.e. not using person weights) transitions
    raw_transitions_occ, raw_transitions_ind, raw_transitions_both = 0, 0, 0
    # coutner for estimated transitions in the use population (using PANLWT)
    transitions_occ, transitions_ind, transitions_both = 0, 0, 0
    # counter for people remaining in same occ or industry
    remained_occ, remained_ind = 0, 0

    n = len(dict_code_index)
    A = np.zeros([n, n])

    # iterate over rows (df is sorted by person id)
    for index, row in df.iterrows():
        person_id_new = row["CPSIDP"]
        # still same person as before
        if person_id_new == person_id_old:
            # check if person changed occupation
            if occ_old != row["OCC"]:
                # add transitions to counters
                raw_transitions_occ +=1
                transitions_occ += 1 * row["PANLWT"]
                occ_old = row["OCC"]
                # check if person also changed industry
                if ind_old != row["IND"]:
                    raw_transitions_both += 1
                    transitions_both += 1 * row["PANLWT"]
            # if did not change occ add in remain
            else:
                remained_occ += 1 * row["PANLWT"]

            # check if person changed industry
            if ind_old != row["IND"]:
                raw_transitions_ind += 1
                transitions_ind += 1 * row["PANLWT"]
            else:
                remained_ind += 1 * row["PANLWT"]
        # if it is not same person a before update id's
        else:
            person_id_old = person_id_new
            occ_old = row["OCC"]
            ind_old = row["IND"]


    return transitions_occ, transitions_ind, transitions_both, \
            raw_transitions_occ, raw_transitions_ind, raw_transitions_both, remained_occ, remained_ind


transitions_occ, transitions_ind, transitions_both, \
    raw_transitions_occ, raw_transitions_ind, raw_transitions_both, remained_occ, remained_ind \
    = calculate_transitions_cps(df[df["YEAR"] == 2019])
print("People that changed occupaiton (millions) Year 2019", transitions_occ/(1e6))
print("Percentage of workforce that switched occupation in a year (w.r.t Feb 2019 workforce)", \
    transitions_occ/ df["PANLWT"][(df["YEAR"] == 2019) & (df["MONTH"] == 2) ].sum())

# # get CPS annual transitions per year
# total_trans = 0
# for y in years:
#     transitions_occ, transitions_ind, transitions_both, \
#         raw_transitions_occ, raw_transitions_ind, raw_transitions_both, remained_occ, remained_ind \
#         = calculate_transitions_cps(df[df["YEAR"] == y])
#     print("People that changed occupaiton (millions) Year " , y, transitions_occ/(1e6))
#     print("Percentage of workforce that switched occupation in a year (w.r.t Feb " + str(y)+ " workforce)", \
#         transitions_occ/ df["PANLWT"][(df["YEAR"] == y) & (df["MONTH"] == 2) ].sum())
#     total_trans += transitions_occ
#
# print("total transitions ", total_trans)

# transitions_occ_alltogether, transitions_ind, transitions_both, \
#     raw_transitions_occ, raw_transitions_ind, raw_transitions_both, remained_occ, remained_ind \
#     = calculate_transitions_cps(df)
transitions_occ_alltogether


# occupation codes in asec
occ_asec = set(df["OCC"].unique())
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

for index, row in df.iterrows():
    dict_occ_emp[row["OCC"]] += row["PANLWT"]/len(years)

df_occ_clean["EMP_2011_2019_avg"] = df_occ_clean["Code"].map(dict_occ_emp)

T_cps = occ_mob_cps(df , dict_code_index)
T_cps.sum()
# plt.imshow(np.sqrt(T_cps))

print("final size ",T_cps.shape)
print("saving results")
df_occ_clean[["Code", "Label", "EMP_2011_2019_avg"]].to_csv(path_data + "occ_names_employment_cps_occ.csv", index=False)
np.savetxt(path_data + "omn_cps_11_19_occ_alltransitions.csv", T_cps, delimiter=",")




# print("Percentage of workforce that switched occupation in a year", \
#     transitions_occ/ df["PANLWT"][(df["LABFORCE"] == 2)].sum())


## make cps network to match asec codes
