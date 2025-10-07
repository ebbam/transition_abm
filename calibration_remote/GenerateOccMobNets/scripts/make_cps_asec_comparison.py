import copy
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import os
import networkx as nx
import scipy.stats


# Get the current working directory
cwd = os.getcwd()
# Print the current working directory
print("Current working directory: {0}".format(cwd))

#defining paths and files
path_data = "../data/"
path_fig = "../results/fig/"
file_omn_asec = "omn_asec_11_19_occ_alltransitions.csv"
file_omn_cps = "omn_cps_11_19_occ_alltransitions.csv"
file_nodes_asec = "occ_names_employment_asec_occ.csv"
file_nodes_cps = "occ_names_employment_cps_occ.csv"

year_start = 2011
year_end = 2019 + 1 # up to 2019
years = [i for i in range(year_start, year_end)]
print("Remove year 2014 since problems in data for ASEC")
years.remove(2014)


A_asec = np.genfromtxt(path_data + file_omn_asec, delimiter = ",")
A_cps = np.genfromtxt(path_data + file_omn_cps, delimiter = ",")
df_asec = pd.read_csv(path_data + file_nodes_asec)
df_cps = pd.read_csv(path_data + file_nodes_cps)

print("nodes in asec omn ", len(df_asec), "matrix ", A_asec.shape)
print("nodes in asec omn ", len(df_cps), "matrix ", A_cps.shape)

print("ratio between asec transitions and cps transitions ", A_asec.sum()/A_cps.sum())

A_asec.sum()/(1e6 * len(years))
A_cps.sum()/(1e6 * len(years) )

# select subset of nodes that are in both networks
nodes_asec = set(df_asec["Code"])
nodes_cps = set(df_cps["Code"])
nodes_common = nodes_asec.intersection(nodes_cps)
nodes_cps.difference(nodes_asec)

df_asec["in_both_cps_asec"] = df_asec["Code"].isin(nodes_common)
df_cps["in_both_cps_asec"] = df_cps["Code"].isin(nodes_common)

selected_asec = df_asec[df_asec["in_both_cps_asec"]].index.tolist()
selected_cps = df_cps[df_cps["in_both_cps_asec"]].index.tolist()

A_asec = A_asec[selected_asec, :][:, selected_asec]
A_cps = A_cps[selected_cps, :][:, selected_cps]

scipy.stats.pearsonr(A_asec.flatten(), A_cps.flatten())

# make scatter plot comparing
plt.scatter(A_asec.flatten(), A_cps.flatten(), s=emp_size)
plt.plot([10 + i*1e7 for i in range(4)], [10 + i*1e7 for i in range(4)], color="g")
plt.xscale("log")
plt.yscale("log")
plt.xlim([10, 1e7])
plt.ylim([10, 1e7])
plt.xlabel("ASEC weights")
plt.ylabel("CPS weights")
# plt.savefig(path_fig + "omn_weights_cps_asec.png")
plt.show()
