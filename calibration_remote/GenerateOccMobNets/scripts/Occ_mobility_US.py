import pandas as pd
path_data = "../data/"
file_asec = "cps_asec.csv"

# Read ipums data
df_asec = pd.read_csv(path_data + file_asec)
# filter by those with ASECFLAG
df_asec = df_asec[df_asec["ASECFLAG"] == 1]
# only consider people 18 or above
df_asec = df_asec[df_asec["AGE"] >=18]
# only consider those employed
employed_codes = [10, 12]
df_asec = df_asec[df_asec["EMPSTAT"].isin(employed_codes)]


# focusing on year 2018
df_asec_2018 = df_asec[df_asec["YEAR"] == 2018]
# Make a column documenting changes in occupations
df_asec_2018["ChangeOcc"] = df_asec_2018["OCC1950"] != df_asec_2018["OCC50LY"]
# Multiply each change in occupation by asec weight
df_asec_2018["ChangeOccWT"] = df_asec_2018["ChangeOcc"] * df_asec_2018["ASECWT"]
# calculate percentage of worker that switched
df_asec_2018["ChangeOccWT"].sum()/df_asec_2018["ASECWT"].sum()


# Questions
# Filtering only those age > 18


# How to recognize imputed values?
# How drop individuals working in government industries or private households
# From Online Appendix B. Figure B.1 bottom left panel suggest that for 2018 the occupational 
# mobility rate is just below 0.12. 
# #I am currently using 1950 occupation codes and filter by age >=18 and employed. With this
# I get mobility of 0.1260