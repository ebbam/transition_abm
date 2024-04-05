# Code and data for article "Occupational mobility and Automation: A data-driven network model"
## authors: R. Maria del Rio-Chanona, Penny Mealy, Mariano Beguerisse-Díaz, François Lafond, and J. Doyne Farmer


### Data
The adjacency matrix of the occupational mobility network is in the data folder. Please cite (Mealy et al. 2018) when using the occupational mobility network, where the construction methodology is explained. <br>
* "occupational_mobility_network.csv" <br>
The characteristics of occupations (e.g. employment, median wage, automation probability, etc.) are in the csv files <br>
 * "ipums_variables.csv" <br>
 * "ipums_lab_mSML_manual.csv"<br>
 * "ipums_employment_2016.csv"<br>
Randomizations or retrained version of the occupational mobility network are also in the data folder.

### Code
The code uses Python 3 (Version 3.8.3) and Julia (Version 1.0.4)<br>
The main functions of the code are in files labornet.py (for solving approximation) and AgentSimulation.jl (for running the agent-based model).<br>
Running the model (approximations). Most of the results presented in the paper originate from the approximations, which one can run using the file "RunNumericalShock.py" for automation shocks, "RunNumericalBeveridgeCurve.py" for the Beveridge curve, and "Calibrate_parameters_withBCurve_exhaustive" for calibration. The results from these files are saved in csv format in folder "results/csv".<br>
Running the agent-based model (simulations). To run the agent based model use file "RunAgentSimulation.jl"<br>

Plots. Once the model has run, the following files reproduce the main plots of the main text. <br>
* "Plotfiles_BeveridgeCurve.py"<br>
* "Plotfiles_aggregate_u_ltu_vs_sim.py"<br>
* "Plotfiles_long-term_unemployment.py"<br>
* "Plotfiles_network_effects_wage.py"<br>

### Results
Folder in which results are saved. The csv file contains the results of the numerical approximations, which we use for most plots. The simulations folder has the results of different runs on the agent-based model.


### References
del Rio-Chanona RM, MealyP, Beguerisse-Díaz M, Lafond F, Farmer JD. 2021 Occupational mobility and automation:a data-driven network model.  *J. R. Soc.Interface* 17: 20200898. https://doi.org/10.1098/rsif.2020.0898 <br>

Mealy P, del Rio-Chanona RM, Farmer JD. 2018 What you do at work matters: new lenses onlabour. *SSRN 3143064*. See SSRN:https://ssrn.com/abstract=3143064. (doi:10.2139/ssrn.3143064)


