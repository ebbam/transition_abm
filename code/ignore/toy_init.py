
T = 1000
delta_u = 0.01
delta_v = 0.005
gamma_u = gamma_v = gamma = 0.01
# Import information about relevant files to employment/unemployment, target demand, vacancies, etc.

A = pd.read_csv(path+"data/small_adj_full.csv", delimiter=';', decimal=',', header=None)
employment = pd.read_csv(path+"data/employed.csv", header = None)
unemployment = pd.read_csv(path+"data/unemployed.csv", header = None)
vacancies = pd.read_csv(path+"data/vacancies.csv", header = None)
demand_target = employment + vacancies
wages = pd.DataFrame(np.round(np.random.normal(50000, 10000, 5)), columns = ['Wages'])
mod_data =  {"A": A, "employment": employment, 
             'unemployment':unemployment, 'vacancies':vacancies, 
             'demand_target': demand_target, 'wages': wages}

