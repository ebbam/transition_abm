### Function and condition to initialise network

def initialise(n_occ, employment, unemployment, vacancies, demand_target, A, wages):
    """ Makes a list of occupations with initial conditions
       Args:
           n_occ: number of occupations initialised
           employment: vector with employment of each occupation
           unemployment: vector with unemployment of each occupation
           vacancies: vector with vacancies of each occupation
           demand_target: vector with (initial) target_demand for each occupation (never updated)
           A: adjacency matrix of network (not including auto-transition probability)
           wages: vector of wages of each occupation

       Returns:
            occupations: list of occupations with above attributes
            vacancies: list of vacancies with occupation id, wage, and list of applicants
       """
    occs = []
    vac_list = []
    ids = 0
    for i in range(0, n_occ):
        # appending relevant number of vacancies to economy-wide vacancy list
        for v in range(round(vacancies.iat[i,0])):
            vac_list.append(vac(i, [], wages.iat[i,0]))
            
        occ = occupation(i, [], A[i] > 0, A[i],
                         (employment.iat[i,0] + vacancies.iat[i,0]), 
                         demand_target.iat[i,0], [], wages.iat[i,0])
        # creating the workers of occupation i and attaching to occupation
        ## adding employed workers
        for e in range(round(employment.iat[i,0])):
            # Assume they have all at least 1 t.s. of employment
            occ.list_of_workers.append(worker(occ.occupation_id, True, False, 1, 0, wages.iat[i,0], False, [occ.occupation_id], random.randint(0, 9)))
            ## adding unemployed workers
            # Could consider adding random initial unemployment durations...for now no one becomes longterm unemployed until 6 time steps in
        for u in range(round(unemployment.iat[i,0])):
            # Assigns time unemployed from absolute value of normal distribution....
            occ.list_of_workers.append(worker(occ.occupation_id, False, False, 0, abs(int(np.random.normal(0, 2))), wages.iat[i,0], False,
                                                     [occ.occupation_id], 
                                                      random.randint(0, 9)))
        occs.append(occ)
        ids += 1
    return occs, vac_list
    
