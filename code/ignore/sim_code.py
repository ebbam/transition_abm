def run_sim(behav_spec, data, time_steps, runs):
    import numpy as np
    import pandas as pd
    # Records variables of interest
    record = pd.DataFrame(columns=['Sim', 'Time', 'Occupation_ID', 'Workers', 'Employment', 'Unemployment', 'Vacancies', 'LT Unemployed Persons', 'Target_Demand'])
    print(record)
    for run in range(runs):
        #print("Running ", init, " model.")
        print("RUN: ", run)
        # Initialise occupational mobility network
        net_temp, vacs = initialise(len(data['A']), data['employment'], data['unemployment'], data['vacancies'], data['demand_target'], data['A'], data['wages'])
        for t in range(time_steps):
            print("TIME: ", t)
            if t == 400 and shock:
                print("initiatied shock!")
                net_temp[0].target_demand += 25
                net_temp[1].target_demand += 50
                net_temp[2].target_demand += 50
                net_temp[3].target_demand += 50
                net_temp[4].target_demand = 100

            # Ensure number of workers in economy has not changed
            assert(sum(map(lambda x: len(x.list_of_workers), net_temp)) == employment.sum().item() + unemployment.sum().item())
            for occ in net_temp:

                ### SEPARATIONS
                occ.separate_workers()

                # Ensure that separated workers have been reassigned appropriately 
                # (ie. that people move witihin the same occupation from employed to unemployed 
                # and that the total number of workers iwthin an occupation is (at this stage) 
                # the same as before separations
                if t > 0:
                    temp = record.loc[(record['Sim'] == run) & (record['Occupation_ID'] == occ.occupation_id) & (record['Time'] == t-1)]
                    assert(temp.Employment.item() - sum(wrkr.employed for wrkr in occ.list_of_workers) ==
                           sum(not(wrkr.employed) for wrkr in occ.list_of_workers) - temp.Unemployment.item())
                    assert(len(occ.list_of_workers) == temp.Workers.item())

                ### APPLICATIONS
                # Questions to verify:
                # - CANNOT be fired and apply in same time step ie. time_unemployed > 0
                # - CAN be rejected and apply in the same time step - no protected attribute
                unemp = [el for el in occ.list_of_workers if not(el.employed) and el.time_unemployed > 0]
                for u in unemp:
                    u.search_and_apply(net_temp, vacs, behav_spec)

            ### HIRING
            # Ordering of hiring randomised to ensure list order does not matter in filling vacancies...
            # ....might be better to do this using an unordered set?
            for v_open in sorted(vacs,key=lambda _: random.random()):
                # Removes any applicants that have already been hired in another vacancy
                v_open.applicants[:] = [app for app in v_open.applicants if not(app.hired)]
                if len([app for app in v_open.applicants if not(app.hired)]) > 0:
                    v_open.hire(net_temp)
                    vacs.remove(v_open)
                    assert(len(v_open.applicants) == 0)
                else:
                    pass

            ### OPEN VACANCIES
            # Update vacancies after all shifts have taken place
            # Could consider making this a function of the class itself
            for occ in net_temp:
                # Update all workers
                occ.update_workers()
                emp = sum(wrkr.employed for wrkr in occ.list_of_workers)
                occ.current_demand = bus_cycle_demand(len([v_open for v_open in vacs if v_open.occupation_id == occ.occupation_id]) + emp, t, 0.9, 15)
                vac_prob = delta_v + ((1 - delta_v) * (gamma * max(0, occ.target_demand - occ.current_demand))) / (emp + 1)
                print("Employment", emp)
                print("Vacancy probability:", vac_prob)
                for v in range(int(np.random.binomial(emp, vac_prob))):
                    vacs.append(vac(occ.occupation_id, [], occ.wage))

                ### UPDATE INDICATOR RECORD
                # Record of indicators of interest (simulation number, occ, # workers, employed, unemployed, vacancies, long_term_unemployed)
                record.loc[len(record)]= [run, 
                                          t,
                                          occ.occupation_id,
                                          len(occ.list_of_workers),
                                          sum(wrkr.employed for wrkr in occ.list_of_workers),
                                          sum(not(wrkr.employed) for wrkr in occ.list_of_workers),
                                          len([v_open for v_open in vacs if v_open.occupation_id == occ.occupation_id]),
                                          sum(wrkr.longterm_unemp for wrkr in occ.list_of_workers),
                                          occ.target_demand]

        print("Done after ", t + 1, " time steps.")
    print("Done after ", run + 1, " time steps.")
    return(record)