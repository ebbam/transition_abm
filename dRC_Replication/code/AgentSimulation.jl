using StatsBase
using LightGraphs
using Distributions

#struct is to make a type, mutable is to be able to mutate its fieldname values.
mutable struct worker <: Any
    occupation_id::Int
    employed::Bool
    longterm_unemp::Bool
    time_employed::Int
    time_unemployed::Int
    #memory::Array{Tuple{Int,Int,Int}}#track occupation it was employed in
end

mutable struct vacancy <: Any
    occupation_id::Int #id of occupation
    applicants::Array{worker,1}# Array{Int}#Array{Int} #
    time_open::Int
    #is_open::Bool
end

mutable struct occupation <: Any
    occupation_id::Int #id of occupation
    list_of_vacancies::Array{vacancy, 1}#Array{vacancy}
    list_of_workers::Array{worker, 1}
    list_of_neigh_bool::BitArray{1}
    list_of_neigh_weights::Array{Float64, 1}
end

function delete_item!(myarray::Array{worker,1}, item::worker)
    """deletes worker from occupation, function defined for efficiency"""
    #deleteat!(myarray, findin(myarray, [item]) )
    deleteat!(myarray, findall((in)([item]), myarray))
end

function delete_item!(myarray::Array{vacancy,1}, item::vacancy)
    """deletes vacancy from occupation, function defined for efficiency"""
    #deleteat!(myarray, findin(myarray, [item]))
    deleteat!(myarray, findall((in)([item]), myarray))
end

function initialise(n_occ::Int, employment::Array{Int,1},
    unemployment::Array{Int,1}, vacancies::Array{Int,1},
     A::Array{Float64,2}, G)
    """ Makes a list of occupations with initial conditions
    Args:
        n_occ: number of occupations initialised
        employment: vector with emploment of each occupation
        unemployment: vector with unemploment of each occupation
        A: adjacency matrix of network
        G: network
    Return:
        all_occupations(Array{occupation,1}): list of initialise occupations
    """
    all_occupations = occupation[]
    #pushing occupations (done once only)
    for i = 1:n_occ
        #creating the occupation
        occ = occupation(i, vacancy[], worker[],
        [A[i, j] > 0 for j =1:n_occ], [A[i, j] for j in outneighbors(G, i)] )
        #creating the workers of occupation i and add them to it's list
        #adding employed workers
        for w =1:employment[i]
            #Assume they have all at least 1 t.s. of employemnt
            push!(occ.list_of_workers, worker(i, true, false, 1, 0) )
        end
        #adding unmeployed workers
        for w =1:unemployment[i]
            #Assume they have 1 t.s. of unemployment (so that they can apply
            #for work)
            push!(occ.list_of_workers, worker(i, false, false, 0, 1) )
        end
        #add vacancies
        for v =1:vacancies[i]
            #Assume they have 1 t.s. of opening (so that they can receive
            #applications)
            push!(occ.list_of_vacancies, vacancy(occ.occupation_id, worker[], 1))
        end
        #adding occupation to list
        push!(all_occupations, occ)
    end
    return all_occupations
end


function serparations_openings(δ_u::Float64, δ_v::Float64,
    γ_u::Float64, γ_v::Float64, employed::Int, Δ_dagger_d::Int)
    """ Draws binomial random variable for the separations
    Args:
        i_occ: occupation id
        n_employed: vector of employment
        δ_u: probability of separation
        δ_v: probabiltiy of opening
    Returns:
        ω(int): number of separations in given occupation
        ν(int):  number of openings in given occupation
    """
    if employed > 0
        g_u = min(employed, γ_u * max(0, Δ_dagger_d))
        g_v = min(employed, γ_v * max(0, -Δ_dagger_d))
        # NOTE with implemented rounding
        # g_u = min(e, convert(Int,γ_u * max(0, Δ)))
        # g_v = min(e, convert(Int,γ_v * max(0, -Δ)))
        p_u = δ_u + (1 - δ_u) * g_u / employed
        p_v =  δ_v + (1 - δ_v) * g_v / employed
        ω = rand(Binomial(employed, p_u))
        ν = rand(Binomial(employed, p_v))
    else
        if Δ_dagger_d < 0
            println("WARNING zero employed")
            ω = 0
            ν = 1
        else
            ω = 0
            ν = 0
        end
    end
    return ω, ν
end


function search!(occupation_list::Array{occupation,1},
    n_vacancies::Array{Int,1}, n_longterm_unemployed::Array{Int,1},
     A, G, threshold_for_ltunemp=5)
    #remember to define types
    """For each occupation, for each worker in the occupation, choose a vacancy
    to apply. Once its chosen, update the list of applicants in that vacancy.
    Choosing occupation (weighted by # vacancies and proximity) is the same
    as choosing a vacancy from all possible (weighted by proximity) but much
    slower. One can check this with the AgentSimulation_slow code.
    Args:
        occupation_list: list of occupation over which it iterates
        n_vacancies: vacancies in each occupation, used for choosing occupation
        n_longterm_unemployed: vector with the longterm unemployed in each
        A: adjacency matrix
        G: network
        threshold_for_ltunemp(int): number of t.s. after which lt unemp
    Updates
        list_of_vacancies: adds applications
        n_longterm_unemployed(maybe): if a worker has no vacancy left to apply
    """
    for occ in occupation_list #run it for all occupations
    	# weight occupation by adjacency matrix and number of vacancies
        neighbors_weights = occ.list_of_neigh_weights .* n_vacancies[occ.list_of_neigh_bool]
        #neighbors_weights = A[w.occupation_id, :].*n_vacancies

        for w in (occ.list_of_workers)
            if w.employed == false #only unemployed workers apply

                #random choice of occupation weighted by adj matrix and number of vac
                chosen_occupation = sample(outneighbors(G, occ.occupation_id),
                Weights(neighbors_weights))

                #if there are vacancies in that occupation
                if length(occupation_list[chosen_occupation].list_of_vacancies) > 0
                    chosen_vacancy = sample(occupation_list[chosen_occupation].list_of_vacancies) #rand choice of vacancy
                #sends application to vacancy
                    push!(chosen_vacancy.applicants, w)

                else
                    #should only happen if there are no vacancies in any occupation
                    println("warning, no vacancies available to apply. Occ ", occ.occupation_id )
                    w.time_unemployed += 1
                    #recording long term unemployment on agent and occupation
                    if w.time_unemployed == threshold_for_ltunemp
                        w.longterm_unemp = true
                        n_longterm_unemployed[w.occupation_id] += 1
                    end
                end
            end
        end
    end
end


function matching!(occupation_list::Array{occupation,1},
    n_employed::Array{Int,1}, n_unemployed::Array{Int,1},
    n_vacancies::Array{Int,1}, n_longterm_unemployed::Array{Int,1},
    threshold_for_ltunemp=5)
    #edit to remove worker from one occupation's list and move to another
    # NOTE as before but doesnt immediately update n_x lists, returns their
    # updates in list to be done in hte update function.
    """
    Hires workers. For each vacancy in occupation, chose an applicant and hire.
    For all other applicants update their status (add unemployment time)
    update the number of vac, employment and unemployment.
    Args:
        occupation_list: list of occupations, it is needed to update the info of long term unemp, etc
        n_employed: entry corresponding to occupation increases by the number of filled vacancies
        n_unemployed: entry corresponding to occupation decreases by the number of filled vacancies
        n_vacancies: entry corresponding to occupation decreases by the number of filled vacancies
        thresh_ltunemp: threshold of time at which an unemployed worker is considered long term unemp
    Modifies:
        worker.employed
        occupation.list_of_workers
        occupation.list_of_vacancies  -- remove vacancy when filled
    Returns:
        None
    """
    hires = zeros(n_occ)
    exports = zeros(n_occ)
    exports_longtermunemp = zeros(n_occ)
    new_longterm = zeros(n_occ)
    for occ in occupation_list
        n_hired = 0
        keep = trues(n_vacancies[occ.occupation_id])
        for (i_v, vacancy) in enumerate(occ.list_of_vacancies)#can be optimised

            #hire an applicant if there was at least one
            if length(vacancy.applicants) > 0
                n_hired += 1
                #choose applicant uniformly at random
                hired_worker = sample(vacancy.applicants)
                if hired_worker.occupation_id != occ.occupation_id
                    #removing the hired worker from it's previous occ list
                    delete_item!(occupation_list[hired_worker.occupation_id].list_of_workers, hired_worker)
                    #add the hired worker into it's new occupation list.
                    push!(occ.list_of_workers, hired_worker)

                end
                #modifying the attributes of the worker
                if hired_worker.longterm_unemp == true
                    #n_longterm_unemployed[hired_worker.occupation_id] += -1
                    hired_worker.longterm_unemp = false
                    exports_longtermunemp[hired_worker.occupation_id] += 1

                end
                hires[occ.occupation_id] += 1
                exports[hired_worker.occupation_id] += 1
                # n_employed[occ.occupation_id] += 1
                # n_vacancies[occ.occupation_id] += -1
                # #note the unemployment is reduced on the previous worker's occupation
                # n_unemployed[hired_worker.occupation_id] += -1
                #note, occ id of worker must be set after n_unemp is decreased
                hired_worker.employed = true
                hired_worker.occupation_id = vacancy.occupation_id
                hired_worker.time_employed = 0
                hired_worker.time_unemployed = 0

                #telling the others they didn't get the job
                delete_item!(vacancy.applicants, hired_worker)
                for worker in vacancy.applicants
                    worker.time_unemployed += 1

                    #recording long term unemployment on agent and occupation
                    if worker.time_unemployed == threshold_for_ltunemp
                        worker.longterm_unemp = true
                        #n_longterm_unemployed[worker.occupation_id] += 1
                        new_longterm[worker.occupation_id] += 1

                    end
                end
                #finally delete the vacancy from the occupation's list
                keep[i_v] = false
            else
                #vacancy is not filled, do nothing
            end

        end
        occ.list_of_vacancies = occ.list_of_vacancies[keep]
    end
    return hires, exports, exports_longtermunemp, new_longterm
end



function update!(occupation::occupation, n_employed::Array{Int,1},
    n_unemployed::Array{Int,1}, n_vacancies::Array{Int,1},
    n_longterm_unemployed::Array{Int,1}, n_separate::Int, n_open::Int,
    hires, exports, exports_longtermunemp, new_longterm)
    # update!(occupation::occupation, n_employed::Array{Int,1},
    # n_unemployed::Array{Int,1}, n_vacancies::Array{Int,1},
    # n_separate::Int, n_open::Int, n_close::Int)
    """variables have been updated due to the search and matching. This function
    update variables due to random fluctuations and market adjustment.
    Args:
        occupation: occupation which gets updated
        n_employed:
        n_unemployed:
        n_vacancies:
    Updates:
        w.employed/w.unemployed: if separted
        w.time_employed: if not separated it increases by 1
        list_of_vacancies: new vacancies open
    """

    # separations
    # only employed workers (not hired in current time step) may be separted
    possible_separations = filter(w -> w.employed==true && w.time_employed > 0,
                                occupation.list_of_workers)

    @assert(length(possible_separations) >= n_separate)
    # if n_separate > length(possible_separations)
    #     # close vacancies to make up for demand reduction
    #     n_close = n_separate - length(possible_separations)
    #     # separate all workers
    #     n_separate = length(possible_separations)
    # else
    #
    #     n_close = 0
    # end

    separated_workers = sample(possible_separations, n_separate, replace=false)
    # update status of separated workers
    for w in separated_workers
        w.employed = false
        w.time_employed = 0
        # NOTE since search and matching have happened we can set their
        # unemployment time to 1
        w.time_unemployed = 1
    end
    # if worker is employed and was not separated employment time increases
    # note this increases employment time for hire in current time step
    # (as desired)
    for w in filter(w -> w.employed==true && w ∉ separated_workers,
        occupation.list_of_workers)
        w.time_employed += 1
    end

    # openings
    # n_openings = n_open - n_close
    n_openings = n_open
    for i = 1:n_openings
        # create vacancies, assume 1 t.s. of opening since search and
        # matching has alrady happened
        push!(occupation.list_of_vacancies,
            vacancy(occupation.occupation_id, worker[], 1))
    end

    n_employed[occupation.occupation_id] += - n_separate
    n_unemployed[occupation.occupation_id] += n_separate
    n_vacancies[occupation.occupation_id] += n_openings
    # updates due to search and matching
    n_employed[occupation.occupation_id] += hires[occupation.occupation_id]
    n_unemployed[occupation.occupation_id] += - exports[occupation.occupation_id]
    n_vacancies[occupation.occupation_id] += - hires[occupation.occupation_id]
    n_longterm_unemployed[occupation.occupation_id] +=
    new_longterm[occupation.occupation_id] - exports_longtermunemp[occupation.occupation_id]

    @assert(exports[occupation.occupation_id] >= exports_longtermunemp[occupation.occupation_id])
    @assert(n_vacancies[occupation.occupation_id] >= 0)
    @assert(n_employed[occupation.occupation_id] >= 0)
    @assert(n_unemployed[occupation.occupation_id] >= 0)
end

function run_time_step!(occupation_list::Array{occupation,1}, δ_u::Float64,
    δ_v::Float64, γ_u::Float64, γ_v::Float64,n_employed::Array{Int,1},
    n_unemployed::Array{Int,1}, n_vacancies::Array{Int,1},
    n_longterm_unemployed::Array{Int,1}, target_demand::Function,
    D_0::Array{Int,1}, D_f::Array{Int,1}, t::Int, t_shock::Int, k::Float64, t0::Int,
    A::Array{Float64,2}, G,threshold_for_ltunemp::Int)
    """
    Runs time step.
    1-2) Happend, first delta and then gamma
    The the applciation process takes place First applications are sent, then workers are hired.
    Then vac and workers are open/fired due to RF and MA.

    Args:
        occupation_list:
        δ_u, δ_v, γ_u, γ_v: parameters for random fluct and market adjustment
        n_employed(), n_unemployed(), n_vacancies():vector with the number of
        employed/unemployed workers/vacancies in each occupation
        n_long_term_unemployed(): vector with # longterm unemployed
        target_demand(): function that gives target demand at each time step
        according to the adoption function (e.g. sigmoid)
        D_0: vector of initial demand of occupations
        D_f: vector with final demand of occupations (after shock)
        t: time of simulation (used for target demand computation)
        t_shock: time at which shock started (used for target demand computation)
        k: sigmoid parameter
        t0: sigmoid parameter

    """
    # note that market adjustment is sequential to random fluctuations
    # likewise matching is sequential to search
    # but market adjusment and random fluct are independent of search and
    # matching and viceversa
    # println("employed at occ 1 ", n_employed[1])
    #All unemployed workers send their applications.
    search!(occupation_list, n_vacancies, n_longterm_unemployed,
            A, G, threshold_for_ltunemp)
    #for all occupations, fill the vacancies with applicants
    # note variables start being updated here
    hires, exports, exports_longtermunemp, new_longterm = matching!(
            occupation_list, n_employed, n_unemployed, n_vacancies,
            n_longterm_unemployed, threshold_for_ltunemp)

    for occ in occupation_list
        i_occ = occ.occupation_id
        # if occ.occupation_id == 1
        #     println("employed at occ 1 after search and match", n_employed[1])
        # end
        employed = n_employed[i_occ]
        Δ_dagger_d = employed + n_vacancies[i_occ] - target_demand(t,
        D_0[i_occ], D_f[i_occ], t_shock, k, t0)

        Δ_dagger_d = convert(Int, Δ_dagger_d)

        ω, ν = serparations_openings(δ_u, δ_v, γ_u, γ_v, employed, Δ_dagger_d)


        # update variables due to random fluctuations and market adjustment
        update!(occ, n_employed, n_unemployed, n_vacancies, n_longterm_unemployed,
                ω, ν, hires, exports, exports_longtermunemp,
                 new_longterm)
        # if occ.occupation_id == 1
        #      println("employed at occ 1 after update", n_employed[1])
        # end
    end

    return occupation_list, n_employed, n_unemployed, n_vacancies, n_longterm_unemployed
end



function run_simulation(n_occ::Int, total_sim_time::Int, δ_u::Float64,
        δ_v::Float64, γ_u::Float64, γ_v::Float64, n_employed::Array{Int,1},
        n_unemployed::Array{Int,1}, n_vacancies::Array{Int,1},
        target_demand::Function, D_0::Array{Int,1}, D_f::Array{Int,1},
        t_shock::Int, k::Float64, t_halfsig::Int, A::Array{Float64,2}, G,
        threshold_for_ltunemp::Int)

    """function that runs a complete simulation. It initialises
    occupation_list and then makes matrix of data for emp, unemp, vac, long-term
        unemployment and applications which in each column has the vector of them
        for a time step.
    Allows setting initial conditions for e, u, and v
    Allows for time dependent shock
    Args
        n_occ():number of occupations
        total_sim_time(): total number of time steps
        δ:
        γ:
        n_employed: inital condition of employment
        n_unemployed: inital condition of unemployment
        n_vacancies: initial condition of vacancies
        D_0: initial demand of labor (before t_of_shock)
        D_f: demand for labor after shock
        t_shock: time at which shock is introduced
        k
        t_halfsig: sigmoid parameter
        A: adjacency matrix
        G: network
        threshold_for_ltunemp: threshold at which workers are considered
        longterm unemp
    """
    occupation_list = initialise(n_occ, n_employed, n_unemployed, n_vacancies, A, G);

    #defining the arrays that will store all variables, forall t forall occ
    employed_sim = zeros(Int, total_sim_time, n_occ)
    unemployed_sim = zeros(Int, total_sim_time, n_occ)
    vacancies_sim = zeros(Int, total_sim_time, n_occ)
    longterm_unemployed_sim = zeros(Int, total_sim_time, n_occ)

    #Setting initial conditions, copy so that we don't modify the i.c.
    employed = copy(n_employed)
    unemployed = copy(n_unemployed)
    vacancies =  copy(n_vacancies)
    longterm_unemployed =  zeros(Int,n_occ)#copy(n_longterm_unemployed)

    for t = 1:total_sim_time
        #saving the variables for the t-th time step
        employed_sim[t, :] = employed
        unemployed_sim[t, :] = unemployed
        vacancies_sim[t, :] =  vacancies
        longterm_unemployed_sim[t, :] = longterm_unemployed

        #runing the step
        occupation_list, employed, unemployed, vacancies, longterm_unemployed =
        run_time_step!(occupation_list, δ_u, δ_v, γ_u, γ_v, employed,
        unemployed, vacancies, longterm_unemployed, target_demand, D_0, D_f, t,
        t_shock, k, t_halfsig, A, G, threshold_for_ltunemp)

    end
    return employed_sim, unemployed_sim, vacancies_sim, longterm_unemployed_sim
end
#
function run_serveral_simulations(n_occ::Int, n_dif_sim::Int, total_sim_time::Int,
    δ_u::Float64, δ_v::Float64, γ_u::Float64, γ_v::Float64,
    employment_0::Array{Int,1}, unemployment_0::Array{Int,1},
    vacancies_0::Array{Int,1}, target_function::Function, D_0::Array{Int,1},
    D_f::Array{Int,1}, t_shock::Int, k::Float64, t_halfsig::Int,
    A::Array{Float64,2}, G, threshold_for_ltunemp::Int)
    """Function that runs several independent simulations. It tracks employment,
    unemployment, vacancies and long term unemployment for each simulation.
    It also allows for time dependent shock.

    n_dif_sim():number of independent simulations
    t_sim(): total number of time steps
    n_occ():number of occupations

    δ_0:
    γ_0:
    employment_0: inital condition of employment
    unemployment_0: inital condition of unemployment
    vacancies_0: inital condition of vacancies
    D_0: initial demand of labor (before t_of_shock)
    D_f: demand for labor after shock
    t_shock(): time at which the shock is introduced
    k
    t_halfsig
    A: adjacency matrix
    G: network
    threshold_for_ltunemp: threshold at which workers are considered longterm
    unemp
    all_info(Bool):

    returns employment, unemployment, vacancies and lt unemployment of
    all points of all the simulations
    """

    #Initiliasing the bid arrays with all the info
    E = zeros(n_dif_sim, total_sim_time, n_occ)
    U = zeros(n_dif_sim, total_sim_time, n_occ)
    V = zeros(n_dif_sim, total_sim_time, n_occ)
    U_longterm = zeros(n_dif_sim, total_sim_time, n_occ)

    for i = 1:n_dif_sim
        println("sim ", i)

        @time employed, unemployed, vacancies, lt = run_simulation(n_occ, total_sim_time,
                δ_u, δ_v, γ_u, γ_v, deepcopy(employment_0),
                deepcopy(unemployment_0), deepcopy(vacancies_0),
                target_function, D_0, D_f, t_shock, k,
                t_halfsig, A, G, threshold_for_ltunemp);

        #saving the simulation results
        E[i, :, :] = employed
        U[i, :, :] = unemployed
        V[i, :, :] = vacancies
        U_longterm[i, :, :] = lt
    end

    return E, U, V, U_longterm
end
