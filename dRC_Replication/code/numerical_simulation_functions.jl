function balls_bins_approx(W, v)
    """Balls and bins matching functions approximated by exponential"""
    return v .* (1 - exp.(-W ./ v) )#v .* (1 - exp.(-W' ./ v) )
end

function balls_bins(W, v)
    """balls and bins matching functions"""
    return v .* (1 .- ( 1 .- 1 ./ v) .^ W' )
end

function pj_balls_bins_approx_0(u, Q, v)
    sj = sum(u .* Q, dims=1)
    return (v .* (1 .- exp.(-sj' ./ v) ))' ./ sj
end

function pj_balls_bins_approx_2(u, Q, v)
    sj = sum(u .* Q, dims=1)
    # To check
    σ2j = sum(u .* Q .* (1 .- Q), dims=1)
    h = (v .* (1 .- exp.(-sj' ./ v) ))' ./ sj
    # To check, each term
    h2 = 0.5 * σ2j' .* ( (- exp.(-sj' ./ v)) ./ (sj' .*v) - 2*exp.(-sj' ./ v) ./ (sj.^2)'
    + 2*v.* ((1 .- exp.(-sj' ./ v) )) ./ (sj.^3)' )
    return h + h2'
end

function pj_balls_bins_0(u, Q, v)
    sj = sum(u .* Q, dims=1)
    return (v .* (1 .- ( 1 .- 1 ./ v) .^ sj' ))' ./ sj
end


function fire_and_hire_workers( δ_u::Float64, δ_v::Float64,
    γ_u::Float64, γ_v::Float64, employment::Array{Float64,2},
    u::Array{Float64,2}, v::Array{Float64,2}, d_dagger_t::Array{Float64,2},
    A::Array{Float64,2}, matching_probability::Function)
    """function that takes the employment, w, v vectors and updates them
    after one time step of the model.
    Note that, when the demand shock is too large additional vacancies may
    be closed.
    """
    n = length(employment)
    Δ_dag_d = employment + v + - d_dagger_t
    # Stages 1 and 2
    g_u = min.(employment, γ_u * max.(zeros(n), Δ_dag_d))
    g_v = min.(employment, γ_v * max.(zeros(n), -Δ_dag_d))
    separated_workers = (1 - δ_u) * g_u + δ_u .* (employment)
    opened_vacancies = (1 - δ_v) * g_v + δ_v .* (employment)

    # Search and Matching
    #application probability matrix
    Av = v'.*A
    Q = Av ./ sum(Av, dims=2)
    # expected number of job applications from i to j
    sij = u .* Q
    # expected probability of application being succesfull
    pj = matching_probability(u, Q, v)
    # expected flow of workers
    F = sij .* pj

    hired_workers = sum(F, dims=1)'
    exported_workers = sum(F, dims=2)

    @assert(minimum(hired_workers) >= 0.0)
    @assert(minimum(exported_workers) >= 0.0)

    #Update e, w and v
    u += separated_workers - exported_workers
    employment += hired_workers - separated_workers
    v += opened_vacancies - hired_workers

    # make sure variables are postive (accounting for floating point error)
    @assert(minimum(v) >= 0.0 || isapprox(v, zeros(n); atol=1e-15, rtol=0))
    @assert(minimum(u) >= 0.0 || isapprox(u, zeros(n); atol=1e-15, rtol=0))
    @assert(minimum(employment) >= 0.0 || isapprox(employment,
            zeros(n); atol=1e-15, rtol=0))
    # now that we know variables are non negative, correct floating point error
    v = max.(1e-15 * ones(n), v)
    u = max.(1e-15 * ones(n), u)
    employment = max.(1e-15 * ones(n), employment)

    return u, v,  employment, separated_workers, exported_workers
end


function run_simulation_job_spell(fire_and_hire_workers::Function, t_sim::Int,
    δ_u::Float64, δ_v::Float64, γ_u::Float64, γ_v::Float64,
    employment_0::Array{Float64,2}, unemployment_0::Array{Float64,2},
    vacancies_0::Array{Float64,2}, target_demand_function::Function,
    D_0::Array{Float64,2}, D_f::Array{Float64,2}, t_shock::Int,
    k::Float64, t_halfsig::Int, matching::Function, A_matrix::Array{Float64,2}, τ::Int)

    """Iterates the firing and hiring process for n_steady steps. Then runs the simulation with the
    restructured employment for n_sim time steps.
    returns unemployment rate, w, v, e and W lists.
    Args:
    fire_and_hire_workers(function): function that updates w, v, e and W.
    A_matrix(array): adjacency matrix of the network
    n_sin(int): number of times the simulation is run (after steady state)
    delta_u, delta_v, gamma_u, gamma_v(Float64): parameters of network
    employment_0(array(n_occ, 1)): number of employed workers at initial time.
    """
    #Defining lists that record unemployed workers, vacancies and employment vectors.
    n_occ = length(employment_0)
    #U_all = [zeros(n_steady + n_sim, length(employment_0)) for i = 1:n_sim + n_steady]

    unemployment, vacancies = copy(unemployment_0), copy(vacancies_0)
    employment = copy(employment_0)
    # initial_eq_employment = employment_0 + w_0
    E = zeros(t_sim, n_occ)
    U = zeros(t_sim, n_occ)
    V = zeros(t_sim, n_occ)

    # recording initial conditions
    U[1, :] = deepcopy(unemployment_0)
    V[1, :] = deepcopy(vacancies_0)
    E[1, :] = deepcopy(employment_0)
    U_all = zeros(t_sim, t_sim, n_occ)

    #a first iteration
    for t = 1:t_sim
        # compute target demand for given time step
        d_dagger_t = target_demand_function(t, D_0, D_f, t_shock,
                    k, t_halfsig)
        # update main variables and get the number of separations
        unemployment, vacancies, employment, separated, exported =
                    fire_and_hire_workers(δ_u, δ_v,γ_u, γ_v, employment,
                    unemployment, vacancies, d_dagger_t, A_matrix, matching)
        # the number of separations = unemployed workers with 1 t.s. of unemp
        U_all[t, 1, :] = separated
        # fill in expected number of unemployed workers with given job spell
        # note that max job spell is time of simulation so far
        for n = 2:t
            # job spell is those of previous job spell - the ones hired
            U_all[t, n, :] = U_all[t - 1, n - 1, :] -
                            U_all[t - 1, n - 1, :] .* exported ./(U[t - 1, :])
        end

        U[t, :] = unemployment
        V[t, :] = vacancies
        E[t, :] = employment

        # if printson
        #     println("t = ", t)
        #     println("unemp rate = ", sum(unemployment)/(sum(employment) + sum(unemployment)))
        #     println("w, v = ", sum(unemployment), " ", sum(vacancies))
        # end

    end
    return U, V, E, U_all
end


function run_simulation_job_spell_demand(fire_and_hire_workers::Function, t_sim::Int,
    δ_u::Float64, δ_v::Float64, γ_u::Float64, γ_v::Float64,
    employment_0::Array{Float64,2}, unemployment_0::Array{Float64,2},
    vacancies_0::Array{Float64,2}, target_demand_function::Function,
    D_0::Array{Float64,2}, D_f::Array{Float64,2}, t_shock::Int,
    k::Float64, t_halfsig::Int, matching::Function, A_matrix::Array{Float64,2}, τ::Int)

    """ Same as before but returns de targe demand as an array
    Iterates the firing and hiring process for n_steady steps. Then runs the simulation with the
    restructured employment for n_sim time steps.
    returns unemployment rate, w, v, e and W lists.
    Args:
    fire_and_hire_workers(function): function that updates w, v, e and W.
    A_matrix(array): adjacency matrix of the network
    n_sin(int): number of times the simulation is run (after steady state)
    delta_u, delta_v, gamma_u, gamma_v(Float64): parameters of network
    employment_0(array(n_occ, 1)): number of employed workers at initial time.
    """
    #Defining lists that record unemployed workers, vacancies and employment vectors.
    n_occ = length(employment_0)
    #U_all = [zeros(n_steady + n_sim, length(employment_0)) for i = 1:n_sim + n_steady]

    unemployment, vacancies = copy(unemployment_0), copy(vacancies_0)
    employment = copy(employment_0)
    # initial_eq_employment = employment_0 + w_0
    E = zeros(t_sim, n_occ)
    U = zeros(t_sim, n_occ)
    V = zeros(t_sim, n_occ)
    D_dagger = zeros(t_sim, n_occ)

    # recording initial conditions
    U[1, :] = deepcopy(unemployment_0)
    V[1, :] = deepcopy(vacancies_0)
    E[1, :] = deepcopy(employment_0)
    U_all = zeros(t_sim, t_sim, n_occ)

    #a first iteration
    for t = 1:t_sim
        # compute target demand for given time step
        d_dagger_t = target_demand_function(t, D_0, D_f, t_shock,
                    k, t_halfsig)
        # update main variables and get the number of separations
        unemployment, vacancies, employment, separated, exported =
                    fire_and_hire_workers(δ_u, δ_v,γ_u, γ_v, employment,
                    unemployment, vacancies, d_dagger_t, A_matrix, matching)
        # the number of separations = unemployed workers with 1 t.s. of unemp
        U_all[t, 1, :] = separated
        # fill in expected number of unemployed workers with given job spell
        # note that max job spell is time of simulation so far
        for n = 2:t
            # job spell is those of previous job spell - the ones hired
            U_all[t, n, :] = U_all[t - 1, n - 1, :] -
                            U_all[t - 1, n - 1, :] .* exported ./(U[t - 1, :])
        end

        U[t, :] = unemployment
        V[t, :] = vacancies
        E[t, :] = employment
        D_dagger[t, :] = d_dagger_t
        # if printson
        #     println("t = ", t)
        #     println("unemp rate = ", sum(unemployment)/(sum(employment) + sum(unemployment)))
        #     println("w, v = ", sum(unemployment), " ", sum(vacancies))
        # end

    end
    return U, V, E, U_all, D_dagger
end


function fast_run_simulation_job_spell_demand(fire_and_hire_workers::Function, t_sim::Int,
    δ_u::Float64, δ_v::Float64, γ_u::Float64, γ_v::Float64,
    employment_0::Array{Float64,2}, unemployment_0::Array{Float64,2},
    vacancies_0::Array{Float64,2}, target_demand_function::Function,
    D_0::Array{Float64,2}, D_f::Array{Float64,2}, t_shock::Int,
    k::Float64, t_halfsig::Int, matching::Function, A_matrix::Array{Float64,2}, τ::Int)

    """ Same as before but returns de targe demand as an array
    Iterates the firing and hiring process for n_steady steps. Then runs the simulation with the
    restructured employment for n_sim time steps.
    returns unemployment rate, w, v, e and W lists.
    Args:
    fire_and_hire_workers(function): function that updates w, v, e and W.
    A_matrix(array): adjacency matrix of the network
    n_sin(int): number of times the simulation is run (after steady state)
    delta_u, delta_v, gamma_u, gamma_v(Float64): parameters of network
    employment_0(array(n_occ, 1)): number of employed workers at initial time.
    """
    #Defining lists that record unemployed workers, vacancies and employment vectors.
    n_occ = length(employment_0)
    #U_all = [zeros(n_steady + n_sim, length(employment_0)) for i = 1:n_sim + n_steady]

    unemployment, vacancies = copy(unemployment_0), copy(vacancies_0)
    employment = copy(employment_0)
    # initial_eq_employment = employment_0 + w_0
    E = zeros(t_sim, n_occ)
    U = zeros(t_sim, n_occ)
    V = zeros(t_sim, n_occ)
    D_dagger = zeros(t_sim, n_occ)

    # recording initial conditions
    U[1, :] = deepcopy(unemployment_0)
    V[1, :] = deepcopy(vacancies_0)
    E[1, :] = deepcopy(employment_0)


    #a first iteration
    for t = 1:t_sim
        # compute target demand for given time step
        d_dagger_t = target_demand_function(t, D_0, D_f, t_shock,
                    k, t_halfsig)
        # update main variables and get the number of separations
        unemployment, vacancies, employment, separated, exported =
                    fire_and_hire_workers(δ_u, δ_v,γ_u, γ_v, employment,
                    unemployment, vacancies, d_dagger_t, A_matrix, matching)


        U[t, :] = unemployment
        V[t, :] = vacancies
        E[t, :] = employment
        D_dagger[t, :] = d_dagger_t
        # if printson
        #     println("t = ", t)
        #     println("unemp rate = ", sum(unemployment)/(sum(employment) + sum(unemployment)))
        #     println("w, v = ", sum(unemployment), " ", sum(vacancies))
        # end

    end
    return U, V, E, 0, D_dagger
end


####
# function to itnerpret simulations that have run
####

function u_jobspell_to_longterm(u_jobspell, n_sim, n_occ, τ)
    """ For Numerical
    function that given the array of unemployed workers and job spells per
    time step returns the number of longterm unemployed per occupation
    per time step
    Args
    n_sim(Int): number of time steps of the whole simulation
    n_occ(Int): number of occupations
    τ(Float): threhold after which worker becomes longterm unemployed
    """
    u_longterm = zeros(n_sim, n_occ)
    for i = 1:n_sim
        # we sum over all unemployment spells after τ (and including)
        u_longterm[i, :] = sum(u_jobspell[i, τ:end, :], dims=1)
        #note correct way to do this is with dims = 1
        # since we have explicitely placed the first index
    end
    return u_longterm
end

function array_to_list(array)
    """ function needed to save arrays of nx2 into data frames. Needed for
    numerical
    """
    n = length(array)
    return [array[i] for i = 1:n]
end
