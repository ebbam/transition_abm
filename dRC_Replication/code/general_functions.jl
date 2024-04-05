##### Functions that are used both for the numerical solution and for the
# agent simulation
using DataFrames
using CSV
using DelimitedFiles
using LinearAlgebra
using LightGraphs
using Distributions
using SimpleWeightedGraphs
using Random

############
# functions for labor restructure
############

function labor_restructure(D_0::Array{Float64,2}, automation_fraction, demand_scale)
    """Given an initial distribution of labor demand and an automation fraction
    of each occupation, returns the employment distribution after shock
    assumes employment stays constant
    Args:
        D_0: original demand
        automation_frection: fraction that each occupation gets automated
        demand_scale: how much demand increases or decreases. If 1 remains constant
    """
    # getting number of occupations and labor force
    n_occ = length(D_0)
    L = sum(D_0)
    #number of hour worked. Has no effect
    x0 = 8;
    #total number of hours per occupation
    h0 = x0 * D_0
    #number of working hours in each occupation after automation [t]
    hf = h0 .* (ones(n_occ, 1) - automation_fraction);
    #new working time [t/e]
    xf = sum(hf) / L;
    #new employment demand of workers [e]
    new_demand = hf / xf;
    return demand_scale*new_demand
end

function labor_restructure(D_0::Array{Int,1}, automation_fraction, demand_scale)
    """Given an initial distribution of labor demand and an automation fraction
    of each occupation, returns the employment distribution after shock
    assumes employment stays constant
    Args:
        D_0: original demand
        automation_frection: fraction that each occupation gets automated
        demand_scale: how much demand increases or decreases. If 1 remains constant
    """
    # getting number of occupations and labor force
    n_occ = length(D_0)
    L = sum(D_0)
    #number of hour worked. Has no effect
    x0 = 8;
    #total number of hours per occupation
    h0 = x0 * D_0
    #number of working hours in each occupation after automation [t]
    hf = h0 .* (ones(n_occ) - automation_fraction);
    #new working time [t/e]
    xf = sum(hf) / L;
    #new employment demand of workers [e]
    new_demand = hf / xf;
    df = [i for i in demand_scale*new_demand]
    return round.(Int, df)
end

function calibrate_sigmoid(shock_duration::Int, automation_level::Float64)
    """
    Args
        shock_duration: time when automation level is reached
        tolerance: float between 0 << t < 1, level of automation that is
        reached
    return: sigmoid growth rate (k) and half life
    """
    sigmoid_half_life = shock_duration/2
    k = - log(1/automation_level - 1) / sigmoid_half_life
    return sigmoid_half_life, k
end



function target_as_sigmoid_shock(t::Int,d_0::Int, d_final::Int, t_shock::Int,
    k::Float64, t_halfsig::Int)
    """function that target demand of time t of sigmoid shock with parameters
    Args:
        d_0: initial demand of occupation, minimum of sigmoid
        d_final: final demand of occupation, maximum of sigmoid
        k:growth rate
        t_halfsig: half growth time
    Returns
        d_dagger(int): demand of occupation at time t
    """
    # set half life considering the time at which the shock starts
    if t < t_shock
        return d_0
    else
        t0 = t_halfsig + t_shock
        d_dagger = round(Int,d_0 + (d_final - d_0) * 1/(1+ℯ^(-k*(t-t0))))
    end
end

function target_as_sigmoid_shock(t::Int, d_0::Array{Float64,2},
    d_final::Array{Float64,2}, t_shock::Int, k::Float64, t_halfsig::Int)
    """ Vectorized version
    function that target demand of time t of sigmoid shock with parameters
    Args:
        d_0: vector of initial demand of occupation, minimum of sigmoid
        d_final: vector of final demand of occupation, maximum of sigmoid
        k:growth rate
        t_halfsig: half growth time
    Returns
        d_dagger(Array{Float64,2}): demand of occupation at time t
    """
    # before shock, return original demand
    if t < t_shock
        return d_0
    else
        # set half life considering the time at which the shock starts
        t0 = t_halfsig + t_shock
        # note if different adoption rate could introduce .* easily.
        d_dagger = d_0 + (d_final - d_0) * 1/(1+ℯ^(-k*(t-t0)))
        return d_dagger
    end
end

function target_as_business_cycle(t::Int, d_0::Int, d_final::Int, t_shock::Int,
    amplitude::Float64, frequency::Float64)
    """function that target demand of time t of sigmoid shock with parameters
    Args:
        d_0: initial demand of occupation, minimum of sigmoid
        d_final: final demand of occupation, maximum of sigmoid
        k:growth rate
        t_halfsig: half growth time
    Returns
        d_dagger(int): demand of occupation at time t
    """
    # set half life considering the time at which the shock starts
    if t < t_shock
        return d_0
    else
        #t0 = t_halfsig + t_shock
        # start the cycle when shock starts
        t0 = t + t_shock
        d_dagger =  d_0 * (1 - amplitude * sin((2*pi / frequency) * t0))

    end
    return round(Int, d_dagger)
end

function target_as_business_cycle(t::Int, d_0::Array{Float64,2},
    d_final::Array{Float64,2}, t_shock::Int, amplitude::Float64,
    frequency::Number)
    """ Vectorized version
    function that target demand of time t of sigmoid shock with parameters
    Args:
        d_0: vector of initial demand of occupation, minimum of sigmoid
        d_final: vector of final demand of occupation, maximum of sigmoid
        k:growth rate
        t_halfsig: half growth time
    Returns
        d_dagger(Array{Float64,2}): demand of occupation at time t
    """
    # before shock, return original demand
    if t < t_shock
        return d_0
    else
        # set half life considering the time at which the shock starts
        t0 = t + t_shock
        d_dagger =  d_0 * (1 - amplitude * sin((2*pi / frequency) * t0))
        # note if different adoption rate could introduce .* easily.
        return d_dagger
    end
end

function target_constant(t::Int, d_0::Int, d_final::Int, t_shock::Int,
    amplitude::Float64, frequency::Float64)
    """function that has target constant
    """
    # set half life considering the time at which the shock starts
    return d_final
end

function target_constant(t::Int, d_0::Array{Float64,2},
    d_final::Array{Float64,2}, t_shock::Int, amplitude::Float64,
    frequency::Number)
    """ Vectorized version
    function that has target constant
    """
    # before shock, return original demand
    return d_final
end

function target_as_linear_shock(t::Int,d_0::Int, d_final::Int, t_shock::Int,
    k::Float64, amplitude::Float64, frequency::Float64)
    """function that returns target demand (for give occupation with d_0
    and d_final) due to a linear shock (relu type)
    """
    linear_increment = (d_0 - d_final) / t_shock_duration
    if t <= t_0
        return d_0
    elseif t <= t_0 + t_shock_duration
        return d_0 + linear_increment * (t - t_0)
    else
        return d_final
    end
end


############
# functions for calibratin of things that depend on tau
############

function shock_duration_timesteps(τ, years)
  """ Function that takes takes the number of years shock should have
  and give the shock duration in time steps
  """
  timestep_duration = 27/τ # 27 weeks of unemployment = lt unemployment
  shock_duration = round(years*52 /timestep_duration) #XX years * 52 weeks
  return convert(Int,shock_duration)
end

function calibrate_selfloop(τ, occ_mobility::Float64=0.19,
    unemployment_rate::Float64=0.06)
    """
    occupational mobility default value is 19%
    unemployment rate is average unemployment rate since 2000
    τ units are in time steps, six percent
    """
    # old way
    # Δ_t units are in weeks
    # Δ_t = 27 / τ
    # # 52 comes from 52 weeks in a year
    # self_loop = 1 - occ_mobility * Δ_t / (52*unemployment_rate)
    # println("r = ", self_loop)
    # return self_loop
    Δ_t = 52/6.75 # time step duration in a year
    x = 1 - occ_mobility
    r = (x^(1/Δ_t) + unemployment_rate - 1) / unemployment_rate
    println("r = ", r)
    return r

end

function bussiness_cycle_freq(τ, start_year::Int=2008, finish_year::Int=2018)
    """
    calibrates how many time steps should bussiness cycle last
    """
    delta_years =finish_year - start_year
    # how much of a business cycle do the years 2008-2018 account for
    # assume it unemployment started at mid point and is now at minimum
    cycle_fraction = 3/4
    # time step in weeks
    Δ_t = 27 / τ
    # one year equals
    one_year_ts = 52 / Δ_t #time steps
    delta_years_in_ts = delta_years * one_year_ts
    # since there has not been a full cycle divide by cycle frac
    # delta_years_in_ts * cycle_fraction = cycle_duration
    cycle_duration = delta_years_in_ts / cycle_fraction
    return round.(Int,cycle_duration)
end

##########
# functions for adjacency matrix
##########


function add_loops(self_p::Float64, A)
 """Function that adds self loop.
 Args:
  self_p(Int): add self loop to all occupation of prob self_loop
 """
 #assert that the matrix is normalised
 for val in sum(A, dims=2)
     if abs(val - 1.0) > 0.000000001
        println("WARNING")
     end
 end
 A_dim = size(A)[1]
 new_A = (1 - self_p) * A #reduce probability to move to other occ
 new_A = new_A + self_p * Matrix(1.0I, A_dim, A_dim) #add self_loop
 return new_A
end

function add_loops(self_p::Array{Float64,2}, A)
 """Function that adds self loop.
 Args
  A(array): transition probability matrix
  self_loop(array): add self loop to all occupation of prob self_loop[i]
 """
    #assert that the matrix is normalised
    for val in sum(A, 2)
        if abs(val - 1.0) > 0.000000001
           println("WARNING")
        end
    end
    new_A = copy(A)
    A_dim = size(A)[1]
    for i = 1:A_dim
        for j = 1:A_dim
            if i != j
              new_A[i, j] = A[i, j]*(1 - self_p[i])
            else
              new_A[i, j] = self_p[i]
            end
        end
    end
    return new_A
end

function add_loops(self_p::Array{Float64,1}, A)
 """Function that adds self loop.
 Args
  A(array): transition probability matrix
  self_loop(array): add self loop to all occupation of prob self_loop[i]
 """
    #assert that the matrix is normalised
    for val in sum(A, 2)
        if abs(val - 1.0) > 0.000000001
           println("WARNING")
        end
    end
    new_A = copy(A)
    A_dim = size(A)[1]
    for i = 1:A_dim
        for j = 1:A_dim
            if i != j
              new_A[i, j] = A[i, j]*(1 - self_p[i])
            else
              new_A[i, j] = self_p[i]
            end
        end
    end
    return new_A
end

################
# Function for initialization
################


function initial_conditions(path_local::String, matrix::String, shock::String,
    δ_u::Float64, δ_v::Float64, γ_u::Float64, γ_v::Float64, τ::Int,
    shock_parameter, matching::String, t_sim::Int, t_shock::Int)
    # (path_local, matrix, t_sim,
    # δ_u, δ_v, γ_u, γ_v, shock, t_shock, shock_params, matching_function, τ)
    """
    path_local = "../" or "../../../Dropbox/TheOccupationSpace/"
    matching_function:: String
    """
    demand_supply_ratio = 1.0 # assume target demand equals labor force
    # paths needed
    path_scripts ="scripts042019/"
    #path where ipums adjacency matrix is stored and its name
    path_adj = "data/"
    ipums_adj_file = "transition_probability_matrix_weights.csv"

    # parameter specifications in name
    parameters_names = "_deltau" * string(δ_u)[4:6] * "v" *string(δ_v)[4:6] *
    "gamma" * string(γ_u)[3:5] * "_tau" * string(round(Int,τ))

    # name of files
    # solution specifications in name. Also sets functions for numerical
    if solution[1:3] == "num"
        # defining matching function
        if matching == "balls_bins_approx_0"
            matching_function = pj_balls_bins_approx_0
        elseif matching == "balls_bins_0"
            matching_function = pj_balls_bins_0
        elseif matching == "balls_bins_approx_2"
            matching_function = pj_balls_bins_approx_2
        else
            print("WARNING, no matching function defined")
        end
        # defining fire and hire function
        fire_and_hire_function = fire_and_hire_workers
        solution_name = "_matching" * matching * "_dimfact" *
                        string(diminishing_factor)[3:end]
    elseif solution[1:3] == "sim"
        solution_name = "_dimfact" *string(diminishing_factor)[3:end]
    end

    if shock == "NoShock"
        shock_function_name = ""
    elseif shock[4:13] == "automation"
        # level of automation reached in shock years
        automation_level = 0.9999
        shock_duration = shock_duration_timesteps(τ, shock_parameter)
        # getting the sigmoid parameters
        sigmoid_half_life, sigmoid_growth_rate =
        calibrate_sigmoid(shock_duration, automation_level)
        shock_function_name = "_yearstoauto" * string(shock_parameter)[1:2]

        # setting change in demand scale if there is such
        if length(shock) == 17
            if shock[15:end] == "095"
                demand_scale = 0.95
            elseif shock[15:end] == "105"
                demand_scale = 1.05
            end
        else
            demand_scale = 1.0
        end

    elseif shock[1:3] == "bev"
        target_function = target_as_business_cycle
        cycle_amp = shock_parameter
        cycle_freq = bussiness_cycle_freq(τ)
        shock_function_name = "_beveridge" * "cycleamp"* string(cycle_amp)[3:end]
    end

    save_name = matrix * shock * parameters_names * shock_function_name * solution_name * ".csv"

    # defining the network
    # set self loop according to τ
    r = calibrate_selfloop(τ)
    if matrix == "OMN"
        # transition_adj = open(readcsv, path_local* path_adj * ipums_adj_file )
        transition_adj = readdlm(path_local* path_adj * ipums_adj_file, ',' ,Float64, '\n')
        A = add_loops(r, transition_adj)
        G = SimpleWeightedDiGraph(A)
    elseif  matrix == "kn"
        A = ones(n_occ, n_occ) / n_occ
        G = SimpleWeightedDiGraph(A)
    elseif matrix == "JS"
        # js_adj = open(readcsv, path_local* path_adj * jobspace_adj_file )
        js_adj = readdlm(path_local* path_adj * jobspace_adj_file, ',' ,Float64, '\n')
        js_adj = js_adj ./ sum(js_adj, dims=2)
        A = add_loops(r, js_adj)
        G = SimpleWeightedDiGraph(A)
    elseif matrix == "ER"
        G_erdos = erdos_renyi(n_occ, mean_degree/n_occ)
        A_erdos = adjacency_matrix(G_erdos) + Matrix(1.0I, n_occ, n_occ);
        G_erdos = DiGraph(A_erdos);
    end

    ########
    # Initial conditions
    ########
    #path where the information of ipums data is stored and the file names
    path_labels = "data/"
    # The file below seems to be corrumpted
    # ipums_lab_file = "gephi_ipums_occ_space_scarcity_tukey_thresh_all_node_attributes_including_communities_blsprojections.csv"
    # this one is better
    ipums_lab_file = "ipums_variables.csv"
    ipums_mSML = "ipums_labs_mSML_manual.csv"
    ipums_employment2016 = "ipums_employment_2016.csv"


    #data frames with info of occupations

    df_labs = CSV.read(path_local*path_labels*ipums_lab_file, DataFrame);
    df_labs_mSML = CSV.read(path_local*path_labels*ipums_mSML, DataFrame);
    df_labs_emp = CSV.read(path_local*path_labels*ipums_employment2016, DataFrame);

    employment_real = df_labs_emp[:IPUMS_CPS_av_monthly_employment_whole_period];
    if solution[1:3] == "num"
        if solution == "num"
            employment_real = [i for i in employment_real]
            employment_0 = reshape(employment_real, (n_occ, 1));
            # The numerical solution also depends on number of agents due to variance
            employment_0 = diminishing_factor * employment_0
            unemployment_0 = δ_u * employment_0 #unemployed vector cannot be of zeros
            vacancies_0 = δ_v * employment_0 #vacancy vector cannot be of zeros
            # the aggregate demand equals labor force multiplied by the demand sup ratio
            D_0 = demand_supply_ratio * (employment_0 + unemployment_0)
        elseif solution == "num_eqemp"
            emp = sum(employment_real)*diminishing_factor/n_occ
            employment_0  = [emp for i =1:n_occ]
            employment_0 = reshape(employment_0 , (n_occ, 1));
        end
        unemployment_0 = δ_u * employment_0 #unemployed vector cannot be of zeros
        vacancies_0 = δ_v * employment_0 #vacancy vector cannot be of zeros
        # the aggregate demand equals labor force multiplied by the demand sup ratio
        D_0 = demand_supply_ratio * (employment_0 + unemployment_0)


    elseif solution[1:3] == "sim"
        if solution == "sim"
            #To make the simulation less heavy we reduce the number of agents
            employment_real_reduced = diminishing_factor*employment_real
            # must make sure there is at least one employee in each occupation
            employment_real_reduced = round.(Int, employment_real_reduced ) .+ 1
            employment_0 = [employment_real_reduced[i] for i=1:n_occ];
            # before I used vector of zeros
        elseif solution == "sim_eqemp"
            emp = round.(Int,sum(employment_real)*diminishing_factor/n_occ)
            employment_0  = [emp for i =1:n_occ]
        end
        # NOTE number of agents depends on δ_u
        unemployment_0 = round.(Int, δ_u * employment_0) #zeros(Int, n_occ);
        vacancies_0 = round.(Int, δ_v * employment_0)  #zeros(Int, n_occ);
        # the aggregate demand equals labor force multiplied by the demand sup ratio
        D_0 = demand_supply_ratio * (employment_0 + unemployment_0)
        D_0 = round.(Int, D_0)
    end

    if shock[1:3] == "SML"
        #p = df_labs_other[:mSML]
        p = df_labs_mSML[:mSML]
        p = [p[i] for i=1:n_occ];
        if solution == "num"
            p = reshape(p, (n_occ, 1));
        end
        p = p / 5;
        D_final = labor_restructure(D_0, p, demand_scale);
        target_function = target_as_sigmoid_shock
    elseif shock == "FO_automation_reshuffled_rand"
        p = df_labs[:auto_prob_average];
        p = [p[i] for i=1:n_occ];
        if solution == "num"
            p = reshape(p, (n_occ, 1));
        end
        p = shuffle(p)
        D_final = labor_restructure(D_0, p, demand_scale);
        target_function = target_as_sigmoid_shock
    elseif shock == "FO_automation_reshuffled_assortative"
        # similar occupations have similar shock
        p = df_labs[:auto_prob_average];
        p = [p[i] for i=1:n_occ];
        if solution == "num"
            p = reshape(p, (n_occ, 1));
        end
        p = sort(p, dims=1)

        # getting the demand
        D_final = labor_restructure(D_0, p, demand_scale);
        target_function = target_as_sigmoid_shock
    elseif shock[1:3] == "FO_"
        p = df_labs[:auto_prob_average];
        p = [p[i] for i=1:n_occ];
        if solution == "num"
            p = reshape(p, (n_occ, 1));
        end
        D_final = labor_restructure(D_0, p, demand_scale);
        target_function = target_as_sigmoid_shock
    elseif shock[1:3] == "bev"
        # only need to set the demand supply_ratio
        D_final = D_0
        target_function = target_as_business_cycle
    elseif shock == "NoShock"
        D_final = D_0
        target_function = target_constant
    end

    # if doing simulations make sure D_final is Int array
    if solution[1:3] == "sim"
        D_final = round.(Int, D_final)
        # make sure an occupation doesn't disappear
        D_final = max.(1, D_final)
    end

    if solution[1:3] == "sim"
        if shock[4:13] == "automation" || shock == "NoShock"
            return save_name, employment_0, unemployment_0, vacancies_0, D_0, D_final,
                target_function, sigmoid_growth_rate, round(Int,sigmoid_half_life), A, G, df_labs
        elseif shock[1:3] == "bev"
            return save_name, employment_0, unemployment_0, vacancies_0, D_0, D_final,
                target_function, cycle_amp, cycle_freq, A, G, df_labs
        end
    elseif solution[1:3] == "num"
        if shock == "NoShock"
            # don't need to return shock parameters thus 0
            return save_name, employment_0, unemployment_0, vacancies_0, D_0, D_final,
            target_function, 0.0, 0, A, fire_and_hire_function, matching_function, df_labs
        elseif  shock[4:13] == "automation"
            return save_name, employment_0, unemployment_0, vacancies_0, D_0, D_final,
            target_function, sigmoid_growth_rate, round(Int,sigmoid_half_life), A, fire_and_hire_function, matching_function, df_labs
        elseif shock[1:3] == "bev"
            return save_name, employment_0, unemployment_0, vacancies_0, D_0, D_final,
            target_function, cycle_amp, cycle_freq, A, fire_and_hire_function, matching_function, df_labs
        end
    end

end


#############
# Functions used for calibration
#############
# stochastic random search
function randomSearchAlgo(f, max_iter, bounds; params=[false])
    """searches randomly for best solution
    Args
        f(function): cost function that takes parameters as input only
        bound(array): array/ list, with lenght = # of parameters.
            each element then has length 2 with the upper and lower bounds of
            parameter
        params(list): plot the search.
    """

    # define whether to plot the results in 2D
    make_plot = params[1]
    # define the dimension of variables through initial point
    len = length(bounds)
    # if dimension is less than 2 warning for plotting
    if len < 1 & make_plot
        print("warning will not plot")
    end

    #define initial random particle
    p_best = Float64[rand(Uniform(bounds[j][1],bounds[j][2])) for j=1:len]
    # define initial evaluated function
    best_result = f(p_best)
    # define vector of all obj valiues
    all_results = zeros(max_iter)
    # define vector of all trials
    all_trials = zeros(len, max_iter)

    # -- main algorithm RS (start) -- #
    # must define dummy variables p_new
    p_new = 0
    max_rep = 0
    for i = 1:max_iter

        # define new random particle
        p_new = Float64[rand(Uniform(bounds[j][1],bounds[j][2])) for j=1:len]
        # obj value for trial point
        test_result = f(p_new)
        # test wether new point is better than best known
        if test_result < best_result
            p_best = p_new
            best_result = test_result
            max_rep = 0
        end
        # compile list of all results for plotting
        all_trials[:,i] = p_new
        all_results[i] = test_result
        # iteration cap on same point
    end
    # -- main algorithm RS (end) -- #

    # plotting
    if make_plot
        scatter(all_trials[1, :], all_trials[2, :], marker=".", c=all_results , cmap="seismic")
        #plot(all_trials[1, :], all_trials[2, :], "o")#, p_new[2])
        show()
    end
    return p_best, best_result, all_trials, all_results
end


function localSearchAlgo_reduce(f, p_init, max_iter, bounds, radius, reduce_iter, reduce_frac; params=[false])
    """Local search for optimun with reduction of radius after reduce_iter
    iterations have failed to improve
    Args:
        f(function): cost functions takes parameters to fit and outputs cost
        p_init: initial point from which to start searching
        max_iter(int): maximum number of iterations
        radius(float): radius from which a random point is drawn to search
        raduce_iter()
        reduce_frac(float): fraction of radius that is reduced.
    """
    # define whether to plot the results in 2D
    make_plot = params[1]
    # define the dimension of variables through initial point
    len = length(p_init)
    # if dimension is less than 2 warning for plotting
    if len < 1 & make_plot
        print("warning will not plot")
    end

    for dim = 1:len
        if bounds[dim][1] < p_init[dim] < bounds[dim][2]
            dummy=1
        else
            println("initial condition outside bounds")
            break
        end
    end

    # define initial evaluated function
    best_result = f(p_init)
    # define vector of all obj valiues
    all_results = zeros(max_iter)
    # define vector of all trials
    all_trials = zeros(len, max_iter)

    # -- main algorithm LS_reduce (start) -- #
    # must define dummy variables p_new
    p_new = 0
    max_rep = 0
    for i = 1:max_iter


        # define search radius
        r_rand = radius * [rand(Uniform(-1,1)) for i=1:len]
        # define new trial point
        p_new = p_init + r_rand
        # obj value for trial point
        test_result = f(p_new)
        # test wether new point is better than best known
        if test_result < best_result
            switch = 1
            for dim = 1:len
                if bounds[dim][1] < p_new[dim] < bounds[dim][2]
                    dummy=1
                else
                    switch = 0
                    break
                end
            end
            if switch == 1
              p_init = p_new
              best_result = test_result
              max_rep = 0
            end
        end
        max_rep += 1
        # compile list of all results for plotting
        all_trials[:,i] = p_new
        all_results[i] = test_result
        # iteration cap on same point
        if max_rep >= reduce_iter
            radius = radius * reduce_frac
            if max_rep >= 200
                println("max iter reached at "*string(i))
                break
          	end
        end

    end
    # -- main algorithm LS_reduce (end) -- #

    # plotting
    if make_plot
        scatter(all_trials[1,2:end], all_trials[2,2:end], marker=".", c=all_results[2:end], cmap="seismic")
        #plot(all_trials[1, :], all_trials[2, :], "o")#, p_new[2])
        show()
    end
    return p_init, best_result, radius, all_trials, all_results
end


function LS_RL(f, max_iter, bounds, radius, reduce_iter, reduce_frac, RS_LS_split; params=[false])
    """
    RL = Remus Lupin
    This combines RS to find initial point, then uses LS where the search radius
    diminishes given a number of iteration withot improvement
    """

    # call RS algorithm to get good initial point
    point_RS, result_RS, trialsRS, resultsRS = randomSearchAlgo(f, convert(Int, round(max_iter * RS_LS_split)), bounds; params=[false]);
    # from good starting point continue with local search
    point_LS, result_LS, radLS, trialsLS, resultsLS = localSearchAlgo_reduce(f,
        point_RS, convert(Int, round(max_iter * (1-RS_LS_split))), bounds, radius, reduce_iter, reduce_frac; params=[false]);

    #concatenating results to plot. Note one is hcat other vcat.
    all_trials = hcat(trialsRS, trialsLS)
    all_results = vcat(resultsRS, resultsLS)

    # define whether to plot the results in 2D
    make_plot = params[1]
    # define the dimension of variables through initial point
    len = length(bounds)
    # if dimension is less than 2 warning for plotting
    if len < 1 & make_plot
        print("warning will not plot")
    end

    # plotting
    if make_plot
        scatter(all_trials[1, :], all_trials[2, :], marker=".", c=all_results , cmap="seismic")
        show()
    end
    return point_LS, result_LS, radLS
end
