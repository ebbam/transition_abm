using StatsBase
using DataFrames
using CSV
using LightGraphs
using Distributions
using Random

path_local = "../"
#path where simulation results are exported/saved
path_exp_sim = "../results/simulations/"
#path where figures are save
path_exp_fig = "../results/fig/"

include("AgentSimulation.jl")
include("general_functions.jl")

#### setting number of simulations and time steps duration
n_dif_sim = 2#10
t_sim = 600 # total simulation time
t_shock = 100#100

# defining the network
matrix = "OMN"
n_occ = 464

# # parameters for OMN
δ_u = 0.01600000001#0.0163001#0.016501#0.01640001#0.0195
δ_v = 0.01200000001#0.0090001#0.100001#0.013001#δ_u
# δ_v = δ_u
γ_u = 10*δ_u
γ_v = γ_u
τ = 4


#defining the shock
# shock = "beveridgeCurve"
shock = "FO_automation"
# shock = "SMLautomation"

diminishing_factor = 0.001
# parameters for automation shock
shock_duration_years = 30
# parameters for Beveridge curve
cycle_amp = 0.065

# parameter that defines the shock
if shock[1:3] == "bev"
    shock_parameter = cycle_amp
else
    shock_parameter = shock_duration_years
end

# change value to true to save simulations in csv file
save_csv = false
# saying if we solve by simulation or numerical
solution = "sim"
# solution = "sim_eqemp"

save_name, employment_0, unemployment_0, vacancies_0, D_0, D_final,
    target_function, param1, param2, A, G, df_labs =
    initial_conditions(path_local, matrix, shock, δ_u,
    δ_v, γ_u, γ_v, τ, shock_parameter, "", t_sim, t_shock)

L = sum(employment_0) + sum(unemployment_0) #labor force is number of all workers
println("reallocation ", sum(abs.(D_0 - D_final[:, :]))/(2*L))
println("all demand ",sum(D_final[:, :]))
println("all demand original ",sum(D_0[:, :]))
##################
# making the shock
##################


println("name = ", save_name)

@time employed_sim, unemployed_sim, vacancies_sim,
longterm_unemployed_sim = run_simulation(n_occ, 2, δ_u,
        δ_v, γ_u, γ_v, employment_0, unemployment_0, vacancies_0,
        target_function, D_0, D_final, t_shock, param1,
        param2, A, G, τ);

#Profile.print()

E, U, V, U_lt = run_serveral_simulations(n_occ,1, 10, δ_u,
        δ_v, γ_u, γ_v, employment_0, unemployment_0, vacancies_0,
        target_function, D_0, D_final, t_shock,param1,
        param2, A, G, τ);

E, U, V, U_lt = run_serveral_simulations(n_occ, n_dif_sim, t_sim, δ_u,
        δ_v, γ_u, γ_v, employment_0, unemployment_0, vacancies_0,
        target_function, D_0, D_final,t_shock, param1,
        param2, A, G, τ);


#####
# ##saving results in csv
#####

for j = 1:n_dif_sim
    df_occ_u = DataFrame(lab=[i for i =1:n_occ])
    df_occ_ltu = DataFrame(lab=[i for i =1:n_occ])
    df_occ_v = DataFrame(lab=[i for i =1:n_occ])
    df_occ_e = DataFrame(lab=[i for i =1:n_occ])
    df_occ_u[:id] = df_labs[:id]
    df_occ_ltu[:id] = df_labs[:id]
    df_occ_v[:id] = df_labs[:id]
    df_occ_e[:id] = df_labs[:id]
    df_occ_u[:label] = df_labs[:label]
    df_occ_ltu[:label] = df_labs[:label]
    df_occ_v[:label] = df_labs[:label]
    df_occ_e[:label] = df_labs[:label]
    for t = 1:t_sim
            df_occ_e[Symbol("t"*string(t))] = E[j, t, :]
            df_occ_u[Symbol("t"*string(t))] = U[j, t, :]
            df_occ_v[Symbol("t"*string(t))] = V[j, t, :]
            df_occ_ltu[Symbol("t"*string(t))] = U_lt[j, t, :]
    end

    if save_csv == true
            @time CSV.write(path_exp_sim * "ltu_per_occ_sim" * string(j) * save_name, df_occ_ltu)
            @time CSV.write(path_exp_sim * "u_per_occ_sim" * string(j) * save_name, df_occ_u)
            @time CSV.write(path_exp_sim * "e_per_occ_sim" * string(j) * save_name, df_occ_e)
            @time CSV.write(path_exp_sim * "v_per_occ_sim" * string(j) * save_name, df_occ_v)
    end
end
