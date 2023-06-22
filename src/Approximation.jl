using JuMP, Gurobi
using Base.Threads

include("Constraints.jl")

function approximate(A, b, box_constraints, V, new_input_dim, approx)
    new_constraints = new_box_constraints(V, box_constraints)
    # return new_constraints[1:new_input_dim, 1:2]
    A₁, b₁ = get_A_b_from_box_alternating(new_constraints[1:new_input_dim, 1:2])
    if approx == 1
        return A₁, b₁
    elseif approx == 2
        A₂, b₂ = approximate_other_dimensions(A, b, new_constraints, new_input_dim)
        print(b₂)
        # return vcat(A₁, A₂), vcat(b₁, b₂)
        return A₂, b₂
    elseif approx == 3
        A₃, b₃ = approximate_new_dimensions(A, b, new_constraints, new_input_dim)
        # return vcat(A₁, A₃), vcat(b₁, b₃)
        print(b₃)
        return A₃, b₃
    else
        throw(ArgumentError("parameter must be between 1 and 3"))
    end
    
end

# apply V on the old bounds to get bounds for the new ̃x
function new_box_constraints(V, bounds)
    n_variables = size(V)[1]
    new_bounds = zeros(n_variables, 2)

    V⁺ = max.(0, V)
    V⁻ = min.(0, V)

    new_bounds[:,1] = V⁺ * bounds[:,1] + V⁻ * bounds[:,2]
    new_bounds[:,2] = V⁺ * bounds[:,2] + V⁻ * bounds[:,1]

    return new_bounds
end

# use the bounds for the n-r dimensions that we want to get rid of
# and subtract that from b
function approximate_other_dimensions(A, b, bounds, new_input_dim)
    A_new = A[:, 1:new_input_dim]
    b_new = b

    b_new -= A[:, new_input_dim + 1:size(A)[2]] * bounds[new_input_dim + 1:size(A)[2], 1]
    return A_new, b_new
end

# similar to new_box_constraints, but this time we optimize the diretions of the new matrix A
function approximate_new_dimensions(A, b, new_constraints, new_input_dim)
    A_new = A[:, 1:new_input_dim]
    b_new = zeros(size(A)[1])

    A⁺ = max.(0, A)
    A⁻ = min.(0, A)

    b_new = A⁺ * new_constraints[:, 2] + A⁻ * new_constraints[:, 1]
    # b_new[:,1] = A⁺ * new_constraints[:,1] + A⁻ * new_constraints[:,2]
    # b_new[:,2] = A⁺ * new_constraints[:,2] + A⁻ * new_constraints[:,1]

    return A_new, b_new
end










# returns as output matrix with upper and lower bounds of the specified variables
function approximate_box(A, b, box_constraints, variables)
    new_bounds = zeros(length(variables), 2)
    model = Model(Gurobi.Optimizer)
    set_attribute(model, "TimeLimit", 100)
    set_attribute(model, "Presolve", 0)

    @variable(model, x[1:size(A, 2)])
    # @variable(model, box_constraints[i, 1] <= x[i = 1:size(A, 2)] <= box_constraints[i, 2])
    @constraint(model, A * x .<= b)
    
    for i in variables
        @objective(model, Min, x[i])
        optimize!(model)
        new_bounds[i, 1] =  value(x[i])
        
        @objective(model, Max, x[i])
        optimize!(model)
        new_bounds[i, 2] =  value(x[i])
    end

    return new_bounds
end


function approximate_box_parallel(A, b, variables)
    ENV["JULIA_NUM_THREADS"] = 8

    new_bounds = zeros(length(variables), 2)
    model = Model(Gurobi.Optimizer)
    set_attribute(model, "TimeLimit", 100)
    set_attribute(model, "Presolve", 0)

    @variable(model, x[1:size(A, 2)])
    @constraint(model, A * x .<= b)
    
    @threads for i in variables
        @objective(model, Min, x[i])
        optimize!(model)
        new_bounds[i, 1] =  value(x[i])
        
        @objective(model, Max, x[i])
        optimize!(model)
        new_bounds[i, 2] =  value(x[i])
    end

    return new_bounds
end

# variables all variables to eliminate


    #=n_variables = size(V)[1]
    new_box_constraints = zeros(n_variables, 2)
    for i in 1:n_variables
        for j in 1:n_variables
            if V[i, j] >= 0
                new_box_constraints[i, 1] += V[i, j] * box_constraints[j, 1]
                new_box_constraints[i, 2] += V[i, j] * box_constraints[j, 2]
            else
                new_box_constraints[i, 1] += V[i, j] * box_constraints[j, 2]
                new_box_constraints[i, 2] += V[i, j] * box_constraints[j, 1]
            end
        end
    end

    for i in eachindex(b)
        for j in (new_input_dim + 1):size(A)[2]
            b_new[i] += bounds[j, 1] * A[i, j]
        end
    end=#