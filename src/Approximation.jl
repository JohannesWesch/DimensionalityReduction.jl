using JuMP, Gurobi
using Base.Threads

include("Constraints.jl")

function approximate(A, b, box_constraints, V, new_input_dim)
    new_constraints = new_box_constraints(V, box_constraints)
    A1, b1 = approximate_other_dimensions(A, b, new_constraints, new_input_dim)
    
    # new_bounds = approximate_box(A, b, new_constraints, variables)
    A2, b2 = get_A_b_from_box_alternating(new_constraints[1:new_input_dim, 1:2])
    A = vcat(A1, A2)
    b = vcat(b1, b2)
    return A, b
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
function approximate_other_dimensions(A, b, bounds, new_input_dim)
    A_new = A[1:end, 1:new_input_dim]
    b_new = b
    
    for i in eachindex(b)
        for j in (new_input_dim + 1):size(A)[2]
            b_new[i] += bounds[j, 1] * A[i, j]
        end
    end
    return A_new, b_new
end

function new_box_constraints(V, box_constraints)
    n_variables = size(V)[1]
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
    return new_box_constraints
end