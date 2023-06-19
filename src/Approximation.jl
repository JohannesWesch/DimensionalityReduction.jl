using JuMP, Gurobi
using Base.Threads

function approximate(A, b, box_constraints, variables)
    approximate_box(A, b, box_constraints, variables)
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
function approximate_other_dimensions(A, b, variables)
    bounds = zeros(size(A)[2], 2)
    A_new = A[1:end, 1:(size(A)[2] - length(variables))]
    b_new = b

    model = Model(Gurobi.Optimizer)
    set_attribute(model, "TimeLimit", 100)
    set_attribute(model, "Presolve", 0)

    @variable(model, x[1:size(A, 2)])
    @constraint(model, A * x .<= b)
    
    for i in eachindex(variables)
        @objective(model, Min, x[i])
        optimize!(model)
        bounds[variables[i], 1] =  value(x[i])
        
        @objective(model, Max, x[i])
        optimize!(model)
        bounds[variables[i], 2] =  value(x[i])
    end

    for i in eachindex(b)
        for j in eachindex(variables)
            if A[i, variables[j]] <= 0
                b_new[i] += bounds[variables[j], 1]
            else
                b_new[i] += bounds[variables[j], 2]
            end
        end
    end

    print(size(A))
    print(size(A_new))

    return A_new, b_new
end
