using JuMP, Gurobi

function eliminate(C, d, variables)

end

function approximate(A, b, variables)
    new_bounds = zeros(length(variables), 2)
    model = Model(Gurobi.Optimizer)
    set_attribute(model, "TimeLimit", 100)
    set_attribute(model, "Presolve", 0)

    @variable(model, x[1:size(A, 2)])
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