using JuMP, Gurobi
using DelimitedFiles

include("Constraints.jl")

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
        @objective(model, Max, x[i])
        optimize!(model)
        new_bounds[i, 2] =  value(x[i])

        @objective(model, Min, x[i])
        optimize!(model)
        new_bounds[i, 1] =  value(x[i])
    end

    return new_bounds
end

#C, d = get_c_d("benchmarks/mnistfc_reduced/test.vnnlib")
#A = readdlm("test/c.txt")
#b = vec(readdlm("test/d.txt"))
#v = readdlm("test/v.txt")
#A = A*transpose(v)
#print(size(A))
#print(size(b))
#print(size(v))

#variables = collect(1:256)
#new_bounds = approximate(A, b, variables)