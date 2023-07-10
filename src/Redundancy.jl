using JuMP, Gurobi
using LazySets

function remove_redundant(constraints)
    backend = Gurobi.Optimizer
    A, b = tosimplehrep(constraints)
   
    m, n = size(A)
    non_redundant_indices = 1:m

    i = 1  # counter over reduced (= non-redundant) constraints

    for j in 1:m # loop over original constraints
        α = A[j, :]
        Ar = A[non_redundant_indices, :]
        br = b[non_redundant_indices]
        @assert br[i] == b[j]
        br[i] += 1
        lp = linprog(-α, Ar, br, backend)
       
        objval = -lp.objval
        if objval <= b[j] + 1
            # the constraint is redundant
            non_redundant_indices = setdiff(non_redundant_indices, j)
        else
            # the constraint is not redundant
            i += 1
         end
        
    end
    
    deleteat!(constraints, setdiff(1:m, non_redundant_indices))
    return tosimplehrep(constraints)
end

function linprog(c, A, b, solver)
    N = length(c)
    model = Model(solver)
    @variable(model, x[i=1:N])
    @objective(model, Min, c' * x)
    @constraint(model, A * x .<= b)
    optimize!(model)
    return (
        status = termination_status(model),
        objval = objective_value(model),
        sol = value.(x)
    )
end

function swap(A, b)
    d = size(A, 1)
    A_swap = zeros(size(A))
    b_swap = zeros(size(b))
    for i in 1:d
        A_swap[i, :] = A[d+1-i, :]
        b_swap[i] = b[d+1-i]
    end
    return A_swap, b_swap
end