using Polyhedra
using LazySets
using JuMP, Gurobi
using Base.Threads

function exact(A, b, new_input_dim)
    b = vec(b)
   
    for i in 1:1 #:size(A, 2) - new_input_dim
        h_polyhedra = Polyhedra.hrep(A, b)
        p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
        reduced = Polyhedra.eliminate(p_polyhedra, FourierMotzkin())
        h_lazy = LazySets.HPolytope(reduced)
        print("redundant")
        # h_lazy = remove_redundant_constraints(h_lazy, backend = Gurobi.Optimizer)
        print("redundant finished")
        A, b = tosimplehrep(h_lazy)
    end
    print(size(A))
    print(size(b))
    return A, b
end

# new_rep = remove_redundant_constraints(final_rep, backend = Gurobi.Optimizer)

function exact_box(A, b, new_input_dim)
    
    b = vec(b)
    P = HPolytope(A, b)
    box = zeros(new_input_dim, 2)

   for i in 1:4 #new_input_dim # Threads.@threads 
        d₋ = zeros(new_input_dim)
        d₋[i] = -1
        s₋ = ρ(d₋, P, solver = Gurobi.Optimizer)

        d₊ = zeros(new_input_dim)
        d₊[i] = 1
        s₊ = ρ(d₊, P, solver = Gurobi.Optimizer)

        box[i, 1] = -s₋
        box[i, 2] = s₊
    end
    println(box)
    return box
end

function permute_A(A)
    d = size(A, 1)
    P = zeros(d, d)

    j = 226

    for i in 1:d
        if i != j && i != 784
            P[i, i] = 1
        else
            P[i, end] = 1
            P[end, i] = 1
        end
    end

    return P*A

end