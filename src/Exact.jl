using Polyhedra
using LazySets
using JuMP, Gurobi
using Base.Threads

function exact(A, b, new_input_dim)
    b = vec(b)
   
    for i in 1:size(A, 2) - new_input_dim
        h_polyhedra = Polyhedra.hrep(A, b)
        p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
        reduced = Polyhedra.eliminate(p_polyhedra, FourierMotzkin())
        h_lazy = LazySets.HPolytope(reduced)
        h_lazy = remove_redundant_constraints(h_lazy, backend = Gurobi.Optimizer)
        A, b = tosimplehrep(h_lazy)
    end
    return A, b
end

function exact_box(A, b, new_input_dim)
    
    b = vec(b)
    P = HPolytope(A, b)
    box = zeros(new_input_dim, 2)

   for i in 1:new_input_dim # Threads.@threads 
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

function get_permutation(dim₁, dim₂)

    P = zeros(dim₂, dim₂)

    for i in 1:dim₁
        P[i, i] = 1
    end

    for i in reverse(dim₁:dim₂)
        for j in dim₁:dim₂
            if(i+j == dim₁ + dim₂ + 1)
                P[j, i] = 1
            end
        end
    end

    return P

end