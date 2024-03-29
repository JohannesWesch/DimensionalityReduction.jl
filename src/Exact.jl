using Polyhedra
using LazySets
using CDDLib

function fourier(A, b, d_to_reduce)
    b = vec(b)
   
    for i in 1:d_to_reduce
        h_polyhedra = Polyhedra.hrep(A, b)
        p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
        reduced = Polyhedra.eliminate(p_polyhedra, FourierMotzkin())
        h_lazy = LazySets.HPolytope(reduced)
        A, b = tosimplehrep(h_lazy)
    end
    return A, b
end

function fourier_redundancy_removal(A, b, d_to_reduce)
    b = vec(b)
   
    for i in 1:d_to_reduce
        h_polyhedra = Polyhedra.hrep(A, b)
        p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
        reduced = Polyhedra.eliminate(p_polyhedra, FourierMotzkin())
        h_lazy = LazySets.HPolytope(reduced)
        LazySets.remove_redundant_constraints!(h_lazy, backend=Gurobi.Optimizer)
        A, b = tosimplehrep(h_lazy)
    end
    return A, b
end

function block(A, b, d_new, d_old)
    b = vec(b)
    h_polyhedra = Polyhedra.hrep(A, b)
    p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
    var = collect(d_new+1:d_old)
    reduced = Polyhedra.eliminate(p_polyhedra, var, BlockElimination())
    h_lazy = LazySets.HPolytope(reduced)
    A, b = tosimplehrep(h_lazy)
    return A, b
end

# apply V on the old bounds to get bounds for the new ̃x
function exact_box(V, bounds)
    n_variables = size(V, 2)
    new_bounds = zeros(n_variables, 2)

    V⁺ = max.(0, V)
    V⁻ = min.(0, V)

    new_bounds[:,1] = V⁺ * bounds[:,1] + V⁻ * bounds[:,2]
    new_bounds[:,2] = V⁺ * bounds[:,2] + V⁻ * bounds[:,1]

    return new_bounds
end

function exact_box_with_optimizer(A, b)
    dim = size(A, 2)
    b = vec(b)
    P = HPolytope(A, b)
    box = zeros(dim, 2)

    Threads.@threads for i in 1:dim
        d₋ = zeros(dim)
        d₋[i] = -1
        s₋ = ρ(d₋, P, solver = Gurobi.Optimizer)

        d₊ = zeros(dim)
        d₊[i] = 1
        s₊ = ρ(d₊, P, solver = Gurobi.Optimizer)

        box[i, 1] = -s₋
        box[i, 2] = s₊
    end
    return box
end