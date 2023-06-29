using Polyhedra
using LazySets

function exact(A, b, new_input_dim)
    b = vec(b)
    for i in 1:size(A, 2) - new_input_dim
        h_polyhedra = Polyhedra.hrep(A, b)
        p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
        reduced = Polyhedra.eliminate(p_polyhedra, FourierMotzkin())
        h_lazy = LazySets.HPolytope(reduced)
        A, b = tosimplehrep(h_lazy)
    end
    return A, b
end

# new_rep = remove_redundant_constraints(final_rep, backend = Gurobi.Optimizer)