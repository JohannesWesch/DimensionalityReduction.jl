using Polyhedra
using LazySets
using CDDLib

include("Constraints.jl")

function fourier(A, b, d_to_reduce)
    b = vec(b)
   
    for i in 1:d_to_reduce
        h_polyhedra = Polyhedra.hrep(A, b)
        p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
        reduced = Polyhedra.eliminate(p_polyhedra, FourierMotzkin())
        h_lazy = LazySets.HPolytope(reduced)
        A, b = tosimplehrep(h_lazy)
    end
    println(size(A))
    return A, b
end

function block(A, b, d_new, d_old)
    b = vec(b)
    h_polyhedra = Polyhedra.hrep(A, b)
    p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
    var = collect(d_new:d_old)
    reduced = Polyhedra.eliminate(p_polyhedra, var, BlockElimination())
    h_lazy = LazySets.HPolytope(reduced)
    A, b = tosimplehrep(h_lazy)

    print(size(A))
    return A, b
end