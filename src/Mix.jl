using Polyhedra
using LazySets
using JuMP, Gurobi
using Base.Threads
using CDDLib


function fourier_approx(A_old, b_old, d_to_reduce, d_old)
    d_old = size(A_old, 2)
    b_old = vec(b_old)
    A = A_old
    b = b_old
   
    for i in 1:d_to_reduce
        h_polyhedra = Polyhedra.hrep(A, b)
        p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
        reduced = Polyhedra.eliminate(p_polyhedra, FourierMotzkin())
        h_lazy = LazySets.HPolytope(reduced)
        A_new, b_new = tosimplehrep(h_lazy)
        A, b = overapprox(A_old, b_old, A_new, b_new, d_old - i)
    end
    printlnln(size(A))
    return A, b
end

function block_approx(A, b, d_new, d_old)
    b = vec(b)
    h_polyhedra = Polyhedra.hrep(A, b)
    p_polyhedra = polyhedron(h_polyhedra, CDDLib.Library())
    var = collect(d_new+1:d_old)
    reduced = Polyhedra.eliminate(p_polyhedra, var, BlockElimination())
    h_lazy = LazySets.HPolytope(reduced)
    A_new, b_new = tosimplehrep(h_lazy)
    A_new, b_new = overapprox(A, b, A_new, b_new, d_new)
    println(size(A_new))
    return A_new, b_new
end

function overapprox(A, b, A_new, b_new, new_dim)
    dim = size(A, 1)
    b = vec(b)
    P = HPolytope(A_new, b_new)
    bₒ = zeros(dim,)

   Threads.@threads for i in 1:dim
        d = A[i, 1:new_dim]
        s = ρ(d, P, solver = Gurobi.Optimizer)
        bₒ[i] = s
    end
    return A[:, 1:new_dim], bₒ
end