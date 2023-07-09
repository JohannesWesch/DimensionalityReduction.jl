using Polyhedra
using LazySets
using JuMP, Gurobi
using Base.Threads
using CDDLib
include("Redundancy.jl")


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
        #A_new, b_new = tosimplehrep(h_lazy)
        #A, b = overapprox(A_old, b_old, A_new, b_new, d_old - i)

        A, b = remove_redundant(constraints_list(h_lazy))
        #A, b = overapprox4(A_new, b_new, d_old - i)
    end
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
    A_new, b_new = overapprox2(A_new, b_new, d_new)
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

# less than x zeros in a row
function overapprox2(A_new, b_new, new_dim)
    m, n = size(A_new)
    A = zeros((0,new_dim))
    b = zeros((0,))

    for i in 1:m
        nonzeros = 0
        vec = A_new[i,:]
        filter!(x->x≠0.0,vec)
        nonzeros= size(vec, 1)
        if nonzeros > new_dim-2
            A = vcat(A, A_new[i, :]')
            b = vcat(b, b_new[i])
        end
    end
    return A, b
end

# b has to be over x
function overapprox3(A_new, b_new, new_dim)
    m = size(A_new, 1)
    A = zeros((0,new_dim))
    b = zeros((0,))

    for i in 1:m
        nonzeros = 0
        if abs(b_new[i]) > 10000
            A = vcat(A, A_new[i, :]')
            b = vcat(b, b_new[i])
        end
    end
    return A, b
end

# only odd rows
function overapprox4(A_new, b_new, new_dim)
    m = size(A_new, 1)
    A = zeros((0,new_dim))
    b = zeros((0,))

    for i in 1:m
        if mod(i, 2) == 1
            A = vcat(A, A_new[i, :]')
            b = vcat(b, b_new[i])
        end
    end
    return A, b
end