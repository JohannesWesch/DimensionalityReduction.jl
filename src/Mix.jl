using Polyhedra
using LazySets
using JuMP, Gurobi
using Base.Threads
using CDDLib

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

function exact_box(A, b)
    dim = size(A, 2)
    
    b = vec(b)
    P = HPolytope(A, b)
    box = zeros(dim, 2)

   for i in 1:dim # Threads.@threads 
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



function test_vrep(A, b, d_reduced)
    b = vec(b)
   
    for i in 1:d_reduced
        h_rep = LazySets.HPolytope(A, b)
        v_rep = LazySets.convert(VPolytope, h_rep)
        #reduced = Polyhedra.eliminate(p_polyhedra, FourierMotzkin())
        #h_lazy = LazySets.HPolytope(reduced)
        #A, b = tosimplehrep(h_lazy)
        #A, b = overapprox(A, b)
        print(size(A))
    end
    print(size(A))
    return A, b
end

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
        println(size(A))
    end
    println(size(A))
    return A, b
end