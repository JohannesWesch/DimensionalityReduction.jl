using LazySets
using Polyhedra
using MultivariateStats
using LinearAlgebra

function calculate_directions(A, b, d_new)
    b = vec(b)
    h_rep = LazySets.HPolytope(A, b)
    v_rep = LazySets.convert(VPolytope, h_rep)
    dim = LazySets.dim(v_rep)
    vertices = LazySets.vertices_list(v_rep)
    data = reshape(collect(Iterators.flatten(vertices)),  (dim, (Int(size(vertices, 1)))))
    model = fit(PCA, data, maxoutdim=d_new)
    pc = projection(model)
    display(pc)
    display(model)
    return pc
end

function pca_approx(A, b, d_new)
    directions = transpose(calculate_directions(A, b, d_new))
    display(directions)
    dim = size(directions, 1)
    b = vec(b)
    P = HPolytope(A, b)
    bₒ = zeros(dim,)

   Threads.@threads for i in 1:dim
        d = directions[i, :]
        s = ρ(d, P, solver = Gurobi.Optimizer)
        bₒ[i] = s
    end
    return directions[:, :], bₒ
end