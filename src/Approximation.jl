using JuMP, Gurobi
using Base.Threads
using LinearAlgebra

# use the bounds for the n-r dimensions that we want to get rid of
# and subtract that from b
function approximate_other_dimensions(A, b, bounds, new_input_dim)
    A_new = A[:, 1:new_input_dim]
    b_new = b

    A⁺ = max.(0, A)
    A⁻ = min.(0, A)

    b_new -= A⁺[:, new_input_dim + 1:end] * bounds[new_input_dim + 1:end, 1]
    b_new -= A⁻[:, new_input_dim + 1:end] * bounds[new_input_dim + 1:end, 2]
    return A_new, b_new
end

# similar to new_box_constraints, but this time we optimize the diretions of the new matrix A
function approximate_new_dimensions(A, b, new_constraints, new_input_dim)
    A_new = A[:, 1:new_input_dim]
    b_new = zeros(size(A, 1))

    A⁺ = max.(0, A)
    A⁻ = min.(0, A)

    new_constraints[new_input_dim + 1:end, 1:2] .= 0.0
    b_new = A⁺ * new_constraints[:, 2] + A⁻ * new_constraints[:, 1]

    return A_new, b_new
end

function approximate_support_function(A, b, new_input_dim)
    j = size(A, 1)
    A_new = A[:, 1:new_input_dim]

    b = vec(b)
    P = HPolytope(A, b)
    b_new = zeros(j,)

    Threads.@threads for i in 1:j
        d = A[i, 1:end]
        d[new_input_dim + 1:end] .= 0.0
        s = ρ(d, P, solver = Gurobi.Optimizer)
        b_new[i] = s
    end
    return A_new, b_new
end

function approximate(A, new_input_dim, V, bounds)
    j = size(A, 1)
    A_new = A[:, 1:new_input_dim]
    b_new = zeros(j,)

    for i in 1:j #Threads.@threads 
        d = A[i, 1:end]
        d[new_input_dim + 1:end] .= 0.0
        s = approximate_direction(V, bounds, d)
        b_new[i] = s
    end
    return A_new, b_new
end

function approximate_direction(V, bounds, direction)
    V = direction' * V
    V[abs.(V) .< 0.001] .= 0.0 #rounding errors ocurred
    V⁺ = max.(0, V)
    V⁻ = min.(0, V)

    s = V⁺ * bounds[:,2] +  V⁻ * bounds[:,1]
    return s[1]
end

function approximate_unitvector(A, new_input_dim, V, bounds)
    j = size(A, 2)
    A_new = zeros(2*new_input_dim, new_input_dim)
    b_new = zeros(2*j,)

    for i in 1:new_input_dim #Threads.@threads 
        d₁ = zeros(j,)
        d₂ = zeros(j,)
        d₁[i] = 1.0
        d₂[i] = -1.0
        
        s₁ = approximate_direction(V, bounds, d₁)
        s₂ = approximate_direction(V, bounds, d₂)
        A_new[2*i - 1, :] = d₁[1:new_input_dim]
        A_new[2*i, :] = d₂[1:new_input_dim]
        b_new[2*i - 1] = s₁
        b_new[2*i] = s₂
    end
    return A_new, b_new
end