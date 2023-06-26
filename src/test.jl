using LazySets
using PyCall
using VNNLib
using Polyhedra
using CDDLib
using JuMP, GLPK, Gurobi

using LazySets.Arrays: SingleEntryVector
using BenchmarkTools

include("Constraints.jl")
include("Approximation.jl")
include("PathGenerator.jl")
include("NetworkUpdate.jl")
include("VNNLibGenerator.jl")

function reduce(onnx_input, vnnlib_input, output, approx=1)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, approx)

    box_constraints, output_dim = get_box_constraints(vnnlib_input)

    Vᵀ, new_input_dim = py"update_network"(onnx_input, onnx_output, box_constraints)
    
    A, b = get_A_b_from_box_alternating(box_constraints)
    A = A * transpose(Vᵀ)
    return A, b, Vᵀ, box_constraints, new_input_dim
end

function default_lp_solver(::Type{<:AbstractFloat})
    return JuMP.optimizer_with_attributes(() -> GLPK.Optimizer(; method=GLPK.SIMPLEX))
end

A, b, Vᵀ, box_constraints, new_input_dim = reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
        "benchmarks/mnistfc/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced", 3)

b = vec(b)
P = HPolytope(A, b)

# MathOptInterface.OptimizerWithAttributes(LazySets.var"#12#13"(), Pair{MathOptInterface.AbstractOptimizerAttribute, Any}[])
# JuMP.optimizer_with_attributes(() -> GLPK.Optimizer(; method=GLPK.SIMPLEX))

d = A[1, 1:end]
d[new_input_dim+1:end] .= 0.0
s = ρ(d, P, solver = Gurobi.Optimizer)
println(s)


#=directions = Vector{Float64}[]

for i in 1:14
    d = A[i, 1:end]
    d[new_input_dim+1:end] .= 0.0
    push!(directions, d)
end

dirs = CustomDirections(directions)

res = Approximations.overapproximate(P, dirs)

println(res)=#

#=proj_mat = [[1. zeros(1, 783)]; [0. 1. zeros(1, 782)]; [0. 0. 1. zeros(1, 781)]; [0. 0. 0. zeros(1, 780) 1.]]

print("hi")

d1 = A[1, 1:end]
d1[new_input_dim+1:end] .= 0.0
d2 = A[2, 1:end]
d2[new_input_dim+1:end] .= 0.0
d3 = A[3, 1:end]
d3[new_input_dim+1:end] .= 0.0
d4 = A[4, 1:end]
d4[new_input_dim+1:end] .= 0.0

dirs = CustomDirections([d1, d2, d3, d4]);
res = Approximations.overapproximate(P, dirs)

print(res)=#


#=for i in 1:15
    d = A[i, 1:end]
    d[new_input_dim+1:end] .= 0.0
    s = ρ(d, P)
    println(s)
end=#

#=
d1 = zeros(size(A, 2))
d1[1] = -1
d2 = zeros(size(A, 2))
d2[1] = 1
d3 = zeros(size(A, 2))
d3[2] = -1
d4 = zeros(size(A, 2))
d4[2] = 1=#


# support functions for the optimization of each variable at a time
#=d = zeros(size(A, 2))
d[1] = -1
s = ρ(d, P)
println(s)

d = zeros(size(A, 2))
d[1] = 1
s = ρ(d, P)
println(s)

d = zeros(size(A, 2))
d[2] = -1
s = ρ(d, P)
println(s)

d = zeros(size(A, 2))
d[2] = 1
s = ρ(d, P)
println(s)

d = zeros(size(A, 2))
d[3] = -1
s = ρ(d, P)
println(s)

d = zeros(size(A, 2))
d[3] = 1
s = ρ(d, P)
println(s)=#





