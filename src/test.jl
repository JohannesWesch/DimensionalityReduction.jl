using LazySets
using PyCall
using VNNLib
using Polyhedra
using CDDLib
using JuMP, GLPK, Gurobi
using LinearAlgebra

using LazySets.Arrays: SingleEntryVector
using BenchmarkTools

include("Constraints.jl")
include("Approximation.jl")
include("PathGenerator.jl")
include("NetworkUpdate.jl")
include("VNNLibGenerator.jl")


onnx_input = "benchmarks/mnistfc/mnist-net_256x2.onnx"
vnnlib_input = "benchmarks/mnistfc/prop_0_0.03.vnnlib"
output = "benchmarks/mnistfc_reduced"
approx = 0
vnnlib = false
nnenum = false

onnx_output = onnx_path(onnx_input, vnnlib_input, output)
vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, approx)

box_constraints, output_dim = get_box_constraints(vnnlib_input)
v, new_input_dim, l, r = update_network(onnx_input, onnx_output, box_constraints)

l, u , p = lu(v)

    





#=function reduce(onnx_input, vnnlib_input, output, approx=1)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, approx)

    box_constraints, output_dim = get_box_constraints(vnnlib_input)
    Vᵀ, new_input_dim = update_network(onnx_input, onnx_output, box_constraints)

    A, b = get_A_b_from_box_alternating(box_constraints)
    # A = A * transpose(Vᵀ)
    A = A * inv(Vᵀ)
    return A, b, Vᵀ, box_constraints, new_input_dim
end

function approximate_support_function(A, b, new_input_dim)
    j = 2 # size(A, 1)
    A_new = A[1:j, 1:new_input_dim]

    b = vec(b)
    P = HPolytope(A, b)
    b_new = zeros(j,)

    println("starting approximation")
    println(j)

    # Threads.@threads 
    # println(i, ", thread: ", Threads.threadid())
    Threads.@threads for i in 1:j
        println(i)
        d = zeros(size(A, 2))
        d[i] = -1
        # d = A[i, 1:end]
        # d[new_input_dim + 1:end] .= 0.0
        s = ρ(d, P, solver = Gurobi.Optimizer)
        println(s)
        b_new[i] = s
    end
    return A_new, b_new
end

A, b, Vᵀ, box_constraints, new_input_dim = reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
        "benchmarks/mnistfc/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced", 3)

approximate_support_function(A, b, new_input_dim)=#



#=b = vec(b)
P = HPolytope(A, b)

d = zeros(size(A, 2))
d[1] = -1
s = ρ(d, P, solver = Gurobi.Optimizer)

d = zeros(size(A, 2))
d[1] = 1
s1 = ρ(d, P, solver = Gurobi.Optimizer)

d = zeros(size(A, 2))
d[2] = -1
s2 = ρ(d, P, solver = Gurobi.Optimizer)

d = zeros(size(A, 2))
d[2] = 1
s3 = ρ(d, P, solver = Gurobi.Optimizer)

d = zeros(size(A, 2))
d[3] = -1
s4 = ρ(d, P, solver = Gurobi.Optimizer)

d = zeros(size(A, 2))
d[3] = 1
s5 = ρ(d, P, solver = Gurobi.Optimizer)

println(s)
println(s1)
println(s2)
println(s3)
println(s4)
println(s5)=#

#=d = A[1, 1:end]
d[new_input_dim+1:end] .= 0.0
s = ρ(d, P, solver = Gurobi.Optimizer)
println(s)=#


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





