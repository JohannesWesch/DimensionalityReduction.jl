using LazySets
using PyCall
using VNNLib
using Polyhedra
using CDDLib

using LazySets.Arrays: SingleEntryVector
using BenchmarkTools

include("Constraints.jl")
include("Approximation.jl")
include("PathGenerator.jl")
@pyinclude("src/NetworkUpdate.py")
@pyinclude("src/VNNLibGenerator.py")

function reduce(onnx_input, vnnlib_input, output, approx=1)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, approx)

    box_constraints, output_dim = get_box_constraints(vnnlib_input, vnnlib_output)

    Vᵀ, new_input_dim = py"update_network"(onnx_input, onnx_output, box_constraints)
    
    A, b = get_A_b_from_box_alternating(box_constraints)
    A = A * transpose(Vᵀ)
    return A, b, Vᵀ, box_constraints, new_input_dim
end

A, b, Vᵀ, box_constraints, new_input_dim = reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
        "benchmarks/mnistfc/prop_3_0.03.vnnlib",
        "benchmarks/mnistfc_reduced", 3)

b = vec(b)
P = HPolytope(A, b)


#=d = A[1, 1:end]
d[new_input_dim+1:end] .= 0.0

s = ρ(d, P)
println(s)
d = A[2, 1:end]
d[new_input_dim+1:end] .= 0.0
s = ρ(d, P)
println(s)
d = A[3, 1:end]
d[new_input_dim+1:end] .= 0.0
s = ρ(d, P)
println(s)=#

d = zeros(size(A, 2))
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
println(s)





