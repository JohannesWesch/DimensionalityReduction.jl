using PyCall

#=include("Constraints.jl")
include("Approximation.jl")
@pyinclude("src/NetworkUpdate.py")

box_constraints = get_box_constraints("benchmarks/mnistfc_reduced/test3.vnnlib")

A, b = get_A_b("benchmarks/mnistfc_reduced/test3.vnnlib")

V = py"update_network"("benchmarks/mnistfc/mnist-net_256x4.onnx",
 "benchmarks/mnistfc_reduced/mnist-net_256x4.onnx", box_constraints)

A = A*transpose(V)

variables = collect(1:3)
new_bounds = approximate(A, b, variables)=#