using ReachabilityAnalysis
using PyCall
using Plots
include("Constraints.jl")
include("Approximation.jl")
@pyinclude("src/NetworkUpdate.py")

# mein Code
box_constraints = get_box_constraints("benchmarks/mnistfc_reduced/test3.vnnlib")

V, new_input_dim = py"update_network"("benchmarks/mnistfc/mnist-net_256x4.onnx",
"benchmarks/mnistfc_reduced/mnist-net_256x4.onnx", box_constraints)

V[new_input_dim + 1:end,1:end] .= 0.0;
V = Float64.(V)

println(V[222, 784])
println(V[223, 784])
println(V[224, 784])
println(V[784, 784])

p = @ivp(x' = V * x, x(0) ∈ Hyperrectangle(box_constraints[1:end, 1], box_constraints[1:end, 2]))

alg = BFFPSV18(; δ=1e-3, setrep=Hyperrectangle, vars=collect(1:223), dim = 223)

solution = solve(p, tspan=(0, 5), alg)

# plot(solution, vars=collect(1:223))

# alg = BFFPSV18(; δ=1e-3, setrep=Hyperrectangle, vars=[3], partition=[[1], [2, 3, 4]])