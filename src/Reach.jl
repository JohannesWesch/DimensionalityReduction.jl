using ReachabilityAnalysis
using PyCall
include("Constraints.jl")
include("Approximation.jl")
@pyinclude("src/NetworkUpdate.py")

# mein Code
box_constraints = get_box_constraints("benchmarks/mnistfc_reduced/test3.vnnlib")
box_constraints = Float32.(box_constraints)


V, new_input_dim = py"update_network"("benchmarks/mnistfc/mnist-net_256x4.onnx",
"benchmarks/mnistfc_reduced/mnist-net_256x4.onnx", box_constraints)

p = @ivp(x' = V * x, x(0) ∈ Hyperrectangle(box_constraints[1:end, 1], box_constraints[1:end, 2]))

alg = BFFPSV18(; δ=1e-3, setrep=Hyperrectangle, vars=[3], partition=[1:24, [25], 26:48])

solution = solve(p, tspan=(0, 5), alg)

# alg = BFFPSV18(; δ=1e-3, vars=collect(1.:2), partition=[[1], [2, 3, 4]])