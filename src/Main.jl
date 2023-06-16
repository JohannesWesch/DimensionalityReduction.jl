using PyCall

include("Constraints.jl")
@pyinclude("src/NetworkUpdate.py")

box_constraints = get_box_constraints("benchmarks/mnistfc/prop_0_0.03.vnnlib")

v = py"update_network"("benchmarks/mnistfc/mnist-net_256x4.onnx",
 "benchmarks/mnistfc_reduced/mnist-net_256x4.onnx", box_constraints)

c, d = get_c_d("benchmarks/mnistfc_reduced/test.vnnlib")

c = c*transpose(v)