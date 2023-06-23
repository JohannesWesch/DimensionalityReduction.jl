using Revise
using VNNLib
include("DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
        "benchmarks/mnistfc/prop_3_0.03.vnnlib",
        "benchmarks/mnistfc_reduced", 4)

# "benchmarks/mnistfc_reduced/mnist-net_256x4_updated.onnx",
# "benchmarks/mnistfc_reduced/prop_0_0.03_updated.vnnlib"

#=reduce_network("benchmarks/mnistfc/mnist-net_256x6.onnx",
        "benchmarks/mnistfc_reduced/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated.onnx",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated_V.txt",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated_dim.txt")=#

#=calculate_polytope("benchmarks/mnistfc_reduced/mnist-net_256x6_updated_V.txt",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated_dim.txt",
        "benchmarks/mnistfc_reduced/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced/prop_0_0.03_updated.vnnlib", 2)=#

# f, n_input, n_output = get_ast("benchmarks/mnistfc_reduced/mnist-net_256x4/prop_0_0.03/prop_0_0.03.vnnlib")
