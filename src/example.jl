using Revise
using DelimitedFiles
include("DimensionalityReduction.jl")
import .DimensionalityReduction: reduce, reduce_network, calculate_polytope

#=reduce("benchmarks/mnistfc/mnist-net_256x4.onnx", 
        "benchmarks/mnistfc_reduced/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced/mnist-net_256x4_updated.onnx",
        "benchmarks/mnistfc_reduced/prop_0_0.03_updated.vnnlib", 2)=#

#=reduce_network("benchmarks/mnistfc/mnist-net_256x6.onnx",
        "benchmarks/mnistfc_reduced/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated.onnx",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated_V.txt",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated_dim.txt")=#

calculate_polytope("benchmarks/mnistfc_reduced/mnist-net_256x6_updated_V.txt",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated_dim.txt",
        "benchmarks/mnistfc_reduced/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced/prop_0_0.03_updated.vnnlib", 2)

