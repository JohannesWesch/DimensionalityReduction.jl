using Revise
include("DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/mnistfc/mnist-net_256x4.onnx", 
        "benchmarks/mnistfc_reduced/prop_0_0.05.vnnlib",
        "benchmarks/mnistfc_reduced/mnist-net_256x4.onnx",
        "benchmarks/mnistfc_reduced/prop_0_0.05_updated.vnnlib")