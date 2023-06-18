using Revise
include("DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/mnistfc/mnist-net_256x4.onnx", 
        "benchmarks/mnistfc_reduced/test3.vnnlib",
        "benchmarks/mnistfc_reduced/mnist-net_256x4.onnx",
        "benchmarks/mnistfc_reduced/test3_updated.vnnlib", collect(1:20))