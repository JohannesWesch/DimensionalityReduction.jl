include("DimensionalityReduction.jl")
using Revise
import .DimensionalityReduction: reduce

reduce("benchmarks/mnist/mnist-net_256x4.onnx", 
        "benchmarks/mnist_reduced/test3.vnnlib",
        "benchmarks/mnist_reduced/mnist-net_256x4.onnx",
        "benchmarks/mnist_reduced/test3_updated.vnnlib", 2)