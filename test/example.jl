include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

#normal example
#=reduce("benchmarks/digits/digit-net_16x4.onnx", 
        "benchmarks/digits/dim16/prop_0_0.20.vnnlib",
        "benchmarks/digits_reduced", method=4, d_to_reduce=2, nnenum=true)=#

#=reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
"benchmarks/mnistfc/prop_5_0.03.vnnlib",
"benchmarks/mnistfc_reduced", method=5, d_to_reduce=10, nnenum=true)=#

#assertion error

#=reduce("benchmarks/digits/digit-net_64x4.onnx", 
        "benchmarks/digits/dim64/prop_0_0.60.vnnlib",
        "benchmarks/digits_reduced", method=4, d_to_reduce=1, nnenum=true)=#

reduce("benchmarks/digits/digit-net_16x4.onnx", 
"benchmarks/digits/dim16/prop_0_0.20.vnnlib",
"benchmarks/digits_reduced", method=1, d_to_reduce=5, nnenum=true, factorization=1)