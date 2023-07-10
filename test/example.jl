include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/digits/digit-net_16x2.onnx", 
        "benchmarks/digits/dim16/prop_0_0.55.vnnlib",
        "benchmarks/digits_reduced", method=5, d_to_reduce=3, nnenum=true, dorefinement=true)

#=reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
"benchmarks/mnistfc/prop_5_0.03.vnnlib",
"benchmarks/mnistfc_reduced", method=5, d_to_reduce=10, nnenum=true)=#