include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/digits/digit-net_16x2.onnx", 
        "benchmarks/digits/dim16/prop_0_0.20.vnnlib",
        "benchmarks/digits_reduced", method=4, d_to_reduce=7, nnenum=true, factorization=0)