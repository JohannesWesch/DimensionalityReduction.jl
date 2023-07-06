include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/digits/test.onnx", 
        "benchmarks/digits/prop_0_0.20.vnnlib",
        "benchmarks/digits_reduced", method=0, d_to_reduce=1, nnenum=true)