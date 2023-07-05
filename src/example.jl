include("DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/digits/test.onnx", 
        "benchmarks/digits/prop_0_1.10.vnnlib",
        "benchmarks/digits_reduced", method=0, d_reduced=1, nnenum=true)