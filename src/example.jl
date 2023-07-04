include("DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/digits/test.onnx", 
        "benchmarks/digits/prop_0_0.80.vnnlib",
        "benchmarks/digits_reduced", 0, false, true)