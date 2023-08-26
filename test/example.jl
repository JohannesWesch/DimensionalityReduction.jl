include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

#normal example
#=reduce("benchmarks/digits/digit-net_64x8x64x64x64x10.onnx", 
        "benchmarks/digits/dim64/prop_5_1.00.vnnlib",
        "benchmarks/digits_reduced", method=2, d_to_reduce=1, nnenum=true, factorization=3)=#

#=reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
"benchmarks/mnistfc/prop_5_0.03.vnnlib",
"benchmarks/mnistfc_reduced", method=5, d_to_reduce=10, nnenum=true)=#

#=reduce("benchmarks/digits/digit-net_64x6.onnx", 
        "benchmarks/digits/dim64/prop_9_0.01.vnnlib",
        "benchmarks/digits_reduced", method=2, d_to_reduce=1, nnenum=true, factorization=1, vnnlib=true)=#

#6
#"benchmarks/digits/dim64/prop_6_0.01.vnnlib"

#7
#"benchmarks/digits/dim64/prop_0_0.01.vnnlib"

#8
#"benchmarks/digits/dim64/prop_4_0.01.vnnlib"

reduce("benchmarks/digits/digit-net_16x2.onnx", 
"benchmarks/digits/dim16/prop_12_0.01.vnnlib",
"benchmarks/digits_reduced", method=1, d_to_reduce=2, nnenum=true, factorization=3)

#5
#"benchmarks/digits/dim16/prop_6_0.01.vnnlib"

#6
#"benchmarks/digits/dim16/prop_3_0.01.vnnlib"

#7
#"benchmarks/digits/dim16/prop_5_0.01.vnnlib"

