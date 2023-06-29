include("DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
        "benchmarks/mnistfc/prop_1_0.03.vnnlib",
        "benchmarks/mnistfc_reduced", 0, true, false)