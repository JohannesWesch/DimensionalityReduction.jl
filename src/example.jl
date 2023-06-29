include("DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

reduce("benchmarks/mnistfc/mnist-net_256x6.onnx", 
        "benchmarks/mnistfc/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced", 0, true, true)