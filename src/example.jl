using Revise
using VNNLib
include("DimensionalityReduction.jl")
include("NNEnum.jl")

import .NNEnum: run_nnenum
import .DimensionalityReduction: reduce

reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
        "benchmarks/mnistfc/prop_4_0.03.vnnlib",
        "benchmarks/mnistfc_reduced", 1, true)

# "benchmarks/mnistfc_reduced/mnist-net_256x4_updated.onnx",
# "benchmarks/mnistfc_reduced/prop_0_0.03_updated.vnnlib"

#=reduce_network("benchmarks/mnistfc/mnist-net_256x6.onnx",
        "benchmarks/mnistfc_reduced/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated.onnx",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated_V.txt",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated_dim.txt")=#

#=calculate_polytope("benchmarks/mnistfc_reduced/mnist-net_256x6_updated_V.txt",
        "benchmarks/mnistfc_reduced/mnist-net_256x6_updated_dim.txt",
        "benchmarks/mnistfc_reduced/prop_0_0.03.vnnlib",
        "benchmarks/mnistfc_reduced/prop_0_0.03_updated.vnnlib", 2)=#

# f, n_input, n_output = get_ast("benchmarks/mnistfc_reduced/mnist-net_256x4/prop_0_0.03/prop_0_0.03.vnnlib")

#=using LazySets
using PyCall
using VNNLib
using Polyhedra
using CDDLib

using LazySets.Arrays: SingleEntryVector
using BenchmarkTools

include("Constraints.jl")
include("Approximation.jl")
include("PathGenerator.jl")
include("Utils.jl")
@pyinclude("src/NetworkUpdate.py")
@pyinclude("src/VNNLibGenerator.py")

function reduce(onnx_input, vnnlib_input, output, approx=1)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, approx)

    box_constraints, output_dim = get_box_constraints(vnnlib_input, vnnlib_output)

    Vᵀ, new_input_dim = py"update_network"(onnx_input, onnx_output, box_constraints)
    
    A, b = get_A_b_from_box_alternating(box_constraints)
    A = A * transpose(Vᵀ)
    A_new, b_new, new_constraints = approximate(A, b, box_constraints, Vᵀ, new_input_dim, approx)
    return A_new, b_new, new_constraints, new_input_dim, onnx_input
end

A, b, box_constraints, new_input_dim, onnx_input = reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
        "benchmarks/mnistfc/prop_3_0.03.vnnlib",
        "benchmarks/mnistfc_reduced", 1)

#print(ENV["OMP_NUM_THREADS"])
#os =pyimport("os")
#print(os.environ["OMP_NUM_THREADS"])
println(size(box_constraints))
println(size(A))
println(size(b))
# out = [(rand([-1,1],(10,10)),rand([-1,1],(10,)))]
out = create_output_matrix("benchmarks/mnistfc/prop_3_0.03.vnnlib")
run_nnenum("benchmarks/mnistfc_reduced/mnist-net_256x2/prop_3_0.03/mnist-net_256x2.onnx",
 box_constraints[1:new_input_dim, 1], box_constraints[1:new_input_dim, 2], A, b[:,1], out)
 =#