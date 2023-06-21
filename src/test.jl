using LazySets
using PyCall
using DelimitedFiles
using Polyhedra
include("Constraints.jl")
include("Approximation.jl")
include("PathGenerator.jl")
@pyinclude("src/NetworkUpdate.py")
@pyinclude("src/VNNLibGenerator.py")

function reduce(onnx_input, vnnlib_input, output, approx=1)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output)

    box_constraints, output_dim = get_box_constraints(vnnlib_input, vnnlib_output)

    V, new_input_dim = py"update_network"(onnx_input, onnx_output, box_constraints)
    A, b = get_A_b_from_box_alternating(box_constraints)
    b = vec(b)

    print("hi")
    H = HPolytope(A,b)
    print("hi")
    V = vertices_list(H)
    # LazySet.AffineMap(V, H)
    print("hi")
end

reduce("benchmarks/mnistfc/mnist-net_256x2.onnx", 
        "benchmarks/mnistfc/prop_3_0.03.vnnlib",
        "benchmarks/mnistfc_reduced", 1)