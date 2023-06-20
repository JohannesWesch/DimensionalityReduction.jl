module DimensionalityReduction

export reduce, reduce_network, calculate_polytope

using PyCall
include("Constraints.jl")
include("Approximation.jl")
@pyinclude("src/NetworkUpdate.py")
@pyinclude("src/VNNLibGenerator.py")

function reduce(onnx_input, vnnlib_input, onnx_output, vnnlib_output)
    box_constraints, n_output = get_box_constraints(vnnlib_input)
    A, b = get_A_b_from_box(box_constraints)
    V, new_input_dim = py"update_network"(onnx_input, onnx_output, box_constraints)
    A = A * transpose(V)
    A_new, b_new = approximate(A, b, box_constraints, V, new_input_dim)
    py"create_vnnlib"(A_new, b_new, new_input_dim, n_output, vnnlib_input, vnnlib_output)
end

function reduce_network(onnx_input, vnnlib_input, onnx_output)
    V, new_input_dim = py"update_network"(onnx_input, onnx_output, box_constraints)
    # speichere V und new_input_dim ab
end

function calculate_polytope(V, new_input_dim, vnnlib_input, vnnlib_output, variables)
    box_constraints = get_box_constraints(vnnlib_input)
    A, b = get_A_b_from_box(box_constraints)

    A = A*transpose(V)

    # variables = collect(1:new_input_dim)
    # variables = collect(1:new_dim)
    new_bounds = approximate(A, b, box_constraints, variables)

    py"create_vnnlib_from_lower_upper_bound"(new_bounds, length(variables), 10, vnnlib_input, vnnlib_output)
end

end # module DimensionalityReduction
