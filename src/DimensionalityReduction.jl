module DimensionalityReduction

export reduce

using PyCall
include("Constraints.jl")
include("Approximation.jl")
@pyinclude("src/NetworkUpdate.py")

function reduce(onnx_input, vnnlib_input, onnx_output, vnnlib_output, new_dim)
    box_constraints = get_box_constraints(vnnlib_input)
    A, b = get_A_b(box_constraints)

    V, new_input_dim = py"update_network"(onnx_input, onnx_output, box_constraints)

    A = A*transpose(V)

    # variables = collect(1:new_input_dim)
    variables = collect(1:new_dim)
    new_bounds = approximate(A, b, variables)

    open(vnnlib_output, "w") do file
        write(file, new_bounds)
    end
end

end # module DimensionalityReduction
