module DimensionalityReduction

export reduce

using PyCall
using DelimitedFiles
include("Constraints.jl")
include("Approximation.jl")
include("PathGenerator.jl")
@pyinclude("src/NetworkUpdate.py")
@pyinclude("src/VNNLibGenerator.py")

function reduce(onnx_input, vnnlib_input, output, approx=1)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, approx)

    box_constraints, output_dim = get_box_constraints(vnnlib_input, vnnlib_output)

    Vᵀ, new_input_dim = py"update_network"(onnx_input, onnx_output, box_constraints)
    A, b = get_A_b_from_box_alternating(box_constraints)
    A = A * transpose(Vᵀ)
    A_new, b_new = approximate(A, b, box_constraints, Vᵀ, new_input_dim, approx)
    py"create_vnnlib"(A_new, b_new, new_input_dim, output_dim, vnnlib_input, vnnlib_output)
    # constraints = approximate(A, b, box_constraints, Vᵀ, new_input_dim, approx)
    # py"create_vnnlib_from_lower_upper_bound"(constraints, new_input_dim, output_dim, vnnlib_input, vnnlib_output)
end

#=function reduce_network(onnx_input, vnnlib_input, onnx_output, V_output, dim_output)
    box_constraints, _ = get_box_constraints(vnnlib_input)
    Vᵀ, new_input_dim = py"update_network"(onnx_input, onnx_output, box_constraints)
   
    open(V_output, "w") do io
        writedlm(io, Vᵀ)
    end

    open(dim_output, "w") do io
        writedlm(io, new_input_dim)
    end
end

function calculate_polytope(V_input, dim_input,vnnlib_input, vnnlib_output, approx=1)
    Vᵀ = readdlm(V_input)
    new_input_dim = Int.(readdlm(dim_input)[1])
    box_constraints, output_dim = get_box_constraints(vnnlib_input)
    A, b = get_A_b_from_box_alternating(box_constraints)
    A = A*transpose(Vᵀ)
    A_new, b_new = approximate(A, b, box_constraints, Vᵀ, new_input_dim, approx)
    py"create_vnnlib"(A_new, b_new, new_input_dim, output_dim, vnnlib_input, vnnlib_output)
end=#

end # module DimensionalityReduction
