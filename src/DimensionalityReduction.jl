module DimensionalityReduction

export reduce

using PyCall
using DelimitedFiles
include("Constraints.jl")
include("Approximation.jl")
include("PathGenerator.jl")
include("NetworkUpdate.jl")
include("VNNLibGenerator.jl")
include("Utils.jl")
include("NNEnum.jl")
import .NNEnum: run_nnenum

function reduce(onnx_input, vnnlib_input, output, approx=1, nnenum=false)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, approx)

    box_constraints, output_dim = get_box_constraints(vnnlib_input, vnnlib_output)
    Vᵀ, new_input_dim = update_network(onnx_input, onnx_output, box_constraints)

    A, b = get_A_b_from_box_alternating(box_constraints)
    A = A * transpose(Vᵀ)

    new_constraints = new_box_constraints(Vᵀ, box_constraints)
    A_new, b_new = approximate(A, b, new_constraints, new_input_dim, approx, nnenum)

    if nnenum
        out = create_output_matrix(vnnlib_input)
        run_nnenum(onnx_output, new_constraints[1:new_input_dim, 1],
        new_constraints[1:new_input_dim, 2], A_new, b_new[:,1], out)
    else
        create_vnnlib(A_new, b_new, new_input_dim, output_dim, vnnlib_input, vnnlib_output)
    end
end

end # module DimensionalityReduction
