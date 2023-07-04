module DimensionalityReduction

export reduce

using PyCall
using Polyhedra
using CDDLib
using Gurobi
using DelimitedFiles
include("Constraints.jl")
include("Approximation.jl")
include("Exact.jl")
include("PathGenerator.jl")
include("NetworkUpdate.jl")
include("VNNLibGenerator.jl")
include("Utils.jl")
include("NNEnum.jl")
include("Svd.jl")
import .NNEnum: run_nnenum

function reduce(onnx_input, vnnlib_input, output, approx=0, vnnlib=false, nnenum=false)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, approx)

    box_constraints, output_dim = get_box_constraints(vnnlib_input)
    O, new_input_dim = update(onnx_input, onnx_output, box_constraints)

    A₁, b = get_A_b_from_box_alternating(box_constraints)
    A = A₁ * inv(O)

    A[abs.(A) .< 0.000000001] .= 0

    # A_new, b_new = approximate(A, b, new_constraints, new_input_dim, approx)
    println("exact reach")
    # A_new, b_new = exact(A, b, new_input_dim)
    A_new = A
    b_new = b

    box_constraints = new_box_constraints(O, box_constraints)
    # println(box_constraints)
    # box_constraints = exact_box(A_new, b_new, new_input_dim)
    # println(box_constraints)
    #box_constraints = new_box_constraints(P, box_constraints)
    #println(box_constraints)

    if nnenum && approx == 0
        out = create_output_matrix(vnnlib_input, output_dim)
        result = run_nnenum(onnx_output, box_constraints[:, 1],
        box_constraints[:, 2], A_new, b_new[:, 1], out)
        #print(result[3])
        #print(A_new * result[3][1, :,1] - b_new)
        #print(maximum(A_new * result[3][1, :,1] - b_new))
        #print(maximum(A₁* (inv(O) * result[3][1, :,1]) - b_new))
        #run_nnenum(onnx_output, new_constraints[1:new_input_dim, 1],
        #new_constraints[1:new_input_dim, 2], zeros((0,new_input_dim)), zeros((0,0)), out)
    elseif nnenum
        out = create_output_matrix(vnnlib_input, output_dim)
        run_nnenum(onnx_output, new_constraints[1:new_input_dim, 1],
        new_constraints[1:new_input_dim, 2], A_new, b_new[:, 1], out)
    end

    if vnnlib && approx == 0
        println("writing vnnlib-file")
        create_vnnlib_from_lower_upper_bound(new_constraints[1:new_input_dim, 1:2],
        new_input_dim, output_dim, vnnlib_input, vnnlib_output)
    elseif vnnlib
        println("writing vnnlib-file")
        create_vnnlib(A_new, b_new, new_input_dim, output_dim, vnnlib_input, vnnlib_output)
    end
end

end # module DimensionalityReduction
