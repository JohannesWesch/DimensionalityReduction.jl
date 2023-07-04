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

function update(onnx_input_filename, onnx_output_filename, box_constraints, d_reduced)
    weights = get_w(onnx_input_filename, onnx_output_filename, box_constraints)
    U, Σ, Vᵀ, d_min = decompose(weights)
    d_old = size(weights, 2)
    d_new = d_old

    println(d_old)
    println(d_reduced)
    println(d_min)

    if (d_old - d_reduced < d_min)
        println("error") # throw exception
    elseif d_reduced == -1
        d_new = d_min
    elseif d_min <= d_old - d_reduced <= d_old
        d_new = d_old - d_reduced
    end

    println(d_new)

    F = lu(Vᵀ, NoPivot())
    P = get_permutation(size(weights, 1), d_old)

    W = U * Σ * F.L * P
    W[abs.(W) .< 0.000000001] .= 0
    
    update_network(onnx_input_filename, onnx_output_filename,  W[:, 1:d_new])
    return P * F.U, d_new
end

function reduce(onnx_input, vnnlib_input, output, approx=0, d_reduced=0, vnnlib=false, nnenum=false)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, approx)

    box_constraints, output_dim = get_box_constraints(vnnlib_input)
    U, d_new = update(onnx_input, onnx_output, box_constraints, d_reduced)

    A, b = get_A_b_from_box_alternating(box_constraints)
    A = A * inv(U)
    A[abs.(A) .< 0.000000001] .= 0

    box_constraints = new_box_constraints(U, box_constraints)
    A_new, b_new = approximate(A, b, box_constraints, d_new, approx)

    if nnenum
        out = create_output_matrix(vnnlib_input, output_dim)
        result = run_nnenum(onnx_output,
        box_constraints[1:d_new, 1],
        box_constraints[1:d_new, 2],
        A_new, b_new[:, 1], out)
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

#print(result[3])
        #print(A_new * result[3][1, :,1] - b_new)
        #print(maximum(A_new * result[3][1, :,1] - b_new))
        #print(maximum(A₁* (inv(O) * result[3][1, :,1]) - b_new))
        #run_nnenum(onnx_output, new_constraints[1:new_input_dim, 1],
        #new_constraints[1:new_input_dim, 2], zeros((0,new_input_dim)), zeros((0,0)), out)
