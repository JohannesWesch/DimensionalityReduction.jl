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

function update(onnx_input, onnx_output, box_constraints, d_reduced, d_old)
    weights = get_w(onnx_input, box_constraints)
    U, Σ, Vᵀ, d_min = decompose(weights)
    
    d_new = get_new_dim(d_old, d_min, d_reduced)

    F = lu(Vᵀ, NoPivot())
    P = get_permutation(size(weights, 1), d_old)

    W = round(U * Σ * F.L * P) #U * Σ * F.L * P
    
    update_network(onnx_input, onnx_output,  W[:, 1:d_new])
    return P * F.U, d_new #P * F.U
end

function reduce(onnx_input, vnnlib_input, output; reduce=true, method=0, d_reduced=0, vnnlib=false, nnenum=false)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, method)
    box_constraints, d_old, output_dim = get_box_constraints(vnnlib_input)

    if reduce
        U, d_new = update(onnx_input, onnx_output, box_constraints, d_reduced, d_old)
        A, b = get_A_b_from_box_alternating(box_constraints)
        A = round(A * inv(U))
        box_constraints = new_box_constraints(U, box_constraints)

        if method == 0
            A_new, b_new = fourier(A, b, d_old-d_new)
        elseif method == 1
            A_new, b_new = block(A, b, d_new, d_old)
        elseif method == 2
            A_new, b_new = approximate(A, b, box_constraints, d_new, approx)
        end

        if nnenum
            out = create_output_matrix(vnnlib_input, output_dim)
            result = run_nnenum(onnx_output, box_constraints[1:d_new, 1], box_constraints[1:d_new, 2], A_new, b_new[:, 1], out)
        end
    
        if vnnlib
            println("writing vnnlib-file")
            create_vnnlib(A_new, b_new, new_input_dim, output_dim, vnnlib_input, vnnlib_output)
        end


    else
        if nnenum
            out = create_output_matrix(vnnlib_input, output_dim)
            run_nnenum(onnx_input, box_constraints[:, 1], box_constraints[:, 2], zeros((0,d_old)), zeros((0,0)), out)
        end
    end
end

function get_new_dim(d_old, d_min, d_reduced)
    d_new = d_old
    if (d_old - d_reduced < d_min)
        println("error") # throw exception
    elseif d_reduced == -1
        d_new = d_min
    elseif d_min <= d_old - d_reduced <= d_old
        d_new = d_old - d_reduced
    end
end

end # module DimensionalityReduction

#print(result[3])
        #print(A_new * result[3][1, :,1] - b_new)
        #print(maximum(A_new * result[3][1, :,1] - b_new))
        #print(maximum(A₁* (inv(O) * result[3][1, :,1]) - b_new))
        #run_nnenum(onnx_output, new_constraints[1:new_input_dim, 1],
        #new_constraints[1:new_input_dim, 2], zeros((0,new_input_dim)), zeros((0,0)), out)
