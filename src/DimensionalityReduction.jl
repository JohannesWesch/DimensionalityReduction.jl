module DimensionalityReduction

export reduce

using PyCall
include("Constraints.jl")
include("Approximation.jl")
include("Exact.jl")
include("NetworkUpdate.jl")
include("VNNLibGenerator.jl")
include("Utils.jl")
include("NNEnum.jl")
include("Svd.jl")
include("Mix.jl")
include("PCA.jl")
import .NNEnum: run_nnenum

function factorize(U, Σ, Vᵀ, d_new, fact, d_old, d_min)
    F = lu(Vᵀ, NoPivot())
    W₁ = U * Σ
    W₂ = Vᵀ
    if fact == 0
        W₁ = round(U * Σ * F.L)
        W₂ = F.U
    elseif fact == 1
        P = get_permutation(d_min, d_old)
        W₁ = round(U * Σ * F.L * P)
        W₂ = round(P * F.U)
    elseif fact == 2
        #do nothing
    end
    return W₁[:, 1:d_new], W₂
end

function update(onnx_input, onnx_output, box_constraints, d_to_reduce, d_old, factorization)
    weights = get_w(onnx_input, box_constraints)
    U, Σ, Vᵀ, d_min = decompose(weights)
    println("Minimal Dimension: ", d_min)
    d_new = get_new_dim(d_old, d_min, d_to_reduce)
    W₁, W₂ = factorize(U, Σ, Vᵀ, d_new, factorization, d_old, d_min)
    update_network(onnx_input, onnx_output,  W₁)
    return W₂, d_new
end

function reduce(onnx_input, vnnlib_input, output; reduce=true, method=0, d_to_reduce=0,
     vnnlib=false, nnenum=false, factorization=0)
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, method)
    box_constraints, d_old, output_dim = get_box_constraints(vnnlib_input)

    if reduce
        W₂, d_new = update(onnx_input, onnx_output, box_constraints, d_to_reduce, d_old, factorization)
        A, b = get_A_b_from_box_alternating(box_constraints)
        A = round(A * inv(W₂))
        new_constraints = exact_box(W₂, box_constraints)

        if method == 0
            A_new, b_new = fourier(A, b, d_to_reduce)
        elseif method == 1
            A_new, b_new = block(A, b, d_new, d_old)
        elseif method == 2
            A_new, b_new = fourier_approx(A, b, d_to_reduce, d_old)
        elseif method == 3
            A_new, b_new = block_approx(A, b, d_new, d_old)
        elseif method == 4
            A_new, b_new = approximate_support_function(A, b, d_new)
        elseif method == 5
            A_new, b_new = pca_approx(A, b, d_new)
        end

        if nnenum
            out = create_output_matrix(vnnlib_input, output_dim)
            result = run_nnenum(onnx_output, new_constraints[1:d_new, 1], new_constraints[1:d_new, 2], A_new, b_new, out)
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

end # module DimensionalityReduction

#print(result[3])
#print(A_new * result[3][1, :,1] - b_new)
#print(maximum(A_new * result[3][1, :,1] - b_new))
#print(maximum(A₁* (inv(O) * result[3][1, :,1]) - b_new))
#run_nnenum(onnx_output, new_constraints[1:new_input_dim, 1],
#new_constraints[1:new_input_dim, 2], zeros((0,new_input_dim)), zeros((0,0)), out)
