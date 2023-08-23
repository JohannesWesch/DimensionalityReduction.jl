module DimensionalityReduction

export reduce

using TimerOutputs
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
include("Refinement.jl")
import .NNEnum: run_nnenum

function factorize(U, Σ, Vᵀ, d_new, fact, d_old, d_min, d_to_reduce)
    F = lu(Vᵀ)
    W₁ = U * Σ
    W₂ = Vᵀ
    if fact == 0
        W₁ = round_matrix(U * Σ * F.L)
        W₂ = F.U * F.P
    elseif fact == 1
        P = get_individual_permutation(d_old, d_min, d_to_reduce)
        W₁ = round_matrix(U * Σ * F.L * P)
        W₂ = round_matrix(transpose(P) * F.U * F.P)
    elseif fact == 2
        P = get_individual_permutation_fourier(d_old, d_min, d_to_reduce)
        W₁ = round_matrix(U * Σ * F.L * P)
        W₂ = round_matrix(transpose(P) * F.U * F.P)
    elseif fact == 3
        # do nothing
    end
    return W₁[:, 1:d_new], W₂
end

function update(onnx_input, onnx_output, box_constraints, d_to_reduce, d_old, factorization)
    first_matrix = 0
    #if (d_old > 700)
    #    first_matrix = 1
    #end

    weights = get_w(onnx_input, box_constraints, first_matrix)
    U, Σ, Vᵀ, d_min = decompose(weights)
    println("Minimal Dimension: ", d_min)
    d_new = get_new_dim(d_old, d_min, d_to_reduce)
    W₁, W₂ = factorize(U, Σ, Vᵀ, d_new, factorization, d_old, d_min, d_to_reduce)
    update_network(onnx_input, onnx_output, W₁, first_matrix)
    return W₂, d_new, d_min
end

function reduce(onnx_input, vnnlib_input, output; doreduction=true, method=0, d_to_reduce=0,
     vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    outputstr = "didn't run nnenum"
    to = TimerOutput()
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, method)
    box_constraints, d_old, output_dim = get_box_constraints(vnnlib_input)

    if doreduction
        W₂, d_new, d_min = update(onnx_input, onnx_output, box_constraints, d_to_reduce, d_old, factorization)
        A, b = get_A_b_from_box_alternating(box_constraints)
        A = round_matrix(A * inv(W₂))
        new_constraints = exact_box(W₂, box_constraints)

        @timeit to "algorithm" begin
            if method == 0
                A_new, b_new = fourier(A, b, d_to_reduce)
            elseif method == 1
                A_new, b_new = block(A, b, d_new, d_old)
            elseif method == 2
                A_new, b_new = approximate(A, d_new, W₂, box_constraints)
            elseif method == 3
                A_new, b_new = approximate_unitvector(A, d_new, W₂, box_constraints)
            elseif method == 4
                A_new, b_new = approximate_support_function(A, b, d_new)
            end
        end
        println(size(A_new))
        result = zeros(4,)
        if nnenum
            out = create_output_matrix(vnnlib_input, output_dim)
            result = run_nnenum(onnx_output, new_constraints[1:d_new, 1], new_constraints[1:d_new, 2], A_new, b_new, out)
            outputstr = result[1]

            if dorefinement && result[1] == "unsafe"
                outputstr = refine(onnx_output, new_constraints[1:d_new, 1], new_constraints[1:d_new, 2], A_new, b_new, out,
                result, d_new, d_old, W₂, box_constraints)
            end
        end
    
        if vnnlib
            println("writing vnnlib-file")
            create_vnnlib(A_new, b_new, d_new, output_dim, vnnlib_input, vnnlib_output)
        end


    else
        if nnenum
            out = create_output_matrix(vnnlib_input, output_dim)
            result = run_nnenum(onnx_input, box_constraints[:, 1], box_constraints[:, 2], zeros((0,d_old)), zeros((0,0)), out)
        end
        return
    end
    safe = 0
    if result[1] == "safe"
        safe = 1
    end
    return (outputstr, d_new, result[2], round(result[4], digits=4),
    round(TimerOutputs.time(to["algorithm"])* 0.000001, digits=4), size(A_new, 1), d_min, safe) #* 0.000000001
end

end # module DimensionalityReduction
