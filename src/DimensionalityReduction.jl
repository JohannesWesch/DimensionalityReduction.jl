module DimensionalityReduction

export reduce

using TimerOutputs
using DelimitedFiles
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

function update(onnx_input, onnx_output, box_constraints, d_to_reduce, d_old, factorization, pertubation)
    weights = get_w(onnx_input, 0)
    if ! (pertubation == zeros(0,))
        weights = weights*pertubation
    end
    remove_zero_activation_weights(weights, box_constraints)

    U, Σ, Vᵀ, d_min = decompose(weights)
    #open("Pertubation784.txt", "w") do io
    #    writedlm(io, Vᵀ)
    #end
    println("Minimal Dimension: ", d_min)
    d_new = get_new_dim(d_old, d_min, d_to_reduce)
    W₁, W₂ = factorize(U, Σ, Vᵀ, d_new, factorization, d_old, d_min, d_to_reduce)
    update_network(onnx_input, onnx_output, W₁, 0)
    return W₂, d_new, d_min
end

function reduce(onnx_input, vnnlib_input, output; doreduction=true, method=0, d_to_reduce=0,
     vnnlib=false, nnenum=false, factorization=0, dorefinement=false, pertubation=zeros(0,))
    outputstr = "didn't run nnenum"
    to = TimerOutput()
    onnx_output = onnx_path(onnx_input, vnnlib_input, output)
    vnnlib_output = vnnlib_path(onnx_input, vnnlib_input, output, method)
    box_constraints, d_old, output_dim = get_box_constraints(vnnlib_input)
    A, b = get_A_b_from_box_alternating(box_constraints)
    #open("PertubationA.txt", "w") do io
    #    writedlm(io, A)
    #end
    #weights = get_w(onnx_input, 0)
    #if !(pertubation == zeros(0,))
    #    print("new weights")
    #    _, _, pertubation,_ = decompose(weights)
    #    open("PertubationA.txt", "w") do io
    #        writedlm(io, pertubation)
    #    end
    #end


    if !(pertubation == zeros(0,))
        box_constraints = exact_box(transpose(pertubation), box_constraints)
        A = A*pertubation
    end

    if doreduction
        W₂, d_new, d_min = update(onnx_input, onnx_output, box_constraints, d_to_reduce, d_old, factorization, pertubation)
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
            elseif method == 5
                A_new, b_new = fourier_redundancy_removal(A, b, d_to_reduce)
            elseif method == 6
                A_new, b_new = approximate_unitvector_lp_solver(A, b, d_new)
            elseif method == 7
                A_new, b_new = zeros((0,d_new)), zeros((0,0))
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
            if pertubation == zeros(0,)
                out = create_output_matrix(vnnlib_input, output_dim)
                result = run_nnenum(onnx_input, box_constraints[:, 1], box_constraints[:, 2], A, b, out)
            else
                w = get_w(onnx_input, 0)
                update_network(onnx_input, onnx_output, w*pertubation, 0)
                out = create_output_matrix(vnnlib_input, output_dim)
                result = run_nnenum(onnx_output, box_constraints[:, 1], box_constraints[:, 2], A, b, out)

            end
            outputstr = result[1]
        end
        safe = 0
        if result[1] == "safe"
        safe = 1
        end
        return (outputstr, d_old, result[2], round(result[4], digits=4),0, 2*d_old, d_old, safe)
    end
    safe = 0
    if result[1] == "safe"
        safe = 1
    end
    return (outputstr, d_new, result[2], round(result[4], digits=4),
    round(TimerOutputs.time(to["algorithm"])* 0.000001, digits=4), size(A_new, 1), d_min, safe) #* 0.000000001
end

end # module DimensionalityReduction
