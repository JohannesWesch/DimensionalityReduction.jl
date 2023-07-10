include("Approximation.jl")

function refine(onnx_output, lower_bounds, upper_bounds, A_new, b_new, out, result, new_input_dim, d_old, V, bounds)
    for i in 1:2
        cex = result[3][1, :, 1]
        direction = zeros(d_old,)
        direction[1:new_input_dim] = cex
        s = approximate_direction(V, bounds, direction)

        A_new = vcat(A_new, cex')
        b_new = vcat(b_new, s)

        result = run_nnenum(onnx_output, lower_bounds, upper_bounds, A_new, b_new, out)

        if (result[1] == "safe")
            break
        end
    end
    return result[1]
end