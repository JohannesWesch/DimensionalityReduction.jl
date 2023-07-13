using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function stars_seconds(onnx_input, vnnlib_input, output, nn, eps, dims; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)

    n = len(dims)

    stars = zeros(n,)
    seconds = zeros(n,)

    for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, vnnlib_input, output; doreduction, method, d_to_reduce=dim,
        vnnlib, nnenum, factorization, dorefinement)
        stars[i] = result[3]
        seconds[i] = result[4]
    end

    p = plot([
        bar(name="Stars", x=dims, y=stars, marker_color="indianred"),
        bar(name="Seconds", x=dims, y=seconds, marker_color="lightsalmon")
    ], Layout(title_text="Neural Network: " * nn * "<br>Epsilon: " * eps))
    relayout!(p, barmode="group")
    p
end

function algorithm_constraints_nnenum(onnx_input, vnnlib_input, output,algorithm, nn, eps, dims; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    n = size(dims, 1)

    algorithm = zeros(n,)
    constraints = zeros(n,)
    nnenum_seconds = zeros(n,)

    for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, vnnlib_input, output; doreduction, method, d_to_reduce=dim,
        vnnlib, nnenum, factorization, dorefinement)
        algorithm[i] = result[5]
        constraints[i] = result[6]
        nnenum_seconds[i] = result[4]
    end

    p = plot([
        #bar(name=algorithm, x=dims, y=algorithm, marker_color="indianred"),
        bar(name="Constraints", x=dims, y=constraints, marker_color="lightsalmon"),
        #bar(name="NNEnum", x=dims, y=nnenum_seconds, marker_color="lightseagreen")
    ], Layout(title_text="Neural Network: " * nn * "<br>Epsilon: " * eps * "<br>Algorithm: " * algorithm))
    relayout!(p, barmode="group")
    p
end

function fourier_block(onnx_input, vnnlib_input, output,algorithm_name, nn, eps, dims; doreduction=true,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    n = size(dims, 1)

    algorithm = zeros(n,)
    constraints = zeros(n,)
    nnenum_seconds = zeros(n,)

    constraints_fourier = zeros(4,)

    for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, vnnlib_input, output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization, dorefinement)
        if (dim <= 2)
            result2 = reduce(onnx_input, vnnlib_input, output; doreduction, method=0, d_to_reduce=dim,
            vnnlib, nnenum, factorization, dorefinement)
            constraints_fourier[i] = result2[6]
        end
        algorithm[i] = result[5]
        constraints[i] = result[6]
        nnenum_seconds[i] = result[4]
    end

    constraints_fourier[4] = 50972460

    p = plot([
        #bar(name=algorithm_name, x=dims, y=algorithm, marker_color="indianred"),
        bar(name="Block-Elimination", x=dims, y=constraints, marker_color="lightsalmon", text=constraints, textposition="outside"),
        bar(name="Fourier-Motzkin-Elimination", x=dims, y=constraints_fourier, marker_color="lightseagreen", text=constraints_fourier ,textposition="outside"),
        #bar(name="NNEnum", x=dims, y=nnenum_seconds, marker_color="lightseagreen")
],  Layout(yaxis_range=[0, 50000], title_text="Neural Network: " * nn * "<br>Epsilon: " * eps))
    relayout!(p, barmode="group")
    p
end

#nn = "64x32x128x128x128x10"
algorithm = "Block Elimination"
nn = "16x8x64x64x64x10"
eps = "0.20"
dims = [0, 1, 2, 3, 4, 5, 6, 7, 8]

#=stars_seconds("benchmarks/digits/digit-net_64x4.onnx", 
"benchmarks/digits/dim64/prop_0_0.20.vnnlib",
"benchmarks/digits_reduced", nn, eps, dims, method=1, nnenum=true)=#

#=algorithm_constraints_nnenum("benchmarks/digits/digit-net_16x4.onnx", 
"benchmarks/digits/dim16/prop_0_0.20.vnnlib",
"benchmarks/digits_reduced",algorithm, nn, eps, dims, method=1, nnenum=true)=#

#=fourier_block("benchmarks/digits/digit-net_16x4.onnx", 
"benchmarks/digits/dim16/prop_0_0.20.vnnlib",
"benchmarks/digits_reduced",algorithm, nn, eps, dims, nnenum=true)=#