using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function stars_seconds(onnx_input, vnnlib_input, output, nn, eps; doreduction=true, method=0, d_to_reduce=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims = collect(0:d_to_reduce)

    stars = zeros(d_to_reduce+1,)
    seconds = zeros(d_to_reduce+1,)

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

function algorithm_constraints_nnenum(onnx_input, vnnlib_input, output, nn, eps; doreduction=true, method=0, d_to_reduce=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims = collect(0:d_to_reduce)

    algorithm = zeros(d_to_reduce+1,)
    constraints = zeros(d_to_reduce+1,)
    nnenum_seconds = zeros(d_to_reduce+1,)

    for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, vnnlib_input, output; doreduction, method, d_to_reduce=dim,
        vnnlib, nnenum, factorization, dorefinement)
        algorithm[i] = result[5]
        constraints[i] = result[6]
        nnenum_seconds[i] = result[4]
    end

    p = plot([
        bar(name="Fourier-Motzkin-Elimination", x=dims, y=algorithm, marker_color="indianred"),
        bar(name="Constraints", x=dims, y=constraints, marker_color="lightsalmon"),
        bar(name="NNEnum", x=dims, y=nnenum_seconds, marker_color="lightseagreen")
    ], Layout(title_text="Neural Network: " * nn * "<br>Epsilon: " * eps))
    relayout!(p, barmode="group")
    p
end

nn = "64x32x128x128x128x10"
eps = "0.20"

#=stars_seconds("benchmarks/digits/digit-net_64x4.onnx", 
"benchmarks/digits/dim64/prop_0_0.20.vnnlib",
"benchmarks/digits_reduced", nn, eps, method=2, d_to_reduce=2, nnenum=true)=#

algorithm_constraints_nnenum("benchmarks/digits/digit-net_16x2.onnx", 
"benchmarks/digits/dim16/prop_1_0.40.vnnlib",
"benchmarks/digits_reduced", nn, eps, method=1, d_to_reduce=10, nnenum=true)
