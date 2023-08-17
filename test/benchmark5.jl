using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block_elimination_time(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims5 = [0,1,2,3,4,5,6,7,8,9,10,11] 
    dims6 = [0,1,2,3,4,5,6,7,8,9,10] 
    dims7 = [0,1,2,3,4,5,6,7,8,9] 
    dims64 = [0,1,2,3] 

    constraints5 = zeros(12,)
    constraints6 = zeros(12,)
    constraints7 = zeros(12,)
    constraints64 = zeros(12,)

    for (i, dim) in enumerate(dims5)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_6_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints5[i] = result[5]
    end

    for (i, dim) in enumerate(dims6)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_3_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints6[i] = result[5]
    end

    for (i, dim) in enumerate(dims7)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_5_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints7[i] = result[5]
    end

    for (i, dim) in enumerate(dims64)
        result = reduce("benchmarks/digits/digit-net_64x4.onnx", "benchmarks/digits/dim64/prop_6_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints64[i] = result[5]
    end

    p = plot([
        scatter(name="Minimal Dimension 5", x=dims5, y=constraints5, marker_color="lightgoldenrodyellow", mode="markers+lines", line=attr(width=3)), #, text=constraints5, textposition="outside"
        scatter(name="Minimal Dimension 6", x=dims6, y=constraints6, marker_color="lightsalmon", mode="markers+lines", line=attr(width=3)), #, text=constraints6, textposition="outside"
        scatter(name="Minimal Dimension 7", x=dims7, y=constraints7, marker_color="indianred", mode="markers+lines", line=attr(width=3)), #, text=constraints7, textposition="outside"
    ], Layout(yaxis=attr(title="Runtime(ms)", linecolor="black", type="log", nticks=5,
    showgrid=true,
    gridcolor="lightslategrey",
    gridwidth=0.1),
    xaxis=attr(title="Reduced Dimensions", linecolor="black",
    showgrid=true,
    gridcolor="lightslategrey",
    gridwidth=0.1) ,
      legend=attr(
    x=1,
    y=1.02,
    yanchor="bottom",
    xanchor="right",
    orientation="h"
    ), plot_bgcolor="white"))
    relayout!(p, barmode="group")
    savefig(p, "test/plot5.svg")
    p
end

block_elimination_time("benchmarks/digits/digit-net_16x2.onnx", "benchmarks/digits_reduced")