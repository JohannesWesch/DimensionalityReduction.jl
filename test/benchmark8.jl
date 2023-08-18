using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block_elimination_time(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims764 = [0,1,2,3,4,5,6,7,8]
    dims864 = [0,1,2,3,4,5,6,7,8]

    constraints764 = zeros(11,)
    constraints864 = zeros(11,)

    for (i, dim) in enumerate(dims864)
        result = reduce(onnx_input, "benchmarks/digits/dim64/prop_5_0.10.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints864[i] = result[4]
    end

    for (i, dim) in enumerate(dims764)
        result = reduce(onnx_input, "benchmarks/digits/dim64/prop_8_0.10.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints764[i] = result[4]
    end



    p = plot([
        scatter(name="64-7", x=dims764, y=constraints764, marker_color="lightgreen", mode="markers+lines", line=attr(width=3)), #, text=constraints7, textposition="outside"
        scatter(name="64-8", x=dims864, y=constraints864, marker_color="lightskyblue", mode="markers+lines", line=attr(width=3)), #, text=constraints7, textposition="outside"
    ], Layout(yaxis=attr(title="Runtime (ms)", linecolor="black", type="log", nticks=4,
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
    savefig(p, "test/plot8.svg")
    p
end

block_elimination_time("benchmarks/digits/digit-net_64x8x64x64x64x10.onnx", "benchmarks/digits_reduced", nnenum=true)