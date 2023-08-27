using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block_elimination_nnenum(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
     
    dims = collect(0:2:188)

    runtime = zeros(100,)
    stars = zeros(100,)
    dummy = zeros(100,)
    color_vec = fill("lightgray", 100)

    for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, "benchmarks/digits/dim196/prop_2_0.007.vnnlib", output; doreduction, method=3, d_to_reduce=dim,
        vnnlib, nnenum, factorization=3, dorefinement)
        runtime[i] = result[4]
        stars[i] = result[3]
        if result[8] == 0
            color_vec[i] = "orangered"
        end
    end


    p = plot([
        scatter(name="runtime", x=dims, y=runtime, marker_color="blue", mode="markers+lines", line=attr(width=3), yaxis="y2"), #, text=constraints5, textposition="outside"
        bar(name="safe", x=dims, y=stars, marker_color=color_vec),
        bar(name="spurious", x=dims, y=dummy, marker_color="orangered"),
    ], Layout(yaxis=attr(title="Stars", linecolor="black", nticks=2, type="log",
    showgrid=true,
    gridcolor="lightslategrey",
    gridwidth=0.1),
    yaxis2=attr(title="Runtime (s)", linecolor="black", nticks=1, type="log", side="right", overlaying="y"),
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
    savefig(p, "test/plot17.svg")
    p
end

block_elimination_nnenum("benchmarks/digits/digit-net_196x8x128x128x128x128x128x10.onnx", "benchmarks/digits_reduced", nnenum=true)