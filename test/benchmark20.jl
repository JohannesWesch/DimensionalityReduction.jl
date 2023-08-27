using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block_elimination_nnenum(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
     
    epsilons = [
                "benchmarks/digits/dim784/prop_7_0.002.vnnlib",
                "benchmarks/digits/dim784/prop_7_0.003.vnnlib",
                "benchmarks/digits/dim784/prop_7_0.004.vnnlib",
                "benchmarks/digits/dim784/prop_7_0.005.vnnlib",
                "benchmarks/digits/dim784/prop_7_0.006.vnnlib",
                "benchmarks/digits/dim784/prop_7_0.007.vnnlib",
                "benchmarks/digits/dim784/prop_7_0.008.vnnlib",
                "benchmarks/digits/dim784/prop_7_0.009.vnnlib",
    ]
    epsilons_text = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]

    runtime = zeros(15,)
    runtime_removed = zeros(15,)
    runtime_nopermute = zeros(15,)
    dummy = zeros(15,)
    color_vec1 = fill("lightgreen", 15)
    color_vec2 = fill("lightgreen", 15)
    color_vec3 = fill("lightgreen", 15)

    for (i, epsilon) in enumerate(epsilons)
        result = reduce(onnx_input, epsilon, output; doreduction, method=2, d_to_reduce=720,
        vnnlib, nnenum, factorization=3, dorefinement)
        runtime_removed[i] = result[4]
        if result[8] == 0 && color_vec1[i] == "darkgray"
            color_vec2[i] = "darkgray"
        elseif result[8] == 0
            color_vec2[i] = "orangered"
        end
    end

    for (i, epsilon) in enumerate(epsilons)
        result = reduce(onnx_input, epsilon, output; doreduction, method=2, d_to_reduce=0,
        vnnlib, nnenum, factorization=3, dorefinement)
        runtime[i] = result[4]
        if result[8] == 0
            color_vec1[i] = "darkgray"
        end
    end


    p = plot([
        scatter(name="without reduction", x=epsilons_text, y=runtime, mode="markers+lines", line=attr(width=8, color="midnightblue"), marker=attr(color=color_vec1, size=8), showlegend=false),
        scatter(name="custom directions", x=epsilons_text, y=runtime_removed, line_color="green", mode="markers+lines",  line=attr(width=8, color="lightseagreen"), marker=attr(color=color_vec2, size=8), showlegend=false),

        scatter(name="without reduction", x=epsilons_text, y=dummy, mode="lines", line=attr(color="midnightblue")),
        scatter(name="custom directions", x=epsilons_text, y=dummy, mode="lines", line=attr(color="lightseagreen")),
        scatter(name="safe", x=epsilons_text, y=dummy, marker_color="lightgreen", mode="markers"),
        scatter(name="spurious", x=epsilons_text, y=dummy, marker_color="orangered", mode="markers"),
        scatter(name="unsafe", x=epsilons_text, y=dummy, marker_color="darkgray", mode="markers"),
    ], Layout(yaxis=attr(title="Runtime (s)", linecolor="black", nticks=4, type="log",
    showgrid=true,
    gridcolor="lightslategrey",
    gridwidth=0.1),
    xaxis=attr(title="Epsilon", linecolor="black",
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
    savefig(p, "test/plot20.svg")
    p
end

block_elimination_nnenum("benchmarks/digits/digit-net_784x64x256x256x256x256x256x10.onnx", "benchmarks/digits_reduced", nnenum=true)
