using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block_elimination_nnenum(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
     
    epsilons = [
                "benchmarks/digits/dim196/prop_12_0.002.vnnlib",
                
                "benchmarks/digits/dim196/prop_12_0.004.vnnlib",
                
                "benchmarks/digits/dim196/prop_12_0.006.vnnlib",
               
                "benchmarks/digits/dim196/prop_12_0.008.vnnlib",
                
                "benchmarks/digits/dim196/prop_12_0.01.vnnlib",
                
                "benchmarks/digits/dim196/prop_12_0.012.vnnlib",
                
                "benchmarks/digits/dim196/prop_12_0.014.vnnlib",
                
                "benchmarks/digits/dim196/prop_12_0.016.vnnlib",
                
                "benchmarks/digits/dim196/prop_12_0.018.vnnlib",
                
                "benchmarks/digits/dim196/prop_12_0.02.vnnlib",
               
                "benchmarks/digits/dim196/prop_12_0.022.vnnlib",
                "benchmarks/digits/dim196/prop_12_0.024.vnnlib",
                "benchmarks/digits/dim196/prop_12_0.026.vnnlib",
                "benchmarks/digits/dim196/prop_12_0.028.vnnlib",
                "benchmarks/digits/dim196/prop_12_0.03.vnnlib",
    ]
    epsilons_text = [0.002, 0.004, 0.006, 0.008,  0.01, 0.012,  0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03]

    runtime = zeros(25,)
    runtime_removed = zeros(25,)
    runtime_nopermute = zeros(25,)
    dummy = zeros(25,)
    color_vec1 = fill("lightgreen", 25)
    color_vec2 = fill("lightgreen", 25)
    color_vec3 = fill("lightgreen", 25)

    for (i, epsilon) in enumerate(epsilons)
        result = reduce(onnx_input, epsilon, output; doreduction, method=2, d_to_reduce=0,
        vnnlib, nnenum, factorization=3, dorefinement)
        runtime[i] = result[4]
        if result[8] == 0
            color_vec1[i] = "darkgray"
        end
    end

    for (i, epsilon) in enumerate(epsilons)
        result = reduce(onnx_input, epsilon, output; doreduction, method=2, d_to_reduce=188,
        vnnlib, nnenum, factorization=3, dorefinement)
        runtime_removed[i] = result[4]
        if result[8] == 0 && color_vec1[i] == "darkgray"
            color_vec2[i] = "darkgray"
        elseif result[8] == 0
            color_vec2[i] = "orangered"
        end
    end

    for (i, epsilon) in enumerate(epsilons)
        result = reduce(onnx_input, epsilon, output; doreduction, method=3, d_to_reduce=188,
        vnnlib, nnenum, factorization=3, dorefinement)
        runtime_nopermute[i] = result[4]
        if result[8] == 0 && color_vec1[i] == "darkgray"
            color_vec3[i] = "darkgray"
        elseif result[8] == 0
            color_vec3[i] = "orangered"
        end
    end


    p = plot([
        scatter(name="without reduction", x=epsilons_text, y=runtime, mode="markers+lines", line=attr(width=8, color="midnightblue"), marker=attr(color=color_vec1, size=8), showlegend=false),
        scatter(name="custom directions", x=epsilons_text, y=runtime_removed, line_color="green", mode="markers+lines",  line=attr(width=8, color="lightseagreen"), marker=attr(color=color_vec2, size=8), showlegend=false),
        scatter(name="unitvector directions", x=epsilons_text, y=runtime_nopermute, line_color="green", mode="markers+lines",  line=attr(width=8, color="lightcoral"), marker=attr(color=color_vec3, size=8), showlegend=false),

        scatter(name="without reduction", x=epsilons_text, y=dummy, mode="lines", line=attr(color="midnightblue")),
        scatter(name="custom directions", x=epsilons_text, y=dummy, mode="lines", line=attr(color="lightseagreen")),
        scatter(name="unitvector directions", x=epsilons_text, y=dummy, mode="lines", line=attr(color="lightcoral")),
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
    savefig(p, "test/plot15.svg")
    p
end

block_elimination_nnenum("benchmarks/digits/digit-net_196x8x128x128x128x128x128x10.onnx", "benchmarks/digits_reduced", nnenum=true)
