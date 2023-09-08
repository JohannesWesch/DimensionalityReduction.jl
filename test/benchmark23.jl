using PlotlyJS
using DelimitedFiles
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
    ]
    epsilons_text = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]

    runtime_overall = zeros(15,)
    runtime_unitvector = zeros(15,)
    runtime_no = zeros(15,)
    dummy = zeros(15,)
    #runtime_overall = vec(readdlm("21-overall.txt", '\t', Float64, '\n'))
    #runtime_unitvector = vec(readdlm("21-unitvector.txt", '\t', Float64, '\n'))
    #runtime_no = vec(readdlm("21-no.txt", '\t', Float64, '\n'))
    #color_vec1 =  vec(readdlm("21-colorvec1.txt", '\t', String, '\n'))
    color_vec1 = fill("lightgreen", 15)
    color_vec2 = fill("lightgreen", 15)


    pertubation = readdlm("Pertubation784.txt", '\t', Float64, '\n')

    for (i, epsilon) in enumerate(epsilons)
        if (runtime_unitvector[i] != 0.0)
            continue
        end
        result = reduce(onnx_input, epsilon, output; doreduction, method=7, d_to_reduce=752,
        vnnlib, nnenum, factorization=3, pertubation=pertubation)
        runtime_unitvector[i] = result[4]
        runtime_overall[i] = result[4] + result[5]/1000
        if result[8] == 0
            color_vec1[i] = "orangered"
        end
        open("21-overall.txt", "w") do io
            writedlm(io, runtime_overall)
        end
        open("21-unitvector.txt", "w") do io
            writedlm(io, runtime_unitvector)
        end
        open("21-colorvec1.txt", "w") do io
            writedlm(io, color_vec1)
        end
    end

    #=for (i, epsilon) in enumerate(epsilons)
        if (runtime_no[i] != 0.0)
            continue
        end
        result = reduce(onnx_input, epsilon, output; doreduction=false, method=2, d_to_reduce=0,
        vnnlib, nnenum, factorization=3, pertubation=pertubation)
        runtime_no[i] = result[4]
        open("21-no.txt", "w") do io
            writedlm(io, runtime_no)
        end
    end=#

    

    p = plot([
        scatter(name="unitvector overall", x=epsilons_text, y=runtime_overall, mode="markers+lines", line=attr(width=8, color="midnightblue"), marker=attr(color=color_vec1, size=8), showlegend=false),
        scatter(name="unitvector verification", x=epsilons_text, y=runtime_unitvector, mode="markers+lines",  line=attr(width=8, color="lightseagreen"), marker=attr(color=color_vec1, size=8), showlegend=false),
        scatter(name="without factorization", x=epsilons_text, y=runtime_no, mode="markers+lines",  line=attr(width=8, color="violet"), marker=attr(color=color_vec2, size=8), showlegend=false),

        scatter(name="without factorization", x=epsilons_text, y=dummy, mode="lines", line=attr(color="violet")),
        scatter(name="unitvector overall", x=epsilons_text, y=dummy, mode="lines", line=attr(color="midnightblue")),
        scatter(name="unitvector verification", x=epsilons_text, y=dummy, mode="lines", line=attr(color="lightseagreen")),
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
    savefig(p, "test/plot23.svg")
    p
end

block_elimination_nnenum("benchmarks/digits/digit-net_784x32x256x256x256x256x256x10.onnx", "benchmarks/digits_reduced", nnenum=true)
