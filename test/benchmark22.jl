#example where the approach also helps solving LPs when checking one-sided ReLUs

using PlotlyJS
using DelimitedFiles
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block_elimination_nnenum(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
     
    epsilons = ["benchmarks/digits/dim784/prop_7_0.007.vnnlib",
                "benchmarks/digits/dim784/prop_7_0.008.vnnlib",
    ]
    epsilons_text = [0.007, 0.008]

    runtime_customoverall = vec(readdlm("test_vectors\\23-customoverall.txt", '\t', Float64, '\n'))
    runtime_custom = vec(readdlm("test_vectors\\23-custom.txt", '\t', Float64, '\n'))
    runtime_unitvectoroverall = vec(readdlm("test_vectors\\23-unitvectoroverall.txt", '\t', Float64, '\n'))
    runtime_unitvector = vec(readdlm("test_vectors\\23-unitvector.txt", '\t', Float64, '\n'))
    runtime_no = vec(readdlm("test_vectors\\23-no.txt", '\t', Float64, '\n'))
    color_vec1 =  vec(readdlm("test_vectors\\23-colorvec1.txt", '\t', String, '\n'))
    color_vec2 =  vec(readdlm("test_vectors\\23-colorvec1.txt", '\t', String, '\n'))
    dummy = zeros(15,)

    pertubation = readdlm("test_vectors\\PertubationA.txt", '\t', Float64, '\n')

    for (i, epsilon) in enumerate(epsilons)
        if (runtime_customoverall[i] != 0.0)
            continue
        end
        result = reduce(onnx_input, epsilon, output; doreduction, method=4, d_to_reduce=720,
        vnnlib, nnenum, factorization=3, pertubation=pertubation)
        runtime_custom[i] = result[4]
        runtime_customoverall[i] = result[4] + result[5]/1000
        if result[8] == 0
            color_vec1[i] = "orangered"
        end
        open("test_vectors\\23-customoverall.txt", "w") do io
            writedlm(io, runtime_customoverall)
        end
        open("test_vectors\\23-custom.txt", "w") do io
            writedlm(io, runtime_custom)
        end
        open("test_vectors\\23-colorvec1.txt", "w") do io
            writedlm(io, color_vec1)
        end
    end

    for (i, epsilon) in enumerate(epsilons)
        if (runtime_no[i] != 0.0)
            continue
        end
        result = reduce(onnx_input, epsilon, output; doreduction=false, method=2, d_to_reduce=0,
        vnnlib, nnenum, factorization=3, pertubation=pertubation)
        runtime_no[i] = result[4]
        open("test_vectors\\23-no.txt", "w") do io
            writedlm(io, runtime_no)
        end
    end

    for (i, epsilon) in enumerate(epsilons)
        if (runtime_unitvector[i] != 0.0)
            continue
        end
        result = reduce(onnx_input, epsilon, output; doreduction, method=6, d_to_reduce=720,
        vnnlib, nnenum, factorization=3, pertubation=pertubation)
        runtime_custom[i] = result[4]
        runtime_customoverall[i] = result[4] + result[5]/1000
        if result[8] == 0
            color_vec2[i] = "orangered"
        end
        open("test_vectors\\23-unitvectoroverall.txt", "w") do io
            writedlm(io, runtime_unitvectoroverall)
        end
        open("test_vectors\\23-unitvector.txt", "w") do io
            writedlm(io, runtime_unitvector)
        end
        open("test_vectors\\23-colorvec2.txt", "w") do io
            writedlm(io, color_vec2)
        end
    end

    p = plot([
        scatter(name="custom directions overall", x=epsilons_text, y=runtime_customoverall, mode="markers+lines", line=attr(width=8, color="lightseagreen"), marker=attr(color=color_vec1, size=8), showlegend=false),
        scatter(name="custom directions verification", x=epsilons_text, y=runtime_custom, mode="markers+lines",  line=attr(width=4, color="lightseagreen"), marker=attr(color=color_vec1, size=8), showlegend=false),
        scatter(name="without factorization", x=epsilons_text, y=runtime_no, mode="markers+lines",  line=attr(width=8, color="violet"), marker=attr(color=color_vec2, size=8), showlegend=false),

        scatter(name="without factorization", x=epsilons_text, y=dummy, mode="lines", line=attr(width=8, color="violet")),
        scatter(name="custom directions overall", x=epsilons_text, y=dummy, mode="lines", line=attr(width=8, color="lightseagreen")),
        scatter(name="custom directions verification", x=epsilons_text, y=dummy, mode="lines", line=attr(width=4, color="lightseagreen")),
        scatter(name="safe", x=epsilons_text, y=dummy, marker_color="lightgreen", mode="markers"),
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

block_elimination_nnenum("benchmarks/digits/digit-net_784x64x256x256x256x256x256x10.onnx", "benchmarks/digits_reduced", nnenum=true)
