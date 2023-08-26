using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block_elimination_time(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims5 = [1,2,3,4,5,6,7,8,9,10] 
    dims6 = [1,2,3,4,5,6,7,8,9,10] 
    dims7 = [1,2,3,4,5,6,7,8,9] 
    dims764 = [1,2,3,4,5,6,7,8,9,10]
    dims864 = [1,2,3,4,5,6,7,8,9,10]

    constraints5 = zeros(10,)
    constraints6 = zeros(10,)
    constraints7 = zeros(10,)
    constraints764 = zeros(10,)
    constraints864 = zeros(10,)

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

    for (i, dim) in enumerate(dims764)
        result = reduce("benchmarks/digits/digit-net_64x4.onnx", "benchmarks/digits/dim64/prop_6_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints764[i] = result[5]
    end

    for (i, dim) in enumerate(dims864)
        result = reduce("benchmarks/digits/digit-net_64x4.onnx", "benchmarks/digits/dim64/prop_5_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints864[i] = result[5]
    end


    p = plot([
        scatter(name="16-5", x=dims5, y=constraints5, marker=attr(color="rosybrown", size=8), mode="markers+lines", line=attr(width=6)), #, text=constraints5, textposition="outside"
        scatter(name="16-6", x=dims6, y=constraints6, marker=attr(color="lightsalmon", size=8), mode="markers+lines", line=attr(width=6)), #, text=constraints6, textposition="outside"
        scatter(name="16-7", x=dims7, y=constraints7, marker=attr(color="indianred", size=8), mode="markers+lines", line=attr(width=6)), #, text=constraints7, textposition="outside"
        scatter(name="64-7", x=dims764, y=constraints764, marker=attr(color="lightgreen", size=8), mode="markers+lines", line=attr(width=6)), #, text=constraints7, textposition="outside"
        scatter(name="64-8", x=dims864, y=constraints864, marker=attr(color="lightskyblue", size=8), mode="markers+lines", line=attr(width=6)), #, text=constraints7, textposition="outside"
    ], Layout(yaxis=attr(title="Runtime (ms)", linecolor="black", type="log", nticks=5,
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