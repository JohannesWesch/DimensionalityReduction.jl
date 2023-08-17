using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block_elimination_time(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims5 = [0,1,2,3,4,5,6,7,8,9,10] 
    dims6 = [0,1, 2]
    #[0,1,2,3,4,5,6,7,8,9,10] 
    dims7 = [0, 1, 2]
    #[0,1,2,3,4,5,6,7,8,9] 
    #dims764 = [0]
    #[0,1,2,3,4,5,6,7,8,9,10]
    dims864 = [0, 1, 2]
    #[0,1,2,3,4,5,6,7,8,9,10]

    constraints5 = zeros(11,)
    constraints6 = zeros(11,)
    constraints7 = zeros(11,)
    #constraints764 = zeros(11,)
    constraints864 = zeros(11,)

    for (i, dim) in enumerate(dims5)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_2_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints5[i] = result[4]
    end

    for (i, dim) in enumerate(dims6)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_0_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints6[i] = result[4]
    end

    for (i, dim) in enumerate(dims7)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_0_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints7[i] = result[4]
    end

    #=for (i, dim) in enumerate(dims764)
        result = reduce("benchmarks/digits/digit-net_64x4.onnx", "benchmarks/digits/dim64/prop_6_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints764[i] = result[4]
    end=#

    for (i, dim) in enumerate(dims864)
        result = reduce("benchmarks/digits/digit-net_64x8x128x10.onnx", "benchmarks/digits/dim64/prop_4_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints864[i] = result[4]
    end


    p = plot([
        scatter(name="16-5", x=dims5, y=constraints5, marker_color="lightgoldenrodyellow", mode="markers+lines", line=attr(width=3)), #, text=constraints5, textposition="outside"
        scatter(name="16-6", x=dims6, y=constraints6, marker_color="lightsalmon", mode="markers+lines", line=attr(width=3)), #, text=constraints6, textposition="outside"
        scatter(name="16-7", x=dims7, y=constraints7, marker_color="indianred", mode="markers+lines", line=attr(width=3)), #, text=constraints7, textposition="outside"
        #scatter(name="64-7", x=dims764, y=constraints764, marker_color="lightgreen", mode="markers+lines", line=attr(width=3)), #, text=constraints7, textposition="outside"
        scatter(name="64-8", x=dims864, y=constraints864, marker_color="lightskyblue", mode="markers+lines", line=attr(width=3)), #, text=constraints7, textposition="outside"
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
    savefig(p, "test/plot7.svg")
    p
end

block_elimination_time("benchmarks/digits/digit-net_16x6.onnx", "benchmarks/digits_reduced", nnenum=true)