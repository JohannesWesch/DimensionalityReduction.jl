using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block_elimination(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims7 = [0,1,2, 4, 5, 6, 7, 8, 9, 10] 
    dims8 = [0,1,2, 4, 5, 6, 7, 8, 9, 10] 

    constraints7 = zeros(11,)
    constraints8 = zeros(11,)

    for (i, dim) in enumerate(dims7)
        result = reduce("benchmarks/digits/digit-net_64x4.onnx", "benchmarks/digits/dim64/prop_6_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints7[i] = result[6]
    end

    for (i, dim) in enumerate(dims8)
        result = reduce("benchmarks/digits/digit-net_64x4.onnx", "benchmarks/digits/dim64/prop_5_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints8[i] = result[6]
    end

    p = plot([
        bar(name="Minimal Dimension 7", x=dims7, y=constraints7, marker_color="lightgreen"), #, text=constraints5, textposition="outside"
        bar(name="Minimal Dimension 8", x=dims8, y=constraints8, marker_color="lightskyblue") #, text=constraints6, textposition="outside"
    ], Layout(yaxis=attr(title="Constraints", linecolor="black",
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
    savefig(p, "test/plot6.svg")
    p
end

block_elimination("benchmarks/digits/digit-net_16x2.onnx", "benchmarks/digits_reduced")