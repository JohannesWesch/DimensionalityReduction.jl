using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function block(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims = [0,1, 2, 3, 4, 5, 6, 7, 8] 
    dims_permute = [0,1, 2, 3, 4, 5, 6, 7, 8] 

    constraints = zeros(9,)
    constraints_permute = zeros(9,)

    for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_0_5.00.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=0, dorefinement)
        constraints[i] = result[6]
    end

    for (i, dim) in enumerate(dims_permute)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_0_5.00.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints_permute[i] = result[6]
    end

    p = plot([
        bar(name="Without Permutation", x=dims, y=constraints, marker_color="indianred"), #, text=constraints, textposition="outside"
        bar(name="With Permutation", x=dims_permute, y=constraints_permute, marker_color="lightsalmon"), #, text=constraints_permute, textposition="outside"
    ], Layout(yaxis=attr(title="Constraints"),
               xaxis=attr(title="Reduced Dimensions") ,
                 legend=attr(
        x=1,
        y=1.02,
        yanchor="bottom",
        xanchor="right",
        orientation="h"
    ))) #title_text="Neural Network: 16x8x64x10 <br>Epsilon: 0.01", 
    relayout!(p, barmode="group")
    savefig(p, "test/plot2.svg")
    p
end

block("benchmarks/digits/digit-net_16x2.onnx", "benchmarks/digits_reduced", nnenum = true)