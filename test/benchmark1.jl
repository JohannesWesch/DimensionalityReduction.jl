using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function fourier(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims = [0,1, 2, 3] 
    dims_permute = [0,1, 2,3] 
    dims_redundant = [0,1,2, 3]

    constraints = [32, 240, 14280, 50969124]
    constraints_text = ["32", "240", "14.280", "50.969.124"]
    constraints_permute = [32, 86, 1344, 443732]
    constraints_permute_text = ["32", "86", "1.344", "443.732"]
    constraints_redundant = [32, 86, 252, 670]
    #constraints_redundant_text = ["32", "86", "1.344", "443.732"]
    #[32, 50, 258, 14416, 51695982] minimal dim5


    #=for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_6_0.01.vnnlib", output; doreduction, method=0, d_to_reduce=dim,
        vnnlib, nnenum, factorization=3, dorefinement)
        constraints[i] = result[6]
    end=#

    #=for (i, dim) in enumerate(dims_permute)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_0_5.00.vnnlib", output; doreduction, method=0, d_to_reduce=dim,
        vnnlib, nnenum, factorization=2, dorefinement)
        constraints_permute[i] = result[6]
    end=#

    #=for (i, dim) in enumerate(dims_redundant)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_0_5.00.vnnlib", output; doreduction, method=5, d_to_reduce=dim,
        vnnlib, nnenum, factorization=2, dorefinement)
        constraints_redundant[i] = result[6]
    end=#

    p = plot([
        bar(name="Without Permutation", x=dims, y=constraints, marker_color="indianred", text=constraints_text, textposition="outside"),
        bar(name="With Permutation", x=dims_permute, y=constraints_permute, marker_color="lightsalmon", text=constraints_permute_text, textposition="outside"),
        bar(name="With Redundancy Removal", x=dims_redundant, y=constraints_redundant, marker_color="rosybrown", text=constraints_redundant, textposition="outside"),
    ], Layout(yaxis=attr(type="log", nticks=4, title="Constraints",linecolor="black",
    showgrid=true,
    gridcolor="lightslategrey",
    gridwidth=0.1),
               xaxis=attr(title="Reduced Dimensions",linecolor="black",
               showgrid=true,
               gridcolor="lightslategrey",
               gridwidth=0.1) ,
                 legend=attr(
        x=1,
        y=1.02,
        yanchor="bottom",
        xanchor="right",
        orientation="h"
    ),plot_bgcolor="white")) #title_text="Neural Network: 16x8x64x10 <br>Epsilon: 0.01", 
    relayout!(p, barmode="group")
    savefig(p, "test/plot1.svg")
    p
end

fourier("benchmarks/digits/digit-net_16x2.onnx", "benchmarks/digits_reduced", nnenum = true)