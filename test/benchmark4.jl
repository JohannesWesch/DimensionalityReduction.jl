using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function vnnlib(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    vnnlib5 = ["benchmarks/digits/dim64/prop_0_0.05.vnnlib",
            "benchmarks/digits/dim64/prop_0_0.10.vnnlib",
             "benchmarks/digits/dim64/prop_0_0.20.vnnlib",
             "benchmarks/digits/dim64/prop_0_0.40.vnnlib",
             "benchmarks/digits/dim64/prop_0_0.60.vnnlib",
             "benchmarks/digits/dim64/prop_0_1.00.vnnlib",
             "benchmarks/digits/dim64/prop_0_2.00.vnnlib",
             "benchmarks/digits/dim64/prop_0_3.00.vnnlib",
             "benchmarks/digits/dim64/prop_0_4.00.vnnlib",
             "benchmarks/digits/dim64/prop_0_5.00.vnnlib",
             "benchmarks/digits/dim64/prop_0_6.00.vnnlib",
             "benchmarks/digits/dim64/prop_0_7.00.vnnlib",
             ] 
    vnnlib6 = ["benchmarks/digits/dim64/prop_1_0.05.vnnlib",
             "benchmarks/digits/dim64/prop_1_0.10.vnnlib",
             "benchmarks/digits/dim64/prop_1_0.20.vnnlib",
             "benchmarks/digits/dim64/prop_1_0.40.vnnlib",
             "benchmarks/digits/dim64/prop_1_0.60.vnnlib",
             "benchmarks/digits/dim64/prop_1_1.00.vnnlib",
             "benchmarks/digits/dim64/prop_1_2.00.vnnlib",
             "benchmarks/digits/dim64/prop_1_3.00.vnnlib",
             "benchmarks/digits/dim64/prop_1_4.00.vnnlib",
             "benchmarks/digits/dim64/prop_1_5.00.vnnlib",
             "benchmarks/digits/dim64/prop_1_6.00.vnnlib",
             "benchmarks/digits/dim64/prop_1_7.00.vnnlib",
             ] 
    vnnlib7 = ["benchmarks/digits/dim64/prop_2_0.05.vnnlib",
             "benchmarks/digits/dim64/prop_2_0.10.vnnlib",
             "benchmarks/digits/dim64/prop_2_0.20.vnnlib",
             "benchmarks/digits/dim64/prop_2_0.40.vnnlib",
             "benchmarks/digits/dim64/prop_2_0.60.vnnlib",
             "benchmarks/digits/dim64/prop_2_1.00.vnnlib",
             "benchmarks/digits/dim64/prop_2_2.00.vnnlib",
             "benchmarks/digits/dim64/prop_2_3.00.vnnlib",
             "benchmarks/digits/dim64/prop_2_4.00.vnnlib",
             "benchmarks/digits/dim64/prop_2_5.00.vnnlib",
             "benchmarks/digits/dim64/prop_2_6.00.vnnlib",
             "benchmarks/digits/dim64/prop_2_7.00.vnnlib",
             ] 
    vnnlib8 = ["benchmarks/digits/dim64/prop_3_0.05.vnnlib",
             "benchmarks/digits/dim64/prop_3_0.10.vnnlib",
             "benchmarks/digits/dim64/prop_3_0.20.vnnlib",
             "benchmarks/digits/dim64/prop_3_0.40.vnnlib",
             "benchmarks/digits/dim64/prop_3_0.60.vnnlib",
             "benchmarks/digits/dim64/prop_3_1.00.vnnlib",
             "benchmarks/digits/dim64/prop_3_2.00.vnnlib",
             "benchmarks/digits/dim64/prop_3_3.00.vnnlib",
             "benchmarks/digits/dim64/prop_3_4.00.vnnlib",
             "benchmarks/digits/dim64/prop_3_5.00.vnnlib",
             "benchmarks/digits/dim64/prop_3_6.00.vnnlib",
             "benchmarks/digits/dim64/prop_3_7.00.vnnlib",
             ] 

             vnnlibx = [0.05, 0.2, 0.4, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    constraints5 = zeros(13,)
    constraints6 = zeros(13,)
    constraints7 = zeros(13,)
    constraints8 = zeros(13,)

    for (i, vnn) in enumerate(vnnlib5)
        result = reduce(onnx_input, vnn, output; doreduction, method=1, d_to_reduce=0,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints5[i] = result[7]
    end

    for (i, vnn) in enumerate(vnnlib6)
        result = reduce(onnx_input, vnn, output; doreduction, method=1, d_to_reduce=0,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints6[i] = result[7]
    end

    for (i, vnn) in enumerate(vnnlib7)
        result = reduce(onnx_input, vnn, output; doreduction, method=1, d_to_reduce=0,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints7[i] = result[7]
    end

    for (i, vnn) in enumerate(vnnlib8)
        result = reduce(onnx_input, vnn, output; doreduction, method=1, d_to_reduce=0,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints8[i] = result[7]
    end

    p = plot([
        scatter(name="Minimal Dimension 5", x=vnnlibx, y=constraints5, marker_color="lightseagreen", mode="markers+lines", line=attr(width=3)), #, text=constraints5, textposition="outside"
        scatter(name="Minimal Dimension 6", x=vnnlibx, y=constraints6, marker_color="lightgray", mode="markers+lines", line=attr(width=3)), #, text=constraints6, textposition="outside"
        scatter(name="Minimal Dimension 7", x=vnnlibx, y=constraints7, marker_color="lightcoral", mode="markers+lines", line=attr(width=3)), #, text=constraints7, textposition="outside"
        scatter(name="Minimal Dimension 8", x=vnnlibx, y=constraints8, marker_color="lightsteelblue", mode="markers+lines", line=attr(width=3)),
    ], Layout(yaxis=attr(title="Minimal Dimension",
                        linecolor="black",
                        showgrid=true,
                        gridcolor="lightslategrey",
                        gridwidth=0.2
    ),
    xaxis=attr(title="Epsilon", linecolor="black", showgrid=true, gridcolor="lightslategrey", gridwidth=0.2
    ), showlegend=false, plot_bgcolor="white",))
    relayout!(p, barmode="group")
    savefig(p, "test/plot4.svg")
    p
end

vnnlib("benchmarks/digits/digit-net_64x32x128x128x128x128x128x10.onnx", "benchmarks/digits_reduced")