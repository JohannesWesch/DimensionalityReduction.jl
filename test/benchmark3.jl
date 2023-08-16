function block_elimination(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims5 = [0,1,2,3,4,5,6,7,8,9,10,11] 
    dims6 = [0,1,2,3,4,5,6,7,8,9,10] 
    dims7 = [0,1,2,3,4,5,6,7,8,9] 

    constraints5 = zeros(12,)
    constraints6 = zeros(12,)
    constraints7 = zeros(12,)

    for (i, dim) in enumerate(dims5)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_6_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints5[i] = result[6]
    end

    for (i, dim) in enumerate(dims6)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_3_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints6[i] = result[6]
    end

    for (i, dim) in enumerate(dims7)
        result = reduce(onnx_input, "benchmarks/digits/dim16/prop_5_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints7[i] = result[6]
    end

    p = plot([
        bar(name="Minimal Dimension 5", x=dims5, y=constraints5, marker_color="lightgoldenrodyellow"), #, text=constraints5, textposition="outside"
        bar(name="Minimal Dimension 6", x=dims6, y=constraints6, marker_color="lightsalmon"), #, text=constraints6, textposition="outside"
        bar(name="Minimal Dimension 7", x=dims7, y=constraints7, marker_color="indianred"), #, text=constraints7, textposition="outside"
    ], Layout(yaxis=attr(title="Constraints"),
    xaxis=attr(title="Reduced Dimensions") ,
      legend=attr(
    x=1,
    y=1.02,
    yanchor="bottom",
    xanchor="right",
    orientation="h"
    )))
    relayout!(p, barmode="group")
    savefig(p, "test/plot3.svg")
    p
end

block_elimination("benchmarks/digits/digit-net_16x2.onnx", "benchmarks/digits_reduced", nnenum = true)