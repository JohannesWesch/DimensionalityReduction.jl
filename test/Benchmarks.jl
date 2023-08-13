using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function stars_seconds(onnx_input, vnnlib_input, output, nn, eps, dims; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)

    n = size(dims, 1)

    stars = zeros(n,)
    seconds = zeros(n,)

    for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, vnnlib_input, output; doreduction, method, d_to_reduce=dim,
        vnnlib, nnenum, factorization, dorefinement)
        stars[i] = result[3]
        seconds[i] = result[4]
    end

    p = plot([
        bar(name="Stars", x=dims, y=stars, marker_color="indianred"),
        bar(name="Seconds", x=dims, y=seconds, marker_color="lightsalmon")
    ], Layout(legend=attr(
        x=1,
        y=1.02,
        yanchor="bottom",
        xanchor="right",
        orientation="h"
    ), title_text="Neural Network: " * nn * "<br>Epsilon: " * eps))
    relayout!(p, barmode="group")
    savefig(p, "plot.svg")
    p
end

function algorithm_constraints_nnenum(onnx_input, vnnlib_input, output,algorithm, nn, eps, dims; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    n = size(dims, 1)

    algorithm = zeros(n,)
    constraints = zeros(n,)
    nnenum_seconds = zeros(n,)

    for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, vnnlib_input, output; doreduction, method, d_to_reduce=dim,
        vnnlib, nnenum, factorization, dorefinement)
        algorithm[i] = result[5]
        constraints[i] = result[6]
        nnenum_seconds[i] = result[4]
    end

    p = plot([
        #bar(name=algorithm, x=dims, y=algorithm, marker_color="indianred"),
        bar(name="Constraints", x=dims, y=constraints, marker_color="lightsalmon"),
        #bar(name="NNEnum", x=dims, y=nnenum_seconds, marker_color="lightseagreen")
    ], Layout(title_text="Neural Network: " * nn * "<br>Epsilon: " * eps * "<br>Algorithm: " * algorithm))
    relayout!(p, barmode="group")
    p
end

function fourier_block(onnx_input, vnnlib_input, output,algorithm_name, nn, eps, dims; doreduction=true,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    n = size(dims, 1)

    algorithm = zeros(n,)
    constraints = zeros(n,)
    nnenum_seconds = zeros(n,)

    constraints_fourier = zeros(n,)

    for (i, dim) in enumerate(dims)
        result = reduce(onnx_input, vnnlib_input, output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization, dorefinement)
        #if (dim <= 2)
            result2 = reduce(onnx_input, vnnlib_input, output; doreduction, method=1, d_to_reduce=dim,
            vnnlib, nnenum, factorization=1, dorefinement)
            constraints_fourier[i] = result2[6]
       # end
        algorithm[i] = result[5]
        constraints[i] = result[6]
        nnenum_seconds[i] = result[4]
    end

    #constraints_fourier[4] = 50972460

    p = plot([
        #bar(name=algorithm_name, x=dims, y=algorithm, marker_color="indianred"),
        bar(name="Block-Elimination", x=dims, y=constraints, marker_color="lightsalmon", text=constraints, textposition="outside"),
        bar(name="Block-Elimination-Permuted", x=dims, y=constraints_fourier, marker_color="lightseagreen", text=constraints_fourier ,textposition="outside"),
        #bar(name="NNEnum", x=dims, y=nnenum_seconds, marker_color="lightseagreen")
],  Layout(title_text="Neural Network: " * nn * "<br>Epsilon: " * eps))
    relayout!(p, barmode="group")
    p
end

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
        #bar(name=algorithm, x=dims, y=algorithm, marker_color="indianred"),
        bar(name="Minimal Dimension 5", x=dims5, y=constraints5, marker_color="indianred"),
        bar(name="Minimal Dimension 6", x=dims6, y=constraints6, marker_color="lightsalmon"),
        bar(name="Minimal Dimension 7", x=dims7, y=constraints7, marker_color="lightseagreen"),
        #bar(name="NNEnum", x=dims, y=nnenum_seconds, marker_color="lightseagreen")
    ], Layout(title_text="Neural Network: 16x8x64x10 <br>Epsilon: 0.01 <br>Algorithm: Block Elimination"))
    relayout!(p, barmode="group")
    p
end

function block_elimination_64(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims6 = [0,1,2,3,4,5,6,7,8,9,10] 
    dims7 = [0,1,2,3,4,5,6,7,8,9] 
    dims8 = [0,1,2,3,4,5,6,7,8] 

    constraints6 = zeros(11,)
    constraints7 = zeros(11,)
    constraints8 = zeros(11,)

    for (i, dim) in enumerate(dims6)
        result = reduce(onnx_input, "benchmarks/digits/dim64/prop_6_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints6[i] = result[6]
    end

    for (i, dim) in enumerate(dims7)
        result = reduce(onnx_input, "benchmarks/digits/dim64/prop_0_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints7[i] = result[6]
    end

    for (i, dim) in enumerate(dims8)
        result = reduce(onnx_input, "benchmarks/digits/dim64/prop_4_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints8[i] = result[6]
    end

    p = plot([
        #bar(name=algorithm, x=dims, y=algorithm, marker_color="indianred"),
        bar(name="Minimal Dimension 6", x=dims6, y=constraints6, marker_color="indianred"),
        bar(name="Minimal Dimension 7", x=dims7, y=constraints7, marker_color="lightsalmon"),
        bar(name="Minimal Dimension 8", x=dims8, y=constraints8, marker_color="lightseagreen"),
        #bar(name="NNEnum", x=dims, y=nnenum_seconds, marker_color="lightseagreen")
    ], Layout(title_text="Neural Network: 64x8x64x64x64x64x64x10 <br>Epsilon: 0.01 <br>Algorithm: Block Elimination"))
    relayout!(p, barmode="group")
    p
end

function block_elimination_64_1(onnx_input, output; doreduction=true, method=0,
    vnnlib=false, nnenum=false, factorization=0, dorefinement=false)
    
    dims6 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] 

    constraints6 = zeros(17,)

    for (i, dim) in enumerate(dims6)
        result = reduce(onnx_input, "benchmarks/digits/dim64/prop_6_0.01.vnnlib", output; doreduction, method=1, d_to_reduce=dim,
        vnnlib, nnenum, factorization=1, dorefinement)
        constraints6[i] = result[6]
    end

    p = plot([
        #bar(name=algorithm, x=dims, y=algorithm, marker_color="indianred"),
        bar(name="Minimal Dimension 6", x=dims6, y=constraints6, marker_color="lightseagreen"),
        #bar(name="NNEnum", x=dims, y=nnenum_seconds, marker_color="lightseagreen")
    ], Layout(title_text="Neural Network: 64x8x64x64x64x64x64x10 <br>Epsilon: 0.01 <br>Algorithm: Block Elimination"))
    relayout!(p, barmode="group")
    p
end

nn = "64x8x64x64x64x64x64x10"
#nn = "64x32x128x128x128x128x128x10"
algorithm = "Block Elimination"
#nn = "16x8x64x64x64x10"
eps = "0.01"
dims = [0, 1, 2]

stars_seconds("benchmarks/digits/digit-net_64x32x128x128x128x10.onnx", 
"benchmarks/digits/dim64/prop_0_0.20.vnnlib",
"benchmarks/digits_reduced", nn, eps, dims, method=2, nnenum=true)

#=stars_seconds("benchmarks/digits/digit-net_64x6.onnx", 
"benchmarks/digits/dim64/prop_0_0.01.vnnlib",
"benchmarks/digits_reduced", nn, eps, dims, method=1, nnenum=true, factorization = 1)=#

#block_elimination("benchmarks/digits/digit-net_16x2.onnx", "benchmarks/digits_reduced", nnenum = true)

#block_elimination_64("benchmarks/digits/digit-net_64x6.onnx", "benchmarks/digits_reduced", nnenum = true)

#block_elimination_64_1("benchmarks/digits/digit-net_64x6.onnx", "benchmarks/digits_reduced", nnenum = true)

#=algorithm_constraints_nnenum("benchmarks/digits/digit-net_16x4.onnx", 
"benchmarks/digits/dim16/prop_0_0.20.vnnlib",
"benchmarks/digits_reduced",algorithm, nn, eps, dims, method=1, nnenum=true)=#

#=fourier_block("benchmarks/digits/digit-net_16x4.onnx", 
"benchmarks/digits/dim16/prop_0_0.20.vnnlib",
"benchmarks/digits_reduced",algorithm, nn, eps, dims, nnenum=true)=#

# to save plots to svg format
#savefig(p, "plot.svg")