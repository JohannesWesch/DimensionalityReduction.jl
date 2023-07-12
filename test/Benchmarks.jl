using PlotlyJS
include("../src/DimensionalityReduction.jl")
import .DimensionalityReduction: reduce

function stars_seconds(dims)
    n = size(dims, 1)

    stars = zeros(n,)
    seconds = zeros(n,)

    for (i, dim) in enumerate(dims)
        result = reduce("benchmarks/digits/digit-net_64x4.onnx", 
            "benchmarks/digits/dim64/prop_0_0.60.vnnlib",
            "benchmarks/digits_reduced", method=2, d_to_reduce=dim, nnenum=true)
        stars[i] = result[3]
        seconds[i] = result[4]
    end

    p = plot([
        bar(name="Stars", x=dims, y=stars, marker_color="indianred"),
        bar(name="Seconds", x=dims, y=seconds, marker_color="lightsalmon")
    ], Layout(title_text="Neural Network: 64x32x128x128x128x10 <br>Epsilon: 0.60"))
    relayout!(p, barmode="group")
    p
end

dims = [0, 1, 2]
stars_seconds(dims)
