module DimensionalityReduction

using PyCall

include("Onnx.jl")

function reduce(x, y)
    v = Onnx.update_network(x, y)
    return v
end

export reduce

end # module DimensionalityReduction
