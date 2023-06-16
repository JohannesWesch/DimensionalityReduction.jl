module DimensionalityReduction

using PyCall

function reduce(x, y)
    v = Onnx.update_network(x, y)
    return v
end

export reduce

end # module DimensionalityReduction
