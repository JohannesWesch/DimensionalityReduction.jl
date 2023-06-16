module DimensionalityReduction

using PyCall

@pyinclude("../src/NetworkUpdate.py")

function reduce(x, y)
    v = py"update_network"(x, y)
    return v
end

export reduce

end # module DimensionalityReduction
