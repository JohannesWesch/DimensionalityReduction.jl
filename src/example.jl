using VNNLib

# "benchmarks/mnistfc_reduced/test.vnnlib"
# "benchmarks/mnistfc/prop_0_0.03.vnnlib"
f, n_input, n_output = get_ast("benchmarks/mnistfc_reduced/test.vnnlib")

print("Number of inputs: ", n_input, "\n")
print("Number of outputs: ", n_output, "\n")

b = []

for (bounds, matrix, bias, num) in f
    global b = bounds[1:n_input, 1:2]
    break
end

print("Bounds of inputs: ", b, "\n")