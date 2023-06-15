using VNNLib

f, n_input, n_output = get_ast("benchmarks/mnistfc/prop_0_0.03.vnnlib")
# network = load_network("benchmarks/mnistfc/mnist-net_256x4.onnx")

println(n_input)