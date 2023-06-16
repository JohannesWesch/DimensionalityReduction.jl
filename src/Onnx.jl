using PyCall

@pyinclude("src/NetworkUpdate.py")

v = py"update_network"("benchmarks/mnistfc/mnist-net_256x4.onnx", "benchmarks/mnistfc_reduced/mnist-net_256x4.onnx")