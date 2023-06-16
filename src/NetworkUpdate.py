import numpy as np
import onnx

def get_num_inputs_outputs(model):
    inputs = model.graph.input
    assert len(inputs) == 1, f"expected single onnx network input, got: {inputs}"
    outputs = model.graph.output
    assert len(outputs) == 1, f"expected single onnx network output, got: {outputs}"

    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inputs[0].type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value if d.dim_value != 0 else 2 for d in outputs[0].type.tensor_type.shape.dim)

    num_inputs = 1
    num_outputs = 1

    for n in inp_shape:
        num_inputs *= n

    for n in out_shape:
        num_outputs *= n

    return num_inputs, num_outputs

def compact_svd(weights):

    u, s, v = np.linalg.svd(weights)
    s = s.round(10)
    s = s[s != 0]
    new_input_dim = s.size
    u = np.delete(u, np.s_[new_input_dim:], 1)
    # v = np.delete(v, np.s_[new_input_dim:], 0)

    s = np.diagflat(s)
    return u, s, v

def update_network(onnx_input_filename, onnx_output_filename):
    # load network
    model = onnx.load(onnx_input_filename)

    init = model.graph.initializer[1] # get first weight matrix
    w = onnx.numpy_helper.to_array(init)

    u, s, v = compact_svd(w) #u, s are compact and v is a square matrix with the size of the input dimension

    # weight update
    name = model.graph.initializer[1].name

    new_weights = np.matmul(u, s)
    tensor = onnx.numpy_helper.from_array(new_weights)

    model.graph.initializer[1].CopyFrom(tensor)
    model.graph.initializer[1].name = name

    new_input_dim = new_weights.shape[1]

    # update input dim
    _input = model.graph.input[0]
    old_input_dim, _ = get_num_inputs_outputs(model)

    for dim in _input.type.tensor_type.shape.dim:
        if dim.dim_value == old_input_dim:
            dim.dim_value = new_input_dim

    onnx.save(model, onnx_output_filename)
    return v