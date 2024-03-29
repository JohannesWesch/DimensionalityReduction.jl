using PyCall

py"""
import numpy as np
import onnx
import numpy as np

def propagate_box(weights, box_constraints):
    num_input_neurons = weights.shape[1]
    num_output_neurons = weights.shape[0]
    activation = np.zeros((num_output_neurons, 2))

    for i in range(0, num_output_neurons):
        for j in range(0, num_input_neurons):
            if weights[i][j] >= 0:
                activation[i][0] += box_constraints[i][0] * weights[i][j]
                activation[i][1] += box_constraints[i][1] * weights[i][j]
            else:
                activation[i][0] += box_constraints[i][1] * weights[i][j]
                activation[i][1] += box_constraints[i][0] * weights[i][j]
    return activation


def remove_zero_activation_weights(weights, box_constraints):
    weights = np.array(weights)
    num_input_neurons = weights.shape[1]
    num_output_neurons = weights.shape[0]
    activation = propagate_box(weights, box_constraints)
    for i in range(0, num_output_neurons):
        if activation[i][1] <= 0:
            weights[i] = np.zeros(num_input_neurons)

    return weights

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

def update_network(onnx_input_filename, onnx_output_filename, new_weights, first_matrix):
    # load network
    model = onnx.load(onnx_input_filename)

    # weight update
    name = model.graph.initializer[first_matrix].name
    new_weights = new_weights.astype(np.single)
    tensor = onnx.numpy_helper.from_array(new_weights)

    model.graph.initializer[first_matrix].CopyFrom(tensor)
    model.graph.initializer[first_matrix].name = name

    new_input_dim = new_weights.shape[1]

    # update input dim
    _input = model.graph.input[0]
    old_input_dim, _ = get_num_inputs_outputs(model)

    for dim in _input.type.tensor_type.shape.dim:
        if dim.dim_value == old_input_dim:
            dim.dim_value = new_input_dim

    onnx.save(model, onnx_output_filename)
    return

def get_w(onnx_input, first_matrix):
    model = onnx.load(onnx_input)
    old_input_dim, _ = get_num_inputs_outputs(model)
    init = model.graph.initializer[first_matrix] # get first weight matrix
    w = onnx.numpy_helper.to_array(init)
    w = np.array(w)
    return w
"""

global remove_zero_activation_weights = py"remove_zero_activation_weights"
global get_w = py"get_w"
global update_network = py"update_network"