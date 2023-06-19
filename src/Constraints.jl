using VNNLib
using PyCall

@pyinclude("src/VNNLibConverter.py")

function get_box_constraints(vnnlib_file)
    
    vnnlib_file_converted = vnnlib_file * "_converted"
    convert_vnnlib(vnnlib_file, vnnlib_file_converted)
    
    f, n_input, _ = get_ast(vnnlib_file_converted)
    global b = []
    
    for (bounds, _, _, _) in f
        global b = bounds[1:n_input, 1:2]
        break
    end
    return b
end

function get_A_b_from_box(box_constraints)
    n_input = size(box_constraints)[1]

    num_constraints = 2 * n_input
    num_variables = n_input

    A = zeros(num_constraints, num_variables)
    b = zeros(num_constraints, 1)

    for i in 1:n_input
        A[i, i] = -1
        A[n_input + i, i] = 1
    end

    for i in 1:n_input
        b[i] = -box_constraints[i, 1]
        b[i + n_input] = box_constraints[i, 2]
    end

    return A, b
end

function convert_vnnlib(vnnlib_file, vnnlib_file_converted)
    s = py"get_input_constraints"(vnnlib_file)
    f = open(vnnlib_file_converted, "w")
    write(f, s)
    close(f)
end