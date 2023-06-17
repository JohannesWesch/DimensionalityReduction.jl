using VNNLib

function get_box_constraints(vnnlib_file)
    f, n_input, _ = get_ast(vnnlib_file)
    global b = []
    
    for (bounds, _, _, _) in f
        global b = bounds[1:n_input, 1:2]
        break
    end
    return b, n_input
end

function get_A_b_from_box(box_constraints)
    n_input = size(box_constraints)[1]
    print(n_input)

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