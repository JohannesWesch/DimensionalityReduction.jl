using VNNLib

function get_box_constraints(vnnlib_file)

    s = get_input_constraints(vnnlib_file)
    path, f = mktemp("benchmarks/buffer"; cleanup=true)
    write(f, s)
    close(f)
    f, n_input, n_output = get_ast(path)
    global b = []
    
    for (bounds, _, _, _) in f
        global b = bounds[1:n_input, 1:2]
        break
    end
    return b, n_input, n_output
end

function get_A_b_from_box(box_constraints)
    n_input = size(box_constraints)[1]

    num_constraints = 2 * n_input
    num_variables = n_input

    A = zeros(num_constraints, num_variables)
    b = zeros(num_constraints,)

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

function get_A_b_from_box_alternating(box_constraints)
    n_input = size(box_constraints)[1]

    num_constraints = 2 * n_input
    num_variables = n_input

    A = zeros(num_constraints, num_variables)
    b = zeros(num_constraints,)

    for i in 1:n_input
        A[2 * i - 1, i] = -1
        A[2 * i, i] = 1
    end

    for i in 1:n_input
        b[2*i - 1] = -box_constraints[i, 1]
        b[2 *i] = box_constraints[i, 2]
    end

    return A, b
end

function get_input_constraints(vnnlib_file)
    s = ""
    for line in eachline(vnnlib_file, keep=true)
        if occursin("Output constraints:", line)
            return s
        end
        s *= line
    end
    return s
end