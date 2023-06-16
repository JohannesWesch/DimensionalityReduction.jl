using VNNLib

function get_c_d(vnnlib_file)
    f, n_input, _ = get_ast(vnnlib_file)

    b =  Matrix{Float64}[]

    for (bounds, matrix, bias, num) in f
    push!(b, bounds)
    break
    end

    b = b[1]
    b = b[:]

    num_constraints = 2 * n_input
    num_variables = n_input

    c = zeros(num_constraints, num_variables)
    d = zeros(num_constraints, 1)

    for i in 1:n_input
    c[i, i] = -1
    c[n_input + i, i] = 1
    end

    for i in 1:n_input
    d[i] = -b[2*i]
    d[i + n_input] = b[2*i + 1]
    end

    return c, d
end

function get_box_constraints(vnnlib_file)
    f, n_input, _ = get_ast(vnnlib_file)

    b =  Matrix{Float64}[]

    for (bounds, matrix, bias, num) in f
    push!(b, bounds)
    break
    end

    b = b[1]
    return b
end