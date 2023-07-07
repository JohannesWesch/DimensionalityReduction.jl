function create_folder_path(onnx_input, vnnlib_input, output)
    onnx = split(onnx_input, "/")
    onnx = split(onnx[end], ".")
    onnx = join(onnx[1:end-1], ".")

    vnnlib = split(vnnlib_input, "/")
    vnnlib = split(vnnlib[end], ".")
    vnnlib = join(vnnlib[1:end-1], ".")

    folder = output * "/" * onnx * "/" * vnnlib
    mkpath(folder)
    return folder
end

function vnnlib_path(onnx_input, vnnlib_input, output, approx)
    folder = create_folder_path(onnx_input, vnnlib_input, output)
    return folder * "/" * string(approx) * "_" * split(vnnlib_input, "/")[end]
end

function onnx_path(onnx_input, vnnlib_input, output)
    folder = create_folder_path(onnx_input, vnnlib_input, output)
    return folder * "/" * split(onnx_input, "/")[end]
end

function create_output_matrix(vnnlib, n_output)
    global property = 0
    open(vnnlib) do io
        firstline = readline(io) # throw out the first line
        global property = parse(Int, firstline[end-1])
    end

    property += 1
    matrix = zeros(n_output - 1,n_output)
    vector = zeros(n_output - 1,)
    matrix[:, property] .= 1
    
    for i in 1:n_output - 1
        for j in 1:n_output
            if j < property && i == j
                matrix[i, j] = -1
            elseif j > property && (i + 1) == j
                matrix[i, j] = -1
            end
        end
    end

    disjunctions = []
    for i in 1:n_output - 1
        push!(disjunctions, (matrix[i:i,1:end], [0]))
    end
    return disjunctions
end

function round(M)
    M[abs.(M) .< 0.00001] .= 0
    return M
end

function get_new_dim(d_old, d_min, d_reduced)
    d_new = d_old
    if (d_old - d_reduced < d_min)
        error("unimplemented")
    elseif d_reduced == -1
        d_new = d_min
    elseif d_min <= d_old - d_reduced <= d_old
        d_new = d_old - d_reduced
    end
    return d_new
end

function get_permutation(dim₁, dim₂)
    P = zeros(dim₂, dim₂)
    for i in 1:dim₁
        P[i, i] = 1
    end
    for i in reverse(dim₁:dim₂)
        for j in dim₁:dim₂
            if(i+j == dim₁ + dim₂ + 1)
                P[j, i] = 1
            end
        end
    end
    return P
end