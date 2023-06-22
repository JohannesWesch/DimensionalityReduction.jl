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