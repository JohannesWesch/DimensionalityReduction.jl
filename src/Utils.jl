using PyCall

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

py"""
def get_input_constraints(vnnlib_file):
    s = ""
    with open(vnnlib_file, 'r') as file:
        for line in file:
            if 'Output constraints:' in line:
                return s
            s += line
           
    return s

"""

global get_input_constraints = py"get_input_constraints"