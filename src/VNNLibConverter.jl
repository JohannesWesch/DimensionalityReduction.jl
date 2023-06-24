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