import nnenum
from nnenum.lp_star import LpStar
from nnenum.specification import Specification
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.onnx_network import load_onnx_network
from nnenum.specification import Specification, DisjunctiveSpec

def generate_input_spec(bounds, matrix, bias, ninputs):
	init_star = LpStar(np.eye(ninputs, dtype=np.float32), np.zeros(ninputs, dtype=np.float32), bounds)
	for a, b in zip(matrix, bias):
		init_star.lpi.add_dense_row(a, b)
	return init_star


Settings.UNDERFLOW_BEHAVIOR = "warn"
Settings.CHECK_SINGLE_THREAD_BLAS = False
Settings.SKIP_CONSTRAINT_NORMALIZATION = False
Settings.PRINT_PROGRESS = True
Settings.PRINT_OUTPUT = True
#Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
Settings.NUM_PROCESSES = 1

# Load network from file path
network = load_onnx_network(filepath)
# Generate input specification stating x is within bounds and matrix * x <= bias and x \in R^ninputs
input_spec = generate_input_spec(bounds, matrix, bias, ninputs)
# Create output specification stating output_matrix * y <= output_bias where y = network(x)
output_spec = Specification(output_matrix, output_bias)
# ALTERNATIVE: Disjuntive specifications:
output_spec = DisjunctiveSpec([
	Specification(output_matrix1, output_bias1),
	Specification(output_matrix2, output_bias2)
])

enumerate_network(input_spec, network, output_spec)