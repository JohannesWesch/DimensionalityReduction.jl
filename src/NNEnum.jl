module NNEnum
	using PyCall

	run_nnenum = nothing

	function __init__()
		# Python Setup
		nnenum_path = joinpath(@__DIR__, "./../nnenum/src")
		py"""
		import sys
		def append_python_path(path):
			sys.path.append(path)
		"""
		append_python_path = py"append_python_path"
		append_python_path(string(nnenum_path))
py"""
import argparse
import numpy as np
import pickle
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path

from nnenum.enumerate import enumerate_network
from nnenum.lp_star import LpStar
from nnenum.onnx_network import load_onnx_network
from nnenum.settings import Settings
from nnenum.specification import DisjunctiveSpec, Specification

def prepare_star(star):
	# Extract Ax <= b
	A = star.lpi.get_constraints_csr().todense()
	b = star.lpi.get_rhs()
	# Extract M*x + c
	M = star.a_mat
	c = star.bias
	# Compute bounds
	dims = star.lpi.get_num_cols()
	should_skip = np.zeros((dims, 2), dtype=bool)
	bounds = star.update_input_box_bounds_old(None, should_skip)
	return (
		A, b,
		M, c,
		bounds,
		star.counter_example
	)

def run_nnenum(model, lb, ub, A_input, b_input, disjunction):
	Settings.UNDERFLOW_BEHAVIOR = "warn"
	# TODO(steuber): Seem to have numerical issue here?
	Settings.SKIP_CONSTRAINT_NORMALIZATION = False
	Settings.PRINT_PROGRESS = True
	Settings.PRINT_OUTPUT = True
	#Settings.RESULT_SAVE_COUNTER_STARS = True
	#Settings.INPUT_SPACE_MINIMIZATION = False
	#??
	Settings.FIND_CONCRETE_COUNTEREXAMPLES = True
	Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
	Settings.NUM_PROCESSES = 0
	#Settings.CHECK_SINGLE_THREAD_BLAS = False
	#Settings.TRY_QUICK_OVERAPPROX = False
	#Settings.SINGLE_SET = False
	
	network = load_onnx_network(model)
	ninputs = A_input.shape[1]

	#b_output+=1e-3
	#b_input+=1e-3

	init_box = np.array(
		list(zip(lb.flatten(), ub.flatten())),
		dtype=np.float32,
	)
	init_star = LpStar(
		np.eye(ninputs, dtype=np.float32), np.zeros(ninputs, dtype=np.float32), init_box
	)
	for a, b in zip(A_input, b_input):
		a_ = a.reshape(network.get_input_shape()).flatten("F")
		init_star.lpi.add_dense_row(a_, b)

	spec_list = []
	for (A_mixed, b_mixed) in disjunction:
		spec_list.append(Specification(A_mixed, b_mixed))
	print("[NNENUM] Spec list length: ", len(spec_list))
	spec = DisjunctiveSpec(spec_list)

	print("[NNENUM] Enumeration in progress... ")
	result = next(enumerate_network(init_star, network, spec))
	print("\n[NNENUM] Result: ")
	print(result)
	print("[NNENUM] Enumeration finished.")
	print(result.result_str)
	cex = None
	counterex_stars = []
	if result.cinput is not None:
		cex = (
			np.array(list(result.cinput))
			.astype(np.float32)
			.reshape(network.get_input_shape())
		)
		print(f"[NNENUM] Found counter-example stars: {len(result.stars)}")
		sys.stdout.flush()
		return (result.result_str, result.total_stars, cex)
	else:
		sys.stdout.flush()
		return (result.result_str, result.total_stars, None)
"""
		global run_nnenum = py"run_nnenum"
	end

	export run_nnenum

	# function to_status(status :: String)
	# 	if status == "safe"
	# 		return Safe
	# 	elseif startswith(status,"unsafe")
	# 		return Unsafe
	# 	else
	# 		return Unknown
	# 	end
	# end

	# function verify_enumerative(model, olnnv_query :: OlnnvQuery)
	# 	print_msg("[NNENUM] Running nnenum now...")
	# 	lb = [b[1] for b in olnnv_query.bounds]
	# 	ub = [b[2] for b in olnnv_query.bounds]
	# 	print_msg("[NNENUM] lb: ", lb)
	# 	print_msg("[NNENUM] ub: ", ub)
	# 	res, _ = iterate(run_nnenum(model, lb, ub, olnnv_query.input_matrix, olnnv_query.input_bias, olnnv_query.disjunction, false))
	# 	if isnothing(res)
	# 		return OlnnvResult()
	# 	else
	# 		return OlnnvResult(to_status(res[1]),res[2],map(Star,res[3][2]))
	# 	end
	# end

end