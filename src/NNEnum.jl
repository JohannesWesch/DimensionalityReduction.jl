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
	Settings.NUM_PROCESSES = 0
	Settings.COMPRESS_INIT_BOX = True
	Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
	#Settings.BRANCH_MODE = Settings.BRANCH_EXACT
	Settings.TRY_QUICK_OVERAPPROX = False

	Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
	Settings.SPLIT_IF_IDLE = False
	Settings.OVERAPPROX_LP_TIMEOUT = np.inf
	Settings.TIMING_STATS = True

	# contraction doesn't help in high dimensions
	#Settings.OVERAPPROX_CONTRACT_ZONO_LP = False
	Settings.CONTRACT_ZONOTOPE = False
	Settings.CONTRACT_ZONOTOPE_LP = False

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
	result = enumerate_network(init_star, network, spec)
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
		return (result.result_str, result.total_stars, cex, result.total_secs)
	else:
		sys.stdout.flush()
		return (result.result_str, result.total_stars, None, result.total_secs)
"""
		global run_nnenum = py"run_nnenum"
end

	export run_nnenum
end