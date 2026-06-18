import argparse
import random

import numpy as np

from modiscolite import tfmodisco
from modiscolite import util
from modiscolite.util import calculate_window_offsets


def _load_npz_array(path):
	data = np.load(path)
	if len(data.files) != 1:
		raise ValueError("{} must contain exactly one array".format(path))

	return data[data.files[0]]


def _load_cli_style_inputs(one_hot_path, hypothetical_contribs_path, window):
	one_hot = _load_npz_array(one_hot_path)
	hypothetical_contribs = _load_npz_array(hypothetical_contribs_path)

	center = one_hot.shape[2] // 2
	start, end = calculate_window_offsets(center, window)
	one_hot = one_hot[:, :, start:end].transpose(0, 2, 1)
	hypothetical_contribs = hypothetical_contribs[:, :, start:end].transpose(0, 2, 1)

	return one_hot.astype("float32"), hypothetical_contribs.astype("float32")


def main():
	parser = argparse.ArgumentParser(
		description="Run a profiled TF-MoDISco baseline on NPZ inputs.")
	parser.add_argument("--one-hot", required=True)
	parser.add_argument("--hypothetical-contribs", required=True)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--max-seqlets-per-metacluster", type=int, default=2000)
	parser.add_argument("--window", type=int, default=200)
	parser.add_argument("--sliding-window-size", type=int, default=20)
	parser.add_argument("--flank-size", type=int, default=5)
	parser.add_argument("--trim-to-window-size", type=int, default=30)
	parser.add_argument("--initial-flank-to-add", type=int, default=10)
	parser.add_argument("--final-flank-to-add", type=int, default=0)
	parser.add_argument("--target-seqlet-fdr", type=float, default=0.05)
	parser.add_argument("--n-leiden-runs", type=int, default=50)
	parser.add_argument(
		"--coarse-affinity-backend",
		choices=["auto", "numba", "inverted"],
		default="auto")
	parser.add_argument(
		"--fine-affinity-backend", choices=["max_only", "tensor"],
		default="max_only")
	parser.add_argument(
		"--density-adaptation-backend",
		choices=["auto", "reference", "optimized"], default="auto")
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)

	one_hot, hypothetical_contribs = _load_cli_style_inputs(
		args.one_hot, args.hypothetical_contribs, args.window)
	profiler = util.ProfileRecorder()

	tfmodisco.TFMoDISco(
		one_hot=one_hot,
		hypothetical_contribs=hypothetical_contribs,
		sliding_window_size=args.sliding_window_size,
		flank_size=args.flank_size,
		trim_to_window_size=args.trim_to_window_size,
		initial_flank_to_add=args.initial_flank_to_add,
		final_flank_to_add=args.final_flank_to_add,
		max_seqlets_per_metacluster=args.max_seqlets_per_metacluster,
		target_seqlet_fdr=args.target_seqlet_fdr,
		n_leiden_runs=args.n_leiden_runs,
		coarse_affinity_backend=args.coarse_affinity_backend,
		fine_affinity_backend=args.fine_affinity_backend,
		density_adaptation_backend=args.density_adaptation_backend,
		profile=profiler)

	for line in profiler.format_summary():
		print(line)


if __name__ == "__main__":
	main()
