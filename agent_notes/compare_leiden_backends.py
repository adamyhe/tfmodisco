"""Compare RAPIDS cuGraph's GPU Leiden implementation against the CPU
`leidenalg` backend this pipeline currently uses, on a REAL affinity graph
captured from an actual TF-MoDISco run, and optionally end-to-end through
the full pipeline.

Context: profiling a real production-scale run (5000 peaks, 20000 max
seqlets, 50 leiden runs -- see agent_notes/leiden_parallel_report.txt)
showed Leiden clustering is ~98% of total wall time, and CPU-side
parallelization across the 50 independent seed runs hits diminishing
returns (and steep memory cost, ~800MB/worker) somewhere around
n_leiden_jobs=8-16, because `leidenalg` holds the GIL and each parallel
worker needs its own full copy of the graph. cuGraph's Leiden runs on the
GPU and only needs the graph resident once, which could sidestep both
limits -- IF its output is close enough to leidenalg's to trust, and IF
it's actually faster on graphs this size/shape (up to ~20k vertices,
~10M edges here).

RESULTS SO FAR (2026-07-18, real L40S run, 5000 peaks / 20000 max seqlets):
a single cugraph.leiden call is ~85x faster than a single leidenalg call
(0.18s vs 15.2s) and lands in the SAME dominant local optimum most
leidenalg seeds find (modularity 0.5188 vs the ~0.5188 that 8/10 leidenalg
seeds also land at; ARI_vs_leidenalg_best=0.90, inside leidenalg's own
seed-to-seed ARI range of 0.65-1.0, just not at the high end). BUT: 3
cugraph trials were bit-identical (trial-to-trial ARI=1.0), which is a red
flag that the script wasn't actually varying any randomness between calls
-- see _detect_cugraph_seed_param below, added to check for and use a real
seed/random-state parameter if cugraph.leiden exposes one, so repeated
trials can actually explore different local optima the way leidenalg's
50-seed strategy does.

KNOWN CAVEAT -- parallel edges: the affinity graph is symmetrized before
reaching LeidenCluster, and the existing igraph-based code
(`cluster.LeidenCluster`) builds its graph from the CSR structure as-is,
which means each undirected edge (u, v) is added TWICE (once per stored
CSR direction) -- igraph keeps these as parallel edges rather than
merging them, and that's what leidenalg has always actually optimized
against in this pipeline. This script mirrors that exactly (same
duplicated edge list) for both leidenalg and cugraph, for a fair
apples-to-apples comparison against production behavior. It is NOT
verified whether cugraph's graph construction treats duplicate undirected
edges as parallel edges (matching igraph) or silently sums/merges them
into one edge with combined weight. If the modularity/ARI numbers look
surprising, check this first.

Three phases (the third is opt-in, via --run-pipeline-comparison, since
it's much more expensive than the first two):
  1. Capture -- run TFMoDISco far enough (with n_seeds=1, so it's cheap)
     to produce a REAL density-adapted affinity graph at production scale,
     and cache it to disk so repeat comparisons don't need to redo this.
  2. Compare (graph-level) -- on that one graph, run leidenalg for
     --n-leiden-runs independent trials to get both a "best" partition and
     a natural seed-to-seed variability baseline, then run cuGraph's
     Leiden --cugraph-trials times (using a real seed/random-state
     parameter if one is found on the installed cugraph.leiden, else
     falling back to identical repeated calls with a clear warning) and
     report wall time, modularity, and clustering agreement (ARI / NMI)
     against both the leidenalg baseline and leidenalg's own seed spread.
  3. Compare (pipeline-level, --run-pipeline-comparison) -- monkeypatch
     cluster.LeidenCluster everywhere (main clustering AND subclustering)
     to route through cugraph.leiden instead of leidenalg, run the FULL
     TFMoDISco() end to end with each backend on the SAME data, and
     compare final pattern counts and per-pattern seqlet-count profiles.
     Graph-level ARI differences might get smoothed over or amplified by
     the downstream pattern-construction/merging/filtering steps -- this
     is the real bar, not just the raw graph partition.
     WARNING: the leidenalg side of this comparison is NOT cheap -- it's
     a real n_leiden_runs-seed run of the full pipeline (the same kind of
     multi-minute-to-hour cost documented in leiden_parallel_report.txt).
     The cugraph side should be fast given the graph-level results above.
     Start with a small --n-leiden-runs (10-20) for a first check before
     using 50 to match the original production report.

Usage (production-scale, matching the report this was motivated by):
    python agent_notes/compare_leiden_backends.py \\
        --n-peaks 5000 --window 400 --max-seqlets-per-metacluster 20000 \\
        --n-leiden-runs 50

Usage (smaller, to sanity check the script itself first):
    python agent_notes/compare_leiden_backends.py \\
        --n-peaks 500 --max-seqlets-per-metacluster 500 --n-leiden-runs 10

Usage (add the full end-to-end pipeline comparison, start small):
    python agent_notes/compare_leiden_backends.py \\
        --n-peaks 5000 --window 400 --max-seqlets-per-metacluster 20000 \\
        --n-leiden-runs 10 --run-pipeline-comparison
"""

import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse

sys.path.insert(0, str(Path(__file__).parent))
from download_and_profile_leiden import download_and_extract, load_arrays  # noqa: E402


def capture_affinity_graph(data_dir, n_peaks, window, max_seqlets_per_metacluster,
        seed, capture_index=0):
    from modiscolite import cluster, tfmodisco

    captured = {}
    call_count = {"n": 0}
    original_leiden_cluster = cluster.LeidenCluster

    def _capture(affinity_mat, n_seeds=2, n_leiden_iterations=-1, n_jobs=1):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx == capture_index and "affmat" not in captured:
            captured["affmat"] = affinity_mat.copy()
        # Keep the whole run cheap: we only want a representative graph,
        # not an actual 50-seed clustering result.
        return original_leiden_cluster(affinity_mat, n_seeds=1,
            n_leiden_iterations=n_leiden_iterations, n_jobs=1)

    cluster.LeidenCluster = _capture
    try:
        seqs_path = Path(data_dir) / "seqs.npy"
        contribs_path = Path(data_dir) / "hyp_contribs.npy"
        one_hot, hyp_contribs = load_arrays(seqs_path, contribs_path, n_peaks,
            window, seed)

        tfmodisco.TFMoDISco(one_hot=one_hot, hypothetical_contribs=hyp_contribs,
            max_seqlets_per_metacluster=max_seqlets_per_metacluster,
            n_leiden_runs=1, verbose=False)
    finally:
        cluster.LeidenCluster = original_leiden_cluster

    if "affmat" not in captured:
        raise RuntimeError(
            f"Never reached capture_index={capture_index} -- only "
            f"{call_count['n']} LeidenCluster call(s) happened. Try a "
            f"smaller --capture-index, or a larger --n-peaks/"
            f"--max-seqlets-per-metacluster so clustering actually runs.")

    return captured["affmat"]


def run_leidenalg_seeds(affmat, n_seeds):
    from modiscolite import cluster as cluster_module

    n_vertices = affmat.shape[0]
    n_cols = affmat.indptr
    sources = np.concatenate([np.ones(n_cols[i+1] - n_cols[i], dtype='int32') * i
        for i in range(n_vertices)])
    targets = affmat.indices
    weights = affmat.data

    results = []
    for seed_idx in range(1, n_seeds + 1):
        seed = seed_idx * 100
        start = time.perf_counter()
        quality, membership = cluster_module._leiden_seed_partition(
            sources, targets, weights, n_vertices, seed, -1)
        wall = time.perf_counter() - start
        results.append({"seed": seed, "quality": float(quality),
            "membership": np.asarray(membership), "wall_seconds": wall})

    return results


# --- cuGraph helpers, shared between the graph-level and pipeline-level
# comparisons below. ---

def _detect_cugraph_seed_param(leiden_fn):
    """Look for a real randomization knob on the installed cugraph.leiden.

    The 3-identical-trials result on 2026-07-18 (ARI=1.0 across trials)
    was a red flag that earlier versions of this script never checked for
    one -- only 'max_iter'/'resolution' were probed. Without a way to vary
    the starting state between calls, repeated trials can't explore
    different local optima the way leidenalg's 50-seed strategy does.
    """
    accepted = set(inspect.signature(leiden_fn).parameters)
    for name in ("random_state", "seed", "random_seed"):
        if name in accepted:
            return name, accepted
    return None, accepted


def _build_cugraph_graph(affmat):
    import cudf
    import cugraph

    n_vertices = affmat.shape[0]
    n_cols = affmat.indptr
    sources = np.concatenate([np.ones(n_cols[i+1] - n_cols[i], dtype='int32') * i
        for i in range(n_vertices)])
    # Mirrors leidenalg's edge list exactly -- see the parallel-edges
    # caveat in the module docstring.
    targets = affmat.indices.astype('int32')
    weights = affmat.data.astype('float64')

    edgelist = cudf.DataFrame({"src": sources, "dst": targets, "weight": weights})
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edgelist, source="src", destination="dst",
        edge_attr="weight", renumber=True)
    return G


def _cugraph_leiden_call(G, seed_param, accepted, seed_value):
    import cugraph

    kwargs = {}
    if "max_iter" in accepted:
        kwargs["max_iter"] = 100
    if "resolution" in accepted:
        kwargs["resolution"] = 1.0
    if seed_param is not None:
        kwargs[seed_param] = seed_value

    try:
        # NOTE: documented as returning (partition_df, modularity_score)
        # on recent RAPIDS releases. If your installed version returns
        # them in the opposite order, swap this line.
        parts_df, modularity = cugraph.leiden(G, **kwargs)
    except TypeError:
        parts_df, modularity = cugraph.leiden(G)

    parts_df = parts_df.sort_values("vertex").reset_index(drop=True)
    # Column name for the partition assignment has varied across RAPIDS
    # versions ('partition' in recent ones) -- pick whichever non-'vertex'
    # column is present rather than hardcoding it.
    partition_col = [c for c in parts_df.columns if c != "vertex"][0]
    membership = parts_df[partition_col].to_numpy()
    return float(modularity), membership


def run_cugraph_leiden(affmat, n_trials=3):
    try:
        import cugraph
    except ImportError as e:
        return None, str(e), None

    G = _build_cugraph_graph(affmat)
    seed_param, accepted = _detect_cugraph_seed_param(cugraph.leiden)

    results = []
    for trial in range(n_trials):
        seed_value = (trial + 1) * 100
        start = time.perf_counter()
        quality, membership = _cugraph_leiden_call(G, seed_param, accepted, seed_value)
        wall = time.perf_counter() - start
        results.append({"trial": trial, "quality": quality, "membership": membership,
            "wall_seconds": wall})

    return results, None, seed_param


def make_cugraph_leiden_cluster():
    """A drop-in replacement for cluster.LeidenCluster backed by
    cugraph.leiden, for monkeypatching into a full pipeline run.

    Preserves the same (affinity_mat, n_seeds, n_leiden_iterations, n_jobs)
    signature and 'try n_seeds candidates, keep the best modularity'
    semantics as the real LeidenCluster, so no calling code needs to
    change -- n_leiden_iterations and n_jobs are accepted but unused (GPU
    calls are already fast and run on a single device, not a CPU process
    pool). Builds the cugraph Graph once and reuses it across the n_seeds
    candidate calls, mirroring how the real LeidenCluster reuses one
    igraph.Graph across its seed loop.
    """
    import cugraph

    seed_param, accepted = _detect_cugraph_seed_param(cugraph.leiden)

    def _leiden_cluster(affinity_mat, n_seeds=2, n_leiden_iterations=-1, n_jobs=1):
        G = _build_cugraph_graph(affinity_mat)

        best_membership = None
        best_quality = None
        for trial in range(max(n_seeds, 1)):
            seed_value = (trial + 1) * 100
            quality, membership = _cugraph_leiden_call(G, seed_param, accepted, seed_value)
            if best_quality is None or quality > best_quality:
                best_quality = quality
                best_membership = membership

        return best_membership

    return _leiden_cluster, seed_param


def run_full_pipeline_comparison(data_dir, n_peaks, window, max_seqlets_per_metacluster,
        n_leiden_runs, seed):
    from modiscolite import cluster, tfmodisco, util

    seqs_path = Path(data_dir) / "seqs.npy"
    contribs_path = Path(data_dir) / "hyp_contribs.npy"
    one_hot, hyp_contribs = load_arrays(seqs_path, contribs_path, n_peaks, window, seed)

    def _run(n_leiden_jobs=1):
        profiler = util.ProfileRecorder()
        start = time.perf_counter()
        pos, neg = tfmodisco.TFMoDISco(one_hot=one_hot, hypothetical_contribs=hyp_contribs,
            max_seqlets_per_metacluster=max_seqlets_per_metacluster,
            n_leiden_runs=n_leiden_runs, n_leiden_jobs=n_leiden_jobs, profile=profiler)
        wall = time.perf_counter() - start
        return {
            "wall_seconds": wall,
            "n_pos": len(pos) if pos else 0,
            "n_neg": len(neg) if neg else 0,
            "pos_sizes": sorted((len(p.seqlets) for p in pos), reverse=True) if pos else [],
            "neg_sizes": sorted((len(p.seqlets) for p in neg), reverse=True) if neg else [],
        }

    print("  [pipeline] running leidenalg (CPU) end to end -- this is the slow part ...")
    cpu_result = _run(n_leiden_jobs=1)

    try:
        cugraph_leiden_cluster, seed_param = make_cugraph_leiden_cluster()
    except ImportError as e:
        return {"cpu": cpu_result, "gpu": None, "gpu_error": str(e), "seed_param": None}

    original_leiden_cluster = cluster.LeidenCluster
    cluster.LeidenCluster = cugraph_leiden_cluster
    try:
        print("  [pipeline] running cuGraph-substituted pipeline end to end ...")
        gpu_result = _run(n_leiden_jobs=1)
    finally:
        cluster.LeidenCluster = original_leiden_cluster

    return {"cpu": cpu_result, "gpu": gpu_result, "gpu_error": None, "seed_param": seed_param}


def summarize(leidenalg_results, cugraph_results, cugraph_error, cugraph_seed_param):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    lines = []
    lines.append("=== leidenalg (CPU, current backend) ===")
    best = max(leidenalg_results, key=lambda r: r["quality"])
    for r in leidenalg_results:
        lines.append(f"  seed={r['seed']}: quality={r['quality']:.6f} "
            f"n_clusters={len(set(r['membership']))} wall={r['wall_seconds']:.3f}s")
    lines.append(f"  best: seed={best['seed']} quality={best['quality']:.6f}")

    if len(leidenalg_results) > 1:
        aris = []
        for i in range(len(leidenalg_results)):
            for j in range(i + 1, len(leidenalg_results)):
                aris.append(adjusted_rand_score(
                    leidenalg_results[i]["membership"],
                    leidenalg_results[j]["membership"]))
        lines.append(f"  seed-to-seed ARI (natural variability baseline): "
            f"mean={np.mean(aris):.4f} min={np.min(aris):.4f} max={np.max(aris):.4f}")

    lines.append("")
    lines.append("=== cuGraph Leiden (GPU) ===")
    if cugraph_results is None:
        lines.append(f"  SKIPPED -- cugraph/cudf not available: {cugraph_error}")
        lines.append("  Install RAPIDS cugraph on a CUDA machine to run this section.")
    else:
        seed_note = cugraph_seed_param or \
            "NONE FOUND on cugraph.leiden -- trials are NOT independently randomized"
        lines.append(f"  seed/random-state parameter used: {seed_note}")
        for r in cugraph_results:
            ari_vs_best = adjusted_rand_score(r["membership"], best["membership"])
            nmi_vs_best = normalized_mutual_info_score(r["membership"], best["membership"])
            lines.append(f"  trial={r['trial']}: quality={r['quality']:.6f} "
                f"n_clusters={len(set(r['membership']))} wall={r['wall_seconds']:.3f}s "
                f"ARI_vs_leidenalg_best={ari_vs_best:.4f} "
                f"NMI_vs_leidenalg_best={nmi_vs_best:.4f}")

        if len(cugraph_results) > 1:
            aris = []
            for i in range(len(cugraph_results)):
                for j in range(i + 1, len(cugraph_results)):
                    aris.append(adjusted_rand_score(
                        cugraph_results[i]["membership"],
                        cugraph_results[j]["membership"]))
            lines.append(f"  cugraph trial-to-trial ARI: mean={np.mean(aris):.4f} "
                f"min={np.min(aris):.4f} max={np.max(aris):.4f}")
            if cugraph_seed_param is None and np.mean(aris) > 0.9999:
                lines.append("  (trial-to-trial ARI is ~1.0 AND no seed parameter was found --")
                lines.append("   these trials are identical repeats, not an independent ensemble.")
                lines.append("   Do not read this as cugraph being deterministic in general.)")

        mean_leidenalg_wall = np.mean([r["wall_seconds"] for r in leidenalg_results])
        mean_cugraph_wall = np.mean([r["wall_seconds"] for r in cugraph_results])
        lines.append("")
        lines.append(f"  mean wall/run: leidenalg={mean_leidenalg_wall:.3f}s "
            f"cugraph={mean_cugraph_wall:.3f}s "
            f"speedup={mean_leidenalg_wall / mean_cugraph_wall:.2f}x")
        lines.append("")
        lines.append("  Interpretation guide: if cuGraph's ARI-vs-leidenalg-best is")
        lines.append("  comparable to leidenalg's OWN seed-to-seed ARI above, cuGraph")
        lines.append("  is landing in the same range of 'plausible good partitions' that")
        lines.append("  leidenalg itself produces across seeds -- not a red flag on its")
        lines.append("  own. If it's substantially lower, treat that as a real behavioral")
        lines.append("  difference to investigate before trusting cuGraph's output.")

    return "\n".join(lines)


def summarize_pipeline_comparison(result):
    lines = []
    lines.append("")
    lines.append("=== Full-pipeline substitution: leidenalg vs cuGraph-everywhere ===")
    cpu = result["cpu"]
    lines.append(f"  leidenalg: wall={cpu['wall_seconds']:.2f}s n_pos={cpu['n_pos']} "
        f"n_neg={cpu['n_neg']}")
    lines.append(f"    pos_pattern_sizes={cpu['pos_sizes']}")

    if result["gpu"] is None:
        lines.append(f"  cuGraph: SKIPPED -- {result['gpu_error']}")
        return "\n".join(lines)

    gpu = result["gpu"]
    seed_note = result["seed_param"] or "NONE FOUND -- see graph-level section above"
    lines.append(f"  cugraph.leiden seed parameter used: {seed_note}")
    lines.append(f"  cuGraph:   wall={gpu['wall_seconds']:.2f}s n_pos={gpu['n_pos']} "
        f"n_neg={gpu['n_neg']}")
    lines.append(f"    pos_pattern_sizes={gpu['pos_sizes']}")
    lines.append(f"  full-pipeline speedup: {cpu['wall_seconds'] / gpu['wall_seconds']:.2f}x")

    counts_match = cpu['n_pos'] == gpu['n_pos'] and cpu['n_neg'] == gpu['n_neg']
    lines.append(f"  pattern count match: {'YES' if counts_match else 'NO -- investigate before trusting'}")
    if counts_match and cpu['pos_sizes'] and cpu['pos_sizes'] == gpu['pos_sizes']:
        lines.append("  pos_pattern_sizes match exactly, in the same sorted order --")
        lines.append("  strong (though not conclusive) sign the substitution is behaving well.")
    elif counts_match:
        lines.append("  pattern counts match but per-pattern sizes differ -- same NUMBER of")
        lines.append("  patterns found, but not necessarily the same patterns. Worth comparing")
        lines.append("  actual pattern content (e.g. via a report or PWM diff) before trusting this.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="agent_notes/data/encode_ENCFF407PRC")
    parser.add_argument("--n-peaks", type=int, default=5000)
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--max-seqlets-per-metacluster", type=int, default=20000)
    parser.add_argument("--n-leiden-runs", type=int, default=50,
        help="Number of independent leidenalg seed trials to run for comparison "
             "(mirrors n_leiden_runs in a real pipeline call). Also used as "
             "n_leiden_runs for --run-pipeline-comparison, where it directly "
             "controls how long the leidenalg side of that comparison takes.")
    parser.add_argument("--cugraph-trials", type=int, default=3,
        help="Number of times to run cugraph.leiden, to gauge its own "
             "run-to-run variability.")
    parser.add_argument("--capture-index", type=int, default=0,
        help="Which LeidenCluster call to capture the affinity graph from "
             "(0 = first call, typically the largest/most representative).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-path", default=None,
        help="Optional path to cache the captured affinity graph (.npz) so "
             "repeat comparisons skip re-running the pipeline.")
    parser.add_argument("--run-pipeline-comparison", action="store_true",
        help="Also run the full TFMoDISco() pipeline end to end with "
             "leidenalg vs. cugraph-substituted-everywhere, and compare "
             "final pattern counts/sizes. The leidenalg side is NOT cheap "
             "-- see the module docstring. Start with a small "
             "--n-leiden-runs before scaling up.")
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args()

    cache_path = Path(args.cache_path) if args.cache_path else \
        Path(args.data_dir) / (
            f"captured_affmat_idx{args.capture_index}"
            f"_peaks{args.n_peaks}_win{args.window}"
            f"_maxseqlets{args.max_seqlets_per_metacluster}_seed{args.seed}.npz")

    if cache_path.exists():
        print(f"Loading cached affinity graph from {cache_path} ...")
        loaded = np.load(cache_path)
        affmat = scipy.sparse.csr_matrix(
            (loaded["data"], loaded["indices"], loaded["indptr"]),
            shape=tuple(loaded["shape"]))
    else:
        download_and_extract(Path(args.data_dir))
        print("Capturing a real affinity graph (n_seeds=1, so this is cheap "
            "relative to a full clustering run) ...")
        affmat = capture_affinity_graph(args.data_dir, args.n_peaks, args.window,
            args.max_seqlets_per_metacluster, args.seed, args.capture_index)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, data=affmat.data, indices=affmat.indices,
            indptr=affmat.indptr, shape=np.array(affmat.shape))
        print(f"Cached to {cache_path}")

    print(f"Graph: {affmat.shape[0]} vertices, {affmat.nnz} edges")

    print(f"Running leidenalg for {args.n_leiden_runs} seed(s) ...")
    leidenalg_results = run_leidenalg_seeds(affmat, args.n_leiden_runs)

    print(f"Running cugraph.leiden for {args.cugraph_trials} trial(s) ...")
    cugraph_results, cugraph_error, cugraph_seed_param = run_cugraph_leiden(
        affmat, args.cugraph_trials)

    report = summarize(leidenalg_results, cugraph_results, cugraph_error, cugraph_seed_param)

    if args.run_pipeline_comparison:
        print("Running full end-to-end pipeline comparison "
            "(leidenalg side is slow -- see module docstring) ...")
        pipeline_result = run_full_pipeline_comparison(args.data_dir, args.n_peaks,
            args.window, args.max_seqlets_per_metacluster, args.n_leiden_runs, args.seed)
        report += "\n" + summarize_pipeline_comparison(pipeline_result)

    print()
    print(report)

    if args.report_path:
        Path(args.report_path).write_text(report)
        print(f"\nReport written to {args.report_path}")


if __name__ == "__main__":
    main()
