"""Compare RAPIDS cuGraph's GPU Leiden implementation against the CPU
`leidenalg` backend this pipeline currently uses, on a REAL affinity graph
captured from an actual TF-MoDISco run.

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

THIS IS UNVERIFIED, BEST-EFFORT CODE FOR THE cuGraph SECTION. It was
written without access to a CUDA/RAPIDS environment (authored on a Mac
laptop) -- the leidenalg side is exercised by this codebase's existing
tests and by a local smoke run of this script, but the cuGraph API calls
have not been run against a real installation. Expect to debug minor
API-version mismatches (`cugraph.leiden`'s signature and return shape
have changed across RAPIDS releases -- see the NOTE comments below).
Run this on a CUDA machine with RAPIDS installed, e.g.:
    pip install cugraph-cu12 cudf-cu12   # match your CUDA version
or follow https://docs.rapids.ai/install for a conda install.

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
into one edge with combined weight (which would change the objective
cuGraph is optimizing relative to what leidenalg optimizes). If the
modularity/ARI numbers look surprising, check this first.

Two phases:
  1. Capture -- run TFMoDISco far enough (with n_seeds=1, so it's cheap)
     to produce a REAL density-adapted affinity graph at production scale,
     and cache it to disk so repeat comparisons don't need to redo this.
  2. Compare -- on that one graph, run leidenalg for --n-leiden-runs
     independent trials (exactly as cluster.LeidenCluster does) to get
     both a "best" partition and a natural seed-to-seed variability
     baseline, then run cuGraph's Leiden --cugraph-trials times and
     report: wall time, modularity, and clustering agreement (Adjusted
     Rand Index / Normalized Mutual Info) against the leidenalg baseline
     AND against leidenalg's own seed-to-seed spread (the natural-noise
     floor to judge cuGraph's output against).

Usage (production-scale, matching the report this was motivated by):
    python agent_notes/compare_leiden_backends.py \\
        --n-peaks 5000 --window 400 --max-seqlets-per-metacluster 20000 \\
        --n-leiden-runs 50

Usage (smaller, to sanity check the script itself first):
    python agent_notes/compare_leiden_backends.py \\
        --n-peaks 500 --max-seqlets-per-metacluster 500 --n-leiden-runs 10
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


def run_cugraph_leiden(affmat, n_trials=3):
    try:
        import cudf
        import cugraph
    except ImportError as e:
        return None, str(e)

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

    accepted = set(inspect.signature(cugraph.leiden).parameters)

    results = []
    for trial in range(n_trials):
        kwargs = {}
        if "max_iter" in accepted:
            kwargs["max_iter"] = 100
        if "resolution" in accepted:
            kwargs["resolution"] = 1.0

        start = time.perf_counter()
        try:
            # NOTE: documented as returning (partition_df, modularity_score)
            # on recent RAPIDS releases. If your installed version returns
            # them in the opposite order, swap this line.
            parts_df, modularity = cugraph.leiden(G, **kwargs)
        except TypeError:
            start = time.perf_counter()
            parts_df, modularity = cugraph.leiden(G)
        wall = time.perf_counter() - start

        parts_df = parts_df.sort_values("vertex").reset_index(drop=True)
        # Column name for the partition assignment has varied across
        # RAPIDS versions ('partition' in recent ones) -- pick whichever
        # non-'vertex' column is present rather than hardcoding it.
        partition_col = [c for c in parts_df.columns if c != "vertex"][0]
        membership = parts_df[partition_col].to_numpy()

        results.append({"trial": trial, "quality": float(modularity),
            "membership": membership, "wall_seconds": wall})

    return results, None


def summarize(leidenalg_results, cugraph_results, cugraph_error):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="agent_notes/data/encode_ENCFF407PRC")
    parser.add_argument("--n-peaks", type=int, default=5000)
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--max-seqlets-per-metacluster", type=int, default=20000)
    parser.add_argument("--n-leiden-runs", type=int, default=50,
        help="Number of independent leidenalg seed trials to run for comparison "
             "(mirrors n_leiden_runs in a real pipeline call).")
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
    cugraph_results, cugraph_error = run_cugraph_leiden(affmat, args.cugraph_trials)

    report = summarize(leidenalg_results, cugraph_results, cugraph_error)
    print()
    print(report)

    if args.report_path:
        Path(args.report_path).write_text(report)
        print(f"\nReport written to {args.report_path}")


if __name__ == "__main__":
    main()
