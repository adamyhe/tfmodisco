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

ALSO COMPARED (2026-07-19, local): igraph itself has a native C Leiden
implementation (`Graph.community_leiden`), separate from the `leidenalg`
package this pipeline currently uses. On a real captured graph it was
~17-20x faster per call than leidenalg, comparable-or-better quality, AND
(unlike cugraph.leiden) responded to seed variation with genuinely different
partitions (ARI=0.46 between two seeds, vs. cugraph's 1.0). It does not
release the GIL either, so process-based parallelism is still needed for
concurrent seed exploration -- this is a per-call speed + diversity win, not
a new parallelism model. See make_igraph_native_leiden_cluster below. This
is a much lower-risk candidate than either cuGraph or a from-scratch
implementation: no new dependency, no CUDA, already installed.

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


def _affmat_edges(affmat):
    n_vertices = affmat.shape[0]
    n_cols = affmat.indptr
    sources = np.concatenate([np.ones(n_cols[i+1] - n_cols[i], dtype='int32') * i
        for i in range(n_vertices)])
    # Mirrors leidenalg's edge list exactly -- see the parallel-edges
    # caveat in the module docstring.
    targets = affmat.indices.astype('int32')
    weights = affmat.data.astype('float64')
    return n_vertices, sources, targets, weights


def _perturb_weights(weights, jitter_scale, seed):
    """Seeded multiplicative jitter on edge weights.

    Varying cugraph.leiden's random_state/seed parameter alone was shown
    (2026-07-18/19 runs) to produce bit-identical output on a fixed graph --
    so it's not a source of real diversity here. Perturbing the input
    graph itself is the other lever available to test whether cugraph's
    local optimum is sensitive to the exact input at all, which is a
    prerequisite for any best-of-N or consensus ensembling to have
    anything real to ensemble over.
    """
    if jitter_scale <= 0:
        return weights
    rng = np.random.RandomState(seed)
    noise = rng.standard_normal(len(weights)) * jitter_scale
    perturbed = weights * (1.0 + noise)
    # Keep weights positive -- a sign flip changes the graph's semantics,
    # not just perturbs it.
    return np.clip(perturbed, 1e-12, None)


def _build_cugraph_graph_from_edges(n_vertices, sources, targets, weights):
    import cudf
    import cugraph

    edgelist = cudf.DataFrame({"src": sources, "dst": targets, "weight": weights})
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edgelist, source="src", destination="dst",
        edge_attr="weight", renumber=True)
    return G


def _build_cugraph_graph(affmat):
    n_vertices, sources, targets, weights = _affmat_edges(affmat)
    return _build_cugraph_graph_from_edges(n_vertices, sources, targets, weights)


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


def run_cugraph_leiden(affmat, n_trials=3, jitter_scale=0.05):
    """Runs BOTH repeat modes as a diagnostic, on the same captured graph:

    - 'seed': graph built once, vary only the random_state/seed parameter
      across trials. Cheap, but per the 2026-07-18/19 runs this gave
      bit-identical results every time -- i.e. no diversity.
    - 'perturbation': rebuild the graph with a small seeded multiplicative
      jitter on edge weights for every trial. Tests whether cugraph's
      local optimum is sensitive to the exact input at all.

    If perturbation-mode trial-to-trial ARI is ALSO ~1.0, this graph likely
    has one dominant modularity basin for cugraph's algorithm, and neither
    best-of-N nor consensus ensembling would have anything real to work
    with. If it's meaningfully lower (but not near 0), there IS real
    diversity to build an ensemble on.
    """
    try:
        import cugraph
    except ImportError as e:
        return None, str(e), None

    n_vertices, sources, targets, weights = _affmat_edges(affmat)
    seed_param, accepted = _detect_cugraph_seed_param(cugraph.leiden)

    seed_mode_results = []
    G = _build_cugraph_graph_from_edges(n_vertices, sources, targets, weights)
    for trial in range(n_trials):
        seed_value = (trial + 1) * 100
        start = time.perf_counter()
        quality, membership = _cugraph_leiden_call(G, seed_param, accepted, seed_value)
        wall = time.perf_counter() - start
        seed_mode_results.append({"trial": trial, "quality": quality,
            "membership": membership, "wall_seconds": wall})

    perturbation_mode_results = []
    for trial in range(n_trials):
        seed_value = (trial + 1) * 100
        perturbed_weights = _perturb_weights(weights, jitter_scale, seed_value)
        G_perturbed = _build_cugraph_graph_from_edges(n_vertices, sources, targets,
            perturbed_weights)
        start = time.perf_counter()
        quality, membership = _cugraph_leiden_call(G_perturbed, seed_param, accepted, seed_value)
        wall = time.perf_counter() - start
        perturbation_mode_results.append({"trial": trial, "quality": quality,
            "membership": membership, "wall_seconds": wall})

    return {"seed": seed_mode_results, "perturbation": perturbation_mode_results}, None, seed_param


def make_cugraph_leiden_cluster(repeat_mode="seed", jitter_scale=0.05):
    """A drop-in replacement for cluster.LeidenCluster backed by
    cugraph.leiden, for monkeypatching into a full pipeline run.

    Preserves the same (affinity_mat, n_seeds, n_leiden_iterations, n_jobs)
    signature and 'try n_seeds candidates, keep the best modularity'
    semantics as the real LeidenCluster, so no calling code needs to
    change -- n_leiden_iterations and n_jobs are accepted but unused (GPU
    calls are already fast and run on a single device, not a CPU process
    pool).

    repeat_mode='seed': builds the graph once and reuses it across the
    n_seeds candidate trials, only varying the random_state/seed param.
    Cheap (one graph build per LeidenCluster call), but per the graph-level
    diagnostic in run_cugraph_leiden this may give zero diversity if the
    graph has one dominant modularity basin for cugraph's algorithm.

    repeat_mode='perturbation': rebuilds the graph with a small seeded
    multiplicative jitter on edge weights for EVERY trial. More expensive
    (one cuDF/graph-construction cost per trial, not amortized), but is
    the only lever confirmed to sometimes vary cugraph's result.
    """
    import cugraph

    if repeat_mode not in ("seed", "perturbation"):
        raise ValueError(f"Unrecognized repeat_mode: {repeat_mode!r}")

    seed_param, accepted = _detect_cugraph_seed_param(cugraph.leiden)

    def _leiden_cluster(affinity_mat, n_seeds=2, n_leiden_iterations=-1, n_jobs=1):
        n_vertices, sources, targets, weights = _affmat_edges(affinity_mat)

        G = None
        if repeat_mode == "seed":
            G = _build_cugraph_graph_from_edges(n_vertices, sources, targets, weights)

        best_membership = None
        best_quality = None
        for trial in range(max(n_seeds, 1)):
            seed_value = (trial + 1) * 100

            if repeat_mode == "perturbation":
                perturbed_weights = _perturb_weights(weights, jitter_scale, seed_value)
                G = _build_cugraph_graph_from_edges(n_vertices, sources, targets,
                    perturbed_weights)

            quality, membership = _cugraph_leiden_call(G, seed_param, accepted, seed_value)
            if best_quality is None or quality > best_quality:
                best_quality = quality
                best_membership = membership

        return best_membership

    return _leiden_cluster, seed_param


# --- igraph-native helpers. igraph itself has a native C Leiden
# implementation (Graph.community_leiden), separate from the leidenalg
# package this pipeline currently uses. Confirmed 2026-07-19 on a real
# captured graph: ~17-20x faster per call than leidenalg, comparable or
# better quality, and -- unlike cugraph.leiden's random_state -- varying
# the seed produces genuinely different partitions (ARI=0.46 between two
# seeds on the same graph, vs. cugraph's 1.0). It does NOT release the GIL
# though (checked the same way as leidenalg: ~1.0x "speedup" with 8
# threads), so process-based parallelism is still needed for concurrent
# seed exploration -- this is a per-call speed + diversity win, not a
# parallelism-model change. No try/except ImportError needed: igraph is
# already a hard dependency of modiscolite itself. ---

def _build_igraph_graph_from_edges(n_vertices, sources, targets):
    import igraph as ig

    g = ig.Graph(directed=None)
    g.add_vertices(n_vertices)
    g.add_edges(zip(sources, targets))
    return g


def _igraph_native_leiden_call(g, weights, seed_value, n_leiden_iterations=-1):
    import random

    # community_leiden has no per-call seed kwarg -- it delegates to
    # Python's global `random` module by default (confirmed: it responds
    # to random.seed() with genuinely different partitions).
    random.seed(seed_value)
    vc = g.community_leiden(objective_function='modularity', weights=weights,
        n_iterations=n_leiden_iterations)
    return float(vc.quality), np.asarray(vc.membership)


def run_igraph_native_leiden(affmat, n_trials=3, jitter_scale=0.05):
    """Mirrors run_cugraph_leiden's structure exactly, for igraph's native
    community_leiden instead of cugraph.leiden -- runs BOTH repeat modes as
    a diagnostic on the same captured graph. See the module-level comment
    above this section for the 2026-07-19 findings this is based on.
    """
    n_vertices, sources, targets, weights = _affmat_edges(affmat)

    seed_mode_results = []
    g = _build_igraph_graph_from_edges(n_vertices, sources, targets)
    for trial in range(n_trials):
        seed_value = (trial + 1) * 100
        start = time.perf_counter()
        quality, membership = _igraph_native_leiden_call(g, weights, seed_value)
        wall = time.perf_counter() - start
        seed_mode_results.append({"trial": trial, "quality": quality,
            "membership": membership, "wall_seconds": wall})

    perturbation_mode_results = []
    for trial in range(n_trials):
        seed_value = (trial + 1) * 100
        perturbed_weights = _perturb_weights(weights, jitter_scale, seed_value)
        g_perturbed = _build_igraph_graph_from_edges(n_vertices, sources, targets)
        start = time.perf_counter()
        quality, membership = _igraph_native_leiden_call(g_perturbed, perturbed_weights,
            seed_value)
        wall = time.perf_counter() - start
        perturbation_mode_results.append({"trial": trial, "quality": quality,
            "membership": membership, "wall_seconds": wall})

    return {"seed": seed_mode_results, "perturbation": perturbation_mode_results}, None


def make_igraph_native_leiden_cluster(repeat_mode="seed", jitter_scale=0.05):
    """A drop-in replacement for cluster.LeidenCluster backed by igraph's
    own native community_leiden. Mirrors make_cugraph_leiden_cluster's
    structure; see run_igraph_native_leiden's docstring for the findings
    behind this.

    repeat_mode='seed': builds the graph once and reuses it across the
    n_seeds candidate trials, reseeding Python's global random module
    before each call. repeat_mode='perturbation': rebuilds the graph with a
    small seeded multiplicative jitter on edge weights for EVERY trial.
    """
    if repeat_mode not in ("seed", "perturbation"):
        raise ValueError(f"Unrecognized repeat_mode: {repeat_mode!r}")

    def _leiden_cluster(affinity_mat, n_seeds=2, n_leiden_iterations=-1, n_jobs=1):
        n_vertices, sources, targets, weights = _affmat_edges(affinity_mat)

        g = None
        if repeat_mode == "seed":
            g = _build_igraph_graph_from_edges(n_vertices, sources, targets)

        best_membership = None
        best_quality = None
        for trial in range(max(n_seeds, 1)):
            seed_value = (trial + 1) * 100
            trial_weights = weights

            if repeat_mode == "perturbation":
                trial_weights = _perturb_weights(weights, jitter_scale, seed_value)
                g = _build_igraph_graph_from_edges(n_vertices, sources, targets)

            quality, membership = _igraph_native_leiden_call(g, trial_weights,
                seed_value, n_leiden_iterations=n_leiden_iterations)
            if best_quality is None or quality > best_quality:
                best_quality = quality
                best_membership = membership

        return best_membership

    return _leiden_cluster


def compare_pattern_sets(patterns_a, patterns_b, label_a="leidenalg", label_b="cugraph",
        sig_threshold=0.05):
    """Cross-compare two sets of discovered patterns with TOMTOM (via
    memelite, in-process -- no external tomtom binary needed), to answer
    the question graph-level ARI/modularity can't: are patterns unique to
    one set genuinely novel/different motifs, or redundant splits/merges
    of a motif the other set already found as one pattern?

    Uses each pattern's contribution-weight matrix (contrib_scores), the
    same representation this codebase's own report.py/descriptive_report.py
    use for TOMTOM-based comparison against reference motif databases --
    here the "database" is just the other backend's pattern set instead of
    an external motif database.
    """
    from memelite import tomtom

    if not patterns_a or not patterns_b:
        return "  (one of the pattern sets is empty -- nothing to compare)"

    Qs = [p.contrib_scores.T.astype('float64') for p in patterns_b]
    Ts = [p.contrib_scores.T.astype('float64') for p in patterns_a]

    best_p_values = tomtom(Qs, Ts, n_jobs=-1)[0]

    lines = []
    lines.append(f"  TOMTOM cross-comparison: {label_b} (query, n={len(Qs)}) vs "
        f"{label_a} (target, n={len(Ts)}), significance threshold p<{sig_threshold}")

    unmatched_b = []
    best_target_for = {}
    for qi in range(len(Qs)):
        row = best_p_values[qi]
        best_ti = int(np.argmin(row))
        best_p = row[best_ti]
        best_target_for[qi] = best_ti
        flag = "" if best_p < sig_threshold else "  <-- NO significant match"
        lines.append(f"    {label_b}[{qi}] (n_seqlets={len(patterns_b[qi].seqlets)}) best match: "
            f"{label_a}[{best_ti}] (n_seqlets={len(patterns_a[best_ti].seqlets)}) "
            f"p={best_p:.2e}{flag}")
        if best_p >= sig_threshold:
            unmatched_b.append(qi)

    unmatched_a = [ti for ti in range(len(Ts)) if np.min(best_p_values[:, ti]) >= sig_threshold]

    lines.append("")
    if unmatched_b:
        lines.append(f"  {label_b} patterns with NO significant match in {label_a}: {unmatched_b}")
        lines.append(f"  -- candidates for motifs genuinely novel to {label_b}, or spurious splits.")
    else:
        lines.append(f"  Every {label_b} pattern has a significant match in {label_a}.")

    if unmatched_a:
        lines.append(f"  {label_a} patterns with NO significant match in {label_b}: {unmatched_a}")
        lines.append(f"  -- candidates for motifs {label_b} missed or merged into something else.")
    else:
        lines.append(f"  Every {label_a} pattern has a significant match in {label_b}.")

    from collections import defaultdict
    target_to_queries = defaultdict(list)
    for qi, ti in best_target_for.items():
        if best_p_values[qi, ti] < sig_threshold:
            target_to_queries[ti].append(qi)

    split_candidates = {ti: qis for ti, qis in target_to_queries.items() if len(qis) > 1}
    if split_candidates:
        lines.append("")
        lines.append(f"  Possible splits -- multiple {label_b} patterns both best-matching the")
        lines.append(f"  same {label_a} pattern (that {label_a} pattern may have been split by "
            f"{label_b}):")
        for ti, qis in split_candidates.items():
            lines.append(f"    {label_a}[{ti}] (n_seqlets={len(patterns_a[ti].seqlets)}) "
                f"<- {label_b}{qis}")

    return "\n".join(lines)


def run_full_pipeline_comparison(data_dir, n_peaks, window, max_seqlets_per_metacluster,
        n_leiden_runs, seed, cugraph_repeat_mode="seed", igraph_native_repeat_mode="seed",
        jitter_scale=0.05):
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
            "patterns_pos": pos or [],
            "profile_summary": profiler.summary(),
        }

    print("  [pipeline] running leidenalg (CPU) end to end -- this is the slow part ...")
    cpu_result = _run(n_leiden_jobs=1)

    result = {"cpu": cpu_result, "cugraph_repeat_mode": cugraph_repeat_mode,
        "igraph_native_repeat_mode": igraph_native_repeat_mode}

    try:
        cugraph_leiden_cluster, seed_param = make_cugraph_leiden_cluster(
            repeat_mode=cugraph_repeat_mode, jitter_scale=jitter_scale)

        original_leiden_cluster = cluster.LeidenCluster
        cluster.LeidenCluster = cugraph_leiden_cluster
        try:
            print("  [pipeline] running cuGraph-substituted pipeline end to end ...")
            gpu_result = _run(n_leiden_jobs=1)
        finally:
            cluster.LeidenCluster = original_leiden_cluster

        print("  [pipeline] cross-comparing leidenalg vs cuGraph patterns with TOMTOM ...")
        result["gpu"] = gpu_result
        result["gpu_error"] = None
        result["seed_param"] = seed_param
        result["tomtom_comparison_cugraph"] = compare_pattern_sets(
            cpu_result["patterns_pos"], gpu_result["patterns_pos"],
            label_a="leidenalg", label_b="cugraph")
    except ImportError as e:
        result["gpu"] = None
        result["gpu_error"] = str(e)
        result["seed_param"] = None
        result["tomtom_comparison_cugraph"] = None

    igraph_native_leiden_cluster = make_igraph_native_leiden_cluster(
        repeat_mode=igraph_native_repeat_mode, jitter_scale=jitter_scale)

    original_leiden_cluster = cluster.LeidenCluster
    cluster.LeidenCluster = igraph_native_leiden_cluster
    try:
        print("  [pipeline] running igraph-native-substituted pipeline end to end ...")
        igraph_native_result = _run(n_leiden_jobs=1)
    finally:
        cluster.LeidenCluster = original_leiden_cluster

    print("  [pipeline] cross-comparing leidenalg vs igraph-native patterns with TOMTOM ...")
    result["igraph_native"] = igraph_native_result
    result["igraph_native_error"] = None
    result["tomtom_comparison_igraph_native"] = compare_pattern_sets(
        cpu_result["patterns_pos"], igraph_native_result["patterns_pos"],
        label_a="leidenalg", label_b="igraph_native")

    return result


def _summarize_candidate_backend(backend_name, candidate_results, candidate_error,
        leidenalg_results, best, seed_note=None, short_name=None):
    """Reports one candidate backend's graph-level comparison against the
    leidenalg baseline -- shared by cuGraph and igraph-native so the two
    report identically-structured, directly comparable sections.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    lines = []
    lines.append("")
    lines.append(f"=== {backend_name} ===")
    if candidate_results is None:
        lines.append(f"  SKIPPED -- not available: {candidate_error}")
        return "\n".join(lines)

    if seed_note:
        lines.append(f"  {seed_note}")

    mode_labels = {
        "seed": "seed (vary randomization only, same graph reused across trials)",
        "perturbation": "perturbation (jitter edge weights, rebuild graph per trial)",
    }
    mode_mean_aris = {}
    for mode_key, mode_label in mode_labels.items():
        mode_results = candidate_results[mode_key]
        lines.append("")
        lines.append(f"  -- repeat mode: {mode_label} --")
        for r in mode_results:
            ari_vs_best = adjusted_rand_score(r["membership"], best["membership"])
            nmi_vs_best = normalized_mutual_info_score(r["membership"], best["membership"])
            lines.append(f"    trial={r['trial']}: quality={r['quality']:.6f} "
                f"n_clusters={len(set(r['membership']))} wall={r['wall_seconds']:.3f}s "
                f"ARI_vs_leidenalg_best={ari_vs_best:.4f} "
                f"NMI_vs_leidenalg_best={nmi_vs_best:.4f}")

        if len(mode_results) > 1:
            mode_aris = []
            for i in range(len(mode_results)):
                for j in range(i + 1, len(mode_results)):
                    mode_aris.append(adjusted_rand_score(
                        mode_results[i]["membership"], mode_results[j]["membership"]))
            mode_mean_aris[mode_key] = np.mean(mode_aris)
            lines.append(f"    trial-to-trial ARI: mean={np.mean(mode_aris):.4f} "
                f"min={np.min(mode_aris):.4f} max={np.max(mode_aris):.4f}")

    lines.append("")
    if mode_mean_aris.get("seed", 0) > 0.9999 and mode_mean_aris.get("perturbation", 0) > 0.9999:
        lines.append("  Diagnostic: BOTH modes show trial-to-trial ARI ~1.0 -- this graph likely")
        lines.append(f"  has one dominant modularity basin for {backend_name}. Best-of-N and")
        lines.append("  consensus ensembling would have nothing real to ensemble over here.")
    elif mode_mean_aris.get("seed", 1.0) < 0.9999:
        lines.append("  Diagnostic: 'seed' mode alone already shows real trial-to-trial diversity")
        lines.append("  -- a genuine best-of-N multi-restart search is possible here.")
    elif mode_mean_aris.get("perturbation", 1.0) < mode_mean_aris.get("seed", 0) - 0.01:
        lines.append("  Diagnostic: 'perturbation' mode shows lower trial-to-trial ARI than 'seed'")
        lines.append("  mode -- there IS real diversity to ensemble over via input perturbation,")
        lines.append("  even though varying the seed alone does not produce any.")
    else:
        lines.append("  Diagnostic: inconclusive from these numbers alone -- inspect the raw ARI")
        lines.append("  values above directly.")

    mean_leidenalg_wall = np.mean([r["wall_seconds"] for r in leidenalg_results])
    mean_candidate_wall = np.mean([r["wall_seconds"] for r in candidate_results["seed"]])
    lines.append("")
    lines.append(f"  mean wall/run: leidenalg={mean_leidenalg_wall:.3f}s "
        f"{short_name or backend_name} (seed mode)={mean_candidate_wall:.3f}s "
        f"speedup={mean_leidenalg_wall / mean_candidate_wall:.2f}x")

    return "\n".join(lines)


def summarize(leidenalg_results, cugraph_results, cugraph_error, cugraph_seed_param,
        igraph_native_results, igraph_native_error):
    from sklearn.metrics import adjusted_rand_score

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

    cugraph_seed_note = "seed/random-state parameter used: " + (cugraph_seed_param or
        "NONE FOUND on cugraph.leiden -- 'seed' mode below is not independently randomized")
    lines.append(_summarize_candidate_backend("cuGraph Leiden (GPU)", cugraph_results,
        cugraph_error, leidenalg_results, best, seed_note=cugraph_seed_note,
        short_name="cugraph"))

    igraph_native_seed_note = ("seed control: Python's global random.seed() -- "
        "community_leiden has no per-call seed kwarg")
    lines.append(_summarize_candidate_backend("igraph native Leiden (CPU, community_leiden)",
        igraph_native_results, igraph_native_error, leidenalg_results, best,
        seed_note=igraph_native_seed_note, short_name="igraph_native"))

    lines.append("")
    lines.append("  Interpretation guide: if a candidate backend's ARI-vs-leidenalg-best is")
    lines.append("  comparable to leidenalg's OWN seed-to-seed ARI above, it's landing in the")
    lines.append("  same range of 'plausible good partitions' leidenalg itself produces across")
    lines.append("  seeds -- not a red flag on its own. If it's substantially lower, treat that")
    lines.append("  as a real behavioral difference to investigate before trusting that")
    lines.append("  backend's output.")

    return "\n".join(lines)


def _format_profile_summary(profile_summary, indent="    "):
    lines = []
    for stage, stats in sorted(profile_summary.items(), key=lambda kv: -kv[1]["seconds"]):
        lines.append(f"{indent}{stage}: {stats['seconds']:.4f}s over {stats['count']} call(s)")
    return lines


def _summarize_pipeline_candidate(name, cpu, candidate_result, candidate_error, tomtom_comparison):
    lines = []
    if candidate_result is None:
        lines.append(f"  {name}: SKIPPED -- {candidate_error}")
        return lines

    lines.append(f"  {name}: wall={candidate_result['wall_seconds']:.2f}s "
        f"n_pos={candidate_result['n_pos']} n_neg={candidate_result['n_neg']}")
    lines.append(f"    pos_pattern_sizes={candidate_result['pos_sizes']}")
    lines.append(f"  {name} per-stage breakdown (sorted by cost):")
    lines.extend(_format_profile_summary(candidate_result["profile_summary"]))
    lines.append(f"  full-pipeline speedup: "
        f"{cpu['wall_seconds'] / candidate_result['wall_seconds']:.2f}x")

    counts_match = cpu['n_pos'] == candidate_result['n_pos'] and \
        cpu['n_neg'] == candidate_result['n_neg']
    lines.append(f"  pattern count match: "
        f"{'YES' if counts_match else 'NO -- investigate before trusting'}")
    if counts_match and cpu['pos_sizes'] and cpu['pos_sizes'] == candidate_result['pos_sizes']:
        lines.append("  pos_pattern_sizes match exactly, in the same sorted order --")
        lines.append("  strong (though not conclusive) sign the substitution is behaving well.")
    elif counts_match:
        lines.append("  pattern counts match but per-pattern sizes differ -- same NUMBER of")
        lines.append("  patterns found, but not necessarily the same patterns.")

    if tomtom_comparison:
        lines.append("")
        lines.append(tomtom_comparison)

    return lines


def summarize_pipeline_comparison(result):
    lines = []
    lines.append("")
    lines.append("=== Full-pipeline substitution: leidenalg vs alternates ===")
    cpu = result["cpu"]
    lines.append(f"  leidenalg: wall={cpu['wall_seconds']:.2f}s n_pos={cpu['n_pos']} "
        f"n_neg={cpu['n_neg']}")
    lines.append(f"    pos_pattern_sizes={cpu['pos_sizes']}")
    lines.append("  leidenalg per-stage breakdown (sorted by cost):")
    lines.extend(_format_profile_summary(cpu["profile_summary"]))

    lines.append("")
    lines.append(f"  -- cuGraph (repeat mode: {result.get('cugraph_repeat_mode', 'seed')}) --")
    if result.get("gpu") is not None:
        lines.append(f"  cugraph.leiden seed parameter used: "
            f"{result['seed_param'] or 'NONE FOUND -- see graph-level section above'}")
    lines.extend(_summarize_pipeline_candidate("cuGraph", cpu, result.get("gpu"),
        result.get("gpu_error"), result.get("tomtom_comparison_cugraph")))

    lines.append("")
    lines.append(f"  -- igraph native Leiden (repeat mode: "
        f"{result.get('igraph_native_repeat_mode', 'seed')}) --")
    lines.extend(_summarize_pipeline_candidate("igraph native", cpu, result.get("igraph_native"),
        result.get("igraph_native_error"), result.get("tomtom_comparison_igraph_native")))

    lines.append("")
    lines.append("  (compare the per-stage breakdowns above to see how much of the gap between")
    lines.append("  full-pipeline speedup and single-call speedup is fixed non-Leiden cost --")
    lines.append("  coarse/fine affinity, unchanged across all backends -- versus per-call")
    lines.append("  graph-construction overhead paid on every subclustering call.)")

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
        help="Number of times to run each candidate backend (cugraph.leiden AND "
             "igraph's native community_leiden) per repeat mode ('seed' and "
             "'perturbation'), to gauge run-to-run variability in the "
             "graph-level comparison.")
    parser.add_argument("--jitter-scale", type=float, default=0.05,
        help="Relative multiplicative jitter applied to edge weights in "
             "'perturbation' repeat mode (0.05 = +/-5%% noise, seeded per "
             "trial for reproducibility).")
    parser.add_argument("--cugraph-repeat-mode", choices=["seed", "perturbation"],
        default="seed",
        help="Which repeat mode --run-pipeline-comparison's cuGraph-substituted "
             "backend uses internally for its 'try n_seeds, keep best' loop. "
             "'seed' is cheap but may have no diversity (see the graph-level "
             "diagnostic printed regardless of this flag); 'perturbation' is "
             "more expensive (rebuilds the graph every trial) but is the "
             "mode confirmed to actually vary cugraph's output.")
    parser.add_argument("--igraph-native-repeat-mode", choices=["seed", "perturbation"],
        default="seed",
        help="Which repeat mode --run-pipeline-comparison's igraph-native-"
             "substituted backend uses internally. Unlike cugraph, 'seed' mode "
             "alone has been shown to give real diversity for igraph's native "
             "community_leiden, so 'seed' is likely sufficient (and cheaper).")
    parser.add_argument("--capture-index", type=int, default=0,
        help="Which LeidenCluster call to capture the affinity graph from "
             "(0 = first call, typically the largest/most representative).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-path", default=None,
        help="Optional path to cache the captured affinity graph (.npz) so "
             "repeat comparisons skip re-running the pipeline.")
    parser.add_argument("--run-pipeline-comparison", action="store_true",
        help="Also run the full TFMoDISco() pipeline end to end with "
             "leidenalg vs. cugraph-substituted-everywhere AND vs. "
             "igraph-native-substituted-everywhere, and compare final "
             "pattern counts/sizes for each. The leidenalg side is NOT "
             "cheap -- see the module docstring. Start with a small "
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

    print(f"Running cugraph.leiden for {args.cugraph_trials} trial(s) per repeat mode "
        "(seed, perturbation) ...")
    cugraph_results, cugraph_error, cugraph_seed_param = run_cugraph_leiden(
        affmat, args.cugraph_trials, jitter_scale=args.jitter_scale)

    print(f"Running igraph native Leiden for {args.cugraph_trials} trial(s) per repeat mode "
        "(seed, perturbation) ...")
    igraph_native_results, igraph_native_error = run_igraph_native_leiden(
        affmat, args.cugraph_trials, jitter_scale=args.jitter_scale)

    report = summarize(leidenalg_results, cugraph_results, cugraph_error, cugraph_seed_param,
        igraph_native_results, igraph_native_error)

    if args.run_pipeline_comparison:
        print(f"Running full end-to-end pipeline comparison (cuGraph repeat_mode="
            f"{args.cugraph_repeat_mode}, igraph_native repeat_mode="
            f"{args.igraph_native_repeat_mode}; leidenalg side is slow -- "
            "see module docstring) ...")
        pipeline_result = run_full_pipeline_comparison(args.data_dir, args.n_peaks,
            args.window, args.max_seqlets_per_metacluster, args.n_leiden_runs, args.seed,
            cugraph_repeat_mode=args.cugraph_repeat_mode,
            igraph_native_repeat_mode=args.igraph_native_repeat_mode,
            jitter_scale=args.jitter_scale)
        report += "\n" + summarize_pipeline_comparison(pipeline_result)

    print()
    print(report)

    if args.report_path:
        Path(args.report_path).write_text(report)
        print(f"\nReport written to {args.report_path}")


if __name__ == "__main__":
    main()
