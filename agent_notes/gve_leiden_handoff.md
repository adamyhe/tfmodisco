# GVE-Leiden-in-Python Handoff

This document hands off context for implementing a fast, genuinely parallel
Leiden community-detection algorithm in Python (inspired by GVE-Leiden), so
the work can restart in a clean project without needing the history of the
investigation that led here. It assumes no prior context beyond what's
written below.

**Recommendation on where this lives:** build the algorithm itself as a
separate, general-purpose project (no dependency on this repo or on
genomics-specific concepts) — it deserves its own test suite against known
graphs, its own README, and room to be reused outside TF-MoDISco. Once it
has a working sequential reference implementation validated against
`leidenalg` on synthetic graphs, integrate it back into *this* repo
(`tfmodisco`) via a thin wrapper reusing the comparison harness described
below — don't rebuild that harness from scratch, and don't let the new
project depend on `modiscolite`.

## 1. Why this matters (background)

Profiling a real production-scale TF-MoDISco run (5000 ENCODE peaks, window
400, `max_seqlets_per_metacluster=20000`, `n_leiden_runs=50`) showed **Leiden
clustering is ~98% of total wall time** — everything else (seqlet extraction,
coarse/fine affinity, density adaptation) is a rounding error by comparison.
See `agent_notes/leiden_parallel_report.txt` for the full breakdown.

Two approaches were already explored in this repo:

1. **CPU seed-loop parallelization** (shipped): `cluster.LeidenCluster` gained
   an `n_jobs` parameter that runs the existing `n_leiden_runs` independent
   `leidenalg` trials across OS processes (via `joblib`/`loky`) instead of
   serially, since `leidenalg` holds the GIL (confirmed both by a threading
   benchmark showing ~1.0x "speedup" with 8 threads, and by checking the
   compiled `_c_leiden.abi3.so` for the absence of
   `PyEval_SaveThread`/`PyEval_RestoreThread`/`Py_BEGIN_ALLOW_THREADS`
   symbols). Real speedup, real memory cost: **~800MB extra RAM per
   additional worker** at this graph scale, because each process needs its
   own full copy of the built graph. Diminishing returns past `n_jobs=8-16`
   given only 50 independent seeds to spread across workers. This is
   solid, low-risk, already shipped, and it's the floor any new approach
   needs to beat.

2. **RAPIDS cuGraph (GPU) substitution** (exploratory, not adopted):
   `agent_notes/compare_leiden_backends.py` swaps `cluster.LeidenCluster`
   for `cugraph.leiden` and compares against the `leidenalg` baseline. A
   single `cugraph.leiden` call is dramatically faster (46-93x depending on
   the run) than a single `leidenalg` call on the same big graph, but the
   **full-pipeline** speedup was only 9-13x — see §3 for why — and,
   critically, **two separate full-pipeline runs of the cuGraph-substituted
   pipeline gave different final pattern counts (18 vs. 21) on nominally
   identical input**, while the unmodified `leidenalg` pipeline was
   bit-identical across the same two runs. That reproducibility gap is the
   main open concern with the GPU approach, and it's part of why GVE-Leiden
   (a real from-scratch algorithm you control end to end) became interesting
   as an alternative.

## 2. What GVE-Leiden actually is

- Paper: Sahu et al., ["GVE-Leiden: Fast Leiden Algorithm for Community
  Detection in Shared Memory Setting"](https://arxiv.org/abs/2312.13936),
  ICPP 2024.
- It is a **CPU, shared-memory, OpenMP-multithreaded** implementation — not
  a GPU algorithm. Same conceptual structure as standard Leiden: local-moving
  → refinement → aggregation, repeated to convergence.
- Headline benchmark: on a 32-core (dual 16-core Xeon Gold 6226R) machine, on
  a **3.8B-edge graph**, it beats sequential Leiden by 436x, `igraph`'s
  Leiden by 104x, NetworKit's Leiden by 8.2x, and — the one that matters
  most here — **cuGraph Leiden on an NVIDIA A100 by 3.0x**, at 403M edges/s.
- Reference C++ implementation:
  [puzzlef/leiden-communities-openmp](https://github.com/puzzlef/leiden-communities-openmp)
  — pure C++ (98.3%), MIT licensed, **no Python bindings, no CLI wrapper**.
  Using it directly would mean writing custom bindings (pybind11/ctypes) and
  taking on a C++ build toolchain as a dependency — that's why "implement it
  in Python" (this project) rather than "bind to their C++" was the chosen
  direction.
- The paper's abstract reports **speed only** — no modularity/quality
  comparison against other implementations. Don't assume correctness parity
  with `leidenalg` is established; that has to be validated from scratch,
  same as the cuGraph work had to be.
- **Scale gap to keep in mind**: their benchmark graph (3.8B edges) is
  roughly 380x bigger than the graphs this pipeline actually produces (up to
  ~20k vertices, ~10M edges for the main clustering graph; much smaller for
  the ~18-21 subclustering graphs). Relative rankings between implementations
  can and do shift at different scales — fixed overheads, thread-launch
  costs, and per-core work granularity all matter more on smaller graphs.
  Don't assume the 3x-over-cuGraph number holds at this pipeline's scale
  without measuring it directly.

## 3. Hard-won lessons from the cuGraph exploration (apply these here too)

These are non-obvious things discovered empirically that will save time if
known up front:

- **A single fast call on one big graph does not mean a fast pipeline.**
  The main clustering call sped up 46x under cuGraph (1589.7s → 34.4s), but
  `detect_spurious_merging` only sped up 4.4x (544.6s → 123.6s) and
  subclustering only 4.9x (347.6s → 70.9s). Both of those stages make
  ~18-21 separate calls on *small* graphs (one per candidate/final pattern),
  and each call pays a roughly fixed per-call construction/dispatch overhead
  that doesn't shrink with graph size — so it dominates on small graphs and
  erases most of the raw per-call speedup. **Benchmark any new
  implementation on this exact "many small graphs" scenario, not just the
  one big graph** — that's where a pure-Python/Numba implementation (no
  GPU transfer, no cuDF DataFrame construction) might actually have an edge
  over cuGraph, which is the more promising angle for GVE-Leiden here.

- **"Deterministic per call" does not mean "reproducible pipeline."**
  Varying `cugraph.leiden`'s `random_state` parameter produced *bit-identical*
  results across trials on a fixed graph (confirmed via ARI=1.0 across
  seeded trials). Yet the full pipeline with cuGraph substituted gave
  *different* results (18 vs. 21 final patterns) across two separate runs
  with nominally identical config, while `leidenalg`'s own result was
  bit-identical across the same two runs. Best working hypothesis: Numba's
  `prange`-parallelized affinity kernels have run-to-run floating-point
  summation-order non-determinism, producing a graph that differs by tiny
  amounts each run; `leidenalg`'s 50-independent-restart search is robust to
  that (it's already sampling broadly), but a single near-deterministic
  `cugraph.leiden` call apparently isn't. **Test reproducibility across
  separate process invocations, not just across trials within one run.**

- **The graph has parallel edges — replicate that, don't "fix" it.** The
  density-adapted affinity matrix gets symmetrized
  (`affmat_nn += affmat_nn.T` in `modiscolite/tfmodisco.py`), and the
  existing `cluster.LeidenCluster` builds its igraph from the CSR structure
  as-is — meaning each undirected edge `(u, v)` gets added *twice* (once per
  stored CSR direction). `igraph` keeps these as parallel edges rather than
  merging them, and that's what `leidenalg` has always actually optimized
  against. Any new implementation being compared against the `leidenalg`
  baseline needs to either replicate this exact edge multiplicity or
  explicitly account for the difference — it's not a bug to fix, it's the
  established (if slightly odd) behavior being compared against.

- **Graph-level agreement (ARI) isn't the real bar — pattern-level agreement
  is.** TOMTOM cross-comparison (via `memelite.tomtom`, in-process, no
  external binary) of the final motifs from a "good" cuGraph run (18 vs. 18
  patterns) showed every pattern had a significant match on both sides, but
  with two ambiguous cases where one `leidenalg` pattern was best-matched by
  *two* `cugraph` patterns (a big primary match plus a small "satellite" —
  e.g. `leidenalg[2]` n=626 matched by both `cugraph[1]` n=960 and
  `cugraph[11]` n=52), and the weakest matches (p≈1e-7 to 1e-8 vs. the ~4e-11
  norm) were concentrated in the smallest/tail patterns. **Validate at the
  pattern level using the existing TOMTOM harness, not just graph-level
  ARI/modularity** — small tail patterns are where disagreement concentrates,
  and that's invisible to graph-level metrics.

## 4. The interface contract to match

`modiscolite/cluster.py::LeidenCluster` is the function any new backend needs
to be comparable against (see the file directly for the current code):

```python
def LeidenCluster(affinity_mat, n_seeds=2, n_leiden_iterations=-1, n_jobs=1):
    # affinity_mat: scipy.sparse.csr_matrix, symmetric, density-adapted
    #   affinity graph. May contain parallel edges per §3.
    # Runs n_seeds independent trials with seeds [100, 200, ..., n_seeds*100],
    # each producing (quality, membership) via modularity-based partitioning.
    # Returns the membership array (length n_vertices) of whichever trial
    # achieved the STRICTLY highest quality, iterating in seed order (so
    # ties resolve to the earliest/lowest-seed trial).
    ...
```

For a genuinely different algorithm, exact bit-compatibility with `leidenalg`
isn't the bar (that's unrealistic and wasn't required of cuGraph either) —
comparable quality/ARI, validated the way described in §5, is.

## 5. Recommended implementation plan

1. **New project scaffold.** Pure Python + NumPy + Numba, no dependency on
   `modiscolite`/TF-MoDISco/genomics concepts at all. Standard package
   layout, `pytest`.

2. **Sequential reference implementation first.** Implement local-moving +
   refinement + aggregation correctly, without worrying about parallelism
   yet — get a working, correct Leiden implementation in plain Python/NumPy.
   Validate against `leidenalg` and `networkx`'s community detection on:
   - Classic tiny benchmark graphs with known structure (Zachary's karate
     club is the standard sanity check most implementations are validated
     against).
   - Synthetic graphs with known ground-truth communities (stochastic block
     models, or LFR benchmark graphs) at a range of sizes.
   Do **not** touch real TF-MoDISco graphs at this stage.

3. **Add real intra-run parallelism to local-moving.** This is the hard
   part flagged repeatedly during the cuGraph investigation: local-moving is
   inherently sequential-dependency-heavy (a node's move decision depends on
   its neighbors' *current* community assignments, which may have just
   changed earlier in the same pass). Real parallelism means batching node
   updates and resolving conflicts (this is what GVE-Leiden's own speedup
   comes from) — a naive `@njit(parallel=True)`/`prange` port of the
   sequential logic will not be faster than compiled C++ `leidenalg`, and
   might even be slower. Validate the parallel version against the
   sequential reference from step 2 with an explicit tolerance for expected
   differences (this repo's own `_sparse_mm_dot_inverted`-vs-`_sparse_mm_dot`
   and `_jaccard_max_only`-vs-`_jaccard` validation tests, in
   `tests/test_affinity_baselines.py`, are a good pattern to follow for
   "does the optimized version match the reference within tolerance").

4. **Benchmark against `leidenalg` and (if available) `cugraph.leiden`** on
   synthetic graphs at increasing scale, **specifically including the
   "many small graphs" scenario** from §3 — that's the scenario where a
   low-per-call-overhead pure-Python/Numba implementation has the best shot
   at beating cuGraph for this pipeline's actual usage pattern, even if it
   loses on one giant graph.

5. **Only once validated on synthetic/general graphs**, integrate with this
   repo's real affinity graphs. Write a `make_gve_leiden_cluster()` wrapper
   matching the `make_cugraph_leiden_cluster()` pattern in
   `agent_notes/compare_leiden_backends.py` (import the new package as a
   dependency of the wrapper, not the other way around), and run it through
   the exact same harness: graph-level ARI/NMI vs. `leidenalg`, the
   seed-vs-perturbation diversity diagnostic, the full-pipeline substitution
   comparison with per-stage profiler breakdown, and the TOMTOM pattern-level
   cross-comparison. See §6 for exactly which pieces to reuse.

6. **Decide on adoption as an actual backend separately and later.** This
   plan stops at "validated and benchmarked." Whether it becomes a real
   `coarse_affinity_backend`-style option in `modiscolite/cluster.py` is a
   later decision, made after real evidence exists — don't design that in
   up front.

## 6. Numbers to beat (the validation bar)

From `agent_notes/leiden_parallel_report.txt` and `agent_notes/cugraph_report.txt`
(5000 ENCODE peaks, window 400, `max_seqlets_per_metacluster=20000`,
`n_leiden_runs=50`):

- **`leidenalg`, `n_jobs=1` (current shipped default):** ~2400-3200s for the
  full positive-metacluster pipeline; found **18 patterns** with seqlet-count
  profile `[1373, 666, 626, 581, 534, 460, 132, 132, 106, 91, 66, 57, 55, 39,
  32, 24, 23, 22]`. This exact result was reproduced bit-for-bit across
  separate runs — that reproducibility is itself part of the bar.
- **`leidenalg`, `n_jobs=8` (CPU parallel, shipped):** ~572s (5.5x), ~8.2GB
  peak RSS. `n_jobs=4` gives 3.25x for ~4.9GB if memory is tighter.
- **cuGraph-substituted-everywhere (exploratory):** 182-277s (9-13x
  full-pipeline speedup depending on the run), 18-21 patterns depending on
  the run (not reproducible run-to-run — see §3).
- Any new implementation should report, at minimum: full-pipeline wall time,
  final pattern count and seqlet-size profile, TOMTOM cross-comparison
  against the `leidenalg` baseline pattern set, and reproducibility across
  at least 2-3 separate full runs.

## 7. Files in this repo worth copying or referencing

**Copy as a starting point once you're ready for real-data integration
(step 5 above), not before:**

- `agent_notes/compare_leiden_backends.py` — the full comparison harness.
  Relevant pieces:
  - `run_leidenalg_seeds` — runs the `leidenalg` baseline and computes
    seed-to-seed ARI.
  - `make_cugraph_leiden_cluster` — the template to copy for a
    `make_gve_leiden_cluster()` (same signature contract, same "try N
    trials, keep best modularity" loop structure).
  - `_perturb_weights` / the seed-vs-perturbation diagnostic in
    `run_cugraph_leiden` and `summarize()` — reuse this pattern to check
    whether your implementation's own randomization actually produces
    diverse candidates (a from-scratch implementation with real multi-start
    local search *should* show real diversity here, unlike cuGraph).
  - `compare_pattern_sets` — the TOMTOM-based pattern-level cross-comparison
    (uses `memelite.tomtom`, already a `modiscolite` dependency, in-process,
    no external binary).
  - `run_full_pipeline_comparison` / `summarize_pipeline_comparison` — the
    full end-to-end substitution comparison with per-stage profiler
    breakdown.
- `agent_notes/download_and_profile_leiden.py` — only needed for the
  real-data integration step; provides `download_and_extract`/`load_arrays`
  for the same ENCODE dataset (ENCFF407PRC) already cached at
  `agent_notes/data/encode_ENCFF407PRC/`.

**Reference, don't copy:**

- `modiscolite/cluster.py` — the interface contract (§4). Read it directly
  rather than copying; it's small.
- `tests/test_cluster.py` — the pattern for validating a parallel version
  against a sequential reference with exact-match tests where correctness
  should be bit-identical (e.g. the `n_jobs` reduction/tie-break logic).
- `agent_notes/leiden_parallel_report.txt`, `agent_notes/cugraph_report.txt` —
  the raw baseline numbers (§6 already extracts the key figures, but these
  have full per-stage breakdowns if more detail is needed).
