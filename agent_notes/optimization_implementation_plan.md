# Optimization Implementation and Testing Plan

This plan describes how to implement CPU-first optimizations for `modiscolite`
while preserving behavior as closely as possible. GPU work is treated as a later
optional backend after CPU profiling and correctness gates are in place.

## Goals

1. Reduce runtime and memory pressure in affinity construction.
2. Preserve current algorithmic semantics wherever possible.
3. Make any expected numerical or tie-ordering differences explicit and tested.
4. Keep optional GPU support behind a backend boundary rather than changing the
   default CPU path.

## Non-goals for the first pass

- Do not rewrite the full pipeline.
- Do not change motif filtering, density adaptation, or clustering heuristics
  unless profiling proves they are dominant after affinity optimization.
- Do not make PyTorch, CuPy, FAISS, or RAPIDS hard dependencies.
- Do not optimize final motif outputs before lower-level affinity equivalence is
  tested.

## Phase 0: Baseline and Instrumentation

Purpose:

- Establish where time and memory are currently spent.
- Create fixtures for comparing optimized paths against the existing
  implementation.

Implementation tasks:

1. Add a lightweight timing helper, gated by an explicit option such as
   `profile=False` or existing `verbose=True`.
2. Time these stages:
   - `extract_seqlets.extract_seqlets`
   - `affinitymat.cosine_similarity_from_seqlets`
   - `affinitymat.jaccard_from_seqlets`
   - `tfmodisco._filter_by_correlation`
   - `tfmodisco._density_adaptation`
   - `cluster.LeidenCluster`
   - `tfmodisco._patterns_from_clusters`
   - `aggregator._detect_spurious_merging`
   - final `pattern.compute_subpatterns`
3. Add small deterministic fixtures for affinity tests:
   - Tiny CSR matrices with nonzero overlaps.
   - Tiny CSR matrices with rows that have fewer than `k` nonzero candidates.
   - Dense seqlet-like arrays with known Jaccard values.
   - Cases with exact ties.
4. Add a development-only benchmark script under `agent_notes/` or
   `benchmarks/` if a benchmarks directory is introduced later.

Testing tasks:

1. Run existing tests before changes.
2. Add tests that call current implementations directly and snapshot:
   - coarse similarity values and neighbor indices,
   - fine Jaccard sparse results,
   - pairwise Jaccard top-k results.
3. Record baseline wall times on at least one realistic dataset if available.

Acceptance criteria:

- Existing test suite passes.
- Baseline timing output can identify the dominant stages.
- New fixtures cover zero-similarity rows and tie cases.

## Phase 1: Deterministic Top-k Helper

Purpose:

- Replace repeated full-row sorting with a reusable partial-selection helper
  without changing tie behavior accidentally.

Implementation tasks:

1. Add an internal helper, for example:

```python
def _topk_descending_stable(scores, k):
    ...
```

2. Requirements:
   - Return exactly `k` indices unless `k > n`, in which case return `n`.
   - Sort by score descending.
   - Break ties by original index ascending to mimic stable descending sort for
     equal values.
   - Use `np.argpartition` only as a candidate preselection step.
   - Stable-sort the selected candidates afterward.
3. Use the helper first in pure Python/NumPy tests, not in hot code.

Testing tasks:

1. Compare helper output to:

```python
np.argsort(-scores, kind="mergesort")[:k]
```

2. Test:
   - all equal values,
   - repeated ties,
   - negative values,
   - `k == 0`,
   - `k == 1`,
   - `k == n`,
   - `k > n`.

Acceptance criteria:

- Helper exactly matches current stable sort behavior on all test cases.
- Helper is documented as the only approved top-k replacement for compatibility
  paths.

## Phase 2: Coarse Nearest-Neighbor CPU Optimization

Purpose:

- Replace the manual all-pairs sparse dot-product loop with optimized SciPy
  sparse matrix multiplication while preserving neighbor selection semantics.

Current target:

- `modiscolite/affinitymat.py::_sparse_mm_dot`
- `modiscolite/affinitymat.py::cosine_similarity_from_seqlets`

Implementation tasks:

1. Keep the current `_sparse_mm_dot` as a reference implementation during
   development.
2. Add a new implementation, for example:

```python
def _sparse_mm_dot_scipy(X, Y, k):
    fwd = X @ X.T
    rev = X @ Y.T
    coarse = fwd.maximum(rev)
    ...
```

3. Carefully handle rows with fewer than `k` explicit nonzero entries:
   - Include all positive/nonzero sparse candidates.
   - Fill remaining slots from the full row index range using implicit zero
     scores.
   - Preserve self-neighbor behavior.
4. Use deterministic ordering:
   - score descending,
   - index ascending for ties.
5. Add a feature switch during development, for example:

```python
coarse_nn_backend = "numba" | "scipy"
```

6. Default to the original implementation until tests and benchmarks pass.

Testing tasks:

1. Unit-test the new implementation against `_sparse_mm_dot` on small matrices.
2. Include cases where:
   - every row has enough nonzero candidates,
   - some rows have fewer than `k` nonzero candidates,
   - all off-diagonal similarities are zero,
   - forward and reverse matrices choose different maxima,
   - ties occur at the top-k boundary.
3. Compare exact neighbor indices and similarity values.
4. Run an integration-level `cosine_similarity_from_seqlets` test on synthetic
   seqlets.

Benchmark tasks:

1. Benchmark old vs new on synthetic CSR matrices with realistic:
   - `n_seqlets`,
   - `max_entries`,
   - sparsity,
   - `k`.
2. Benchmark on a real or representative TF-MoDISco run.

Acceptance criteria:

- Exact match to reference implementation on deterministic tests.
- No regressions in existing tests.
- Meaningful speedup on realistic sizes.
- If exact match cannot be preserved due to sparse kernel ordering, document
  the difference and keep the original backend available.

## Phase 3: Fine Jaccard Max-only CPU Kernel

Purpose:

- Avoid allocating the full `(n_seqlets, k_neighbors, n_shifts)` tensor when
  `return_sparse=True`.

Current target:

- `modiscolite/affinitymat.py::jaccard`
- `modiscolite/affinitymat.py::_jaccard`

Implementation tasks:

1. Keep the existing path for `return_sparse=False`.
2. Add a new max-only Numba kernel for `return_sparse=True`, for example:

```python
@njit(parallel=True)
def _jaccard_max_only(X, Y, neighbors, out):
    ...
```

3. Preserve current behavior:
   - `X.astype("float32")`
   - `Y.astype("float32")`
   - same padding width and mode,
   - same min/max continuous Jaccard formula,
   - same neighbor indexing.
4. Keep output shape identical to current `scores.max(axis=-1)`.
5. Add a temporary fallback flag if useful:

```python
jaccard_backend = "tensor" | "max_only"
```

Testing tasks:

1. Compare max-only output to existing tensor path exactly or within tight
   tolerance.
2. Test:
   - no padding,
   - `min_overlap` padding,
   - forward and reverse data,
   - negative values,
   - zero arrays,
   - multiple neighbors,
   - multiple shifts with tied maxima.
3. Run `jaccard_from_seqlets` tests to confirm the combined forward/reverse max
   is unchanged.

Benchmark tasks:

1. Measure runtime and peak memory on synthetic dense seqlet arrays.
2. Measure within `seqlets_to_patterns` on a representative run.

Acceptance criteria:

- Matches current `return_sparse=True` output within tolerance.
- Existing `return_sparse=False` behavior remains unchanged.
- Peak memory drops materially for large `n_seqlets * k_neighbors * n_shifts`.
- Runtime improves or remains neutral while memory improves.

## Phase 4: Pairwise Jaccard Top-k Optimization

Purpose:

- Reduce sorting overhead in subpattern clustering while preserving top-k
  neighbor semantics.

Current target:

- `modiscolite/affinitymat.py::pairwise_jaccard`
- `modiscolite/core.py::SeqletSet.compute_subpatterns`

Implementation tasks:

1. Replace full `np.argsort` inside `pairwise_jaccard` with deterministic
   top-k selection.
2. Because `pairwise_jaccard` is Numba-compiled, evaluate two options:
   - implement a small Numba-compatible top-k insertion/select routine,
   - or compute scores in Numba and do stable top-k outside Numba.
3. Prefer exact behavior over maximum speed for the first implementation.

Testing tasks:

1. Compare old and new `pairwise_jaccard` outputs on small dense arrays.
2. Include ties, negative values, and all-zero rows.
3. Confirm `SeqletSet.compute_subpatterns` still produces valid subclusters on
   a small fixture.

Benchmark tasks:

1. Benchmark old vs new on synthetic flattened seqlet data.
2. Profile final `pattern.compute_subpatterns` before and after.

Acceptance criteria:

- Exact or documented-tolerance match on deterministic tests.
- No existing test regressions.
- Measurable speedup when `n` is large enough for sorting to matter.

## Phase 5: Pattern-merging Cache Cleanup

Purpose:

- Reduce repeated array conversion and repeated within-pattern similarity work
  in `SimilarPatternsCollapser`.

Current target:

- `modiscolite/aggregator.py::SimilarPatternsCollapser`

Implementation tasks:

1. Add iteration-scoped caches only.
2. Cache:
   - subsampled `SeqletSet` data arrays,
   - flattened pattern arrays,
   - within-pattern Jaccard distributions.
3. Clear caches after each merge round.
4. Do not cache across pattern mutation or polishing.

Testing tasks:

1. Add tests that run `SimilarPatternsCollapser` before/after on a small
   deterministic pattern set.
2. Verify outputs are unchanged or differences are understood.
3. Add a test that forces at least one merge round, so cache invalidation is
   exercised.

Benchmark tasks:

1. Profile `_detect_spurious_merging` before and after.
2. Measure with several values of `merging_max_seqlets_subsample`.

Acceptance criteria:

- No stale-cache behavior.
- No existing test regressions.
- Measurable reduction in repeated conversion/comparison cost.

## Phase 6: Optional GPU Prototype

Purpose:

- Evaluate whether GPU acceleration is worth maintaining after CPU
  optimization.

Backend recommendation:

- Start with CuPy, not PyTorch, for the first prototype because the package is
  currently NumPy/SciPy-shaped and CuPy has closer SciPy sparse interop.
- PyTorch remains viable for dense Jaccard kernels, especially for users whose
  attribution workflows already use torch, but it should not be the first
  compatibility-preserving backend.

Implementation tasks:

1. Define a backend boundary in `affinitymat.py`, limited initially to:
   - fine Jaccard max-only,
   - pairwise Jaccard.
2. Add optional dependency detection:

```python
try:
    import cupy as cp
except ImportError:
    cp = None
```

3. Do not import GPU libraries at package import time unless requested.
4. Implement a CuPy path for `jaccard(..., return_sparse=True)`.
5. Return NumPy arrays to existing CPU downstream code.
6. Keep CPU as the default backend.

Testing tasks:

1. Mark GPU tests as optional, skipped when CuPy/CUDA is unavailable.
2. Compare CuPy output to CPU max-only output within tolerance.
3. Test host/device transfer behavior.
4. Include small and medium tensor sizes.

Benchmark tasks:

1. Measure transfer overhead separately from kernel runtime.
2. Benchmark cases where data starts on CPU.
3. If torch is later considered, benchmark torch vs CuPy on the same dense
   Jaccard fixture, including transfer costs.

Acceptance criteria:

- CPU-only users see no dependency or behavior change.
- GPU output matches CPU output within tolerance.
- End-to-end speedup is positive after transfer overhead.
- The implementation is small enough to justify maintenance.

## Release Strategy

Recommended PR sequence:

1. Profiling and fixtures only.
2. Deterministic top-k helper and tests.
3. Coarse nearest-neighbor SciPy backend behind a switch.
4. Fine Jaccard max-only CPU path.
5. Pairwise Jaccard top-k optimization.
6. Pattern-merging cache improvements.
7. Optional GPU prototype.

Each PR should include:

- correctness tests,
- performance notes,
- a short explanation of expected numerical differences,
- rollback path or backend switch if behavior is not exactly preserved.

## Final Validation Checklist

Before making any optimized path the default:

1. Existing tests pass.
2. New affinity-level tests pass.
3. Old and new coarse neighbors match on deterministic fixtures.
4. Old and new fine Jaccard sparse outputs match within tolerance.
5. Old and new pairwise Jaccard outputs match within tolerance.
6. Density-adapted graph inputs are unchanged or differences are documented.
7. Representative end-to-end motif outputs are reviewed.
8. Benchmarks show improvement large enough to justify the change.
9. Documentation describes backend switches and expected differences.
