# tfmodisco Optimization Handoff

This document summarizes the main performance findings from a codebase review of
`modiscolite`, with an emphasis on CPU-only acceleration first and optional GPU
support later.

## Executive Summary

CPU-only acceleration is very plausible and should be the first optimization
track. The dominant runtime appears to be affinity construction:

1. Coarse nearest-neighbor search over gapped-kmer representations.
2. Fine sliding Jaccard affinity over each seqlet and its candidate neighbors.
3. Repeated all-pairs Jaccard during subclustering.
4. Pattern merging and AUROC checks as a second-tier bottleneck.

GPU support is feasible, especially for the Jaccard kernels, but a CPU-first
refactor should provide a cleaner baseline and likely substantial speedups
without new deployment complexity.

## Main Pipeline

The primary entry point is `modiscolite/tfmodisco.py::TFMoDISco`.

The expensive path is:

1. Extract seqlets from attribution tracks.
2. Split positive and negative seqlets.
3. For each metacluster, call `seqlets_to_patterns`.
4. Build coarse nearest neighbors from gapped-kmer cosine similarity.
5. Build fine affinity from sliding continuous Jaccard.
6. Filter by coarse/fine correlation.
7. Density-adapt the sparse affinity matrix.
8. Cluster with Leiden.
9. Build patterns from clusters.
10. Detect spurious merging and compute final subpatterns.

## Hotspots

### 1. Coarse nearest-neighbor search

Files:

- `modiscolite/affinitymat.py::_sparse_mm_dot`
- `modiscolite/affinitymat.py::cosine_similarity_from_seqlets`
- `modiscolite/gapped_kmer.py::_seqlet_to_gkmers`

Current behavior:

- Gapped-kmer features are represented as CSR matrices with shape
  `(n_seqlets, 5 ** max_len)`.
- `_sparse_mm_dot` computes all pairwise sparse dot products manually in Numba.
- For each row, it performs a full `np.argsort` and keeps the top `k`.
- It computes both forward-forward and forward-reverse similarities and takes
  the max.

Why it is expensive:

- Complexity is effectively all-pairs over seqlets.
- Full sorting is unnecessary when only top-k neighbors are needed.
- Manual sparse pairwise dot products may underuse optimized SciPy sparse
  kernels.

CPU-only candidate:

- Replace the manual all-pairs loop with sparse matrix multiplication:

```python
fwd = X @ X.T
rev = X @ Y.T
coarse = fwd.maximum(rev)
```

- Extract top-k per row from the resulting sparse matrix.
- Use `np.argpartition` where dense row top-k is unavoidable.
- Preserve output shape and ordering expectations:
  `(sims, neighbors)` with `k = min(n_neighbors + 1, n)`.

Notes:

- Behavior may change subtly if zero-valued missing entries are not handled like
  the current dense all-pairs implementation. The existing implementation can
  choose zero-similarity neighbors because it considers every pair. A sparse-only
  top-k implementation needs a fallback to fill missing neighbors if a row has
  fewer than `k` nonzero entries.

### 2. Fine sliding Jaccard affinity

Files:

- `modiscolite/affinitymat.py::jaccard`
- `modiscolite/affinitymat.py::_jaccard`
- `modiscolite/affinitymat.py::jaccard_from_seqlets`

Current behavior:

- `jaccard` pads `Y` when `min_overlap` is specified.
- It allocates:

```python
scores = np.zeros((Y.shape[0], seqlet_neighbors.shape[1], len_output), dtype="float32")
```

- `_jaccard` fills the full score tensor over all shifts.
- For `return_sparse=True`, only `scores.max(axis=-1)` is returned.

Why it is expensive:

- The full shift tensor is allocated even when only the maximum shift score is
  needed.
- The hot path in `seqlets_to_patterns` calls `jaccard_from_seqlets` with
  `return_sparse=True`, so most intermediate values are disposable.

CPU-only candidate:

- Add a specialized max-only kernel for `return_sparse=True`.
- Compute only the best score per `(seqlet, neighbor)` pair.
- Avoid allocating the full `n_seqlets x k_neighbors x n_shifts` tensor.

Expected benefit:

- Lower peak memory.
- Better cache behavior.
- Less NumPy allocation overhead.
- Same algorithmic result if implemented carefully.

### 3. Subpattern all-pairs Jaccard

Files:

- `modiscolite/core.py::SeqletSet.compute_subpatterns`
- `modiscolite/affinitymat.py::pairwise_jaccard`

Current behavior:

- Subpattern computation flattens seqlet data.
- `pairwise_jaccard` computes every pairwise Jaccard score.
- Each row is fully sorted with `np.argsort`, then truncated to `k`.

CPU-only candidate:

- Replace full row sorting with `np.argpartition` plus a small sort of only the
  selected top-k entries.
- Consider a parallel max/top-k strategy similar to the fine Jaccard max-only
  kernel.

Expected benefit:

- Reduced `O(n log n)` sorting overhead per row when only top-k is required.
- This is especially useful for large patterns during final subclustering.

### 4. Density adaptation

File:

- `modiscolite/tfmodisco.py::_density_adaptation`

Current behavior:

- Converts nearest-neighbor rows to CSR.
- Symmetrizes with its transpose.
- Performs per-row binary search for perplexity.
- Loops over every CSR edge to compute adapted affinities.

CPU-only candidate:

- Vectorize CSR construction from rectangular neighbor/value arrays.
- Convert `filtered_affmat_nn` and `seqlet_neighbors` to arrays earlier when
  possible.
- Consider Numba-compiling the final edge loop.

Priority:

- Medium. This may matter, but affinity construction is likely bigger.

### 5. Pattern merging and AUROC checks

File:

- `modiscolite/aggregator.py::SimilarPatternsCollapser`

Current behavior:

- Iterates over all directed pairs of patterns.
- Aligns patterns.
- Shifts seqlets.
- Rebuilds seqlet arrays.
- Computes between-pattern and within-pattern Jaccard distributions.
- Runs AUROC.

Why it is expensive:

- Quadratic in number of patterns.
- Can become quadratic in sampled seqlets within a pattern.
- Uses Python object loops heavily.

CPU-only candidates:

- Cache `util.get_2d_data_from_patterns(...)` results for subsampled patterns
  during an iteration.
- Avoid recomputing within-pattern similarities for the same pattern across
  multiple comparisons.
- Use `np.argpartition` or approximate screening to reduce pairwise comparisons
  before AUROC.

Priority:

- Second wave. Optimize affinity construction first, then profile again.

## CPU-Only Roadmap

Recommended order:

1. Add lightweight timing around major stages:
   - `cosine_similarity_from_seqlets`
   - `jaccard_from_seqlets`
   - `_density_adaptation`
   - `cluster.LeidenCluster`
   - `_patterns_from_clusters`
   - `_detect_spurious_merging`
2. Replace full sorts with top-k partial selection:
   - `_sparse_mm_dot`
   - `pairwise_jaccard`
   - gapped-kmer feature selection if measurable.
3. Replace manual sparse all-pairs coarse similarity with SciPy sparse matrix
   multiplication.
4. Add a max-only Jaccard path for `return_sparse=True`.
5. Add caching inside `SimilarPatternsCollapser`.
6. Re-profile and decide whether remaining bottlenecks justify GPU support.

## GPU Support Assessment

GPU acceleration is possible but should be built behind an optional backend,
for example:

```python
backend = "cpu" | "cuda" | "auto"
```

The most important design principle is to keep the existing Python object model
and CPU Leiden path stable while moving only the high-volume numeric kernels.
The first GPU implementation should return the same NumPy/SciPy-compatible
outputs that the rest of the pipeline already expects.

Best initial GPU candidates:

- `affinitymat.jaccard(..., return_sparse=True)`
- `affinitymat.pairwise_jaccard`
- Coarse nearest-neighbor search only if the gapped-kmer representation can be
  handled efficiently on GPU.

### GPU target 1: fine Jaccard max-only kernel

Files:

- `modiscolite/affinitymat.py::jaccard`
- `modiscolite/affinitymat.py::_jaccard`
- `modiscolite/affinitymat.py::jaccard_from_seqlets`

This is the best GPU target. The current hot path computes all shift scores for
each `(seqlet, neighbor)` pair and then keeps only the maximum when
`return_sparse=True`.

GPU shape:

- Inputs: dense `float32` arrays for forward/reverse seqlet data and `int32`
  neighbor indices.
- Output: dense `float32` array shaped `(n_seqlets, k_neighbors)`.
- One kernel can assign work by `(seqlet, neighbor)` and loop over shifts, or by
  `(seqlet, neighbor, shift)` with a reduction. The first option avoids writing
  the full shift tensor and is likely simpler.

Compatibility goal:

- Preserve current padding semantics.
- Preserve `float32` input dtype.
- Match `scores.max(axis=-1)` from the CPU implementation within tolerance.

### GPU target 2: pairwise subpattern Jaccard

Files:

- `modiscolite/core.py::SeqletSet.compute_subpatterns`
- `modiscolite/affinitymat.py::pairwise_jaccard`

This kernel computes all pairwise continuous Jaccard scores on flattened seqlet
data, then keeps top-k neighbors. It is also GPU-friendly, but should come after
the fine-affinity path because it is used later and on smaller subsets.

GPU shape:

- Input: dense `float32` or `float64` flattened seqlet data.
- Output: `(jaccards, neighbors)` with shape `(n, k)`.
- A custom kernel or batched tensor operation can compute the pairwise scores.
- Top-k selection can use the backend's top-k primitive if tie behavior is
  acceptable, or return scores to CPU for deterministic tie-breaking.

### GPU target 3: coarse nearest-neighbor search

Files:

- `modiscolite/affinitymat.py::cosine_similarity_from_seqlets`
- `modiscolite/gapped_kmer.py::_seqlet_to_gkmers`

This is attractive because it is very expensive, but it is less straightforward
than Jaccard:

- Gapped-kmer features are extremely high-dimensional CSR matrices.
- The feature dimension is `5 ** max_len`, which is huge even though each row is
  sparse.
- The current implementation compares every pair and includes zero-similarity
  candidates, so sparse GPU methods need deterministic fallback behavior.

Possible GPU strategies:

- Use CuPy sparse CSR operations for `X @ X.T` and `X @ Y.T`.
- Use FAISS GPU only if the representation is transformed into a dense or
  compact embedding where inner-product search is meaningful.
- Keep coarse nearest-neighbor search CPU-only initially and GPU only the fine
  Jaccard kernel.

Less attractive initial GPU candidates:

- Python object orchestration in `core.py` and `aggregator.py`.
- Leiden clustering, because the existing dependency is CPU-based
  `igraph`/`leidenalg`.

### CuPy backend

CuPy is probably the best fit for the first CUDA implementation.

Pros:

- NumPy-like API, so code can stay close to existing NumPy implementation.
- Supports `cupyx.scipy.sparse`, which maps naturally to the current
  SciPy-sparse code.
- Offers raw CUDA kernels when vectorized CuPy expressions are not enough.
- Easy conversion back to NumPy/SciPy for existing downstream code.

Cons:

- Adds a CUDA-specific dependency.
- Sparse API coverage is not identical to SciPy.
- Some operations may require explicit host/device transfers.

Best use:

- Implement `jaccard(..., return_sparse=True)` max-only CUDA path.
- Consider sparse coarse affinity after the CPU sparse-matmul path is stable.

### PyTorch backend

PyTorch is viable for some GPU work, but it is not as natural a fit as CuPy for
this package.

Good fits:

- Dense tensor math for fine Jaccard and pairwise Jaccard.
- `torch.topk` for GPU top-k selection.
- Environments where users already have PyTorch installed for attribution/model
  inference.
- Future integration with model-analysis pipelines that already keep tensors on
  GPU.

Weak fits:

- SciPy-compatible sparse matrix interop is less direct.
- PyTorch sparse support is improving but is not a drop-in replacement for
  SciPy CSR workflows.
- The package currently has no torch dependency, and adding torch is a much
  heavier dependency than adding optional CuPy.
- Exact tie handling in `torch.topk` may differ from NumPy stable sort.
- Moving results back to SciPy/NumPy is still required for density adaptation
  and CPU Leiden unless those are also ported.

Recommendation:

- PyTorch is viable if the main user base already runs torch-based attribution
  workflows and wants fewer GPU libraries.
- CuPy is preferable for an implementation that wants to look like the existing
  NumPy/SciPy code and keep sparse-matrix options open.
- A clean backend abstraction could support both later, but the first GPU
  backend should be one implementation, not two.

### FAISS backend

FAISS GPU may help with nearest-neighbor search if the coarse representation is
converted to a dense or compact vector form. It is not a direct replacement for
the current sparse gapped-kmer dot-product path.

Pros:

- Excellent GPU top-k nearest-neighbor performance.
- Useful for inner product / cosine search over dense vectors.

Cons:

- Does not naturally preserve the current sparse gapped-kmer semantics.
- Approximate indexes may change neighbor sets.
- Exact flat search may require dense matrices that are too large.

Recommendation:

- Treat FAISS as an optional experimental coarse-neighbor backend, not the first
  compatibility-preserving optimization.

### RAPIDS cuGraph backend

RAPIDS cuGraph includes Leiden for undirected weighted graphs, so it could
replace `igraph`/`leidenalg` eventually.

Pros:

- Moves graph clustering to GPU.
- Natural fit if affinity construction also stays on GPU.

Cons:

- Large dependency footprint.
- Likely different clustering implementation details and stochastic behavior.
- Requires converting the adapted affinity graph into cuGraph data structures.

Recommendation:

- Do not start here. Keep Leiden on CPU until affinity construction is faster
  and profiling shows clustering is the remaining bottleneck.

### Numba CUDA

Avoid relying on the old built-in `numba.cuda` path for new work. If using Numba
for CUDA, evaluate NVIDIA's separate `numba-cuda` package.

Numba CUDA may still be useful for small custom kernels, but CuPy raw kernels or
a dedicated extension are likely cleaner for long-term maintenance.

## Expected Differences From CPU Optimizations

The CPU optimization path can be made close to behavior-preserving, but it
should be treated as numerically compatible rather than guaranteed bit-for-bit
identical.

### Highest-risk differences

#### Sparse matmul coarse nearest neighbors

The current `_sparse_mm_dot` implementation compares every seqlet pair,
including pairs with zero similarity, and then stable-sorts the full dense row.
A sparse matrix multiplication replacement naturally sees only nonzero entries.

Risk:

- Rows with fewer than `k` nonzero candidates may get different fallback
  neighbors unless explicitly filled.
- Zero-score tie ordering may differ.
- Different neighbor sets can change fine affinities, density adaptation,
  Leiden clusters, and final motifs.

Mitigation:

- Preserve `k = min(n_neighbors + 1, n)`.
- Always include the self-neighbor.
- Fill missing candidates from the full index range in the same order the old
  stable descending sort would have produced for zero scores.
- For selected candidates, sort by `(-score, index)` to make tie behavior
  deterministic.

#### `argpartition` top-k selection

Replacing full `np.argsort(..., kind="mergesort")` with `np.argpartition` can
change ties because `argpartition` is not stable.

Risk:

- Equal-score neighbors can appear in a different order.
- If only the first `k` tied neighbors are kept, the selected set can differ.

Mitigation:

- Use `argpartition` only to get a candidate superset.
- Then run a deterministic sort on that small set using score descending and
  original index ascending.
- Add tests with deliberate ties.

#### Fine Jaccard max-only path

The proposed max-only CPU kernel should be exact in principle, but it changes
evaluation structure.

Risk:

- Differences in dtype, accumulation order, or NaN/zero-denominator behavior can
  alter scores near thresholds.
- If the optimized path updates the running maximum differently from
  `scores.max(axis=-1)`, tie behavior for alignments may differ in non-sparse
  uses.

Mitigation:

- Keep the existing tensor path for `return_sparse=False`.
- For `return_sparse=True`, match current input casts to `float32`.
- Compare the new max-only output against the old tensor output within tight
  tolerance on deterministic fixtures.

### Medium-risk differences

#### Floating-point summation order

Sparse matrix multiplication and vectorized kernels may sum products in a
different order than the current explicit loops.

Risk:

- Tiny differences can change rank order when scores are nearly tied.
- Downstream clustering can amplify small graph changes.

Mitigation:

- Validate both affinity matrices and final motifs.
- Use deterministic tie-breaking so tiny differences do not produce arbitrary
  ordering changes.

#### Density adaptation refactors

Vectorizing `_density_adaptation` can change duplicate handling, sparse row
ordering, or zero elimination behavior.

Risk:

- Adapted graph weights can differ even when raw affinity values match.

Mitigation:

- Keep CSR construction semantics unchanged unless separately tested.
- Compare CSR `.indices`, `.indptr`, and `.data` against baseline on small
  examples.

#### Pattern-merging caches

Caching in `SimilarPatternsCollapser` is safe only within a well-defined
iteration.

Risk:

- `SeqletSet` objects are copied, polished, and replaced during merging.
- A cache that outlives the current pattern state can reuse stale arrays.

Mitigation:

- Scope caches to one merge iteration.
- Key caches by object identity only when object mutation/replacement is
  understood.
- Clear caches after each merge round.

### Low-risk differences

- Timing instrumentation should not affect results unless it changes random
  state or evaluation order.
- Replacing full sorting in `pairwise_jaccard` is low risk if deterministic
  tie-breaking is preserved.
- Caching pure conversions from seqlet objects to arrays is low risk when cache
  lifetime is short.

### Validation strategy for differences

Use a layered validation approach:

1. Compare old and new affinity outputs directly.
2. Compare neighbor sets and order, including tie cases and zero-similarity
   rows.
3. Compare density-adapted CSR graph structure and weights.
4. Compare Leiden inputs separately from Leiden outputs.
5. Compare final motifs only after lower-level numeric outputs are understood.

Final motifs should be considered an integration-level check, not the first
debugging signal, because clustering can amplify small affinity changes.

## Correctness Risks

Important behavior to preserve:

- Neighbor arrays include self-neighbors because callers use
  `n_neighbors + 1`.
- Coarse affinity currently considers every pair, including zero-similarity
  pairs. A sparse matrix multiplication replacement must handle rows with fewer
  than `k` nonzero candidates.
- Sorting stability may affect tie handling. Current code uses
  `kind="mergesort"` in several places.
- Floating-point differences in Jaccard can affect downstream clustering.
- Leiden clustering is stochastic across seeds, so validation should compare
  affinities directly as well as high-level motif outcomes.

## Suggested Tests

Add small deterministic tests for:

- `_sparse_mm_dot` replacement returns the same neighbor sets and similarities
  on tiny CSR matrices, including rows with all-zero similarities.
- `jaccard(..., return_sparse=True)` max-only path matches the current tensor
  path within tolerance.
- `pairwise_jaccard` top-k partial selection matches the current full-sort path,
  including ties if preserving tie behavior is required.
- `seqlets_to_patterns` produces stable outputs on a small fixture, allowing
  reasonable tolerance for floating-point and clustering variability.

## Practical Recommendation

Start with CPU-only work. The likely best first patch is:

1. Add timing instrumentation controlled by `verbose` or a small internal
   profiling helper.
2. Implement `argpartition` top-k helpers.
3. Replace coarse nearest-neighbor all-pairs loops with sparse matrix
   multiplication plus careful sparse/dense top-k fill.
4. Implement a max-only Jaccard kernel for the `return_sparse=True` path.

That path should reduce runtime and memory pressure while keeping deployment
simple. Once that lands, benchmark real analyses and decide whether a CUDA
backend is still worth the maintenance cost.
