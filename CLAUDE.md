# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TF-MoDISco (Transcription Factor Motif Discovery from Importance Scores) discovers sequence motifs from
per-nucleotide importance/attribution scores (e.g. DeepLIFT, SHAP) computed on genomic sequences. The
installed package/CLI is called `modisco`; the Python package is `modiscolite` (the "lite" reimplementation
described in the README — this is v2 of the project, not the original `tfmodisco` v0).

## Setup and commands

Install the package editable with test dependencies (uses standard `pip`/`venv`, or `uv`):

```bash
pip install -e '.[test]'
```

Run the full test suite:

```bash
pytest
```

Run a single test file or test:

```bash
pytest tests/test_motif.py
pytest tests/test_motif.py::test_modisco_motif
```

Note: `tests/conftest.py` downloads a shared fixture dataset (`tests/data/ohe1.npz`,
`tests/data/hypscores1.npz`) from a remote URL on first run (session-scoped fixture `data_ohe_hyps`), so the
first test run needs network access; subsequent runs reuse the cached files.

CLI entry point (installed via `pyproject.toml` `[project.scripts]`): `modisco`, backed by
`modiscolite/cli.py`. Subcommands: `motifs` (run discovery), `report` / `report-simple` (HTML reports),
`convert` / `convert-backward` (HDF5 format migration), `meme` (export MEME motif files), `seqlet-bed` /
`seqlet-fasta` (export seqlet coordinates/sequences).

## Architecture

The pipeline has two conceptual stages, both driven from `modiscolite/tfmodisco.py`:

1. **Seqlet extraction** (`extract_seqlets.py`): given per-position contribution scores
   (`one_hot * hypothetical_contribs`, summed over channels), find high-importance windows ("seqlets") using
   a Laplacian-null FDR threshold, then split them into positive and negative sets based on total
   attribution ("metaclusters"). Entry point: `TFMoDISco()`.

2. **Seqlets → patterns** (`tfmodisco.py::seqlets_to_patterns`, called once per metacluster sign): the core
   clustering loop, run twice (`round_idx in range(2)`) to refine seqlet membership:
   - **Coarse affinity** (`affinitymat.cosine_similarity_from_seqlets`): approximate nearest neighbors via
     cosine similarity between gapped k-mer representations (`gapped_kmer.py`).
   - **Fine affinity** (`affinitymat.jaccard_from_seqlets`): exact continuous Jaccard similarity between each
     seqlet and its coarse neighbors, computed by sliding alignment over all offsets satisfying
     `min_overlap_while_sliding`.
   - **Correlation filtering** (round 0 only): drop seqlets whose fine/coarse affinity rows correlate poorly
     (`affmat_correlation_threshold`), since these are likely noise.
   - **Density adaptation** (`_density_adaptation`): t-SNE-style perplexity-based rescaling of the sparse
     affinity graph (symmetrized, RBF-kernelized via per-row binary-searched beta).
   - **Leiden clustering** (`cluster.LeidenCluster`): community detection on the density-adapted graph,
     run with multiple random seeds (`n_leiden_runs`) for stability.
   - **Pattern construction** (`_patterns_from_clusters` → `aggregator.merge_in_seqlets_filledges` +
     `aggregator.polish_pattern`): seqlets in each cluster are aligned/merged into a `core.SeqletSet`
     (PFM/CWM/hCWM), then trimmed to a high-information-content window.

   After the two refinement rounds: `aggregator._detect_spurious_merging` subclusters each pattern to check
   whether it should be split; `_filter_patterns` drops low-support or low-information-content patterns; each
   surviving pattern gets flanks re-expanded and `pattern.compute_subpatterns` run for reporting.

3. **Output/reporting**: `io.py` reads/writes the HDF5 result format (`pos_patterns/neg_patterns` groups of
   `pattern_i/{sequence,contrib_scores,hypothetical_contribs,seqlets/...}`, see README for the full layout)
   and converts to/from the legacy format. `report.py` / `descriptive_report.py` render HTML reports (logos
   via `logomaker`, TOMTOM motif matches via `memelite`) from an HDF5 results file — `report.py` is the
   legacy/"simple" report, `descriptive_report.py` (+ `templates/descriptive_report.html`) is the current
   default. `meme_writer.py`, `bed_writer.py`, `fasta_writer.py` handle the `meme`/`seqlet-bed`/`seqlet-fasta`
   CLI export subcommands.

Core data types live in `core.py`: `TrackSet` holds the full one-hot/contrib/hypothetical-contrib arrays and
constructs `Seqlet`s from coordinates; a `SeqletSet` (pattern) aggregates seqlets into PFM/CWM/hCWM summary
tracks and can recursively hold `subpattern_i` `SeqletSet`s.

### Pluggable backends and profiling

Several hot paths in `affinitymat.py` and `tfmodisco.py` have multiple implementations selected by a
`backend` argument (`'auto'` picks based on input size), used to compare optimized vs. reference numerics
without changing default behavior:

- `coarse_affinity_backend`: `'numba'` (original) vs `'inverted'` (exact inverted-index) — `auto` chooses.
- `fine_affinity_backend`: `'max_only'` (optimized) vs `'tensor'` (reference).
- `density_adaptation_backend`: `'reference'` vs `'optimized'` (Numba CSR) — `auto` chooses by graph size.

`util.ProfileRecorder` provides a `with profiler.time("label"):` context manager used throughout
`tfmodisco.py` to record per-stage timings; pass `profile=True` to `TFMoDISco`/`seqlets_to_patterns` to print
a summary, or pass an existing `ProfileRecorder` to accumulate across calls. `agent_notes/` contains a
profiling harness (`profile_baseline.py`) and a walkthrough (`profiling_quickstart.md`) for benchmarking
these backends against each other — read those before making backend-switching or performance-sensitive
changes to `affinitymat.py`, `cluster.py`, or the density adaptation code. `agent_notes/optimization_handoff.md`
and `optimization_implementation_plan.md` record the current CPU-optimization plan and rationale (coarse NN
search, fine Jaccard, subclustering Jaccard, and pattern-merging are the known bottlenecks).

Numba (`@njit`) is used for hot inner loops (`affinitymat.py`, `tfmodisco.py` density adaptation, `cluster.py`
LeidenCluster) — first invocation in a process pays JIT compilation cost, which matters when comparing
profiling runs (run twice, compare second run).
