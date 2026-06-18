# Profiling Quickstart

This note gives a minimal way to collect stage-level runtime profiles for the
CPU optimization work. It assumes the project `.venv` has already been created
with `uv` and the package is installed editable.

## Environment

Use the project virtual environment:

```bash
UV_CACHE_DIR=/Users/adamhe/github/tfmodisco/.uv-cache \
  uv pip install --python .venv/bin/python -e '.[test]'
```

Use a writable matplotlib cache directory to avoid noisy import warnings:

```bash
export MPLCONFIGDIR=/private/tmp
```

## Existing Example Dataset

The current test fixture uses:

```text
tests/data/ohe1.npz
tests/data/hypscores1.npz
```

If those files are missing, run the test suite once with network access so
`tests/conftest.py` can download them:

```bash
MPLCONFIGDIR=/private/tmp .venv/bin/python -m pytest
```

## Smoke Profile

Use this to confirm profiling works and get a quick stage breakdown:

```bash
MPLCONFIGDIR=/private/tmp .venv/bin/python agent_notes/profile_baseline.py \
  --one-hot tests/data/ohe1.npz \
  --hypothetical-contribs tests/data/hypscores1.npz \
  --window 200 \
  --max-seqlets-per-metacluster 500 \
  --n-leiden-runs 5
```

Suggested use:

- Good for checking that the profiling path runs.
- Fast enough for repeated local development.
- Not enough seqlets or Leiden runs to represent full production cost.

## Quick Representative Profile

Use this for a more useful CPU optimization baseline:

```bash
MPLCONFIGDIR=/private/tmp .venv/bin/python agent_notes/profile_baseline.py \
  --one-hot tests/data/ohe1.npz \
  --hypothetical-contribs tests/data/hypscores1.npz \
  --window 200 \
  --max-seqlets-per-metacluster 2000 \
  --n-leiden-runs 10
```

Suggested use:

- Better for comparing Phase 1-3 affinity changes.
- Keeps runtime manageable.
- Uses fewer Leiden runs than the library default of 50 so affinity costs remain
  easier to inspect during development.

## Heavier Profile

Use this when an optimization looks promising and needs a stronger check:

```bash
MPLCONFIGDIR=/private/tmp .venv/bin/python agent_notes/profile_baseline.py \
  --one-hot tests/data/ohe1.npz \
  --hypothetical-contribs tests/data/hypscores1.npz \
  --window 200 \
  --max-seqlets-per-metacluster 5000 \
  --n-leiden-runs 25
```

Suggested use:

- Better signal for coarse nearest-neighbor and fine Jaccard scaling.
- Still below the default `max_seqlets_per_metacluster=20000`.
- Run before making an optimized backend the default.

## What To Look For

The profiler prints lines like:

```text
TFMoDISco.extract_seqlets: 0.123456s over 1 call(s)
seqlets_to_patterns.coarse_affinity: 1.234567s over 2 call(s)
seqlets_to_patterns.fine_affinity: 2.345678s over 2 call(s)
seqlets_to_patterns.density_adaptation: 0.345678s over 2 call(s)
seqlets_to_patterns.leiden_cluster: 0.456789s over 2 call(s)
seqlets_to_patterns.detect_spurious_merging: 0.567890s over 1 call(s)
compute_subpatterns.pairwise_jaccard: 0.678901s over 3 call(s)
```

For the planned CPU work, the most important lines are:

- `seqlets_to_patterns.coarse_affinity`
- `seqlets_to_patterns.fine_affinity`
- `compute_subpatterns.pairwise_jaccard`
- `seqlets_to_patterns.detect_spurious_merging`

## Comparing Changes

Run each profile at least twice before comparing numbers, because first-run
Numba compilation can dominate the first timing.

Recommended comparison flow:

1. Run the smoke profile once to warm Numba.
2. Run the quick representative profile twice on the baseline branch.
3. Apply one optimization.
4. Run the same quick representative profile twice.
5. Compare the second run from each side.

Keep the exact command, git commit, and profile output together in notes. Small
runtime differences are normal; large changes in final motif behavior should be
debugged at the affinity-test level before trusting end-to-end output.

## Backend Comparisons

The profiling helper exposes backend switches for the affinity experiments:

```bash
--coarse-affinity-backend auto|numba|inverted
--fine-affinity-backend max_only|tensor
--density-adaptation-backend auto|reference|optimized
```

Current recommendations:

- Use `--fine-affinity-backend max_only` for the optimized fine Jaccard path.
- Use `--fine-affinity-backend tensor` as the reference path for comparisons.
- Use `--density-adaptation-backend auto` for safe default behavior.
- Use `--density-adaptation-backend optimized` to test the Numba CSR density
  adaptation path on larger runs; it was slower on the small smoke profile.
- Use `--coarse-affinity-backend auto` for safe default behavior.
- Use `--coarse-affinity-backend inverted` to explicitly select the optimized
  exact inverted-index coarse path.
- Use `--coarse-affinity-backend numba` as the original reference path.

Example fine-affinity comparison:

```bash
MPLCONFIGDIR=/private/tmp .venv/bin/python agent_notes/profile_baseline.py \
  --one-hot tests/data/ohe1.npz \
  --hypothetical-contribs tests/data/hypscores1.npz \
  --window 200 \
  --max-seqlets-per-metacluster 500 \
  --n-leiden-runs 5 \
  --fine-affinity-backend tensor

MPLCONFIGDIR=/private/tmp .venv/bin/python agent_notes/profile_baseline.py \
  --one-hot tests/data/ohe1.npz \
  --hypothetical-contribs tests/data/hypscores1.npz \
  --window 200 \
  --max-seqlets-per-metacluster 500 \
  --n-leiden-runs 5 \
  --fine-affinity-backend max_only
```
