import numpy as np

from modiscolite import tfmodisco


def test_density_adaptation_matches_reference():
    affmat_nn = np.array([
        [1.0, 0.7, 0.2],
        [1.0, 0.6, 0.3],
        [1.0, 0.5, 0.4],
        [1.0, 0.8, 0.1],
    ], dtype=np.float64)
    seqlet_neighbors = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 1, 0],
        [3, 0, 1],
    ], dtype=np.int32)

    reference = tfmodisco._density_adaptation_reference(
        affmat_nn, seqlet_neighbors, tsne_perplexity=2.0)
    optimized = tfmodisco._density_adaptation(
        affmat_nn, seqlet_neighbors, tsne_perplexity=2.0,
        backend="optimized")

    np.testing.assert_array_equal(optimized.indptr, reference.indptr)
    np.testing.assert_array_equal(optimized.indices, reference.indices)
    np.testing.assert_allclose(optimized.data, reference.data)
