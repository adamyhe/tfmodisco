import numpy as np
import scipy.sparse

from modiscolite import affinitymat


def test_topk_stable_matches_full_stable_argsort():
    cases = [
        (np.array([1.0, 3.0, 2.0, 3.0, 0.0]), 3),
        (np.array([0.0, 0.0, 0.0, 0.0]), 2),
        (np.array([-1.0, -3.0, -2.0, -1.0, 0.0]), 4),
        (np.array([5.0, 1.0, 5.0, 4.0, 4.0, 3.0]), 4),
        (np.array([2.0, 1.0, 0.0]), 3),
    ]

    for scores, k in cases:
        expected = np.argsort(-scores, kind="mergesort")[:k]
        np.testing.assert_array_equal(
            affinitymat._topk_stable(scores, k), expected)
        np.testing.assert_array_equal(
            affinitymat._topk_stable_numba(scores, k), expected)


def test_sparse_mm_dot_baseline_with_zero_similarity_rows():
    X = scipy.sparse.csr_matrix(np.array([
        [1.0, 0.0, 2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]))
    Y = scipy.sparse.csr_matrix(np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]))

    X.indices = X.indices.astype(np.int64)
    X.indptr = X.indptr.astype(np.int64)
    Y.indices = Y.indices.astype(np.int64)
    Y.indptr = Y.indptr.astype(np.int64)

    sims, neighbors = affinitymat._sparse_mm_dot(
        X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, 3)

    np.testing.assert_allclose(sims, np.array([
        [5.0, 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]))
    np.testing.assert_array_equal(neighbors, np.array([
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [0, 1, 2],
    ], dtype=np.int32))

    scipy_sims, scipy_neighbors = affinitymat._sparse_mm_dot_scipy(X, Y, 3)
    np.testing.assert_allclose(scipy_sims, sims)
    np.testing.assert_array_equal(scipy_neighbors, neighbors)

    X_keys, X_starts, X_ends, X_rows, X_values = affinitymat._build_inverted_index(X)
    Y_keys, Y_starts, Y_ends, Y_rows, Y_values = affinitymat._build_inverted_index(Y)
    inverted_sims, inverted_neighbors = affinitymat._sparse_mm_dot_inverted(
        X.data, X.indices, X.indptr,
        X_keys, X_starts, X_ends, X_rows, X_values,
        Y_keys, Y_starts, Y_ends, Y_rows, Y_values, 3)
    np.testing.assert_allclose(inverted_sims, sims)
    np.testing.assert_array_equal(inverted_neighbors, neighbors)


def test_sparse_mm_dot_scipy_matches_numba_with_negative_values():
    X = scipy.sparse.csr_matrix(np.array([
        [1.0, 0.0, -2.0],
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]))
    Y = scipy.sparse.csr_matrix(np.array([
        [-1.0, 0.0, 2.0],
        [1.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
    ]))

    X.indices = X.indices.astype(np.int64)
    X.indptr = X.indptr.astype(np.int64)
    Y.indices = Y.indices.astype(np.int64)
    Y.indptr = Y.indptr.astype(np.int64)

    numba_sims, numba_neighbors = affinitymat._sparse_mm_dot(
        X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, 3)
    scipy_sims, scipy_neighbors = affinitymat._sparse_mm_dot_scipy(X, Y, 3)
    X_keys, X_starts, X_ends, X_rows, X_values = affinitymat._build_inverted_index(X)
    Y_keys, Y_starts, Y_ends, Y_rows, Y_values = affinitymat._build_inverted_index(Y)
    inverted_sims, inverted_neighbors = affinitymat._sparse_mm_dot_inverted(
        X.data, X.indices, X.indptr,
        X_keys, X_starts, X_ends, X_rows, X_values,
        Y_keys, Y_starts, Y_ends, Y_rows, Y_values, 3)

    np.testing.assert_allclose(scipy_sims, numba_sims)
    np.testing.assert_array_equal(scipy_neighbors, numba_neighbors)
    np.testing.assert_allclose(inverted_sims, numba_sims)
    np.testing.assert_array_equal(inverted_neighbors, numba_neighbors)


def test_jaccard_baselines_for_sparse_and_full_outputs():
    X = np.array([
        [[1.0, -0.5], [0.0, 0.5], [2.0, 0.0]],
        [[0.5, 0.5], [-1.0, 1.0], [0.0, 0.0]],
    ])
    Y = np.array([
        [[0.5, -0.5], [0.0, 0.5], [1.5, 0.0], [0.0, 1.0]],
        [[0.5, 0.0], [-1.0, 1.0], [0.0, 0.5], [0.5, 0.5]],
    ])
    neighbors = np.array([[0, 1], [1, 0]], dtype=np.int32)

    sparse_scores = affinitymat.jaccard(
        X, Y, min_overlap=0.5, seqlet_neighbors=neighbors,
        return_sparse=True)
    tensor_sparse_scores = affinitymat.jaccard(
        X, Y, min_overlap=0.5, seqlet_neighbors=neighbors,
        return_sparse=True, sparse_backend="tensor")
    full_scores = affinitymat.jaccard(
        X, Y, min_overlap=0.5, seqlet_neighbors=neighbors,
        return_sparse=False)

    np.testing.assert_allclose(sparse_scores, np.array([
        [0.75, 0.375],
        [0.71428573, 0.16666667],
    ]))
    np.testing.assert_allclose(sparse_scores, tensor_sparse_scores)
    np.testing.assert_allclose(full_scores, np.array([
        [[0.75, 0.0], [0.375, 2.0]],
        [[0.71428573, 0.0], [0.16666667, 0.0]],
    ]))


def test_pairwise_jaccard_baseline_with_ties_and_zero_row():
    X = np.array([
        [1.0, 0.0, -1.0],
        [0.5, 0.0, -0.5],
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])

    jaccards, neighbors = affinitymat.pairwise_jaccard(X, 3)

    np.testing.assert_allclose(jaccards, np.array([
        [1.0, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 0.0, -0.5],
        [0.0, 0.0, 0.0],
    ]))
    np.testing.assert_array_equal(neighbors, np.array([
        [0, 1, 3],
        [1, 0, 3],
        [2, 3, 1],
        [0, 1, 2],
    ], dtype=np.int32))
