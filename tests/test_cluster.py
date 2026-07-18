import numpy as np
import scipy.sparse

from modiscolite import cluster


def _make_two_blob_affinity_mat(seed=0):
    # Two well-separated dense blocks plus a couple of cross-block edges,
    # weighted so Leiden has an unambiguous best partition to find but
    # enough seeds are needed for at least one to land on it.
    rng = np.random.RandomState(seed)
    n_per_block = 8
    n = 2 * n_per_block

    affmat = np.zeros((n, n))
    for block in range(2):
        idx = np.arange(block * n_per_block, (block + 1) * n_per_block)
        for i in idx:
            for j in idx:
                if i != j:
                    affmat[i, j] = 0.9 + 0.05 * rng.rand()

    affmat[0, n_per_block] = affmat[n_per_block, 0] = 0.01
    affmat[1, n_per_block + 1] = affmat[n_per_block + 1, 1] = 0.01

    return scipy.sparse.csr_matrix(affmat)


def test_leiden_cluster_parallel_matches_serial():
    affmat = _make_two_blob_affinity_mat()

    serial = cluster.LeidenCluster(affmat, n_seeds=12, n_jobs=1)
    parallel = cluster.LeidenCluster(affmat, n_seeds=12, n_jobs=2)

    np.testing.assert_array_equal(serial, parallel)


def test_leiden_cluster_parallel_matches_serial_many_jobs():
    affmat = _make_two_blob_affinity_mat(seed=1)

    serial = cluster.LeidenCluster(affmat, n_seeds=12, n_jobs=1)
    parallel = cluster.LeidenCluster(affmat, n_seeds=12, n_jobs=4)

    np.testing.assert_array_equal(serial, parallel)


def test_leiden_cluster_finds_two_blocks():
    affmat = _make_two_blob_affinity_mat()

    membership = cluster.LeidenCluster(affmat, n_seeds=12, n_jobs=1)

    assert len(set(membership[:8])) == 1
    assert len(set(membership[8:])) == 1
    assert membership[0] != membership[8]
