# cluster.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar

import leidenalg
import numpy as np
import igraph as ig

from joblib import Parallel, delayed


def _find_partition(graph, weights, seed, n_leiden_iterations):
    partition = leidenalg.find_partition(
        graph=graph,
        partition_type=leidenalg.ModularityVertexPartition,
        weights=weights,
        n_iterations=n_leiden_iterations,
        initial_membership=None,
        seed=seed)

    quality = np.array(partition.quality())
    membership = np.array(partition.membership)
    return quality, membership


def _leiden_seed_partition(sources, targets, weights, n_vertices, seed,
        n_leiden_iterations):
    # Graph is rebuilt per call (instead of pickling a shared igraph.Graph)
    # so that parallel worker processes only need cheap, memmap-friendly
    # numpy arrays; each process ends up with its own copy of the graph.
    g = ig.Graph(directed=None)
    g.add_vertices(n_vertices)
    g.add_edges(zip(sources, targets))
    return _find_partition(g, weights, seed, n_leiden_iterations)


def LeidenCluster(affinity_mat, n_seeds=2, n_leiden_iterations=-1, n_jobs=1):
    n_vertices = affinity_mat.shape[0]
    n_cols = affinity_mat.indptr
    sources = np.concatenate([np.ones(n_cols[i+1] - n_cols[i], dtype='int32') * i for i in range(n_vertices)])
    targets = affinity_mat.indices
    weights = affinity_mat.data

    seeds = [seed * 100 for seed in range(1, n_seeds + 1)]

    if n_jobs == 1:
        # Build the graph once and reuse it across all seeds -- same cost
        # as the original serial implementation.
        g = ig.Graph(directed=None)
        g.add_vertices(n_vertices)
        g.add_edges(zip(sources, targets))
        results = [_find_partition(g, weights, seed, n_leiden_iterations)
            for seed in seeds]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_leiden_seed_partition)(sources, targets, weights,
                n_vertices, seed, n_leiden_iterations) for seed in seeds)

    best_clustering = None
    best_quality = None

    # Reduce in seed order (not completion order) so that ties are broken
    # identically to the original serial loop regardless of n_jobs.
    for quality, membership in results:
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_clustering = membership

    return best_clustering
