# affinitymat.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import sklearn
import sklearn.manifold

import numpy as np

import scipy
from scipy.sparse import coo_matrix

from numba import njit
from numba import prange

from . import util
from . import gapped_kmer


@njit('float64(float64[:], int64[:], int64[:], float64[:], int64[:], int64[:], int64, int64)')
def _sparse_vv_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, i, j):
	xi = X_indptr[i]
	yj = Y_indptr[j]
	dot = 0.0

	while xi < X_indptr[i+1] and yj < Y_indptr[j+1]:
		x_col = X_indices[xi]
		x_data = X_data[xi]

		y_col = Y_indices[yj]
		y_data = Y_data[yj]

		if x_col == y_col:
			dot += x_data * y_data
			xi += 1
			yj += 1

		elif x_col < y_col:
			xi += 1

		else:
			yj += 1

	return dot

@njit(parallel=True)
def _sparse_mm_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, k):
	n_rows = len(Y_indptr) - 1

	neighbors = np.empty((n_rows, k), dtype='int32')
	sims = np.empty((n_rows, k), dtype='float64')

	for i in prange(n_rows):
		dot = np.zeros(n_rows, dtype='float64')

		for j in range(n_rows):
			xdot = _sparse_vv_dot(X_data, X_indices, X_indptr, X_data, X_indices, X_indptr, i, j)
			ydot = _sparse_vv_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, i, j)
			dot[j] = max(xdot, ydot)

		dot_argsort = np.argsort(-dot, kind='mergesort')[:k]
		neighbors[i] = dot_argsort
		sims[i] = dot[dot_argsort]

	return sims, neighbors

@njit
def _argsort_stable_numba(scores, k):
	idxs = np.argsort(-scores, kind='mergesort')[:k]
	result = np.empty(k, dtype='int32')
	for i in range(k):
		result[i] = idxs[i]

	return result

@njit
def _topk_stable_numba(scores, k):
	n = len(scores)
	if k >= n or n < 2000:
		return _argsort_stable_numba(scores, min(k, n))

	kth_score = np.partition(scores, n-k)[n-k]
	above_idxs = np.empty(k, dtype='int32')
	above_scores = np.empty(k, dtype='float64')
	n_above = 0

	for idx in range(n):
		score = scores[idx]
		if score > kth_score:
			above_scores[n_above] = score
			above_idxs[n_above] = idx
			n_above += 1

	result = np.empty(k, dtype='int32')
	above_order = np.argsort(-above_scores[:n_above], kind='mergesort')
	for idx in range(n_above):
		result[idx] = above_idxs[above_order[idx]]

	result_idx = n_above
	for idx in range(n):
		if scores[idx] == kth_score:
			result[result_idx] = idx
			result_idx += 1
			if result_idx == k:
				break

	return result

def _topk_stable(scores, k):
	if k >= len(scores) or len(scores) < 2000:
		return np.argsort(-scores, kind='mergesort')[:k]

	kth_score = np.partition(scores, len(scores)-k)[len(scores)-k]
	above = np.flatnonzero(scores > kth_score)
	tied = np.flatnonzero(scores == kth_score)
	above = above[np.argsort(-scores[above], kind='mergesort')]
	return np.concatenate([above, tied[:k-len(above)]])

def _merge_sparse_rows_to_dense(X_row, Y_row, n_cols):
	x_dot = np.zeros(n_cols, dtype='float64')
	y_dot = np.zeros(n_cols, dtype='float64')

	if X_row.nnz > 0:
		x_dot[X_row.indices] = X_row.data

	if Y_row.nnz > 0:
		y_dot[Y_row.indices] = Y_row.data

	return np.maximum(x_dot, y_dot)

def _sparse_mm_dot_scipy(X, Y, k, block_size=32):
	n_rows = X.shape[0]

	neighbors = np.empty((n_rows, k), dtype='int32')
	sims = np.empty((n_rows, k), dtype='float64')

	for start in range(0, n_rows, block_size):
		end = min(start + block_size, n_rows)
		X_block = X[start:end]
		fwd = X_block @ X.T
		rev = X_block @ Y.T

		for block_i, i in enumerate(range(start, end)):
			dot = _merge_sparse_rows_to_dense(
				fwd.getrow(block_i), rev.getrow(block_i), n_rows)
			dot_argsort = _topk_stable(dot, k)
			neighbors[i] = dot_argsort.astype('int32')
			sims[i] = dot[dot_argsort]

	return sims, neighbors

def _build_inverted_index(X):
	row_counts = np.diff(X.indptr)
	rows = np.repeat(np.arange(X.shape[0], dtype='int64'), row_counts)
	keys = X.indices.astype('int64', copy=False)
	data = X.data.astype('float64', copy=False)

	order = np.argsort(keys, kind='mergesort')
	sorted_keys = keys[order]
	sorted_rows = rows[order]
	sorted_data = data[order]

	unique_keys, starts = np.unique(sorted_keys, return_index=True)
	ends = np.empty_like(starts)
	ends[:-1] = starts[1:]
	ends[-1] = len(sorted_keys)

	return unique_keys, starts.astype('int64'), ends.astype('int64'), sorted_rows, sorted_data

@njit
def _find_key_idx(keys, key):
	left = 0
	right = len(keys)

	while left < right:
		mid = (left + right) // 2
		if keys[mid] < key:
			left = mid + 1
		else:
			right = mid

	if left < len(keys) and keys[left] == key:
		return left

	return -1

@njit
def _sparse_mm_dot_inverted(X_data, X_indices, X_indptr,
	X_keys, X_starts, X_ends, X_rows, X_values,
	Y_keys, Y_starts, Y_ends, Y_rows, Y_values, k):

	n_rows = len(X_indptr) - 1

	neighbors = np.empty((n_rows, k), dtype='int32')
	sims = np.empty((n_rows, k), dtype='float64')

	xdot = np.zeros(n_rows, dtype='float64')
	ydot = np.zeros(n_rows, dtype='float64')
	x_seen = np.zeros(n_rows, dtype='int64')
	y_seen = np.zeros(n_rows, dtype='int64')

	pos_idxs = np.empty(n_rows, dtype='int32')
	pos_scores = np.empty(n_rows, dtype='float64')
	neg_idxs = np.empty(n_rows, dtype='int32')
	neg_scores = np.empty(n_rows, dtype='float64')

	for i in range(n_rows):
		epoch = i + 1

		for row_idx in range(X_indptr[i], X_indptr[i+1]):
			col = X_indices[row_idx]
			value = X_data[row_idx]

			x_key_idx = _find_key_idx(X_keys, col)
			if x_key_idx != -1:
				for idx in range(X_starts[x_key_idx], X_ends[x_key_idx]):
					row = X_rows[idx]
					if x_seen[row] != epoch:
						x_seen[row] = epoch
						xdot[row] = 0.0

					xdot[row] += value * X_values[idx]

			y_key_idx = _find_key_idx(Y_keys, col)
			if y_key_idx != -1:
				for idx in range(Y_starts[y_key_idx], Y_ends[y_key_idx]):
					row = Y_rows[idx]
					if y_seen[row] != epoch:
						y_seen[row] = epoch
						ydot[row] = 0.0

					ydot[row] += value * Y_values[idx]

		n_pos = 0
		n_neg = 0
		for row in range(n_rows):
			x_score = 0.0
			y_score = 0.0
			if x_seen[row] == epoch:
				x_score = xdot[row]
			if y_seen[row] == epoch:
				y_score = ydot[row]

			score = max(x_score, y_score)
			if score > 0.0:
				pos_idxs[n_pos] = row
				pos_scores[n_pos] = score
				n_pos += 1
			elif score < 0.0:
				neg_idxs[n_neg] = row
				neg_scores[n_neg] = score
				n_neg += 1

		result_count = 0

		pos_order = np.argsort(-pos_scores[:n_pos], kind='mergesort')
		for order_idx in range(len(pos_order)):
			if result_count == k:
				break

			pos_idx = pos_order[order_idx]
			row = pos_idxs[pos_idx]
			neighbors[i, result_count] = row
			sims[i, result_count] = pos_scores[pos_idx]
			result_count += 1

		if result_count < k:
			for row in range(n_rows):
				x_score = 0.0
				y_score = 0.0
				if x_seen[row] == epoch:
					x_score = xdot[row]
				if y_seen[row] == epoch:
					y_score = ydot[row]

				if max(x_score, y_score) == 0.0:
					neighbors[i, result_count] = row
					sims[i, result_count] = 0.0
					result_count += 1
					if result_count == k:
						break

		if result_count < k:
			neg_order = np.argsort(-neg_scores[:n_neg], kind='mergesort')
			for order_idx in range(len(neg_order)):
				if result_count == k:
					break

				neg_idx = neg_order[order_idx]
				row = neg_idxs[neg_idx]
				neighbors[i, result_count] = row
				sims[i, result_count] = neg_scores[neg_idx]
				result_count += 1

	return sims, neighbors

def cosine_similarity_from_seqlets(seqlets, n_neighbors, sign, topn=20, 
	min_k=4, max_k=6, max_gap=15, max_len=15, max_entries=500, 
	alphabet_size=4, backend='auto'):

	X_fwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, 
		min_k, max_k, max_gap, max_len, max_entries, True, sign)

	X_bwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, min_k, max_k, max_gap, 
			max_len, max_entries, False, sign)

	X = sklearn.preprocessing.normalize(X_fwd, norm='l2', axis=1)
	Y = sklearn.preprocessing.normalize(X_bwd, norm='l2', axis=1)

	n, d = X.shape
	k = min(n_neighbors+1, n)
	if backend == 'auto':
		backend = 'inverted'

	if backend == 'numba':
		return _sparse_mm_dot(
			X.data, X.indices.astype('int64'), X.indptr.astype('int64'),
			Y.data, Y.indices.astype('int64'), Y.indptr.astype('int64'), k)
	elif backend == 'inverted':
		X_keys, X_starts, X_ends, X_rows, X_values = _build_inverted_index(X)
		Y_keys, Y_starts, Y_ends, Y_rows, Y_values = _build_inverted_index(Y)
		return _sparse_mm_dot_inverted(
			X.data, X.indices.astype('int64'), X.indptr.astype('int64'),
			X_keys, X_starts, X_ends, X_rows, X_values,
			Y_keys, Y_starts, Y_ends, Y_rows, Y_values, k)
	else:
		raise ValueError("Unrecognized backend: {}".format(backend))


def jaccard_from_seqlets(seqlets, min_overlap, filter_seqlets=None, 
	seqlet_neighbors=None, sparse_backend='max_only'):

	all_fwd_data, all_rev_data = util.get_2d_data_from_patterns(seqlets)

	if filter_seqlets is None:
		filter_seqlets = seqlets
		filters_all_fwd_data = all_fwd_data
		filters_all_rev_data = all_rev_data
	else:
		filters_all_fwd_data, filters_all_rev_data = util.get_2d_data_from_patterns(filter_seqlets)

	if seqlet_neighbors is None:
		seqlet_neighbors = [list(range(len(filter_seqlets)))
							for x in seqlets] 

	#apply the cross metric
	affmat_fwd = jaccard(seqlet_neighbors=seqlet_neighbors, 
		X=filters_all_fwd_data,
		Y=all_fwd_data, min_overlap=min_overlap, func=int, 
		return_sparse=True, sparse_backend=sparse_backend)

	affmat_rev = jaccard(seqlet_neighbors=seqlet_neighbors,
		X=filters_all_rev_data, Y=all_fwd_data,
		min_overlap=min_overlap, func=int,
		return_sparse=True, sparse_backend=sparse_backend) 

	affmat = np.maximum(affmat_fwd, affmat_rev)
	return affmat


def jaccard(X, Y, min_overlap=None, seqlet_neighbors=None, func=np.ceil, 
	return_sparse=False, sparse_backend='max_only'):

	if seqlet_neighbors is None:
		seqlet_neighbors = np.tile(np.arange(X.shape[0]), (Y.shape[0], 1))

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		pad_width = ((0,0), (n_pad, n_pad), (0,0)) 
		Y = np.pad(array=Y, pad_width=pad_width, mode="constant")
	else:
		n_pad = 0 

	len_output = 1 + Y.shape[1] - X.shape[1] 

	X = X.astype('float32')
	Y = Y.astype('float32')

	seqlet_neighbors = seqlet_neighbors.astype('int32')

	if return_sparse == True and sparse_backend == 'max_only':
		scores = np.zeros((Y.shape[0], seqlet_neighbors.shape[1]), dtype='float32')
		_jaccard_max_only(X, Y, seqlet_neighbors, scores)
		return scores
	elif sparse_backend != 'max_only' and sparse_backend != 'tensor':
		raise ValueError("Unrecognized sparse_backend: {}".format(sparse_backend))

	scores = np.zeros((Y.shape[0], seqlet_neighbors.shape[1], len_output), dtype='float32')
	_jaccard(X, Y, seqlet_neighbors, scores)

	if return_sparse == True:
		return scores.max(axis=-1)

	argmaxs = np.argmax(scores, axis=-1)
	idxs = np.arange(seqlet_neighbors.shape[1])
	results = np.zeros((Y.shape[0], seqlet_neighbors.shape[1], 2))
	for i in range(Y.shape[0]):
		results[i, :, 0] = scores[i][idxs, argmaxs[i]]
		results[i, :, 1] = argmaxs[i] - n_pad

	return results

@njit(parallel=True)
def pairwise_jaccard(X, k):
	n, m = X.shape

	jaccards = np.empty((n, k), dtype='float64')
	neighbors = np.empty((n, k), dtype='int32')

	for i in prange(n):
		jaccard_ = np.empty(n, dtype='float64')

		for j in range(n):
			min_sum = 0.0
			max_sum = 0.0

			for l in range(m):
				sign = np.sign(X[i, l]) * np.sign(X[j, l])
				xi = abs(X[i, l])
				xj = abs(X[j, l])

				if xi > xj:
					min_sum += xj * sign
					max_sum += xi
				else:
					min_sum += xi * sign
					max_sum += xj 

			jaccard_[j] = min_sum / max_sum

		idxs = np.argsort(-jaccard_, kind='mergesort')[:k]

		jaccards[i] = jaccard_[idxs]
		neighbors[i] = idxs

	return jaccards, neighbors


@njit('void(float32[:, :, :], float32[:, :, :], int32[:, :], float32[:, :, :])', parallel=True)
def _jaccard(X, Y, neighbors, scores):
	nx, d, m = X.shape
	ny = Y.shape[0]
	len_output = scores.shape[-1]

	for l in prange(ny):
		for idx in range(len_output):
			for i in range(neighbors.shape[1]):
				min_sum = 0.0
				max_sum = 0.0
				neighbor_li = neighbors[l, i]

				for j in range(idx, idx+d):
					j_idx = j - idx

					for k in range(m):
						sign = np.sign(X[neighbor_li, j_idx, k]) * np.sign(Y[l, j, k])

						x = abs(X[neighbor_li, j_idx, k])
						y = abs(Y[l, j, k])

						if y > x:
							min_sum += x * sign
							max_sum += y
						else:
							min_sum += y * sign
							max_sum += x

				scores[l, i, idx] = min_sum / max_sum


@njit('void(float32[:, :, :], float32[:, :, :], int32[:, :], float32[:, :])', parallel=True)
def _jaccard_max_only(X, Y, neighbors, scores):
	nx, d, m = X.shape
	ny = Y.shape[0]
	len_output = 1 + Y.shape[1] - X.shape[1]

	for l in prange(ny):
		for i in range(neighbors.shape[1]):
			best_score = -np.inf
			has_nan = False
			neighbor_li = neighbors[l, i]

			for idx in range(len_output):
				min_sum = 0.0
				max_sum = 0.0

				for j in range(idx, idx+d):
					j_idx = j - idx

					for k in range(m):
						sign = np.sign(X[neighbor_li, j_idx, k]) * np.sign(Y[l, j, k])

						x = abs(X[neighbor_li, j_idx, k])
						y = abs(Y[l, j, k])

						if y > x:
							min_sum += x * sign
							max_sum += y
						else:
							min_sum += y * sign
							max_sum += x

				score = min_sum / max_sum
				if np.isnan(score):
					has_nan = True
				elif score > best_score:
					best_score = score

			if has_nan:
				scores[l, i] = np.nan
			else:
				scores[l, i] = best_score



def pearson_correlation(X, Y, min_overlap=None, func=np.ceil):
	if X.ndim == 2:
		X = X[None, :, :]
	if Y.ndim == 2:
		Y = Y[None, :, :]

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		pad_width = ((0, 0), (n_pad, n_pad), (0, 0)) 
		Y = np.pad(array=Y, pad_width=pad_width, mode="constant")

	n, d, _ = X.shape
	len_output = 1 + Y.shape[1] - d 
	scores = np.zeros((n, len_output))

	for idx in range(len_output):
		Y_ = Y[:, idx:idx+d]

		scores_ = np.dot((X / np.linalg.norm(X)).ravel(),
				  (Y_ / np.linalg.norm(Y_)).ravel()) 
		scores_ = np.nan_to_num(scores_)
		scores[:,idx] = scores_

	argmaxs = np.argmax(scores, axis=1)
	idxs = np.arange(len(scores))
	return np.array([[scores[idxs, argmaxs], argmaxs - n_pad]]).transpose(0, 2, 1)


class NNTsneConditionalProbs():
	def __init__(self, perplexity):
		self.perplexity = perplexity 

	def __call__(self, affinity_mat, nearest_neighbors):
		distmat_nn = np.log((1.0/(0.5*np.maximum(affinity_mat, 0.0000001)))-1)
		distmat_nn = np.maximum(distmat_nn, 0.0) #eliminate tiny neg floats

		# Compute the number of nearest neighbors to find.
		# LvdM uses 3 * perplexity as the number of neighbors.
		# In the event that we have very small # of points
		# set the neighbors to n - 1.
		n_samples = distmat_nn.shape[0]
		k = min(n_samples - 1, int(3. * self.perplexity + 1))

		P = self.tsne_probs_calc(distances_nn=distmat_nn[:,1:(k+1)],
								 neighbors_nn=[row[1:(k+1)] for row in 
											   nearest_neighbors])
		return P

	def tsne_probs_calc(self, distances_nn, neighbors_nn):
		# Compute conditional probabilities such that they approximately match
		# the desired perplexity
		n_samples, k = len(neighbors_nn),len(neighbors_nn[0])
		distances = distances_nn.astype(np.float32, copy=False)
		neighbors = neighbors_nn
		
		conditional_P = sklearn.manifold._utils._binary_search_perplexity(
			distances, self.perplexity, verbose=False)

		eps = 1e-8
		marginal_sum = conditional_P.sum(axis=-1)
		marginal_sum[marginal_sum < eps] = eps

		#normalize the conditional_P to sum to 1 across the rows
		conditional_P = conditional_P / marginal_sum[:,None]

		data = []
		rows = []
		cols = []
		for row_idx,(ps,neigh_row) in enumerate(zip(conditional_P, neighbors)):
			data.extend([p for p,neighbor in zip(ps, neigh_row)])
			rows.extend([row_idx for neighbor in neigh_row])
			cols.extend([neighbor for neighbor in neigh_row])

		P = coo_matrix((data, (rows, cols)),
					   shape=(len(neighbors), len(neighbors)))
		return P
