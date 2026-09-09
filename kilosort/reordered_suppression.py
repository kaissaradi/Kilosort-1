"""Experimental exact universal suppression, temporal maximum first.

For time-independent neighbor sets, max_time(max_neighbor(A)) equals
max_neighbor(max_time(A)). Zero the batch edges BEFORE either temporal
maximum. Only positions with the original A > threshold need neighbor reads.

This eager implementation is a candidate for GPU measurement, not a claimed
speedup. It keeps score/argmax arithmetic unchanged and bounds the sparse
gather allocation. It needs one extra full-size float buffer for temporal
maxima, plus candidate indices and chunk-sized gather temporaries.
"""
import torch
from torch.nn.functional import max_pool1d


def mask(scores, neighbors, nt, radius, threshold, *, workspace=None,
         chunk_size=16384):
    """Match spatial-max -> edge-zero -> time-max, without dense spatial max.

    neighbors has shape (neighbor_count, n_templates). Duplicate neighbors and
    graphs without self-edges are supported. The full mask preserves the
    original filter-major/time-major nonzero order. workspace is overwritten;
    scores are never modified.
    """
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError('scores must be a nonempty two-dimensional time series')
    if neighbors.ndim != 2 or neighbors.shape[0] == 0 \
            or neighbors.shape[1] != scores.shape[0]:
        raise ValueError('neighbors must have shape (n_neighbors, n_templates)')
    if nt < 1 or radius < 0:
        raise ValueError('nt must be positive and radius nonnegative')
    if workspace is None:
        workspace = scores.clone()
    else:
        if (workspace.shape != scores.shape or workspace.dtype != scores.dtype
                or workspace.device != scores.device):
            raise ValueError('workspace must match scores')
        if workspace.untyped_storage().data_ptr() == scores.untyped_storage().data_ptr():
            raise ValueError('workspace must not share storage with scores')
        workspace.copy_(scores)
    workspace[:, :nt] = 0
    workspace[:, -nt:] = 0
    temporal = max_pool1d(workspace.unsqueeze(0), 2 * radius + 1,
                          stride=1, padding=radius).squeeze(0)
    candidates = (scores > threshold).nonzero()
    out = torch.zeros_like(scores, dtype=torch.bool)
    for start in range(0, candidates.shape[0], chunk_size):
        rows, times = candidates[start:start + chunk_size].unbind(1)
        maxima = temporal[neighbors[:, rows], times.unsqueeze(0)].max(0).values
        out[rows, times] = maxima == scores[rows, times]
    return out
