"""Resolve "collision units": a unit that is the sum of two neighbouring units.

When two neighbouring neurons fire within a fraction of a millisecond, the
summed waveform is a consistent shape of its own. The template-matching peel
explains such an event with ONE template, so one spike is recorded where two
neurons fired, and the final clustering groups these events into a unit of
their own. Both neurons lose those spikes. On hybrid ground truth
(rat 2026-10-15-0/data001, injected pairs 60 um apart) this cost 25-50 % of
the spikes that fall within 0.25 ms of the partner's spike.

`resolve_collision_units` runs after the final merges. It finds each such
unit C and its pair (A, B), gives every C spike to A and to B at the fitted
shifts, and removes C.

A unit C is a collision unit when ALL of these hold, for some neighbours A, B:
  * C's mean waveform is fitted by shift(A) + shift(B) (least squares,
    |shift| <= MAX_SHIFT) with R2 >= R2_MIN and both weights in [W_LO, W_HI];
  * no single shifted neighbour fits C within GAIN of that R2;
  * fewer than PRESENT of C's spikes have a spike of A, or of B, where the
    fit puts it (the pair's spikes are missing exactly at C's events);
  * C has fewer than SIZE x min(n_A, n_B) spikes: a pair cannot collide more
    often than its rarer member fires. This keeps a real neuron whose shape
    happens to resemble two neighbours.

These units form in the FINAL clustering, not the first: a filter on the
first-pass templates found none (2026-11-10), so the step sits here.
"""
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

NEIGHBOUR_UM = 200.0
MAX_SHIFT = 20          # samples
R2_MIN = 0.92
GAIN = 0.08
W_LO, W_HI = 0.5, 1.6
PRESENT = 0.30
TOL = 6                 # samples
SIZE = 0.5
MIN_SPIKES = 30
N_CHANNELS = 24         # strongest channels of C used for the fit


def _shifted(x, shifts):
    """x: (nt, nc). Returns (len(shifts), nt * nc), zero padded."""
    out = np.zeros((len(shifts),) + x.shape, dtype=x.dtype)
    for k, s in enumerate(shifts):
        if s > 0:
            out[k, s:] = x[:-s]
        elif s < 0:
            out[k, :s] = x[-s:]
        else:
            out[k] = x
    return out.reshape(len(shifts), -1)


def _present(tc, tu, shift):
    if tu.size == 0 or tc.size == 0:
        return 0.0
    t = tc + shift
    k = np.searchsorted(tu, t)
    near = ((np.abs(t - tu[np.clip(k - 1, 0, tu.size - 1)]) <= TOL)
            | (np.abs(tu[np.clip(k, 0, tu.size - 1)] - t) <= TOL))
    return float(near.mean())


def find_collision_templates(waveforms, positions, trains, near_misses=None):
    """Indices of collision templates, with the pair that explains each.

    waveforms : (n_units, nt, n_channels) array, one common amplitude unit.
    positions : (n_channels, 2) array, um.
    trains    : list of sorted int64 spike-time arrays (samples), per unit.

    Returns a list of dicts: unit, a, b, shift_a, shift_b, r2, r2_single, n.
    shift_x is where unit x's spike lies relative to C's spike time.
    near_misses : optional list; units whose best pair fit reaches R2 >= 0.80
    but which fail a rule are appended to it with the rule, for diagnosis.
    """
    wf = np.asarray(waveforms, dtype=np.float64)
    n_units = wf.shape[0]
    counts = np.array([t.size for t in trains])
    ptp = wf.max(1) - wf.min(1)                       # (n_units, n_channels)
    main = ptp.argmax(1)
    live = np.flatnonzero((counts >= MIN_SPIKES) & (ptp.max(1) > 0))
    shifts = np.arange(-MAX_SHIFT, MAX_SHIFT + 1)
    found = []
    for c in live:
        dist = np.hypot(*(positions[main[live]] - positions[main[c]]).T)
        nb = live[(dist <= NEIGHBOUR_UM) & (live != c)
                  & (counts[live] * SIZE > counts[c])]
        if nb.size < 2:
            continue
        ch = np.argsort(-ptp[c])[:N_CHANNELS]
        y = wf[c][:, ch].ravel()
        yy = float(y @ y)
        if yy <= 0:
            continue
        X = np.concatenate([_shifted(wf[u][:, ch], shifts) for u in nb])
        owner = np.repeat(np.arange(nb.size), shifts.size)
        shift_of = np.tile(shifts, nb.size)
        g = X @ y
        G = X @ X.T
        dg = np.diag(G).copy()
        r1 = float((np.where(dg > 0, g * g / np.maximum(dg, 1e-12), 0.0)).max() / yy)
        det = dg[:, None] * dg[None, :] - G * G
        ok = ((det > 1e-9 * dg[:, None] * dg[None, :])
              & (owner[:, None] != owner[None, :]))
        safe = np.where(ok, det, 1.0)
        wi = (dg[None, :] * g[:, None] - G * g[None, :]) / safe
        wj = (dg[:, None] * g[None, :] - G * g[:, None]) / safe
        ok &= (wi >= W_LO) & (wi <= W_HI) & (wj >= W_LO) & (wj <= W_HI)
        r2 = np.where(ok, (wi * g[:, None] + wj * g[None, :]) / yy, -1.0)
        i, j = np.unravel_index(int(r2.argmax()), r2.shape)
        best = float(r2[i, j])
        if best < 0.80:
            continue
        a, b = int(nb[owner[i]]), int(nb[owner[j]])
        sa, sb = int(shift_of[i]), int(shift_of[j])
        row = dict(unit=int(c), a=a, b=b, shift_a=sa, shift_b=sb,
                   r2=best, r2_single=r1, n=int(counts[c]))
        if best < R2_MIN:
            why = 'r2'
        elif best - r1 < GAIN:
            why = 'gain'
        elif not (_present(trains[c], trains[a], sa) < PRESENT
                  and _present(trains[c], trains[b], sb) < PRESENT):
            why = 'present'
        else:
            found.append(row)
            continue
        if near_misses is not None:
            near_misses.append(dict(row, failed=why))
    return found


def resolve_collision_units(ops, Wall, clu, st, tF):
    """Give each collision unit's spikes to the two units it is the sum of.

    Wall : (n_units, n_channels, n_pcs) cluster templates.
    clu, st, tF : labels, spike table (column 0 = sample) and features.
    Returns (Wall, clu, st, tF) with the collision units removed, labels
    renumbered and rows sorted by time. A C spike becomes an A spike at
    t + shift_a (the same row, relabelled) and a B spike at t + shift_b (a
    copy of the row); either is skipped where that unit already has a spike
    within TOL samples. The copy keeps the event's features, so its position
    estimate is the collision's, between the two neurons.
    Check-only switch: KILOSORT_NO_COLLISION_UNITS=1 skips the step.
    """
    ops['collision_units'] = []
    if os.environ.get('KILOSORT_NO_COLLISION_UNITS'):
        logger.info('collision units: skipped by KILOSORT_NO_COLLISION_UNITS')
        return Wall, clu, st, tF
    import torch
    n_units = int(Wall.shape[0])
    clu = np.asarray(clu)
    lab = clu.astype(np.int64)
    if lab.size == 0 or int(lab.max()) >= n_units:
        return Wall, clu, st, tF
    times = np.asarray(st)[:, 0].astype(np.int64)
    order = np.argsort(lab, kind='stable')
    edges = np.searchsorted(lab[order], np.arange(n_units + 1))
    rows = [order[edges[u]:edges[u + 1]] for u in range(n_units)]
    trains = [np.sort(times[r]) for r in rows]
    W = ops['wPCA'].to(Wall.device).float()
    wf = torch.einsum('icp,pt->itc', torch.nan_to_num(Wall.float()), W)
    # The fit uses the whitened waveforms as they are. A fit on unwhitened
    # waveforms was tried (2026-11-10): it caught no further known collision
    # unit on the hybrids and resolved 3 more units on recordings without
    # forced collisions, which could not be verified. Not adopted.
    positions = np.stack([np.asarray(ops['xc']), np.asarray(ops['yc'])], axis=1)
    near = []
    found = find_collision_templates(wf.cpu().numpy(), positions, trains,
                                     near_misses=near)
    # A unit named as a member of one pair cannot itself be resolved away.
    members = {f['a'] for f in found} | {f['b'] for f in found}
    near += [dict(f, failed='member') for f in found if f['unit'] in members]
    found = [f for f in found if f['unit'] not in members]
    ops['collision_units'] = found
    ops['collision_near_misses'] = near
    if not found:
        logger.info('collision units: none found')
        return Wall, clu, st, tF

    def has_spike(u, t):
        tu = trains[u]
        if tu.size == 0:
            return np.zeros(t.size, dtype=bool)
        k = np.searchsorted(tu, t)
        return ((np.abs(t - tu[np.clip(k - 1, 0, tu.size - 1)]) <= TOL)
                | (np.abs(tu[np.clip(k, 0, tu.size - 1)] - t) <= TOL))

    st = np.array(st, copy=True)
    new_lab = lab.copy()
    drop = np.zeros(lab.size, dtype=bool)
    copy_rows, copy_lab, copy_shift = [], [], []
    n_added = 0
    for f in found:
        r = rows[f['unit']]
        t = times[r]
        in_a = has_spike(f['a'], t + f['shift_a'])
        in_b = has_spike(f['b'], t + f['shift_b'])
        # the row itself becomes A's spike
        new_lab[r] = f['a']
        st[r, 0] += f['shift_a']
        drop[r[in_a]] = True
        # a copy becomes B's spike
        rb = r[~in_b]
        copy_rows.append(rb)
        copy_lab.append(np.full(rb.size, f['b'], dtype=np.int64))
        copy_shift.append(np.full(rb.size, f['shift_b'] - f['shift_a']))
        n_added += int((~in_a).sum()) + rb.size
    copy_rows = np.concatenate(copy_rows)
    st_copy = st[copy_rows].copy()
    st_copy[:, 0] += np.concatenate(copy_shift)
    keep = np.flatnonzero(~drop)
    take = np.concatenate([keep, copy_rows])
    st = np.concatenate([st[keep], st_copy])
    new_lab = np.concatenate([new_lab[keep], np.concatenate(copy_lab)])
    gone = np.zeros(n_units, dtype=bool)
    gone[[f['unit'] for f in found]] = True
    new_lab = (np.cumsum(~gone) - 1)[new_lab]
    Wall = Wall[torch.as_tensor(~gone, device=Wall.device)]
    by_time = np.argsort(st[:, 0], kind='stable')
    st = st[by_time]
    new_lab = new_lab[by_time].astype(clu.dtype)
    tF = tF[torch.as_tensor(take[by_time], device=tF.device)]
    logger.info(f'collision units: resolved {len(found)} units '
                f'({sum(f["n"] for f in found)} spikes) into '
                f'{n_added} spikes of their pairs')
    return Wall, new_lab, st, tF
