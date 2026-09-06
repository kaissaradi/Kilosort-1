"""Statement-level profile of run_matching's peel loop (the learned pass).

The validation notes attribute the learned pass coarsely (peel_subtract 65.9%,
torch.max 10.3%, rest 23.8%). That "rest" is now the largest single unexplained
block after the peel fusion landed, so this pins every statement.

CUDA-syncs around each statement, so absolute totals are inflated by sync
overhead; the SHARES are what to read. Values are untouched -- this runs the
real loop and returns the real results.
"""
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import conv1d, max_pool1d

sys.path.insert(0, '/home/localadmin/Downloads/Kilosort-1')
from kilosort import template_matching, fused_peel          # noqa: E402
from kilosort.template_matching import _matching_unit_cache  # noqa: E402
from kilosort.run_kilosort import run_kilosort               # noqa: E402

T = defaultdict(float)
N = defaultdict(int)


class Tic:
    __slots__ = ('k', 't')

    def __init__(self, k):
        self.k = k

    def __enter__(self):
        torch.cuda.synchronize()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        torch.cuda.synchronize()
        T[self.k] += time.perf_counter() - self.t
        N[self.k] += 1
        return False


def run_matching_prof(ops, X, U, ctc, device=torch.device('cuda'), unit_cache=None):
    Th = ops['Th_learned']
    nt = ops['nt']
    max_peels = ops['max_peels']
    if unit_cache is None:
        unit_cache = _matching_unit_cache(ops, U)
    s = unit_cache['s']
    Us = unit_cache['Us']
    U_time = unit_cache['U_time']
    W = unit_cache['W']
    trange = unit_cache.get('trange')
    tiwave = unit_cache.get('tiwave')
    if trange is None or tiwave is None:
        trange = torch.arange(-nt, nt + 1, device=device)
        tiwave = torch.arange(-(nt // 2), nt // 2 + 1, device=device)
        unit_cache['trange'] = trange
        unit_cache['tiwave'] = tiwave

    with Tic('01_conv1d'):
        B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt // 2)
    with Tic('02_einsum'):
        B = torch.einsum('ijk, kjl -> il', Us, B)

    NT = int(X.shape[-1])
    peel_cap = min(100000, max(2048, NT // 8))
    st = torch.zeros((peel_cap, 2), dtype=torch.int64, device=device)
    amps = torch.zeros((peel_cap, 1), dtype=torch.float, device=device)
    th_amps = torch.zeros((peel_cap, 1), dtype=torch.float, device=device)
    k = 0
    Xres = X
    Th2 = Th * Th
    ctc_p = ctc.permute(1, 0, 2)

    for t in range(max_peels):
        with Tic('03_max'):
            Cfmax, imax = torch.max(B, 0)
        with Tic('04_TAIL_cond'):
            Cfmax = torch.relu(Cfmax)
            Cfmax.mul_(Cfmax)
            Cfmax[:nt] = 0
            Cfmax[-nt:] = 0
            Cmax = max_pool1d(Cfmax.view(1, 1, -1), (2 * nt + 1), stride=1, padding=(nt))
            cmax = Cmax[0, 0]
            cnd1 = cmax > Th2
            cnd2 = torch.abs(cmax - Cfmax) < 1e-9
            both = cnd1 & cnd2
            xs = torch.nonzero(both)
        with Tic('08_len_sync'):
            nxs = len(xs)
        if nxs == 0:
            break
        with Tic('09_TAIL_store'):
            iX = xs[:, :1]
            iY = imax[iX]
            nsp = len(iX)
        need = k + nsp
        if need > st.shape[0]:
            with Tic('10_grow'):
                new_cap = max(need, st.shape[0] * 2)
                extra = new_cap - st.shape[0]
                st = torch.cat((st, st.new_zeros((extra, 2))), 0)
                amps = torch.cat((amps, amps.new_zeros((extra, 1))), 0)
                th_amps = torch.cat((th_amps, th_amps.new_zeros((extra, 1))), 0)
        with Tic('09_TAIL_store'):
            st[k:k + nsp, 0] = iX[:, 0]
            st[k:k + nsp, 1] = iY[:, 0]
            amps[k:k + nsp] = B[iY, iX] * s[iY]
            amp = amps[k:k + nsp]
            th_amps[k:k + nsp] = cmax[iX[:, 0], None] ** .5
        k += nsp
        with Tic('12_peel_subtract'):
            fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc, ctc_p,
                                     tiwave, trange, nt, 2)

    st = st[:k]
    amps = amps[:k]
    th_amps = th_amps[:k]
    return st, amps, th_amps, Xres


template_matching.run_matching = run_matching_prof

prod = np.load(sys.argv[1], allow_pickle=True).item()
settings = dict(prod['settings'])
data = sys.argv[2]
settings['filename'] = data
settings['data_dir'] = None
settings.pop('probe', None)

t0 = time.time()
run_kilosort(settings=settings, probe=prod['probe'], filename=data,
             results_dir=sys.argv[3], do_CAR=prod['do_CAR'],
             device=torch.device('cuda'), save_preprocessed_copy=False,
             clear_cache=False)
print(f'\nTOTAL_WALL_SECONDS {time.time() - t0:.2f}')

tot = sum(T.values())
print(f'\n=== run_matching statement profile (synced; total {tot:.1f}s) ===')
print(f'{"statement":<20} {"s":>9} {"share":>7} {"calls":>8} {"us/call":>9}')
for kk in sorted(T):
    print(f'{kk:<20} {T[kk]:>9.2f} {100*T[kk]/tot:>6.1f}% {N[kk]:>8} '
          f'{1e6*T[kk]/max(N[kk],1):>9.1f}')
